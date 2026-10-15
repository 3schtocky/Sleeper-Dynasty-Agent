"""`dynasty-agent chat`: ask about the league in plain English.

Each question goes: the local model picks a tool (llm.route), Python runs it
on the real league (tools.run_tool) and prints the full numbers block, then
the model streams a 1-3 sentence take on that block (llm.stream). The take is
released a sentence at a time and every sentence passes the grounding check
first, so a number the model made up is never shown. A stats line under each
answer shows the model, its writing speed and the tool used.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import re
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from dynasty_agent import explain, grounding, knowledge, llm, refresh, tools
from dynasty_agent.errors import AgentError

ROUTE_SYSTEM = (
    "You are a dynasty fantasy football assistant for the user's Sleeper league. Answer every question about the "
    "user's team by calling exactly one tool; never answer from memory and never guess a number. The current week, "
    "the season and the user's roster are already known to the tools, so never ask for them. For a trade, send is "
    "what the user gives away and receive is what the user gets: \"I'm offered X for Y\" means receive X, send Y. "
    "If the question is not about the user's fantasy team, reply in one short sentence without a tool and mention "
    "what you can help with: lineups, trades, team outlook, waiver pickups, draft picks, taxi and IR."
)

CHAT_SYSTEM = (
    "You are a friendly, knowledgeable dynasty fantasy football assistant chatting with the user about their Sleeper "
    "league. Talk like a person: warm and direct. Answer first in the opening sentence, then give the two or three "
    "most useful reasons, in at most five sentences and one short paragraph. Never repeat a point or contradict the "
    "reference. End with at most one short question, only when it would help. You can discuss anything fantasy football: strategy, rules, scoring, positions, dynasty philosophy, "
    "buying and selling, how to think about a decision. You do not have live data in this reply, so never state a "
    "specific player's stat, value, projection, injury or roster status, and never give a number you are not sure "
    "of; if the user needs one, say you can check it for them and ask which player or question. Do not make a "
    "trade, lineup or waiver recommendation about a named player without that check. If the user asks about "
    "something that is not fantasy football, say so kindly in one sentence and steer back. Write plain text only: "
    "no markdown, no bold, no headings, no bullet symbols, no emoji."
)
TOPIC_SYSTEM = (
    "Classify the user's message with exactly one word. fantasy: about fantasy football or the NFL: players, "
    "positions (QB, RB, WR, TE), rules and terms (flex, PPR, superflex, waivers, taxi, FAAB), aging, injuries, lineups, trades, drafts, picks, dynasty or strategy. When unsure "
    "between fantasy and other, answer fantasy. smalltalk: a greeting, thanks or a goodbye. other: anything else, "
    "including trivia, math, poems, geography and other sports. Reply with only the word."
)
OFF_TOPIC_REPLY = (
    "That one's outside my lane. I'm here for your fantasy football team: lineups, trades, waivers, picks, "
    "taxi and IR, or strategy talk. What would you like to dig into?"
)

TAKE_SYSTEM = (
    "You are a friendly dynasty fantasy football assistant talking to the user. They have just been shown the full "
    "numbers from a tool. The tool result has a list called facts: plain statements already checked, your only source. "
    "Reply in natural, conversational sentences, two to four in all, in plain text. Start with your direct answer "
    "to their question, taken from the first fact or two (a warning or note comes first and must be said in plain "
    "words), then give the one or two facts that matter most in your own friendly words. Never say the word "
    "facts. Copy any number character for character from the facts (write 16.5, +17.7 not 17, never 16 or about "
    "16), at most three numbers in all and none twice. Never calculate, round or invent a number, never add a reason the facts don't give, and "
    "never mention a player or pick they don't name. Trade-value points are not dollars; a bid is what you pay, "
    "not what a player is worth. Recommend a move (a trade, a pickup, a taxi or IR move) only when the facts "
    "clearly support it; if they are close, mixed or unclear, say so plainly. Only when you recommend making a move, add that the user "
    "makes it in the Sleeper app. Do not ask questions. No markdown, lists or emoji."
)

EXPLAIN_SYSTEM = (
    "The user asked how a number is calculated or what a term means. You are given the calculation as ordered "
    "steps, already worked out. Walk them through in plain, friendly language in the same order, in at most six "
    "sentences: say what goes in, what each step does and why, and land on the result. Only use numbers that "
    "appear in the steps, copied character for character; never calculate, round or invent a number, and never add "
    "a reason the steps don't give. Write a multiplication as 'times' or 'x', and never put a minus sign on a number "
    "the steps don't show with one. Market and FantasyCalc values are trade-value points, not dollars. If the steps "
    "list things to keep in mind, mention the most important one, and finish your last sentence."
)

# How many earlier answers a follow-up can refer back to.
MEMORY_TURNS = 3
EXPLAIN_HINT = "Ask about any step and I'll go deeper, for example: why does age matter for a running back?"

HELP = """Ask in plain English, for example:
  Who should I start this week?
  Should I trade Jonah Coleman and my 2028 2nd for a 2027 1st?
  Am I a contender?        Who should I pick up?        How much should I bid on Trey Benson?
  Should I sell my first?  Anyone I should put on taxi?
Ask how it works, too:
  How did you get Trey Benson's win-now value?     Why is his 3yr lower?     What is situation score?
  Why? (after a trade)     How is that win probability calculated?     How did you size that bid?
  How did you decide I'm a contender?     Why is that a sell?
Commands:
  /raw          what the model was given for the last answer (tool, arguments, summary)
  /stats off    hide the stats line (/stats on to show it)
  /help         this list
  /quit         leave (or Ctrl-D)
Moves are yours to make in the Sleeper app: Sleeper's API is read-only."""

LOADING = "loading model..."


def format_stats(stats: llm.Stats | None, tool: str | None, width: int) -> str:
    """The right-aligned line under each answer: model · exact tok/s · first
    words (or "loading model..." after a cold start) · tool used."""
    if stats is None:
        return ""
    parts = [stats.model]
    if stats.tokens_per_second:
        parts.append(f"{stats.tokens_per_second:.1f} tok/s")
    if stats.loaded_model:
        parts.append(f"{LOADING} {stats.load_duration_ns / 1e9:.0f}s")
    elif stats.first_token_s is not None:
        parts.append(f"first words {stats.first_token_s:.1f}s")
    parts.append(f"tool: {tool or 'none'}")
    line = " · ".join(parts)
    return line.rjust(width) if len(line) < width else line


def _sentence_end(buffer: str) -> int | None:
    """Index just past the first finished sentence in buffer, or None. A
    period inside a number ("78.1") or after an initial ("T.J.") doesn't end
    one; ".", "!" or "?" followed by a space does."""
    for i in range(len(buffer) - 1):
        ch = buffer[i]
        if ch not in ".!?" or buffer[i + 1] != " ":
            continue
        initial = ch == "." and i > 0 and buffer[i - 1].isupper() and (i < 2 or buffer[i - 2] in " .")
        if not initial:
            return i + 1
    return None


def sentences(pieces: Iterable[str]) -> Iterator[str]:
    """Regroup streamed text into whole sentences, so each can be checked
    before it's shown."""
    buffer = ""
    for piece in pieces:
        buffer += piece
        cut = _sentence_end(buffer)
        while cut is not None:
            yield buffer[:cut].strip()
            buffer = buffer[cut:]
            cut = _sentence_end(buffer)
    if buffer.strip():
        yield buffer.strip()


# A follow-up that only makes sense with the last question in view: "what if I add Lamb?", "and him?", "what about Puka instead".
_FOLLOW_UP = re.compile(r"^(and|but|what if|what about|how about|ok(ay)?,? (and|what|how)|instead|also)\b|\b(him|her|them|that (trade|player|pick|deal)|instead)\b", re.I)
# A "player" the router invented from a description ("28 year old RB", "a young WR"): a strategy question, not a lookup.
_GENERIC_PLAYER = re.compile(r"\d|\b(year|old|young|veteran|rookie|aging|a|an|some|any|guy|player)\b", re.I)
# Earlier turns the talk path can see.
TALK_TURNS = 6

# The model sometimes writes markdown anyway; a terminal shows it literally.
_PLAIN = re.compile(r"\*+|__|`|^#+\s*", re.M)


def _is_generic(call: dict) -> bool:
    """True when a tool call names a description instead of a player, so the question is strategy talk."""
    args = call.get("arguments") or {}
    names = [*(args.get("send_players") or []), *(args.get("receive_players") or []), args.get("player") or ""]
    return any(isinstance(n, str) and not tools._LOOKS_LIKE_PICK.search(n) and _GENERIC_PLAYER.search(n) for n in names)


@dataclass
class Turn:
    question: str
    tool: str | None = None
    arguments: dict = field(default_factory=dict)
    compact: dict = field(default_factory=dict)
    reply: str = ""
    clarification: bool = False
    result: tools.ToolResult | None = None  # what the user was shown, for a follow-up to explain


class ChatSession:
    def __init__(self, conn: sqlite3.Connection, client: llm.OllamaClient, out=None, width: int | None = None):
        self.conn = conn
        self.client = client
        self.out = out or sys.stdout
        self.width = width or shutil.get_terminal_size((100, 20)).columns
        self.show_stats = True
        self.history: list[Turn] = []
        self.last_dropped: list[str] = []

    def say(self, text: str = "", end: str = "\n") -> None:
        self.out.write(text + end)
        self.out.flush()

    def _dim(self, text: str) -> str:
        return f"\033[2m{text}\033[0m" if getattr(self.out, "isatty", lambda: False)() else text

    def _recent_results(self) -> list[tools.ToolResult]:
        return [t.result for t in self.history[-MEMORY_TURNS:] if t.result is not None]

    def _router_messages(self, question: str) -> list[dict]:
        """Each question is routed on its own, the way the bake-off scored
        16/16: earlier answers in the context made the model imitate them
        instead of calling a tool. The one exception is an answer to a
        question the agent just asked ("Which Kenneth Walker?" "the RB"),
        which only makes sense with that exchange in view."""
        messages = [{"role": "system", "content": ROUTE_SYSTEM}]
        last = self.history[-1] if self.history else None
        if last is not None and last.clarification:
            messages.append({"role": "user", "content": last.question})
            messages.append({"role": "assistant", "content": last.reply})
        elif last is not None and last.tool and _FOLLOW_UP.search(question):
            question = f"Earlier question: {last.question}\nFollow-up: {question}"
        messages.append({"role": "user", "content": question})
        return messages

    def ask(self, question: str) -> Turn:
        """Answer one question end to end, writing to self.out."""
        turn = Turn(question)
        recent = self._recent_results()
        last = self.history[-1] if self.history else None
        answering_explain_question = last is not None and last.tool == tools.EXPLAIN and last.clarification
        if answering_explain_question or explain.detect(question, has_recent=bool(recent)):
            # Python recognizes these itself: the model routes only the six real tools, and routes them best alone.
            topic = f"{last.question} {question}" if answering_explain_question else question
            return self._answer(turn, {"name": tools.EXPLAIN, "arguments": {"topic": topic}}, None, recent)

        calls, text, route_stats = self.client.route(self._router_messages(question), tools.TOOLS)
        if not calls and last is not None and last.clarification and last.tool:
            # The reply to a question the agent asked belongs to that question: say so in one message and route again.
            merged = f"{last.question} ({question})"
            calls, text, route_stats = self.client.route(
                [{"role": "system", "content": ROUTE_SYSTEM}, {"role": "user", "content": merged}], tools.TOOLS)
        if not calls or _is_generic(calls[0]):
            return self._talk(turn)
        return self._answer(turn, calls[0], route_stats, recent)

    def _answer(self, turn: Turn, call: dict, route_stats: llm.Stats | None, recent: list[tools.ToolResult]) -> Turn:
        question = turn.question
        turn.tool, turn.arguments = call["name"], call["arguments"]
        result = tools.run_tool(self.conn, call["name"], call["arguments"], recent=recent)
        if result.clarification:
            turn.reply, turn.clarification = result.clarification, True
            self.say(result.clarification)
            self._stats(route_stats, turn.tool)
            self.history.append(turn)
            return turn

        turn.compact, turn.result = result.compact, result
        self.say(result.numbers)
        if result.name == "set_lineup":
            self.say(self._dim(tools.win_caveat(result.data)))
        self.say()
        if not result.take:
            # Python's own text already answers (a definition): nothing for the model to add.
            turn.reply = result.numbers
            self._stats(route_stats, turn.tool)
            self.history.append(turn)
            return turn
        turn.reply = self._take(question, call, result, recent)
        if result.name == tools.EXPLAIN:
            self.say(self._dim(EXPLAIN_HINT))
        self.history.append(turn)
        return turn

    def _take(self, question: str, call: dict, result: tools.ToolResult, recent: list[tools.ToolResult]) -> str:
        """Stream the model's take, one checked sentence at a time. An
        explanation may also quote numbers from the recent answers it is
        explaining, so those count as grounded for that turn."""
        payload = json.dumps(result.compact)
        explaining = result.name == "explain"
        messages = [
            {"role": "system", "content": EXPLAIN_SYSTEM if explaining else TAKE_SYSTEM},
            {"role": "user", "content": question},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": call["name"], "arguments": call["arguments"]}}]},
            {"role": "tool", "content": payload, "tool_name": call["name"]},
        ]
        earlier = [r.numbers for r in recent] if explaining else []
        allowed = grounding.allowed_numbers(result.numbers, payload, question, *earlier)
        shown, stats = self._stream_checked(messages, allowed, llm.EXPLAIN_OPTIONS if explaining else None)
        self._stats(stats, call["name"])
        return shown

    def _talk(self, turn: Turn) -> Turn:
        """No tool fits: chat about fantasy football in the model's own words.
        Plain counts and ages pass; a stat or value must come from the question,
        the vetted reference or the recent conversation (grounding.unsupported_claims)."""
        if not self._on_topic(turn.question, self.history[-1].question if self.history else None):
            turn.reply = OFF_TOPIC_REPLY
            self.say(OFF_TOPIC_REPLY)
            self.history.append(turn)
            return turn
        reference = knowledge.reference(self.conn)
        system = f"{CHAT_SYSTEM}\n\nReference (true; use it and never contradict it):\n{reference}"
        earlier = [t for t in self.history[-TALK_TURNS:] if t.reply and not t.clarification]
        messages = [{"role": "system", "content": system}]
        for t in earlier:
            messages += [{"role": "user", "content": t.question}, {"role": "assistant", "content": t.reply}]
        messages.append({"role": "user", "content": turn.question})
        # Numbers the user was already shown, or the model already said, stay quotable.
        known = [r.numbers for r in self._recent_results()] + [t.reply for t in earlier]
        reply, stats = self._stream_checked(
            messages, grounding.allowed_numbers(turn.question, reference, *known), llm.CHAT_OPTIONS,
            check=grounding.unsupported_claims)
        turn.reply = reply or "I can help with lineups, trades, team outlook, waivers, draft picks, taxi and IR."
        if not reply:
            self.say(turn.reply)
        self._stats(stats, None)
        self.history.append(turn)
        return turn

    def _on_topic(self, question: str, previous: str | None = None) -> bool:
        """A one-word model call, far more reliable on a 4B model than an
        instruction buried in the chat prompt. Anything but "other" passes.
        The previous question is shown so a follow-up ("is that a big gap?")
        is judged in context."""
        shown = f"Previous message: {previous}\nMessage to classify: {question}" if previous else question
        messages = [{"role": "system", "content": TOPIC_SYSTEM}, {"role": "user", "content": shown}]
        text = "".join(e for e in self.client.stream(messages, options=llm.TOPIC_OPTIONS) if isinstance(e, str))
        return "other" not in text.lower()

    def _stream_checked(self, messages: list[dict], allowed: set[str], options: dict | None,
                        check=grounding.ungrounded) -> tuple[str, llm.Stats | None]:
        """Stream a reply one checked sentence at a time; a sentence `check` flags is left out."""
        stats: list[llm.Stats] = []

        def text_only(events):
            for event in events:
                if isinstance(event, llm.Stats):
                    stats.append(event)
                else:
                    yield event

        shown = []
        self.last_dropped = []
        for sentence in sentences(text_only(self.client.stream(messages, options=options))):
            sentence = _PLAIN.sub("", sentence)
            bad = check(sentence, allowed)
            if bad:
                # Skip just this sentence; the rest may be fine. The user sees nothing about it, /raw lists it.
                self.last_dropped.append(f"{sentence.strip()} [numbers not allowed: {', '.join(sorted(bad))}]")
                continue
            shown.append(sentence)
            self.say(sentence, end=" ")
        if shown:
            self.say()
        return " ".join(shown), (stats[-1] if stats else None)

    def _stats(self, stats: llm.Stats | None, tool: str | None) -> None:
        if self.show_stats and stats is not None:
            self.say(self._dim(format_stats(stats, tool, self.width)))

    def command(self, line: str) -> bool:
        """Handle a /command. False means quit."""
        word, _, rest = line[1:].partition(" ")
        if word in ("quit", "exit", "q"):
            return False
        if word == "help":
            self.say(HELP)
        elif word == "stats":
            self.show_stats = rest.strip().lower() != "off"
            self.say(f"Stats line {'on' if self.show_stats else 'off'}.")
        elif word == "raw":
            last = next((t for t in reversed(self.history) if t.tool), None)
            if last is None:
                self.say("Nothing yet: ask a question first.")
            else:
                self.say(json.dumps({"tool": last.tool, "arguments": last.arguments, "summary_given_to_model": last.compact,
                                     "sentences_left_out": self.last_dropped}, indent=2))
        else:
            self.say(f"Unknown command /{word}. Try /help.")
        return True


def run(conn: sqlite3.Connection, do_refresh: bool = True) -> None:
    """Start a chat session: check Ollama, refresh the data while the model
    loads, then answer questions until /quit."""
    client = llm.OllamaClient()
    client.ensure_ready()  # AgentError with the fix if Ollama or the model is missing

    warm_error: list[Exception] = []
    warmer = threading.Thread(target=lambda: _warm(client, warm_error), daemon=True)
    warmer.start()
    if do_refresh:
        print("Refreshing league data while the model loads...", flush=True)
        results = refresh.run(conn)
        failed = [r for r in results if not r["ok"]]
        print(f"Data: {len(results) - len(failed)} of {len(results)} refresh steps ok."
              + "".join(f"\n  FAILED {r['step']}: {r['detail']}" for r in failed))
    warmer.join()
    if warm_error:
        raise warm_error[0]

    try:
        import readline  # noqa: F401  (arrow keys and history for input(), stdlib)
    except ImportError:
        pass
    session = ChatSession(conn, client)
    print(f"\nTalking to {client.model} on this Mac. Ask about your league; /help for examples, /quit to leave.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not session.command(line):
                break
            continue
        try:
            session.ask(line)
        except KeyboardInterrupt:
            print("\n(stopped)")
        except AgentError as e:
            print(f"{e}")
        print()
    client.close()


def _warm(client: llm.OllamaClient, errors: list) -> None:
    try:
        client.warm()
    except Exception as e:  # reported on the main thread
        errors.append(e)
