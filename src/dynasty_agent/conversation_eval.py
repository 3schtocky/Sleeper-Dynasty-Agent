"""The chat's conversation evaluation: scripted multi-turn talks on the real
league, each step checked by plain rules.

chat_eval.py scores one cold question at a time (did it reach the right
tool). This scores what a person actually experiences: follow-ups that need
the earlier answer, strategy talk with no tool, off-topic redirects, a trade
the user cannot make, and answers that must agree with the facts Python
wrote. Run it live with `dynasty-agent chat-eval --conversations`; it needs
Ollama and the real database, so it is a measurement, not a unit test (the
rules themselves are tested offline in tests/test_conversation_eval.py).

A check returns None when the step is fine or a short reason when it is not.
The scenarios name real players on the real roster; edit them if the
roster changes.
"""

from __future__ import annotations

import io
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from dynasty_agent import chat, grounding, knowledge

Check = Callable[["chat.Turn", str, object], "str | None"]  # (turn, what the user saw for this step, database)


def tool_is(name: str | None) -> Check:
    def check(turn, shown, conn):
        return None if turn.tool == name else f"used tool {turn.tool}, wanted {name}"
    return check


def asks_back(*words: str) -> Check:
    """The agent asked a question (no verdict) and the question names these words."""
    def check(turn, shown, conn):
        if not turn.clarification:
            return "gave an answer instead of asking"
        missing = [w for w in words if w.lower() not in turn.reply.lower()]
        return f"question does not mention {missing}" if missing else None
    return check


def args_include(text: str) -> Check:
    """The tool call carried this text (the follow-up kept its context)."""
    def check(turn, shown, conn):
        return None if text.lower() in str(turn.arguments).lower() else f"tool arguments {turn.arguments} lack {text!r}"
    return check


def answers() -> Check:
    def check(turn, shown, conn):
        return "asked a question instead of answering" if turn.clarification else None
    return check


def says(*alternatives: str) -> Check:
    """The reply contains at least one of these (case-insensitive)."""
    def check(turn, shown, conn):
        text = turn.reply.lower()
        return None if any(a.lower() in text for a in alternatives) else f"reply says none of {list(alternatives)}"
    return check


def never_says(*phrases: str) -> Check:
    def check(turn, shown, conn):
        text = turn.reply.lower()
        hit = [p for p in phrases if p.lower() in text]
        return f"reply says {hit}" if hit else None
    return check


def redirects() -> Check:
    def check(turn, shown, conn):
        return None if turn.reply == chat.OFF_TOPIC_REPLY else "did not redirect"
    return check


def does_not_redirect() -> Check:
    def check(turn, shown, conn):
        return "redirected a fantasy football or small-talk message" if turn.reply == chat.OFF_TOPIC_REPLY else None
    return check


def plain_and_short(max_sentences: int = 8) -> Check:
    """A person-sized reply: non-empty, plain text, not a wall."""
    def check(turn, shown, conn):
        if not turn.reply.strip():
            return "empty reply"
        if re.search(r"\*|^#|^\s*[-•]\s|[\U0001F300-\U0001FAFF]", turn.reply, re.M):
            return "markdown, a list or an emoji in the reply"
        n = len(re.findall(r"[.!?](?:\s|$)", turn.reply))
        return f"{n} sentences, more than {max_sentences}" if n > max_sentences else None
    return check


def grounded() -> Check:
    """Re-check the reply with the same rule the chat enforces, so removing that guard fails the eval."""
    def check(turn, shown, conn):
        if turn.clarification:
            return None
        if turn.result is not None:
            allowed = grounding.allowed_numbers(turn.result.numbers, str(turn.result.compact), turn.question)
            bad = grounding.ungrounded(turn.reply, allowed)
        else:
            allowed = grounding.allowed_numbers(turn.question, knowledge.reference(conn))
            bad = grounding.unsupported_claims(turn.reply, allowed)
        return f"quotes numbers it was not given: {sorted(bad)}" if bad else None
    return check


def agrees_with_facts() -> Check:
    """A pick or trade answer must not contradict the call Python wrote."""
    def check(turn, shown, conn):
        facts = " ".join(turn.compact.get("facts", [])).lower()
        reply = turn.reply.lower()
        if "sell (worth trading away now)" in facts and "hold" in reply and "sell" not in reply:
            return "facts say SELL, reply says hold"
        if "favor the other team on both" in facts and re.search(r"good deal|make (the|this) trade", reply):
            return "facts say the other team wins, reply calls it good"
        if "favor you on both" in facts and re.search(r"bad deal|don't make|do not make|pass on", reply):
            return "facts say it favors you, reply calls it bad"
        return None
    return check


@dataclass
class Step:
    question: str
    checks: list[Check] = field(default_factory=list)


@dataclass
class Scenario:
    name: str
    steps: list[Step]


# Real players on the recorded roster (QB Stafford, WR Lamb/Rice/Odunze, RB Johnson) and on others (Puka Nacua,
# Jonah Coleman is on DannyBoss).
SCENARIOS: list[Scenario] = [
    Scenario("strategy talk", [
        Step("How should I think about trading a 28 year old RB for a young WR?",
             [tool_is(None), answers(), plain_and_short(), grounded()]),
        Step("Why is age such a big deal for running backs?", [tool_is(None), plain_and_short(), grounded()]),
    ]),
    Scenario("league facts", [
        Step("Is my league superflex?", [tool_is(None), says("one quarterback", "one qb", "1qb", "not superflex", "only one"),
                                         never_says("is a superflex league")]),
        Step("What does PPR do to running backs?", [tool_is(None), plain_and_short(), grounded()]),
    ]),
    Scenario("follow-up memory", [
        Step("Should I trade Rome Odunze for Puka Nacua?", [tool_is("evaluate_trade"), grounded(), agrees_with_facts()]),
        Step("what if I add my 2028 2nd?", [tool_is("evaluate_trade"), args_include("2028"), args_include("Puka"), grounded()]),
        Step("Is that a big gap?", [tool_is(None), does_not_redirect(), plain_and_short()]),
        Step("Why?", [tool_is("explain")]),
    ]),
    Scenario("a trade the user cannot make", [
        Step("Should I trade Jonah Coleman and my 2028 2nd for a 2027 1st?", [asks_back("Jonah Coleman")]),
        Step("I meant I'd get Jonah Coleman for my 2028 2nd", [tool_is("evaluate_trade"), answers(), grounded()]),
    ]),
    Scenario("a reversed trade is still read right", [
        Step("Would you trade my 2027 1st for Puka Nacua?", [tool_is("evaluate_trade"), answers(), agrees_with_facts()]),
    ]),
    Scenario("off topic and small talk", [
        Step("What's the capital of France?", [redirects()]),
        Step("Write me a poem about autumn", [redirects()]),
        Step("hey there", [does_not_redirect(), plain_and_short(4)]),
        Step("thanks!", [does_not_redirect(), plain_and_short(4)]),
    ]),
    Scenario("tool answers agree with their facts", [
        Step("What should I do with my draft picks?", [tool_is("pick_advice"), grounded(), agrees_with_facts(), plain_and_short(6)]),
        Step("Who should I start this week?", [tool_is("set_lineup"), grounded(), plain_and_short(6)]),
        Step("Am I a contender?", [tool_is("my_team"), grounded(), plain_and_short(6)]),
        Step("Anyone I should put on taxi?", [tool_is("taxi_plan"), plain_and_short(6)]),
    ]),
]


# Tougher scenarios, added after the first 19 steps all passed: vague asks, pronouns, pushback, mixed topics.
HARD_SCENARIOS: list[Scenario] = [
    Scenario("pushback", [
        Step("Who should I start this week?", [tool_is("set_lineup"), grounded()]),
        Step("are you sure?", [does_not_redirect(), plain_and_short(6), grounded()]),
        Step("why not start Lamar Jackson instead?", [does_not_redirect(), plain_and_short(6), grounded()]),
    ]),
    Scenario("vague asks", [
        Step("help", [does_not_redirect(), plain_and_short(6)]),
        Step("I need help with my team", [does_not_redirect(), plain_and_short(6)]),
        Step("what should I do", [does_not_redirect(), plain_and_short(6)]),
    ]),
    Scenario("needs the roster", [
        Step("I need a WR upgrade, who should I target?", [does_not_redirect(), plain_and_short(8), grounded()]),
        Step("how deep is my receiver room?", [does_not_redirect(), plain_and_short(8), grounded()]),
    ]),
    Scenario("pronouns after a tool answer", [
        Step("Should I trade Rome Odunze for Puka Nacua?", [tool_is("evaluate_trade"), grounded()]),
        Step("is he worth it?", [does_not_redirect(), plain_and_short(6), grounded()]),
        Step("what about Jaxon Smith-Njigba instead?", [tool_is("evaluate_trade"), args_include("Jaxon"), grounded()]),
    ]),
    Scenario("mixed topics", [
        Step("Am I a contender?", [tool_is("my_team"), grounded()]),
        Step("what's the best pizza topping", [redirects()]),
        Step("ok back to football, should I be selling picks then?", [does_not_redirect(), plain_and_short(8), grounded()]),
    ]),
    Scenario("rules and terms", [
        Step("what's FAAB?", [does_not_redirect(), plain_and_short(6), grounded(), never_says("real money is")]),
        Step("how many taxi spots do I have?", [says("3", "three"), plain_and_short(6)]),
        Step("when is my trade deadline?", [says("week 9", "9"), plain_and_short(6)]),
    ]),
    Scenario("an opinion about a named player", [
        Step("is Puka Nacua a buy or a sell?", [does_not_redirect(), plain_and_short(8), grounded()]),
        Step("Tell me about Jaxon Smith-Njigba", [does_not_redirect(), never_says("rookie", "breakout", "season", "years old", "career"), grounded()]),
        Step("will CeeDee Lamb score 25 points this week?", [does_not_redirect(), says("can't", "cannot", "can’t", "don't have", "don’t have", "unable", "no way to"),
                                                  never_says("yes, he will", "he will score", "he'll score"), grounded()]),
    ]),
    Scenario("asking it to invent", [
        Step("just guess what my win probability is, don't look anything up", [does_not_redirect(), grounded()]),
        Step("what's Rashee Rice's exact trade value?", [does_not_redirect(), grounded()]),
    ]),
]


# The trade-idea finder and the player check (Python-detected, so the router never sees them).
FINDER_SCENARIOS: list[Scenario] = [
    Scenario("trade ideas", [
        Step("help me find good trade ideas, I need some new WR's in my lineup",
             [tool_is("find_trades"), answers(), grounded(), never_says("make the trade", "make that trade", "make this trade"),
              plain_and_short(6)]),
        Step("what about running back?", [tool_is("find_trades"), grounded()]),
        Step("I need a tight end", [tool_is("find_trades"), grounded()]),
    ]),
    Scenario("player check", [
        Step("Is Puka Nacua a buy or a sell?", [tool_is("player_check"), grounded(), never_says("pickup"), plain_and_short(6)]),
        Step("what about Rashee Rice?", [tool_is("player_check"), grounded()]),
        Step("What's CeeDee Lamb's trade value?", [tool_is("player_check"), grounded()]),
    ]),
]


def run(conn, client, scenarios: list[Scenario] | None = None) -> dict:
    """Play every scenario on a fresh session; return per-step results."""
    results, started = [], time.monotonic()
    for scenario in scenarios if scenarios is not None else SCENARIOS:
        out = io.StringIO()
        session = chat.ChatSession(conn, client, out=out, width=100)
        for step in scenario.steps:
            before = out.tell()
            turn = session.ask(step.question)
            shown = out.getvalue()[before:]
            failures = [f for f in (check(turn, shown, conn) for check in step.checks) if f]
            results.append({"scenario": scenario.name, "question": step.question, "ok": not failures,
                            "why": "; ".join(failures), "reply": turn.reply})
    return {"passed": sum(r["ok"] for r in results), "total": len(results),
            "seconds": round(time.monotonic() - started, 1), "results": results}

