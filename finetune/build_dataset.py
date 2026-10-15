"""Build the fine-tuning dataset for the chat model from a synthetic league (see make_synthetic_league.py).

Four jobs, each in the exact message format the chat sends the model, so training matches inference:
  route  the router: a question to one of six tool calls, or a short no-tool reply when it is not about the team
  topic  the one-word classifier: fantasy / smalltalk / other
  take   a short friendly reply from the facts a tool returned
  talk   free fantasy-football conversation

Routing and topic answers are right by construction (the sides of every trade come from who owns whom in the
synthetic league, and each trade is run through the real tool). Take and talk replies are sampled from the base model
and kept only if the project's own checks pass: every number grounded, plain text, sentence limits, no contradiction
of the facts, no trade endorsed by the trade finder. Nothing comes from a real league or a real chat log.

Questions that appear in the repo's own evals are excluded, so the gates stay honest.

Usage: python finetune/build_dataset.py <synthetic.db> <out_dir> [--seed 11] [--model qwen3:4b-instruct]
Writes train.jsonl, eval.jsonl, probes.jsonl and build_report.json. Resumable: reruns skip finished tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dynasty_agent import chat, chat_eval, config, conversation_eval, grounding, knowledge, llm, risk_talk, tools, valuation

SAMPLES = 3  # candidate replies per take or talk task
YEARS = (2027, 2028, 2029)
ROUNDS = {1: ("1st", "first"), 2: ("2nd", "second"), 3: ("3rd", "third")}
SMALLTALK = ["hey", "hi there", "hello!", "good morning", "thanks!", "thank you so much", "thanks, that helps", "appreciate it",
             "cool, thanks", "ok great", "bye", "see you later", "have a good one", "yo", "hey, how's it going", "what's up",
             "good evening", "nice one", "perfect, thanks", "talk later"]
OTHER = ["What's the capital of Australia?", "Write me a haiku about winter", "Who won the 1998 World Series?", "What is 18 times 12?",
         "How do I bake sourdough bread?", "Explain how a car engine works", "Who is the best NBA player right now?",
         "What's the weather like on Mars?", "Recommend a good movie for tonight", "How do I fix a leaky faucet?",
         "Translate 'good morning' into Spanish", "What year did the Berlin Wall fall?", "Tell me a joke about cats",
         "Who is the president of France?", "What is the boiling point of water in Fahrenheit?", "Help me write a cover letter",
         "Who won the last Premier League title?", "How many ounces are in a pound?", "What's a good name for a puppy?",
         "Explain photosynthesis simply", "Who painted the Mona Lisa?", "What's the tallest mountain in Europe?",
         "How do I learn guitar quickly?", "What are the rules of chess castling?", "Who won the Stanley Cup in 2010?",
         "Summarize the plot of Hamlet", "What is the square root of 144?", "How do I change a tire?",
         "Best pizza toppings?", "Who invented the telephone?"]
STRATEGY = [
    "How should I think about trading a 30 year old RB for a younger WR?", "Is it better to build through the draft or through trades in dynasty?",
    "Why do running backs lose value so fast after 27?", "When should a dynasty team switch from contending to rebuilding?",
    "How much should I value a late first round rookie pick?", "What does PPR do to the value of a pass catching back?",
    "Is a one QB league different from superflex for roster building?", "Should I hold my rookie picks or trade them for veterans?",
    "How do I decide between two similar players at the same position?", "What makes a tight end valuable when there is no TE premium?",
    "How aggressive should I be on waivers in a deep league?", "What is a taxi squad for and how should I use it?",
    "Why is receiver depth so important in full PPR?", "How should I approach the trade deadline if I'm 4 and 5?",
    "What is the difference between win-now value and long-term value?", "Should I ever trade a first round pick for an aging star?",
    "How do I avoid overpaying for a player coming off a career year?", "What should I look for in a rookie wide receiver?",
    "Is it smart to stash a handcuff running back?", "How do injuries change how I should build my roster?",
    "What does FAAB mean and how much should I bid?", "How do I know if I should sell a player at his peak?",
    "Should I trade for a quarterback in a one QB league?", "Why does age matter more at running back than at receiver?",
    "How important is opportunity compared to talent for a young player?", "What is a good way to value future draft picks?",
    "Should I consolidate two good players into one great one?", "How do I tell a buy low candidate from a player who is just bad?",
    "When is it right to start a rookie over a veteran?", "How many running backs should I carry on a deep bench?",
    "What does it mean to be a contender in dynasty?", "Is tanking ever a good idea in a dynasty league?",
    "How should the scoring format change my draft strategy?", "What are the warning signs that a receiver's role is shrinking?",
    "Should I trade for a player on a bad offense?", "How do bye weeks change my roster decisions?",
    "What is the best way to use the taxi squad in a rookie only setup?", "Do tight ends take a long time to develop?",
    "How much does team situation matter for a running back?", "Should I take the safe player or the high upside player?",
    "What should I do with a roster full of aging veterans?", "How do I rebuild without losing every week?",
    "Is it worth paying for a player's name value in a trade?", "What matters more for a receiver, targets or yards per route run?",
    "How should I think about trading with a team that is rebuilding?", "How early should I trade for next year's rookie class?",
    "What is the biggest mistake new dynasty managers make?", "How do I handle a player who might retire soon?",
    "Is depth or star power better for winning a title?", "Why do dynasty managers overvalue draft picks?",
    "what should I do", "any advice for my team", "I'm not sure what to focus on this week", "give me some tips",
    "How do I get better at this game?", "should I be worried about my team", "what's the plan for the rest of the season",
    "Do you think I should be buying or selling right now?", "what do you think about the state of my league",
    "How do you compare a rookie to an established veteran?", "Which matters more, the next two years or the next five?",
    "What do you look at first when you evaluate a trade?",
]
NAMED_TALK = ["What do you think of {p}?", "Is {p} a good player?", "Tell me about {p}", "Do you like {p} this year?",
              "How good is {p} going to be?", "What's {p}'s outlook?"]
TRADE_Q = [
    ("Would you trade {m} for {t}?", "mt"), ("Can I get {t} for {m}?", "mt"), ("Should I send {m} for {t}?", "mt"),
    ("{m} for {t} straight up, good idea?", "mt"), ("Is {m} for {t} a fair swap?", "mt"), ("What about {m} for {t}?", "mt"),
    ("Thinking about moving {m} to get {t}. Worth it?", "mt"), ("Does giving up {m} for {t} make sense?", "mt"),
    ("Someone offered me {t} for {m}. Should I do it?", "mt"), ("I'm offered {t} for {m}. Take it?", "mt"),
    ("They'll give me {t} if I send {m}. Yes or no?", "mt"), ("Would you give {m} and {m2} for {t}?", "mmt"),
    ("Is {m} plus {m2} enough for {t}?", "mmt"), ("Could I land {t} and {t2} for {m}?", "mtt"),
    ("Would you trade {pk} for {t}?", "pt"), ("Should I flip {pk} for {t}?", "pt"), ("Is {t} worth {pk}?", "pt"),
    ("Would giving up {pk} get me {fa}?", "pf"), ("Should I trade {pk} for {fa}?", "pf"), ("{pk} for {fa}, yes or no?", "pf"),
    ("A guy offered me his {opk} for {m}, should I accept?", "mO"), ("Should I trade {m} for a {opk}?", "mO"),
    ("Would you sell {m} for {opk}?", "mO"), ("Someone wants {m} and will give me {opk}. Good deal?", "mO"),
    ("Someone wants my {pk} for their {t}. Good deal?", "pt"), ("I'm offered {t} for {m} and {pk}. Take it?", "mpt"),
    ("Would {m} and {pk} get me {t}?", "mpt"),
]


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def eval_questions() -> set[str]:
    qs = {norm(c.question) for c in chat_eval.CASES}
    for group in (conversation_eval.SCENARIOS, conversation_eval.HARD_SCENARIOS, conversation_eval.FINDER_SCENARIOS,
                  conversation_eval.INJURY_SCENARIOS):
        qs |= {norm(s.question) for sc in group for s in sc.steps}
    return qs


class World:
    """The synthetic league: who is mine, who is theirs, who is a free agent."""

    def __init__(self, db: Path):
        self.conn = sqlite3.connect(db)
        self.conn.row_factory = sqlite3.Row
        self.me_user = self.conn.execute("SELECT value FROM meta WHERE key = 'synthetic_user_id'").fetchone()[0]
        config.SLEEPER_USER_ID = self.me_user
        self.me = self.conn.execute("SELECT roster_id FROM rosters WHERE owner_id = ?", (self.me_user,)).fetchone()[0]
        rows = self.conn.execute(
            "SELECT p.player_id, p.full_name, p.position, mv.value, rp.roster_id FROM players p "
            "JOIN market_values mv ON mv.player_id = p.player_id AND mv.as_of_date = "
            "(SELECT max(as_of_date) FROM market_values m2 WHERE m2.player_id = p.player_id) "
            "LEFT JOIN roster_players rp ON rp.player_id = p.player_id "
            "WHERE p.position IN ('QB','RB','WR','TE') AND length(p.full_name) > 6 ORDER BY mv.value DESC").fetchall()
        self.mine, self.theirs, self.free = [], [], []
        seen = set()
        for r in rows:
            name = r["full_name"]
            if name in seen or not self._unique(name, r["player_id"]):
                continue
            seen.add(name)
            (self.mine if r["roster_id"] == self.me else self.theirs if r["roster_id"] else self.free).append(name)
        self.mine, self.theirs, self.free = self.mine[:30], self.theirs[:90], self.free[:70]
        self.players = self.mine + self.theirs + self.free

    def _unique(self, name: str, pid: str) -> bool:
        try:
            return valuation.resolve_player(self.conn, name)["player_id"] == pid
        except Exception:
            return False


def pick_spec(rng: random.Random) -> tuple[str, str, str]:
    """(arg string, my phrasing, their phrasing)"""
    year, rnd = rng.choice(YEARS), rng.choice((1, 1, 2, 2, 3))
    short, word = ROUNDS[rnd]
    spec = f"{year} {short}"
    mine = rng.choice([f"my {year} {word}", f"my {year} {short}", f"my {year} round {rnd} pick", f"my {year} {word} round pick"])
    theirs = rng.choice([f"a {year} {short}", f"a {year} {word}", f"their {year} {word}", f"a {year} round {rnd} pick"])
    return spec, mine, theirs


def trade_task(w: World, rng: random.Random, banned: set[str]):
    """(question, arguments) for one trade question, sides right by construction, or None."""
    template, shape = rng.choice(TRADE_Q)
    m, m2 = rng.sample(w.mine[:20], 2)
    t, t2 = rng.sample(w.theirs[:50], 2)
    fa = rng.choice(w.free[:30])
    spec, my_pk, _ = pick_spec(rng)
    ospec, _, their_pk = pick_spec(rng)
    q = template.format(m=m, m2=m2, t=t, t2=t2, fa=fa, pk=my_pk, opk=their_pk)
    args = {"send_players": [], "send_picks": [], "receive_players": [], "receive_picks": []}
    parts = {"m": ("send_players", [m]), "t": ("receive_players", [t]), "f": ("receive_players", [fa])}
    for ch in shape:
        if ch == "m":
            args["send_players"].append(m if args["send_players"] == [] else m2)
        elif ch == "t":
            args["receive_players"].append(t if args["receive_players"] == [] else t2)
        elif ch == "f":
            args["receive_players"].append(fa)
        elif ch == "p":
            args["send_picks"].append(spec)
        elif ch == "O":
            args["receive_picks"].append(ospec)
    if norm(q) in banned:
        return None
    return q, args


SIMPLE_ROUTES = {
    "my_team": ["Am I a contender or a rebuilder?", "How does my roster look?", "Should I be buying or selling this year?",
                "Where does my team stand?", "Give me a read on my team", "Is my roster any good?", "Am I built to win now?",
                "How strong is my team compared to the league?", "Should I tear it down?", "Tell me about my roster",
                "Is this a contending team?", "What's my team's outlook?", "How am I looking this year?", "Rate my roster"],
    "set_lineup": ["Who should I start this week?", "Set my lineup", "What's my best lineup for Sunday?", "Who do I sit this week?",
                   "Optimize my starters", "Which flex should I play?", "What are my chances this week?",
                   "Build me the best lineup", "Who starts for me?", "Help me pick my starters", "Lineup check please",
                   "Who's my best RB2 play this week?", "Should I start my bench guys?"],
    "pick_advice": ["Should I trade my rookie picks?", "What should I do with my draft picks?", "Are my picks worth keeping?",
                    "Buy, hold or sell on my picks?", "Is my first round pick worth more than a vet?",
                    "Should I cash in my 2027 picks?", "Tell me about my draft capital", "Do my picks have good value?",
                    "Is it time to sell my picks?", "How are my rookie picks looking?"],
    "taxi_plan": ["Who should I put on taxi?", "Any IR moves I should make?", "How do I free up a roster spot?",
                  "Should I stash anyone on the taxi squad?", "Who can go on IR?", "My roster is full, what can I move?",
                  "Which rookies belong on taxi?", "Help me clear a spot on my bench"],
}


def route_example(question: str, name: str | None, args: dict | None) -> dict:
    msgs = [{"role": "system", "content": chat.ROUTE_SYSTEM}, {"role": "user", "content": question}]
    if name is None:
        msgs.append({"role": "assistant", "content": args})
    else:
        clean = {k: v for k, v in (args or {}).items() if v not in ([], "", None)}
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": clean}}]})
    return {"kind": "route", "tools": tools.TOOLS, "messages": msgs, "meta": {"question": question, "tool": name}}


def topic_example(text: str, label: str, previous: str | None = None) -> dict:
    shown = f"Previous message: {previous}\nMessage to classify: {text}" if previous else text
    return {"kind": "topic", "messages": [{"role": "system", "content": chat.TOPIC_SYSTEM}, {"role": "user", "content": shown},
                                          {"role": "assistant", "content": label}], "meta": {"question": text, "label": label}}


MARKDOWN = re.compile(r"\*|^#|^\s*[-•]\s|[\U0001F300-\U0001FAFF]|—|–", re.M)


def sentence_count(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text))


def take_ok(reply: str, result: tools.ToolResult, question: str, tool: str) -> bool:
    payload = json.dumps(result.compact)
    allowed = grounding.allowed_numbers(result.numbers, payload, question)
    facts = " ".join(result.compact.get("facts", []))
    if not reply or MARKDOWN.search(reply) or "?" in reply or re.search(r"\bfacts?\b", reply, re.I):
        return False
    if not 2 <= sentence_count(reply) <= 4:
        return False
    if grounding.ungrounded(reply, allowed) or risk_talk.contradicts(reply, facts):
        return False
    nums = grounding.numbers_in(reply)
    if len(nums) > 3:
        return False
    if tool == tools.FIND_TRADES and chat._TELLS_TO_TRADE.search(reply):
        return False
    return True


def talk_ok(reply: str, allowed: set[str]) -> bool:
    if not reply or MARKDOWN.search(reply) or sentence_count(reply) > chat.TALK_SENTENCES or sentence_count(reply) < 2:
        return False
    return not grounding.unsupported_claims(reply, allowed) and not risk_talk.talk_claims_risk(reply)


def sample(client, messages, options, n):
    """n candidate replies, requested at the same time: Ollama serves concurrent requests on one GPU, which gave
    about 2.3 times the throughput of asking one after another."""
    def one(_):
        text = "".join(e for e in client.stream(messages, options=options) if isinstance(e, str)).strip()
        return re.sub(r"\s+", " ", text)

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(one, range(n)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("db", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--model", default=llm.LLM_MODEL)
    ap.add_argument("--route", type=int, default=330)
    ap.add_argument("--topic", type=int, default=130)
    ap.add_argument("--take", type=int, default=520)
    ap.add_argument("--talk", type=int, default=300)
    ap.add_argument("--probes", type=int, default=80)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    w = World(a.db)
    banned = eval_questions()
    done_path = a.out / "examples.jsonl"
    done = [json.loads(l) for l in done_path.open(encoding="utf-8")] if done_path.exists() else []
    have = {d["key"] for d in done}
    report = {"rejected": {"route": 0, "take": 0, "talk": 0}, "excluded_as_eval_questions": 0}

    def add(key: str, example: dict) -> None:
        example["key"] = key
        done.append(example)
        have.add(key)
        with done_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(example) + "\n")

    def key_of(*parts) -> str:
        return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]

    # ---- route (by construction; every trade is run through the real tool)
    n_route = sum(d["kind"] == "route" for d in done)
    simple_budget = a.route // 3
    seen_q = {d["meta"]["question"] for d in done if d["kind"] == "route"}
    while n_route < a.route:
        if rng.random() < 0.25 and sum(d["meta"].get("tool") not in ("evaluate_trade", None) for d in done if d["kind"] == "route") < simple_budget:
            tool = rng.choice(list(SIMPLE_ROUTES) + ["waiver_targets"] * 2)
            if tool == "waiver_targets":
                kind = rng.choice(("open", "pos", "player"))
                if kind == "open":
                    q, args = rng.choice(["Who's worth picking up on waivers?", "Who should I add from the wire?", "Any free agents I should grab?",
                                          "Who should I pick up?", "Best waiver pickups right now?"]), {}
                elif kind == "pos":
                    pos = rng.choice(["QB", "RB", "WR", "TE"])
                    q, args = rng.choice([f"Any {pos}s worth picking up?", f"Who's the best {pos} on waivers?", f"I need a {pos}, who's out there?"]), {"position": pos}
                else:
                    p = rng.choice(w.free[:25])
                    q, args = rng.choice([f"How much should I bid on {p}?", f"What's a fair FAAB bid for {p}?", f"Is {p} worth a waiver claim?"]), {"player": p}
            else:
                q, args = rng.choice(SIMPLE_ROUTES[tool]), {}
        else:
            tool = "evaluate_trade"
            t = trade_task(w, rng, banned)
            if t is None:
                report["excluded_as_eval_questions"] += 1
                continue
            q, args = t
            r = tools.run_tool(w.conn, "evaluate_trade", args)
            if r.clarification or not r.numbers:
                report["rejected"]["route"] += 1
                continue
        if norm(q) in banned or q in seen_q:
            report["excluded_as_eval_questions"] += norm(q) in banned
            if len(seen_q) > 3000:
                break
            continue
        seen_q.add(q)
        add(key_of("route", q, json.dumps(args, sort_keys=True)), route_example(q, tool, args))
        n_route += 1
    no_tool = [route_example(q, None, rng.choice([
        "That's outside what I do. I can help with lineups, trades, team outlook, waiver pickups, draft picks, taxi and IR.",
        "I'm built for your fantasy team, so that one isn't mine. Ask me about lineups, trades, waivers, picks or taxi and IR.",
        "Not something I can help with. I cover lineups, trades, your team's outlook, waivers, picks and taxi and IR."]))
               for q in rng.sample(OTHER, 25) if norm(q) not in banned]
    for ex in no_tool:
        k = key_of("route-none", ex["meta"]["question"])
        if k not in have:
            add(k, ex)

    # ---- topic
    fantasy_pool = STRATEGY + [t[0].format(m=rng.choice(w.mine), m2="x", t=rng.choice(w.theirs), t2="x", fa="x", pk="my 2027 1st", opk="a 2028 1st")
                                for t in TRADE_Q[:8]] + [f"{rng.choice(w.players)} or {rng.choice(w.players)} this week?" for _ in range(12)]
    topic_items = ([(q, "fantasy") for q in fantasy_pool] + [(q, "smalltalk") for q in SMALLTALK] + [(q, "other") for q in OTHER])
    rng.shuffle(topic_items)
    for text, label in topic_items:
        if norm(text) in banned:
            continue
        prev = rng.choice(STRATEGY) if rng.random() < 0.2 else None
        k = key_of("topic", text, prev)
        if sum(d["kind"] == "topic" for d in done) >= a.topic:
            break
        if k not in have:
            add(k, topic_example(text, label, prev))

    # ---- take and talk need the model
    client = llm.OllamaClient(model=a.model)
    client.ensure_ready()
    client.warm()
    started = time.monotonic()

    def take_task(question: str, call: dict) -> None:
        k = key_of("take", question, json.dumps(call, sort_keys=True))
        if k in have:
            return
        result = tools.run_tool(w.conn, call["name"], call["arguments"], recent=[])
        if result.clarification or not result.take or not result.compact:
            report["rejected"]["take"] += 1
            return
        payload = json.dumps(result.compact)
        messages = [{"role": "system", "content": chat.TAKE_SYSTEM}, {"role": "user", "content": question},
                    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": call["name"], "arguments": call["arguments"]}}]},
                    {"role": "tool", "content": payload, "tool_name": call["name"]}]
        good = [s for s in sample(client, messages, {**llm.TAKE_OPTIONS, "temperature": 0.6}, SAMPLES) if take_ok(s, result, question, call["name"])]
        if not good:
            report["rejected"]["take"] += 1
            return
        reply = min(good, key=lambda s: abs(sentence_count(s) - 3))
        ex_messages = [m if m["role"] != "assistant" or "tool_calls" not in m else
                       {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]}
                       for m in messages] + [{"role": "assistant", "content": reply}]
        add(k, {"kind": "take", "messages": ex_messages, "meta": {"question": question, "tool": call["name"]}})

    take_plan = []
    while len(take_plan) < a.take * 1.5:
        roll = rng.random()
        if roll < 0.40:
            t = trade_task(w, rng, banned)
            if t:
                take_plan.append((t[0], {"name": "evaluate_trade", "arguments": {k: v for k, v in t[1].items() if v}}))
        elif roll < 0.55:
            tool = rng.choice(list(SIMPLE_ROUTES))
            take_plan.append((rng.choice(SIMPLE_ROUTES[tool]), {"name": tool, "arguments": {}}))
        elif roll < 0.67:
            pos = rng.choice([None, "QB", "RB", "WR", "TE"])
            q = f"Who should I pick up{f' at {pos}' if pos else ''}?" if rng.random() < 0.5 else f"Any {pos or 'good'} free agents worth a bid?"
            take_plan.append((q, {"name": "waiver_targets", "arguments": {"position": pos} if pos else {}}))
        elif roll < 0.74:
            p = rng.choice(w.free[:25])
            take_plan.append((rng.choice([f"How much should I bid on {p}?", f"What's a fair FAAB bid for {p}?"]), {"name": "waiver_targets", "arguments": {"player": p}}))
        elif roll < 0.88:
            q = rng.choice(["Tell me about {p}", "How is {p} valued?", "What do you think of {p} for my team?", "Is {p} worth owning?", "Give me the rundown on {p}"]).format(p=rng.choice(w.players))
            call = tools.detect_action(w.conn, q)
            if call:
                take_plan.append((q, call))
        else:
            pos = rng.choice(["QB", "RB", "WR", "TE"])
            q = rng.choice([f"Help me find a {pos} to trade for", f"Find me some trade ideas at {pos}", f"I need a {pos}, who can I get?"])
            call = tools.detect_action(w.conn, q)
            if call:
                take_plan.append((q, call))
    for q, call in take_plan:
        if sum(d["kind"] == "take" for d in done) >= a.take:
            break
        if norm(q) not in banned:
            take_task(q, call)

    reference = knowledge.reference(w.conn)
    system = f"{chat.CHAT_SYSTEM}\n\nReference (true; use it and never contradict it):\n{reference}"
    talk_qs = [q for q in STRATEGY if norm(q) not in banned]
    talk_pool = talk_qs + [rng.choice(NAMED_TALK).format(p=rng.choice(w.players)) for _ in range(60)]
    rng.shuffle(talk_pool)
    for i, q in enumerate(talk_pool * 6):
        if sum(d["kind"] == "talk" for d in done) >= a.talk or i > len(talk_pool) * 6:
            break
        prior = None
        if rng.random() < 0.25:
            prior = rng.choice(talk_qs)
        k = key_of("talk", q, prior, i // len(talk_pool))
        if k in have:
            continue
        msgs = [{"role": "system", "content": system}]
        allowed_src = [q, reference]
        if prior:
            first = [s for s in sample(client, msgs + [{"role": "user", "content": prior}], llm.CHAT_OPTIONS, 2) if talk_ok(s, grounding.allowed_numbers(prior, reference))]
            if not first:
                continue
            msgs += [{"role": "user", "content": prior}, {"role": "assistant", "content": first[0]}]
            allowed_src.append(first[0])
        msgs.append({"role": "user", "content": q})
        allowed = grounding.allowed_numbers(*allowed_src)
        good = [s for s in sample(client, msgs, llm.CHAT_OPTIONS, SAMPLES) if talk_ok(s, allowed)]
        if not good:
            report["rejected"]["talk"] += 1
            continue
        add(k, {"kind": "talk", "messages": msgs[:] + [{"role": "assistant", "content": min(good, key=lambda s: abs(sentence_count(s) - 4))}],
                "meta": {"question": q}})
    client.close()

    # ---- split: examples about held-out players (and a random slice) go to eval; probes come from a separate seed
    held = set(rng.sample(w.players, max(6, len(w.players) // 10)))
    train, evals = [], []
    for ex in done:
        text = " ".join(m["content"] for m in ex["messages"] if m["role"] == "user" and isinstance(m["content"], str))
        about_held = any(p in text for p in held)
        (evals if about_held or int(hashlib.sha1(ex["key"].encode()).hexdigest(), 16) % 14 == 0 else train).append(ex)
    for ex in train + evals:
        ex.pop("key", None)

    def write(name, rows):
        with (a.out / name).open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: v for k, v in r.items() if k != "meta"} | ({"tools": r["tools"]} if "tools" in r else {})) + "\n")

    write("train.jsonl", train)
    write("eval.jsonl", evals)
    report["counts"] = {"train": len(train), "eval": len(evals), **{k: sum(d["kind"] == k for d in done) for k in ("route", "topic", "take", "talk")}}
    report["held_out_players"] = sorted(held)
    report["seconds"] = round(time.monotonic() - started)
    (a.out / "build_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["counts"]), "rejected:", report["rejected"])


if __name__ == "__main__":
    main()
