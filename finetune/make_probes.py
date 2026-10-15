"""Build probes.jsonl: fresh prompts for checking a fine-tuned model, none of them in the training data or the repo's evals.

Each row holds the messages up to the model's turn, the tools when the router is being tested, and what a good
answer must satisfy. score_probes.py grades the model's outputs against those.

Usage: python finetune/make_probes.py <synthetic.db> <dataset_dir> [--seed 99]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import build_dataset as bd
from dynasty_agent import chat, grounding, knowledge, tools


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db", type=Path)
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--route", type=int, default=34)
    ap.add_argument("--topic", type=int, default=12)
    ap.add_argument("--take", type=int, default=20)
    ap.add_argument("--talk", type=int, default=14)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    w = bd.World(a.db)
    used = {bd.norm(json.loads(l)["meta"]["question"]) for l in (a.dataset / "examples.jsonl").open(encoding="utf-8")}
    banned = bd.eval_questions() | used
    rows: list[dict] = []

    def add(kind: str, messages: list[dict], expect: dict, tools_list=None) -> None:
        rows.append({"id": f"{kind}-{len(rows):03d}", "kind": kind, "messages": messages, "expect": expect,
                     **({"tools": tools_list} if tools_list else {})})

    n = 0
    while n < a.route:
        t = bd.trade_task(w, rng, banned)
        if t is None or bd.norm(t[0]) in banned:
            continue
        q, args = t
        banned.add(bd.norm(q))
        add("route", [{"role": "system", "content": chat.ROUTE_SYSTEM}, {"role": "user", "content": q}],
            {"tool": "evaluate_trade", "send_players": args["send_players"], "receive_players": args["receive_players"],
             "send_picks": args["send_picks"], "receive_picks": args["receive_picks"]}, tools.TOOLS)
        n += 1
    for tool, qs in bd.SIMPLE_ROUTES.items():
        for q in rng.sample(qs, 2):
            if bd.norm(q) not in banned:
                banned.add(bd.norm(q))
                add("route", [{"role": "system", "content": chat.ROUTE_SYSTEM}, {"role": "user", "content": q}], {"tool": tool}, tools.TOOLS)
    for q in rng.sample(bd.OTHER, 4):
        add("route", [{"role": "system", "content": chat.ROUTE_SYSTEM}, {"role": "user", "content": q}], {"tool": None}, tools.TOOLS)

    items = ([(q, "fantasy") for q in rng.sample(bd.STRATEGY, 5)] + [(q, "smalltalk") for q in rng.sample(bd.SMALLTALK, 3)]
             + [(q, "other") for q in rng.sample(bd.OTHER, 4)])
    for q, label in items[:a.topic]:
        add("topic", [{"role": "system", "content": chat.TOPIC_SYSTEM}, {"role": "user", "content": q}], {"label": label})

    made = 0
    while made < a.take:
        t = bd.trade_task(w, rng, banned)
        if t is None:
            continue
        q, args = t
        call = {"name": "evaluate_trade", "arguments": {k: v for k, v in args.items() if v}}
        result = tools.run_tool(w.conn, call["name"], call["arguments"], recent=[])
        if result.clarification or not result.compact:
            continue
        banned.add(bd.norm(q))
        payload = json.dumps(result.compact)
        add("take", [{"role": "system", "content": chat.TAKE_SYSTEM}, {"role": "user", "content": q},
                     {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]},
                     {"role": "tool", "content": payload, "tool_name": call["name"]}],
            {"allowed_text": [result.numbers, payload, q], "facts": " ".join(result.compact.get("facts", [])), "tool": call["name"]})
        made += 1

    reference = knowledge.reference(w.conn)
    system = f"{chat.CHAT_SYSTEM}\n\nReference (true; use it and never contradict it):\n{reference}"
    pool = [q for q in bd.STRATEGY if bd.norm(q) not in used]
    for q in rng.sample(pool, min(len(pool), a.talk)):
        add("talk", [{"role": "system", "content": system}, {"role": "user", "content": q}], {"allowed_text": [q, reference]})
    with (a.dataset / "probes.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print({k: sum(r["kind"] == k for r in rows) for k in ("route", "topic", "take", "talk")}, "->", a.dataset / "probes.jsonl")


if __name__ == "__main__":
    main()
