"""Run probes.jsonl through a model served by Ollama and grade it with the project's own checks.

  python finetune/score_probes.py <probes.jsonl> <out.jsonl> --model dynasty-qwen3-4b:v2 [--limit N]

Routing and classification run greedy (temperature 0, as the chat does); takes and talk at the chat's own settings.
Reports: share of tool-call rows whose call is valid and sends the right items to the right side, topic accuracy,
share of takes that are grounded, plain and the right length, share of talk replies that pass, and any reply with
markdown, a list or an emoji.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import build_dataset as bd
from dynasty_agent import grounding, llm, risk_talk, tools


def norm_names(values) -> list[str]:
    return sorted(str(v).lower() for v in tools._as_list(values))


def grade_route(row: dict, calls: list[dict], text: str) -> tuple[bool, str]:
    want = row["expect"]
    if want["tool"] is None:
        return (not calls, "called a tool on an off-topic row" if calls else "")
    if not calls:
        return False, "no tool call"
    call = tools.repair_trade_call(row["messages"][-1]["content"], calls[0])
    if call["name"] != want["tool"]:
        return False, f"routed to {call['name']}"
    if want["tool"] == "evaluate_trade":
        args = call["arguments"]
        for side in ("send_players", "receive_players", "send_picks", "receive_picks"):
            if norm_names(args.get(side)) != norm_names(want[side]):
                return False, f"{side} was {args.get(side)}, wanted {want[side]}"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("probes", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    rows = [json.loads(l) for l in a.probes.open(encoding="utf-8")][: a.limit]
    stats = {k: [0, 0] for k in ("route", "topic", "take", "talk")}
    bad_format, results = [], []
    with llm.OllamaClient(model=a.model) as client:
        client.ensure_ready()
        client.warm()
        for row in rows:
            kind = row["kind"]
            if kind == "route":
                calls, text, _ = client.route(row["messages"], row["tools"])
                ok, why = grade_route(row, calls, text)
                reply = json.dumps(calls)
            else:
                options = llm.TOPIC_OPTIONS if kind == "topic" else llm.TAKE_OPTIONS if kind == "take" else llm.CHAT_OPTIONS
                reply = re.sub(r"\s+", " ", "".join(e for e in client.stream(row["messages"], options=options) if isinstance(e, str)).strip())
                why = ""
                if kind == "topic":
                    ok = reply.lower().strip(" .") == row["expect"]["label"]
                elif kind == "take":
                    allowed = grounding.allowed_numbers(*row["expect"]["allowed_text"])
                    ok = (2 <= bd.sentence_count(reply) <= 4 and "?" not in reply and not bd.MARKDOWN.search(reply)
                          and not grounding.ungrounded(reply, allowed) and not risk_talk.contradicts(reply, row["expect"]["facts"]))
                else:
                    allowed = grounding.allowed_numbers(*row["expect"]["allowed_text"])
                    ok = bd.talk_ok(reply, allowed)
                if bd.MARKDOWN.search(reply):
                    bad_format.append(row["id"])
            stats[kind][0] += ok
            stats[kind][1] += 1
            results.append({"id": row["id"], "kind": kind, "ok": ok, "why": why, "reply": reply})
            print(f"{'ok  ' if ok else 'FAIL'} {row['id']} {why}", flush=True)
    a.out.write_text("\n".join(json.dumps(r) for r in results) + "\n", encoding="utf-8")
    print("\nmodel:", a.model)
    for k, (good, total) in stats.items():
        if total:
            print(f"  {k:6} {good}/{total}")
    print("  replies with markdown, a list or an emoji:", bad_format or "none")


if __name__ == "__main__":
    main()
