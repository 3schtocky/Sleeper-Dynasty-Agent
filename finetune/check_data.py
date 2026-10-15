"""Validate a training set before spending hours on it: every row is JSON, renders through the chat template, ends with
an assistant turn and has assistant tokens to learn from. Prints token-length statistics. Exits 1 on any problem.

  python finetune/check_data.py <train.jsonl> <eval.jsonl> [--max-len 2048]

It reuses train.py's own tokenizing code (the part above `load`), so it checks exactly what training will see.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs=2, type=Path, metavar=("TRAIN", "EVAL"))
ap.add_argument("--max-len", type=int, default=2048)
ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
src = (Path(__file__).parent / "train.py").read_text(encoding="utf-8")
ns = {"json": json, "tok": tok, "args": argparse.Namespace(max_len=a.max_len)}
exec(src[src.index("def norm_messages"):src.index("def load(path)")], ns)

problems = 0
for path in a.files:
    kinds, lengths, labelled, bad = Counter(), [], [], 0
    for i, line in enumerate(path.open(encoding="utf-8"), 1):
        try:
            row = json.loads(line)
            e = ns["encode"](row)
            assert row["messages"][-1]["role"] == "assistant", "the last message is not the assistant's"
            n_lab = sum(1 for l in e["labels"] if l != -100)
            assert n_lab > 0, "no assistant tokens to train on"
            kinds[row.get("kind", "?")] += 1
            text = tok.apply_chat_template(ns["norm_messages"](row["messages"]), tools=row.get("tools") or None, tokenize=False)
            lengths.append(len(tok(text, add_special_tokens=False)["input_ids"]))
            labelled.append(n_lab)
        except Exception as ex:  # noqa: BLE001
            bad += 1
            problems += 1
            print(f"MALFORMED {path.name} line {i}: {type(ex).__name__}: {ex}")
    lengths.sort()

    def pct(q):
        return lengths[int(q * (len(lengths) - 1))] if lengths else 0

    over = sum(l > a.max_len for l in lengths)
    print(f"{path.name}: {len(lengths)} ok, {bad} malformed | kinds {dict(kinds)} | tokens p50={pct(.5)} p90={pct(.9)} "
          f"p99={pct(.99)} max={max(lengths) if lengths else 0} | over {a.max_len}: {over} | "
          f"mean trained tokens per example {sum(labelled) / max(1, len(labelled)):.0f}")
    if over:
        print(f"  note: {over} examples in {path.name} exceed {a.max_len} tokens and will be truncated")
sys.exit(1 if problems else 0)
