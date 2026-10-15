"""`dynasty-agent finetune`: the fine-tuning recipe in finetune/ as four stages you can run one at a time or together.

  data     build a synthetic league from your database, then the training set and the probes (needs Ollama)
  train    validate the data, then QLoRA-train on it (needs a CUDA GPU and the `finetune` extra)
  package  merge the adapter, convert it to GGUF, quantize it and load it into Ollama (needs llama.cpp)
  gates    score the new model against the base model; a pass means it may replace the default
  all      data, train, package, gates

Each stage runs the scripts in finetune/ with the current Python, so a stage can also be run by hand. Nothing here
publishes anything: the gates decide only whether the model is worth keeping.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dynasty_agent import config
from dynasty_agent.errors import AgentError

STAGES = ("data", "train", "package", "gates", "all")
BASE_MODEL = "qwen3:4b-instruct"
DEFAULT_NAME = "dynasty-qwen3-4b:v2"
GGUF_NAME = "dynasty-qwen3-4b-v2.Q4_K_M.gguf"
REQUIRED_MODULES = ("torch", "transformers", "peft", "datasets", "accelerate", "bitsandbytes")
SPEED_TOLERANCE = 0.90  # the new model must generate at least this share of the base model's tokens per second
SPEED_PROMPTS = ["In four sentences, how should a dynasty team decide when to rebuild?",
                 "Explain why running backs lose value after 27 in three sentences.",
                 "Give two reasons to hold a rookie first round pick."]


@dataclass
class Layout:
    work: Path

    @property
    def synthetic_db(self) -> Path: return self.work / "synthetic.db"
    @property
    def dataset(self) -> Path: return self.work / "dataset"
    @property
    def runs(self) -> Path: return self.work / "runs"
    @property
    def latest_run(self) -> Path: return self.work / "latest_run.txt"
    @property
    def merged(self) -> Path: return self.work / "merged"
    @property
    def package(self) -> Path: return self.work / "package"
    @property
    def gates_report(self) -> Path: return self.work / "gates_report.json"


def finetune_dir() -> Path:
    path = config.PROJECT_ROOT / "finetune"
    if not (path / "train.py").exists():
        raise AgentError("The finetune/ folder is missing. This command works from a clone of the repository, "
                         "not a copy that only has the installed package.")
    return path


def script(name: str) -> str:
    return str(finetune_dir() / name)


# ---- the gates: a pure decision, tested without a GPU ---------------------------------------------------

def decide(base: dict, new: dict) -> list[tuple[str, bool, str]]:
    """(gate, passed, detail) for each gate. `base` and `new` hold: routing [passed, total], conversations
    [passed, total], probes {kind: [passed, total]}, bad_format [row ids], tokens_per_second."""
    gates = []
    r = new["routing"]
    gates.append(("Routing: every question reaches the right tool", r[0] == r[1] and r[1] > 0, f"{r[0]} of {r[1]} (base {base['routing'][0]} of {base['routing'][1]})"))
    c = new["conversations"]
    gates.append(("Conversations: every step passes", c[0] == c[1] and c[1] > 0, f"{c[0]} of {c[1]} (base {base['conversations'][0]} of {base['conversations'][1]})"))
    worse = []
    for kind, (good, total) in new["probes"].items():
        b_good, b_total = base["probes"].get(kind, [0, 0])
        if total and b_total and good / total < b_good / b_total:
            worse.append(f"{kind} {good}/{total} vs base {b_good}/{b_total}")
    gates.append(("Probes: no kind of answer worse than the base model", not worse, "; ".join(worse) or "none worse"))
    gates.append(("Format: no markdown, list or emoji in any reply", not new["bad_format"], ", ".join(new["bad_format"]) or "none"))
    need = SPEED_TOLERANCE * base["tokens_per_second"]
    gates.append((f"Speed: at least {SPEED_TOLERANCE:.0%} of the base model's tokens per second",
                  new["tokens_per_second"] >= need, f"{new['tokens_per_second']:.1f} tok/s (base {base['tokens_per_second']:.1f})"))
    return gates


# ---- running things --------------------------------------------------------------------------------------

Runner = Callable[[list[str], Path | None], int]


def _run(cmd: list[str], cwd: Path | None = None) -> int:
    return subprocess.run(cmd, cwd=cwd).returncode


class Flow:
    def __init__(self, args, run: Runner = _run):
        self.args = args
        self.run = run
        self.layout = Layout(Path(args.workdir).resolve())
        self.dry = bool(getattr(args, "dry_run", False))

    def step(self, label: str, cmd: list[str], cwd: Path | None = None) -> None:
        shown = " ".join(f'"{c}"' if " " in c else c for c in cmd)
        print(f"\n> {label}\n  {shown}")
        if self.dry:
            return
        code = self.run(cmd, cwd)
        if code != 0:
            raise AgentError(f"'{label}' failed (exit code {code}). Fix that and run the same command again; "
                             f"finished work in {self.layout.work} is kept.")

    # data
    def data(self) -> None:
        L, py = self.layout, sys.executable
        L.work.mkdir(parents=True, exist_ok=True)
        if not self.dry and not config.DB_PATH.exists():
            raise AgentError("There is no league database to build from. Run `dynasty-agent setup` first.")
        if L.synthetic_db.exists() and not self.args.rebuild_data:
            print(f"\n> Synthetic league already built ({L.synthetic_db}); --rebuild-data to redo it")
        else:
            self.step("Build the synthetic league (and scan it for leaks)", [py, script("make_synthetic_league.py"), str(config.DB_PATH), str(L.synthetic_db)])
        self.step("Build the training set (samples the base model through Ollama, so this takes hours)",
                  [py, script("build_dataset.py"), str(L.synthetic_db), str(L.dataset), "--model", self.args.base])
        self.step("Build the probes (fresh prompts, none in the training set)", [py, script("make_probes.py"), str(L.synthetic_db), str(L.dataset)])

    # train
    def train(self) -> None:
        L, py = self.layout, sys.executable
        if not self.dry:
            missing = [m for m in REQUIRED_MODULES if importlib.util.find_spec(m) is None]
            if missing:
                raise AgentError(
                    f"Training needs {', '.join(missing)}. Install them with:  pip install -e \".[finetune]\"\n"
                    "On an NVIDIA GPU install the CUDA build of PyTorch first, from https://pytorch.org/get-started/locally/ "
                    "(the default pip build is CPU only).")
            if not (L.dataset / "train.jsonl").exists():
                raise AgentError("No training set yet. Run `dynasty-agent finetune data` first.")
        run_dir = L.runs / time.strftime("%Y%m%d-%H%M%S")
        self.step("Check the training data", [py, script("check_data.py"), str(L.dataset / "train.jsonl"), str(L.dataset / "eval.jsonl")])
        self.step("Train (QLoRA, about 2 to 3 hours on an 8 GB card)",
                  [py, script("train.py"), "--train", str(L.dataset / "train.jsonl"), "--eval", str(L.dataset / "eval.jsonl"), "--run-dir", str(run_dir)])
        if not self.dry:
            L.latest_run.write_text(str(run_dir), encoding="utf-8")

    # package
    def package(self) -> None:
        L, py = self.layout, sys.executable
        llama = Path(self.args.llama_cpp) if self.args.llama_cpp else None
        convert = (llama / "convert_hf_to_gguf.py") if llama else None
        quantize = shutil.which("llama-quantize") or (str(next(iter(llama.glob("llama-quantize*")), "")) if llama else "")
        if self.dry:
            quantize = quantize or "llama-quantize"
        if not self.dry:
            if convert is None or not convert.exists() or not quantize:
                raise AgentError(
                    "Packaging needs llama.cpp: a folder holding convert_hf_to_gguf.py and the llama-quantize program. "
                    "Download a release (and the source of the same tag) from https://github.com/ggml-org/llama.cpp/releases, "
                    "put them in one folder and pass it as --llama-cpp <folder>.")
            if not L.latest_run.exists():
                raise AgentError("No trained adapter yet. Run `dynasty-agent finetune train` first.")
        run_dir = Path(L.latest_run.read_text(encoding="utf-8")) if L.latest_run.exists() else L.runs / "<run>"
        f16, q4 = L.package / "model-f16.gguf", L.package / GGUF_NAME
        L.package.mkdir(parents=True, exist_ok=True)
        self.step("Merge the adapter into the base model (on the CPU)", [py, script("merge.py"), str(run_dir / "best_adapter"), str(L.merged)])
        self.step("Convert to GGUF (16-bit)", [py, str(convert), str(L.merged), "--outfile", str(f16), "--outtype", "f16"])
        self.step("Quantize to Q4_K_M", [quantize, str(f16), str(q4), "Q4_K_M"])
        modelfile = L.package / "Modelfile"
        if not self.dry:
            shutil.copy(finetune_dir() / "Modelfile", modelfile)
        self.step(f"Load it into Ollama as {self.args.name}", ["ollama", "create", self.args.name, "-f", "Modelfile"], cwd=L.package)

    # gates
    def measure(self, model: str, probes: Path) -> dict:
        from dynasty_agent import chat_eval, conversation_eval, llm
        from dynasty_agent.db import get_db

        sys.path.insert(0, str(finetune_dir()))
        import score_probes

        print(f"\n> Measuring {model}")
        conn = get_db()
        with llm.OllamaClient(model=model) as client:
            client.ensure_ready()
            client.warm()
            routing = chat_eval.run_live(client, conn=conn)
            scenarios = (conversation_eval.SCENARIOS + conversation_eval.HARD_SCENARIOS + conversation_eval.FINDER_SCENARIOS
                         + conversation_eval.INJURY_SCENARIOS)
            conversations = conversation_eval.run(conn, client, scenarios)
            speeds = []
            for prompt in SPEED_PROMPTS:
                stats = [e for e in client.stream([{"role": "user", "content": prompt}], options=llm.CHAT_OPTIONS) if isinstance(e, llm.Stats)]
                if stats and stats[-1].tokens_per_second:
                    speeds.append(stats[-1].tokens_per_second)
        probe = score_probes.score(probes, model, verbose=False)
        return {"routing": [routing["passed"], routing["total"]], "conversations": [conversations["passed"], conversations["total"]],
                "probes": probe["stats"], "bad_format": probe["bad_format"],
                "tokens_per_second": sum(speeds) / len(speeds) if speeds else 0.0}

    def gates(self) -> None:
        L = self.layout
        probes = L.dataset / "probes.jsonl"
        if self.dry:
            print(f"\n> Would measure {self.args.base} and {self.args.name} on routing, conversations, probes and speed, then decide.")
            return
        if not probes.exists():
            raise AgentError("No probes yet. Run `dynasty-agent finetune data` first.")
        base, new = self.measure(self.args.base, probes), self.measure(self.args.name, probes)
        results = decide(base, new)
        L.gates_report.write_text(json.dumps({"base": base, "new": new, "gates": results}, indent=1), encoding="utf-8")
        print("\nGATES")
        for name, ok, detail in results:
            print(f"  {'PASS' if ok else 'FAIL'}  {name}\n        {detail}")
        if all(ok for _, ok, _ in results):
            print(f"\nEvery gate passed. {self.args.name} may replace {self.args.base}: set LLM_MODEL={self.args.name} in .env.")
        else:
            print(f"\nA gate failed. Keep {self.args.base} as the default. The numbers are in {L.gates_report}.")
            raise SystemExit(1)


def run(args, runner: Runner = _run) -> None:
    flow = Flow(args, runner)
    stages = ["data", "train", "package", "gates"] if args.stage == "all" else [args.stage]
    for stage in stages:
        getattr(flow, stage)()
