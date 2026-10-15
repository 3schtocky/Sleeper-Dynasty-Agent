"""`dynasty-agent finetune`: the gate decision and the commands each stage runs, without a GPU, a model or a network."""

import argparse
import json
from pathlib import Path

import pytest

from dynasty_agent import config, finetune_flow
from dynasty_agent.errors import AgentError


def metrics(routing=(39, 39), conv=(57, 57), probes=None, bad=(), speed=80.0):
    return {"routing": list(routing), "conversations": list(conv),
            "probes": probes or {"route": [30, 34], "topic": [12, 12], "take": [18, 20], "talk": [12, 14]},
            "bad_format": list(bad), "tokens_per_second": speed}


def verdict(base, new):
    return {name.split(":")[0]: ok for name, ok, _ in finetune_flow.decide(base, new)}


def test_a_model_that_matches_the_base_on_every_gate_passes_all_five():
    assert all(verdict(metrics(), metrics()).values())


def test_a_model_that_is_better_than_the_base_also_passes():
    better = metrics(probes={"route": [34, 34], "topic": [12, 12], "take": [20, 20], "talk": [14, 14]}, speed=84)
    assert all(verdict(metrics(), better).values())


@pytest.mark.parametrize("new, failed", [
    (metrics(routing=(38, 39)), "Routing"),
    (metrics(conv=(56, 57)), "Conversations"),
    (metrics(probes={"route": [30, 34], "topic": [12, 12], "take": [17, 20], "talk": [12, 14]}), "Probes"),
    (metrics(bad=["talk-070"]), "Format"),
    (metrics(speed=71.9), "Speed"),  # 90% of 80 is 72
])
def test_each_gate_fails_on_its_own_and_only_its_own(new, failed):
    result = verdict(metrics(), new)
    assert [k for k, ok in result.items() if not ok] == [failed]


def test_an_empty_measurement_never_passes():
    assert not verdict(metrics(), metrics(routing=(0, 0)))["Routing"]
    assert not verdict(metrics(), metrics(conv=(0, 0)))["Conversations"]


def make_args(tmp_path, stage, **kw):
    return argparse.Namespace(stage=stage, workdir=str(tmp_path / "work"), base="qwen3:4b-instruct", name="dynasty-qwen3-4b:v2",
                              llama_cpp=None, rebuild_data=False, dry_run=False, **kw)


class Recorder:
    def __init__(self, code=0):
        self.calls, self.code = [], code

    def __call__(self, cmd, cwd=None):
        self.calls.append((list(cmd), cwd))
        return self.code

    def scripts(self):
        return [Path(c[1]).name for c, _ in self.calls if len(c) > 1 and c[1].endswith(".py")]


@pytest.fixture
def league_db(tmp_path, monkeypatch):
    db = tmp_path / "dynasty.db"
    db.write_bytes(b"")
    monkeypatch.setattr(config, "DB_PATH", db)
    return db


def test_data_builds_the_synthetic_league_then_the_dataset_then_the_probes(tmp_path, league_db):
    rec = Recorder()
    finetune_flow.run(make_args(tmp_path, "data"), rec)
    assert rec.scripts() == ["make_synthetic_league.py", "build_dataset.py", "make_probes.py"]
    build = rec.calls[1][0]
    assert build[build.index("--model") + 1] == "qwen3:4b-instruct"
    assert str(league_db) in rec.calls[0][0]  # builds from the real database, writing only under the work folder


def test_data_keeps_a_synthetic_league_it_already_built_unless_asked(tmp_path, league_db):
    work = tmp_path / "work"
    work.mkdir()
    (work / "synthetic.db").write_bytes(b"")
    rec = Recorder()
    finetune_flow.run(make_args(tmp_path, "data"), rec)
    assert "make_synthetic_league.py" not in rec.scripts()
    redo = make_args(tmp_path, "data")
    redo.rebuild_data = True
    third = Recorder()
    finetune_flow.run(redo, third)
    assert "make_synthetic_league.py" in third.scripts()


def test_data_needs_a_league_database(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "missing.db")
    with pytest.raises(AgentError, match="dynasty-agent setup"):
        finetune_flow.run(make_args(tmp_path, "data"), Recorder())


def test_a_failing_script_stops_the_stage_and_names_it(tmp_path, league_db):
    rec = Recorder(code=2)
    with pytest.raises(AgentError, match="Build the synthetic league.*exit code 2"):
        finetune_flow.run(make_args(tmp_path, "data"), rec)
    assert len(rec.calls) == 1


def test_train_without_the_gpu_libraries_says_how_to_install_them(tmp_path, monkeypatch):
    monkeypatch.setattr(finetune_flow.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(AgentError, match=r"(?s)pip install -e \".\[finetune\]\".*CUDA"):
        finetune_flow.run(make_args(tmp_path, "train"), Recorder())


def test_train_needs_the_dataset_first(tmp_path, monkeypatch):
    monkeypatch.setattr(finetune_flow.importlib.util, "find_spec", lambda name: object())
    with pytest.raises(AgentError, match="finetune data"):
        finetune_flow.run(make_args(tmp_path, "train"), Recorder())


def test_train_checks_the_data_then_trains_and_remembers_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(finetune_flow.importlib.util, "find_spec", lambda name: object())
    data = tmp_path / "work" / "dataset"
    data.mkdir(parents=True)
    (data / "train.jsonl").write_text("{}\n")
    rec = Recorder()
    finetune_flow.run(make_args(tmp_path, "train"), rec)
    assert rec.scripts() == ["check_data.py", "train.py"]
    run_dir = rec.calls[1][0][rec.calls[1][0].index("--run-dir") + 1]
    assert (tmp_path / "work" / "latest_run.txt").read_text() == run_dir


def test_package_without_llama_cpp_says_where_to_get_it(tmp_path):
    with pytest.raises(AgentError, match="llama.cpp"):
        finetune_flow.run(make_args(tmp_path, "package"), Recorder())


def test_package_merges_converts_quantizes_and_loads_the_model_into_ollama(tmp_path):
    work = tmp_path / "work"
    run = work / "runs" / "r1"
    run.mkdir(parents=True)
    (work / "latest_run.txt").write_text(str(run))
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "convert_hf_to_gguf.py").write_text("")
    (llama / "llama-quantize.exe").write_text("")
    args = make_args(tmp_path, "package")
    args.llama_cpp = str(llama)
    rec = Recorder()
    finetune_flow.run(args, rec)
    steps = [c for c, _ in rec.calls]
    assert [Path(s[1]).name if s[1].endswith(".py") else Path(s[0]).stem for s in steps] == ["merge.py", "convert_hf_to_gguf.py", "llama-quantize", "ollama"]
    assert steps[0][2] == str(run / "best_adapter")
    assert steps[2][-1] == "Q4_K_M" and steps[2][2].endswith(finetune_flow.GGUF_NAME)
    assert steps[3][:3] == ["ollama", "create", "dynasty-qwen3-4b:v2"] and rec.calls[3][1] == work / "package"
    assert (work / "package" / "Modelfile").exists()


def test_dry_run_prints_every_stage_and_runs_nothing(tmp_path, capsys):
    rec = Recorder()
    args = make_args(tmp_path, "all")
    args.dry_run = True
    args.llama_cpp = str(tmp_path / "llama")
    finetune_flow.run(args, rec)
    out = capsys.readouterr().out
    assert rec.calls == []
    for expected in ("make_synthetic_league.py", "build_dataset.py", "check_data.py", "train.py", "merge.py", "ollama", "Would measure"):
        assert expected in out


def test_gates_write_the_report_and_exit_nonzero_when_one_fails(tmp_path, monkeypatch, capsys):
    data = tmp_path / "work" / "dataset"
    data.mkdir(parents=True)
    (data / "probes.jsonl").write_text("")
    results = {"qwen3:4b-instruct": metrics(), "dynasty-qwen3-4b:v2": metrics(routing=(38, 39))}
    monkeypatch.setattr(finetune_flow.Flow, "measure", lambda self, model, probes: results[model])
    with pytest.raises(SystemExit) as e:
        finetune_flow.run(make_args(tmp_path, "gates"), Recorder())
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "FAIL  Routing" in out and "Keep qwen3:4b-instruct" in out
    report = json.loads((tmp_path / "work" / "gates_report.json").read_text())
    assert report["new"]["routing"] == [38, 39]


def test_gates_that_all_pass_say_how_to_switch(tmp_path, monkeypatch, capsys):
    data = tmp_path / "work" / "dataset"
    data.mkdir(parents=True)
    (data / "probes.jsonl").write_text("")
    monkeypatch.setattr(finetune_flow.Flow, "measure", lambda self, model, probes: metrics())
    finetune_flow.run(make_args(tmp_path, "gates"), Recorder())
    assert "LLM_MODEL=dynasty-qwen3-4b:v2" in capsys.readouterr().out
