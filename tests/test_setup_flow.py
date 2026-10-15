"""`dynasty-agent setup` with a stand-in for the cli module: order, skipping, failure and the init handoff."""

import argparse

import pytest

from dynasty_agent import config, setup_flow
from dynasty_agent.errors import AgentError

COMMANDS = ["cmd_sync", "cmd_ingest_nflverse", "cmd_ingest_draft_data", "cmd_ingest_college", "cmd_backfill_history",
            "cmd_fit_prospect_model", "cmd_calibrate_matchup_model", "cmd_fit_availability", "cmd_refresh"]


class FakeCommands:
    def __init__(self, fail_on=None, fail_with=AgentError("boom")):
        self.calls, self.fail_on, self.fail_with = [], fail_on, fail_with
        for name in COMMANDS + ["cmd_init"]:
            setattr(self, name, self._make(name))

    def _make(self, name):
        def command(ns):
            self.calls.append((name, vars(ns)))
            if name == self.fail_on:
                raise self.fail_with
        return command

    @property
    def names(self):
        return [n for n, _ in self.calls]


@pytest.fixture
def ready(conn, monkeypatch):
    """A connected league (config set) on a database that knows the NFL season is 2026, regular season."""
    monkeypatch.setattr(setup_flow, "get_db", lambda: conn)
    monkeypatch.setattr(config, "SLEEPER_USERNAME", "me")
    monkeypatch.setattr(config, "SLEEPER_USER_ID", "u1")
    monkeypatch.setattr(config, "LEAGUE_ID", "L1")
    conn.execute("INSERT INTO nfl_state (fetched_at, season, week, season_type, display_week) VALUES ('2026-10-01', 2026, 6, 'regular', 6)")
    return conn


def args(**kw):
    return argparse.Namespace(**{"username": None, "league_id": None, "refit": False, **kw})


def test_every_step_runs_once_in_an_order_that_works_on_an_empty_database(ready, capsys):
    fake = FakeCommands()
    setup_flow.run(args(), fake)
    assert fake.names == COMMANDS
    by_name = dict(fake.calls)
    assert by_name["cmd_ingest_nflverse"]["season"] == 2025  # the last completed season, from the synced NFL state
    assert by_name["cmd_ingest_college"] == {"season": 2008, "through": 2026, "force": False}
    # history is loaded before the models that are scored on it
    assert fake.names.index("cmd_backfill_history") < fake.names.index("cmd_fit_prospect_model")
    assert fake.names.index("cmd_backfill_history") < fake.names.index("cmd_calibrate_matchup_model")
    assert "dynasty-agent chat" in capsys.readouterr().out


def test_a_model_that_is_already_fitted_is_skipped_unless_refit(ready, capsys):
    cols = [r[1] for r in ready.execute("PRAGMA table_info(calibration_params)") if r[1] != "id"]
    ready.execute(f"INSERT INTO calibration_params ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                  ["2026-10-01" if c == "fitted_at" else 1 for c in cols])
    fake = FakeCommands()
    setup_flow.run(args(), fake)
    assert "cmd_calibrate_matchup_model" not in fake.names
    assert "already fitted" in capsys.readouterr().out
    again = FakeCommands()
    setup_flow.run(args(refit=True), again)
    assert "cmd_calibrate_matchup_model" in again.names


def test_a_failing_step_stops_setup_and_says_how_to_continue(ready, capsys):
    fake = FakeCommands(fail_on="cmd_ingest_draft_data")
    with pytest.raises(SystemExit) as e:
        setup_flow.run(args(), fake)
    assert e.value.code == 1
    assert fake.names[-1] == "cmd_ingest_draft_data"  # nothing after the failure ran
    out = capsys.readouterr().out
    assert "Step 3 failed: boom" in out and "dynasty-agent setup" in out


def test_a_failed_final_refresh_is_a_warning_not_the_end_of_setup(ready, capsys):
    fake = FakeCommands(fail_on="cmd_refresh", fail_with=SystemExit(1))
    setup_flow.run(args(), fake)  # does not raise
    out = capsys.readouterr().out
    assert fake.names == COMMANDS and "warning" in out and "Setup complete" in out


def test_a_command_that_exits_with_an_error_code_also_stops_setup(ready):
    fake = FakeCommands(fail_on="cmd_sync", fail_with=SystemExit(1))
    with pytest.raises(SystemExit) as e:
        setup_flow.run(args(), fake)
    assert e.value.code == 1 and fake.names == ["cmd_sync"]


def test_no_league_and_no_username_asks_for_the_username(conn, monkeypatch):
    monkeypatch.setattr(config, "LEAGUE_ID", None)
    monkeypatch.setattr(setup_flow, "get_db", lambda: conn)
    with pytest.raises(AgentError, match="--username"):
        setup_flow.run(args(), FakeCommands())


def test_a_username_connects_the_league_first_then_loads_everything(conn, monkeypatch):
    monkeypatch.setattr(setup_flow, "get_db", lambda: conn)
    monkeypatch.setattr(config, "LEAGUE_ID", None)
    monkeypatch.setattr(config, "SLEEPER_USERNAME", None)
    monkeypatch.setattr(config, "SLEEPER_USER_ID", None)
    conn.execute("INSERT INTO nfl_state (fetched_at, season, week, season_type, display_week) VALUES ('2026-10-01', 2026, 6, 'regular', 6)")

    def reload():  # what config.reload does after init has written .env
        monkeypatch.setattr(config, "SLEEPER_USERNAME", "me")
        monkeypatch.setattr(config, "SLEEPER_USER_ID", "u1")
        monkeypatch.setattr(config, "LEAGUE_ID", "L1")

    monkeypatch.setattr(config, "reload", reload)
    fake = FakeCommands()
    setup_flow.run(args(username="me", league_id="L1"), fake)
    assert fake.names[0] == "cmd_init" and fake.names[1:] == COMMANDS
    assert dict(fake.calls)["cmd_init"] == {"username": "me", "league_id": "L1"}


def test_username_is_ignored_when_a_league_is_already_connected(ready, capsys):
    fake = FakeCommands()
    setup_flow.run(args(username="someone-else"), fake)
    assert "cmd_init" not in fake.names and "already connected" in capsys.readouterr().out


def test_missing_pieces_names_what_the_chat_is_waiting_for(conn):
    assert setup_flow.missing_pieces(conn)[0] == "your league"
    assert len(setup_flow.missing_pieces(conn)) == 6
    conn.execute("INSERT INTO prospect_model (variant, features_json, weights_json, n_rows, cv_mae, cv_r2, training_classes, ridge_lambda, fitted_at) "
                 "VALUES ('post_draft', '[]', '[]', 1, 1, 1, '[]', 1, 'x')")
    assert "the rookie model" not in setup_flow.missing_pieces(conn)


def test_config_reload_rereads_the_env_file(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("SLEEPER_USERNAME=newuser\nSLEEPER_USER_ID=42\nLEAGUE_ID=99\nDRAFT_ID=\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env)
    for name in ("SLEEPER_USERNAME", "SLEEPER_USER_ID", "LEAGUE_ID", "DRAFT_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "LEAGUE_ID", None)
    config.reload()
    assert (config.SLEEPER_USERNAME, config.SLEEPER_USER_ID, config.LEAGUE_ID) == ("newuser", "42", "99")
    assert config.missing_config() == []
    for name in ("SLEEPER_USERNAME", "SLEEPER_USER_ID", "LEAGUE_ID", "DRAFT_ID"):  # leave the process as it was
        monkeypatch.delenv(name, raising=False)
