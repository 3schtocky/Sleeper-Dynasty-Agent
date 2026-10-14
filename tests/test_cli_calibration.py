import argparse

import pytest

from dynasty_agent import calibration, cli, config, context, nfl_extra, nflverse, weekly
from dynasty_agent.errors import AgentError
from tests.test_simulate import league, schedule  # noqa: F401  (schedule is a fixture)


@pytest.fixture
def db(conn, monkeypatch):
    monkeypatch.setattr(cli, "get_db", lambda: conn)
    monkeypatch.setattr(context, "require_config", lambda: None)
    return conn


def test_backfill_ingests_missing_seasons_skips_present_ones_and_adds_injuries(db, monkeypatch, capsys):
    db.execute("INSERT INTO league (league_id, season, scoring_settings_json, roster_positions_json, settings_json, fetched_at) "
               "VALUES ('L1', '2026', '{}', '[]', '{}', 't')")
    db.execute("INSERT INTO weekly_stats (player_id, season, week, fantasy_points, fetched_at) VALUES ('p', '2018', 1, 5, 't')")
    ingested, injuries = [], []
    monkeypatch.setattr(nflverse, "ingest_season", lambda conn, season, scoring, force=False: ingested.append(season) or f"Ingested {season}")
    monkeypatch.setattr(nfl_extra, "ingest_injuries", lambda conn, season: injuries.append(season) or 100)
    cli.cmd_backfill_history(argparse.Namespace(start_season=2017, end_season=2019, force=False))
    out = capsys.readouterr().out
    assert ingested == [2017, 2019] and injuries == [2017, 2018, 2019]
    assert "2018: weekly stats already in the database" in out
    assert "calibrate-matchup-model --start-season 2018 --end-season 2019" in out
    ingested.clear()
    cli.cmd_backfill_history(argparse.Namespace(start_season=2018, end_season=2018, force=True))
    assert ingested == [2018]


SUMMARY = {"start_season": 2018, "end_season": 2025, "sample_size": 2100, "games_considered": 2160, "platt_a": 0.31,
           "platt_b": -0.01, "brier_before": 0.26, "brier_after": 0.23, "log_loss_before": 0.82, "log_loss_after": 0.65,
           "accuracy_before": 0.61, "accuracy_after": 0.61}


def test_calibrate_prints_the_fit_the_scores_and_what_it_means(db, monkeypatch, capsys):
    monkeypatch.setattr(calibration, "fit_and_store_calibration", lambda conn, s, e: SUMMARY)
    cli.cmd_calibrate_matchup_model(argparse.Namespace(start_season=2018, end_season=2025))
    out = capsys.readouterr().out
    assert "2100 real NFL games (2018-2025; 60 skipped" in out
    assert "a = 0.310, b = -0.010" in out
    assert "0.2600" in out and "0.2300" in out and "61.0%" in out
    assert "too sure of itself" in out and "Refit after a new season" in out


def test_calibrate_with_nothing_to_test_is_a_clear_error(db, monkeypatch):
    def nothing(conn, s, e):
        raise ValueError("No backtestable real games found for 2018-2025")
    monkeypatch.setattr(calibration, "fit_and_store_calibration", nothing)
    with pytest.raises(AgentError, match="No backtestable"):
        cli.cmd_calibrate_matchup_model(argparse.Namespace(start_season=2018, end_season=2025))


def test_calibrate_defaults_to_the_last_completed_season(db, monkeypatch):
    seen = []
    monkeypatch.setattr(context, "latest_complete_season", lambda conn: 2025)
    monkeypatch.setattr(calibration, "fit_and_store_calibration", lambda conn, s, e: seen.append((s, e)) or SUMMARY)
    cli.cmd_calibrate_matchup_model(argparse.Namespace(start_season=2018, end_season=None))
    assert seen == [(2018, 2025)]


def test_simulate_prints_a_table_marks_my_team_and_warns_when_uncalibrated(db, schedule, monkeypatch, capsys):  # noqa: F811
    league(db)
    db.execute("UPDATE rosters SET owner_id = 'me' WHERE roster_id = 2")
    monkeypatch.setattr(config, "SLEEPER_USER_ID", "me")
    monkeypatch.setattr(weekly, "sync_matchups_or_note", lambda conn, week: None)
    cli.cmd_simulate_season(argparse.Namespace(season=None, vegas_season=None, simulations=500, seed=3))
    out = capsys.readouterr().out
    assert "Season simulation, 500 runs, from week 4 through 5 (4 real remaining matchups), top 2 make the playoffs." in out
    assert "Not calibrated yet" in out
    rows = [line for line in out.splitlines() if "roster " in line and "%" in line]
    assert len(rows) == 4 and sum(line.startswith("*") for line in rows) == 1 and "* roster 2" in out
    assert out.index("roster 1") < out.index("roster 4")  # best playoff odds first


def test_simulate_says_when_it_is_calibrated(db, schedule, monkeypatch, capsys):  # noqa: F811
    league(db)
    db.execute("UPDATE rosters SET owner_id = 'me' WHERE roster_id = 1")
    monkeypatch.setattr(config, "SLEEPER_USER_ID", "me")
    monkeypatch.setattr(weekly, "sync_matchups_or_note", lambda conn, week: None)
    db.execute("INSERT INTO calibration_params (fitted_at, start_season, end_season, sample_size, platt_a, platt_b, "
               "brier_before, brier_after, log_loss_before, log_loss_after, accuracy_before, accuracy_after) "
               "VALUES ('t', 2018, 2025, 100, 0.3, 0, 0.3, 0.2, 0.9, 0.6, 0.6, 0.6)")
    cli.cmd_simulate_season(argparse.Namespace(season=None, vegas_season=None, simulations=200, seed=3))
    assert "Win probabilities are calibrated" in capsys.readouterr().out


def test_the_three_commands_are_registered(monkeypatch, capsys):
    import dynasty_agent

    monkeypatch.setattr("sys.argv", ["dynasty-agent", "--help"])
    with pytest.raises(SystemExit):
        dynasty_agent.main()
    out = capsys.readouterr().out
    for name in ("backfill-history", "calibrate-matchup-model", "simulate-season"):
        assert name in out


def test_a_slope_only_fit_prints_just_the_slope(db, monkeypatch, capsys):
    monkeypatch.setattr(calibration, "fit_and_store_calibration", lambda conn, s, e: {**SUMMARY, "platt_b": 0.0})
    cli.cmd_calibrate_matchup_model(argparse.Namespace(start_season=2018, end_season=2025))
    out = capsys.readouterr().out
    assert "sigmoid(a x logit(raw)) with a = 0.310\n" in out and "b =" not in out

