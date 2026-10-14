import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dynasty_agent import calibration, nflverse
from dynasty_agent.metrics import platt_scale


def write_games(path, games):
    """games: (season, week, home, away, home_score, away_score)"""
    pq.write_table(pa.table({
        "game_type": ["REG"] * len(games),
        "season": [g[0] for g in games],
        "week": [g[1] for g in games],
        "home_team": [g[2] for g in games],
        "away_team": [g[3] for g in games],
        "home_score": pa.array([g[4] for g in games], pa.float64()),
        "away_score": pa.array([g[5] for g in games], pa.float64()),
        "spread_line": pa.array([None] * len(games), pa.float64()),
        "total_line": pa.array([None] * len(games), pa.float64()),
    }), path)


def add_stats(conn, pid, team, season, week, points):
    conn.execute(
        "INSERT INTO weekly_stats (player_id, gsis_id, season, week, team, position, fantasy_points, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, 'WR', ?, 't')", (pid, "g" + pid, str(season), week, team, points))


@pytest.fixture
def world(conn, tmp_path, monkeypatch):
    """Two teams, 2024 (the prior) and 2025. KC's receiver scores 20 a week, BUF's 10.
    2025 week 3 is the game under test."""
    games = tmp_path / "games.parquet"
    write_games(games, [(2025, 3, "KC", "BUF", 30.0, 20.0)])
    monkeypatch.setattr(nflverse, "ensure_games_cached", lambda force=False: games)
    for season in (2024, 2025):
        for week in range(1, 4):
            add_stats(conn, "kc1", "KC", season, week, 20.0 + (week % 2))
            add_stats(conn, "bu1", "BUF", season, week, 10.0 + (week % 2))
    return conn


def test_a_game_is_projected_from_what_was_known_before_it(world):
    before, _ = calibration.build_backtest_sample(world, 2025, 2025)
    # Change the answer: the game's own week and any later week must not matter.
    world.execute("UPDATE weekly_stats SET fantasy_points = 99 WHERE season = '2025' AND week >= 3")
    after, _ = calibration.build_backtest_sample(world, 2025, 2025)
    assert before[0]["raw_win_probability_a"] == pytest.approx(after[0]["raw_win_probability_a"])


def test_the_better_side_is_favored_and_the_outcome_follows_the_orientation(world):
    rows, considered = calibration.build_backtest_sample(world, 2025, 2025)
    assert considered == 1 and len(rows) == 1
    r = rows[0]
    kc_is_a = r["team_a"] == "KC"
    assert r["team_a_is_home"] == kc_is_a  # KC is the home team
    assert (r["raw_win_probability_a"] > 0.5) == kc_is_a  # KC projects far ahead
    assert r["actual_a_win"] == (1.0 if kc_is_a else 0.0)  # KC won 30-20
    assert (r["team_a_players_found"], r["team_b_players_found"]) == (1, 1)


def test_orientation_is_seeded_and_the_table_is_replaced_not_doubled(world):
    first, _ = calibration.build_backtest_sample(world, 2025, 2025, seed=7)
    again, _ = calibration.build_backtest_sample(world, 2025, 2025, seed=7)
    assert [r["team_a"] for r in first] == [r["team_a"] for r in again]
    assert world.execute("SELECT COUNT(*) FROM backtest_games").fetchone()[0] == 1
    flips = {calibration.build_backtest_sample(world, 2025, 2025, seed=s)[0][0]["team_a"] for s in range(20)}
    assert flips == {"KC", "BUF"}  # both orientations occur across seeds


def test_a_game_with_a_side_that_has_no_stats_is_skipped(world, tmp_path, monkeypatch):
    games = tmp_path / "more.parquet"
    write_games(games, [(2025, 3, "KC", "BUF", 30.0, 20.0), (2025, 3, "DEN", "LV", 17.0, 14.0)])
    monkeypatch.setattr(nflverse, "ensure_games_cached", lambda force=False: games)
    rows, considered = calibration.build_backtest_sample(world, 2025, 2025)
    assert (len(rows), considered) == (1, 2)


def test_an_injury_report_lowers_that_weeks_projection(world):
    plain = calibration.predicted_team_distribution(world, 2025, 3, "KC", {}, {})[0]
    world.execute("INSERT INTO nfl_injuries VALUES (2025, 3, 'gkc1', 'KC', 'WR', 'Doubtful', 't')")
    hurt = calibration.predicted_team_distribution(world, 2025, 3, "KC", {}, {})[0]
    assert hurt < plain


def test_fit_stores_a_row_and_the_newest_fit_is_the_one_in_use(world, monkeypatch):
    assert calibration.current_calibration(world) is None
    # An overconfident heuristic: always 90% sure, right half the time.
    rows = [{"raw_win_probability_a": 0.9, "actual_a_win": float(i % 2)} for i in range(40)]
    monkeypatch.setattr(calibration, "build_backtest_sample", lambda conn, s, e: (rows, 40))
    first = calibration.fit_and_store_calibration(world, 2018, 2025)
    assert first["platt_a"] > 0 and first["sample_size"] == 40
    assert first["brier_after"] < first["brier_before"]  # the correction helps
    assert calibration.current_calibration(world) == (first["platt_a"], first["platt_b"])
    assert platt_scale(0.9, first["platt_a"], first["platt_b"]) == pytest.approx(0.5, abs=0.05)
    rows2 = [{"raw_win_probability_a": 0.8, "actual_a_win": float(i % 2)} for i in range(40)]
    monkeypatch.setattr(calibration, "build_backtest_sample", lambda conn, s, e: (rows2, 40))
    second = calibration.fit_and_store_calibration(world, 2018, 2025)
    assert world.execute("SELECT COUNT(*) FROM calibration_params").fetchone()[0] == 2
    assert calibration.current_calibration(world) == (second["platt_a"], second["platt_b"])


def test_fit_with_nothing_to_backtest_says_to_backfill(world, monkeypatch):
    monkeypatch.setattr(calibration, "build_backtest_sample", lambda conn, s, e: ([], 120))
    with pytest.raises(ValueError, match="backfill-history --start-season 2017"):
        calibration.fit_and_store_calibration(world, 2018, 2025)
    assert calibration.current_calibration(world) is None


def test_a_non_positive_slope_is_stored_but_warned_about(world, monkeypatch, capsys):
    # The heuristic is anti-correlated with results: favorites lose.
    rows = [{"raw_win_probability_a": 0.9, "actual_a_win": 0.0} for _ in range(20)] + \
           [{"raw_win_probability_a": 0.1, "actual_a_win": 1.0} for _ in range(20)]
    monkeypatch.setattr(calibration, "build_backtest_sample", lambda conn, s, e: (rows, 40))
    summary = calibration.fit_and_store_calibration(world, 2018, 2025)
    assert summary["platt_a"] <= 0
    assert "a=" in capsys.readouterr().err
    assert calibration.current_calibration(world) is not None


def test_blend_before_week_leaves_the_default_unchanged(conn):
    from dynasty_agent import blend
    for week, pts in ((1, 10.0), (2, 20.0), (3, 30.0)):
        add_stats(conn, "p", "KC", 2025, week, pts)
    assert blend.weekly_values(conn, "p", 2025) == [10.0, 20.0, 30.0]
    assert blend.weekly_values(conn, "p", 2025, before_week=3) == [10.0, 20.0]
    assert blend.player_distribution(conn, "p", 2025)["current_games"] == 3
    assert blend.player_distribution(conn, "p", 2025, before_week=2)["current_games"] == 1
