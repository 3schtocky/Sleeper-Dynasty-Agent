import sqlite3

from dynasty_agent.config import MIGRATIONS_DIR
from dynasty_agent.db import apply_migrations


def test_migrations_apply_cleanly_and_are_idempotent():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    first_pass = apply_migrations(conn, MIGRATIONS_DIR)
    assert first_pass, "expected at least one migration to apply on a fresh database"

    second_pass = apply_migrations(conn, MIGRATIONS_DIR)
    assert second_pass == [], "a migration already recorded must not run twice"

    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    for expected in (
        "players",
        "league",
        "rosters",
        "roster_players",
        "traded_picks",
        "weekly_stats",
        "market_values",
        "api_cache",
    ):
        assert expected in tables


def test_calibration_tables_exist_with_their_keys():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_migrations(conn, MIGRATIONS_DIR)

    def columns(table):
        return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]

    assert {"season", "week", "gsis_id", "report_status"} <= set(columns("nfl_injuries"))
    assert {"team_a_is_home", "raw_win_probability_a", "actual_a_win"} <= set(columns("backtest_games"))
    assert {"platt_a", "platt_b", "brier_before", "brier_after"} <= set(columns("calibration_params"))

    # A re-run of the same injury row replaces it; a second fit adds a row.
    conn.execute("INSERT INTO nfl_injuries VALUES (2024, 5, 'g1', 'KC', 'WR', 'Out', 't')")
    conn.execute("INSERT OR REPLACE INTO nfl_injuries VALUES (2024, 5, 'g1', 'KC', 'WR', 'Questionable', 't')")
    assert conn.execute("SELECT COUNT(*), MAX(report_status) FROM nfl_injuries").fetchone()[:] == (1, "Questionable")
    for a in (0.3, 0.4):
        conn.execute(
            "INSERT INTO calibration_params (fitted_at, start_season, end_season, sample_size, platt_a, platt_b, "
            "brier_before, brier_after, log_loss_before, log_loss_after, accuracy_before, accuracy_after) "
            "VALUES ('t', 2018, 2025, 10, ?, 0, 0.3, 0.2, 0.9, 0.6, 0.6, 0.6)", (a,))
    assert conn.execute("SELECT COUNT(*) FROM calibration_params").fetchone()[0] == 2
