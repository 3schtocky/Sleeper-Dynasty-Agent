-- Calibration of matchup.py's win-probability heuristic against real
-- historical NFL games (see calibration.py for the method). Ported from the
-- earlier worktree-phase4-college-data branch, renumbered to follow 0008.

-- Real per-week injury designations (nflverse `injuries`, 2009+). The
-- `players` table only holds TODAY's status, useless for a past week.
CREATE TABLE IF NOT EXISTS nfl_injuries (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    gsis_id TEXT NOT NULL,
    team TEXT,
    position TEXT,
    report_status TEXT,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (season, week, gsis_id)
);

CREATE INDEX IF NOT EXISTS idx_nfl_injuries_gsis ON nfl_injuries (gsis_id);

-- One row per backtested real game. Which real side is "team A" is a seeded
-- coin flip, so a real home-field effect is not baked into the correction.
CREATE TABLE IF NOT EXISTS backtest_games (
    season INTEGER NOT NULL,
    week INTEGER NOT NULL,
    team_a TEXT NOT NULL,
    team_b TEXT NOT NULL,
    team_a_is_home INTEGER NOT NULL,   -- 1/0, keeps the random assignment auditable
    raw_win_probability_a REAL,
    predicted_margin REAL,
    predicted_std REAL,
    team_a_players_found INTEGER,      -- coverage signal, not hidden
    team_b_players_found INTEGER,
    actual_a_win REAL,                 -- 1.0 / 0.0 / 0.5 for a real tie
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (season, week, team_a, team_b)
);

-- Every fit is kept; the newest row is the one in use.
CREATE TABLE IF NOT EXISTS calibration_params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fitted_at TEXT NOT NULL,
    start_season INTEGER NOT NULL,
    end_season INTEGER NOT NULL,
    sample_size INTEGER NOT NULL,
    platt_a REAL NOT NULL,
    platt_b REAL NOT NULL,
    brier_before REAL NOT NULL,
    brier_after REAL NOT NULL,
    log_loss_before REAL NOT NULL,
    log_loss_after REAL NOT NULL,
    accuracy_before REAL NOT NULL,
    accuracy_after REAL NOT NULL
);
