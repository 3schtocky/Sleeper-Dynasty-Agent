-- Fitted availability model (see availability.py): how many games a
-- skill-position player is expected to miss. Every fit is kept; the newest
-- row is the one in use, the same convention as calibration_params.
CREATE TABLE IF NOT EXISTS availability_model (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fitted_at TEXT NOT NULL,
    start_season INTEGER NOT NULL,
    end_season INTEGER NOT NULL,
    sample_size INTEGER NOT NULL,      -- starter-level player-seasons the fit used
    kappa REAL NOT NULL,               -- games of evidence the age/position prior is worth
    lambda REAL NOT NULL,              -- yearly decay of a player's own history
    history_rmse REAL NOT NULL,        -- held-out, the fitted history model
    prior_rmse REAL NOT NULL,          -- held-out, position and age only
    league_rmse REAL NOT NULL,         -- held-out, one league-wide number for everyone
    model_json TEXT NOT NULL           -- prior table, in-season status table, label cuts, backtests
);
