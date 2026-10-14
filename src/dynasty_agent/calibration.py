"""Backtests matchup.py's win-probability heuristic against real historical
NFL games and fits a calibration correction. Ported from the earlier
worktree-phase4-college-data branch and rebuilt on this branch's blended
projections.

**Why it exists:** the heuristic turns a projected-points gap into a win
probability with a normal curve, and nothing ever checked that curve against
real results. The backtest found it badly overconfident, so every win
probability the tool prints is now shown beside a Platt-scaled version,
sigmoid(a * logit(raw) + b), fitted here.

**What a backtested game uses:** for week W of season S, each side's
projection is built the way the live tool builds one before that week:
last season's real games as a prior worth a few games, plus this season's
weeks before W (blend.player_distribution with before_week=W), then the same
injury and Vegas adjustments matchup.py applies. Nothing from week W or later
reaches the projection.

**A real design decision, stated plainly:** orienting every game as
home=team_a would let Platt scaling's global `b` silently absorb any real
home-field effect (the raw heuristic has no home/away concept). That
correction then gets applied everywhere, including `optimize-lineup`, where
"team A vs team B" means your fantasy roster vs your opponent's and
home/away is meaningless. Fix: which real side plays "team A" is picked by
`random.Random(seed)`, a fixed, documented seed, so any home-field effect is
absorbed as noise across both classes instead of baked into a bias.

**Stated limitations, not hidden:**
- This corrects the heuristic's overall over/under-confidence. It adds no
  new information; home-field advantage stays unmodeled.
- Hindsight roster: a historical team's "roster" for one game is whoever
  recorded a real weekly_stats row for that team that week, not a true
  pre-game depth chart. There is no historical lineup for an arbitrary past
  NFL team, so this is the only real roster concept available. It leaves out
  anyone who was inactive and never played.
- Rookies are projected from their games so far only (no prospect-model
  projection for a past class), so their first weeks are thin.
- It is fitted on NFL team games and applied to fantasy lineups, whose
  points are noisier in a different way. Treat the calibrated number as an
  honest correction in direction and rough size, not as exact.
"""

from __future__ import annotations

import random
import sqlite3
import sys

import duckdb

from dynasty_agent import blend, matchup, nfl_extra, nflverse
from dynasty_agent.db import utcnow
from dynasty_agent.metrics import (
    brier_score,
    fit_platt_scaling,
    injury_adjusted_mean,
    injury_adjusted_variance,
    log_loss,
    matchup_win_probability,
    platt_scale,
    vegas_week_multiplier,
)

# The schedules file keeps the code a team played under (Oakland in 2018-2019),
# while the stats files use the current one for every year. Found live: all 32
# Raiders games of 2018-2019 failed to match any player stats.
STATS_TEAM = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Fixed and documented, not left to chance: the random team_a/team_b
# assignment is reproducible, so a re-run over the same seasons rebuilds the
# same backtest_games rows.
DEFAULT_SEED = 20260902


def predicted_team_distribution(
    conn: sqlite3.Connection,
    season: int,
    week: int,
    team: str,
    week_implied: dict,
    season_avg_implied: dict,
) -> tuple[float, float, int]:
    """One historical team's projected mean and variance for one real game,
    from the same blend, injury and Vegas machinery matchup.py uses, driven
    by the hindsight roster (see module docstring). Returns (mean_total,
    variance_total, players_found); players_found is stored downstream as a
    coverage signal."""
    rows = conn.execute(
        "SELECT DISTINCT player_id, gsis_id FROM weekly_stats "
        "WHERE team = ? AND season = ? AND week = ? AND gsis_id IS NOT NULL",
        (STATS_TEAM.get(team, team), str(season), week),
    ).fetchall()

    vegas_mult = vegas_week_multiplier(week_implied.get(team), season_avg_implied.get(team))
    mean_total, variance_total = 0.0, 0.0
    for player_id, gsis_id in rows:
        dist = blend.player_distribution(conn, player_id, season, before_week=week)
        injury_status = nfl_extra.report_status_for_week(conn, season, week, gsis_id)
        raw_mean = dist["mean"] or 0.0
        raw_variance = dist["variance"]
        mean_total += injury_adjusted_mean(raw_mean, injury_status) * vegas_mult
        variance_total += injury_adjusted_variance(raw_variance, injury_status) if raw_variance is not None else 0.0
    return mean_total, variance_total, len(rows)


def build_backtest_sample(
    conn: sqlite3.Connection, start_season: int, end_season: int, seed: int = DEFAULT_SEED
) -> tuple[list[dict], int]:
    """For each real completed regular-season game in [start_season,
    end_season] (from the cached schedules file): project both sides, record
    the raw win probability and the real outcome. Games where either side has
    no weekly_stats rows to project from are skipped, there is nothing real
    to backtest. Replaces any earlier backtest_games rows for this season
    range. Returns (rows, games_considered)."""
    games_path = str(nflverse.ensure_games_cached())
    con = duckdb.connect()
    try:
        games = con.execute(
            """
            SELECT season, week, home_team, away_team, home_score, away_score
            FROM read_parquet(?)
            WHERE game_type = 'REG' AND season >= ? AND season <= ?
              AND home_score IS NOT NULL AND away_score IS NOT NULL
            ORDER BY season, week, home_team
            """,
            [games_path, start_season, end_season],
        ).fetchall()
    finally:
        con.close()

    rng = random.Random(seed)
    fetched_at = utcnow()
    implied_cache: dict[tuple[int, int], tuple[dict, dict]] = {}
    rows = []

    for season, week, home_team, away_team, home_score, away_score in games:
        if (season, week) not in implied_cache:
            implied_cache[(season, week)] = (
                matchup.team_week_implied_points(season, week),
                matchup.team_season_avg_implied_points(season, week),
            )
        week_implied, season_avg_implied = implied_cache[(season, week)]

        # Drawn for every game, played or skipped, so one skipped game never
        # shifts the orientation of the games after it.
        team_a_is_home = rng.random() < 0.5
        team_a, team_b = (home_team, away_team) if team_a_is_home else (away_team, home_team)

        a_mean, a_var, a_found = predicted_team_distribution(conn, season, week, team_a, week_implied, season_avg_implied)
        b_mean, b_var, b_found = predicted_team_distribution(conn, season, week, team_b, week_implied, season_avg_implied)
        if a_found == 0 or b_found == 0:
            continue  # nothing real to project one side from, skip rather than record a meaningless 50/50

        mean_diff = a_mean - b_mean
        std_diff = (a_var + b_var) ** 0.5
        a_score, b_score = (home_score, away_score) if team_a_is_home else (away_score, home_score)
        rows.append({
            "season": season, "week": week, "team_a": team_a, "team_b": team_b,
            "team_a_is_home": team_a_is_home,
            "raw_win_probability_a": matchup_win_probability(mean_diff, std_diff),
            "predicted_margin": mean_diff, "predicted_std": std_diff,
            "team_a_players_found": a_found, "team_b_players_found": b_found,
            "actual_a_win": 1.0 if a_score > b_score else (0.0 if a_score < b_score else 0.5),
            "fetched_at": fetched_at,
        })

    conn.execute("DELETE FROM backtest_games WHERE season >= ? AND season <= ?", (start_season, end_season))
    conn.executemany(
        """
        INSERT OR REPLACE INTO backtest_games (
            season, week, team_a, team_b, team_a_is_home, raw_win_probability_a,
            predicted_margin, predicted_std, team_a_players_found, team_b_players_found,
            actual_a_win, fetched_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (r["season"], r["week"], r["team_a"], r["team_b"], int(r["team_a_is_home"]),
             r["raw_win_probability_a"], r["predicted_margin"], r["predicted_std"],
             r["team_a_players_found"], r["team_b_players_found"], r["actual_a_win"], r["fetched_at"])
            for r in rows
        ],
    )
    conn.commit()
    return rows, len(games)


def _accuracy(pairs: list[tuple[float, float]]) -> float:
    """Fraction of real games where the predicted favorite (probability >=
    0.5) matches the actual winner. A real tie counts as a win for side A
    here for simplicity; a diagnostic next to Brier score and log-loss, not
    load-bearing anywhere."""
    if not pairs:
        return 0.0
    return sum(1 for p, y in pairs if (p >= 0.5) == (y >= 0.5)) / len(pairs)


def fit_and_store_calibration(conn: sqlite3.Connection, start_season: int, end_season: int) -> dict:
    """Builds the backtest, fits a global Platt-scaling correction, scores the
    heuristic before and after on the same real games, and stores one new
    calibration_params row. Raises ValueError if the range produced no
    backtestable games (the history has not been backfilled). Warns on stderr
    about, rather than silently accepting, a fitted a <= 0:
    weekly.optimize_lineup's argmax shortcut depends on the correction staying
    monotonic increasing."""
    rows, games_considered = build_backtest_sample(conn, start_season, end_season)
    pairs = [(r["raw_win_probability_a"], r["actual_a_win"]) for r in rows]

    if not pairs:
        raise ValueError(
            f"No backtestable real games found for {start_season}-{end_season} ({games_considered} completed games "
            f"in the schedule, none with player stats for both sides). Run "
            f"`dynasty-agent backfill-history --start-season {start_season - 1} --end-season {end_season}` first."
        )

    platt_a, platt_b = fit_platt_scaling(pairs)
    if platt_a <= 0:
        print(
            f"Warning: fitted calibration has a={platt_a:.4f} <= 0, so the raw heuristic's ranking is not preserved by "
            f"this correction and optimize-lineup's argmax shortcut assumes a > 0. Stored anyway since it is the real "
            f"fit, however treat the result as suspect.",
            file=sys.stderr,
        )

    calibrated_pairs = [(platt_scale(p, platt_a, platt_b), y) for p, y in pairs]
    summary = {
        "start_season": start_season,
        "end_season": end_season,
        "sample_size": len(pairs),
        "games_considered": games_considered,
        "platt_a": platt_a,
        "platt_b": platt_b,
        "brier_before": brier_score(pairs),
        "brier_after": brier_score(calibrated_pairs),
        "log_loss_before": log_loss(pairs),
        "log_loss_after": log_loss(calibrated_pairs),
        "accuracy_before": _accuracy(pairs),
        "accuracy_after": _accuracy(calibrated_pairs),
    }
    conn.execute(
        """
        INSERT INTO calibration_params (
            fitted_at, start_season, end_season, sample_size, platt_a, platt_b,
            brier_before, brier_after, log_loss_before, log_loss_after, accuracy_before, accuracy_after
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utcnow(), start_season, end_season, summary["sample_size"], platt_a, platt_b,
            summary["brier_before"], summary["brier_after"], summary["log_loss_before"],
            summary["log_loss_after"], summary["accuracy_before"], summary["accuracy_after"],
        ),
    )
    conn.commit()
    return summary


def current_calibration(conn: sqlite3.Connection) -> tuple[float, float] | None:
    """The latest fitted (platt_a, platt_b), or None if calibrate-matchup-model
    has never been run. Callers fall back to the raw heuristic on None and
    never guess a correction, the same convention vegas_week_multiplier's
    neutral 1.0 uses for a missing input."""
    row = conn.execute("SELECT platt_a, platt_b FROM calibration_params ORDER BY id DESC LIMIT 1").fetchone()
    return (row["platt_a"], row["platt_b"]) if row else None


def usable(params: tuple[float, float] | None) -> tuple[float, float] | None:
    """The fitted pair if applying it is safe, else None. A slope a <= 0 would
    reverse or flatten the heuristic's ranking, so it is never applied;
    fit_and_store_calibration warned when it stored one."""
    return params if params is not None and params[0] > 0 else None
