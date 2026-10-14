"""Monte Carlo of the league's remaining regular season: playoff odds, average
final wins and rank, last-place odds. Ported from the earlier
worktree-phase4-college-data branch and rebuilt on this branch's projections.

Sleeper pre-generates every regular-season week's pairings at the start of
the season (confirmed against the real endpoint), so each remaining week has
real, complete matchups to simulate. "Playoff odds" is a regular-season
outcome (who finishes in a qualifying seed), so only weeks before
playoff_week_start are read; playoff-week pairings are never used.

**How a matchup is simulated:** each remaining matchup gets ONE win
probability, computed once: both rosters projected for that week, the gap and
combined spread turned into a probability, then calibrated if a fit is
stored (calibration.py). Each trial is a single weighted coin flip per
matchup, not a resample of every player's distribution, which the win
probability already collapses.

**Stated assumptions, not hidden:**
- Each team fields its best healthy lineup every week (weekly.best_points_lineup
  over its starters and bench), so a starter on a bye or out is replaced. Real
  managers set lineups less well, so every team's odds are slightly flattered.
- Rosters hold as they are today: no trades, waiver pickups or injuries after
  this week are simulated.
- Ties in the standings break on points scored so far, a stand-in for Sleeper's
  own rule, and that tiebreak is static (not simulated).
- A simulated game never ties.
"""

from __future__ import annotations

import json
import random
import sqlite3

from dynasty_agent.calibration import current_calibration, usable
from dynasty_agent.matchup import team_season_avg_implied_points, team_week_implied_points, teams_playing
from dynasty_agent.metrics import matchup_win_probability, platt_scale, starting_slot_counts
from dynasty_agent.weekly import best_points_lineup, position_variance_fallback, project_player

# Players who can start. Taxi and IR (reserve) players cannot be in a lineup.
LINEUP_SLOTS = ("starter", "bench")


def _team_projection(
    conn: sqlite3.Connection,
    stats_season: int,
    roster_id: int,
    slot_counts: dict[str, int],
    week_implied: dict,
    season_avg_implied: dict,
    playing: set[str],
    fallback: dict[str, float],
) -> tuple[float, float]:
    """(mean, variance) of this roster's best healthy lineup for one week, with
    weekly.project_player (reused, not reimplemented) projecting each player."""
    player_ids = [
        r[0]
        for r in conn.execute(
            f"SELECT player_id FROM roster_players WHERE roster_id = ? AND slot IN ({','.join('?' * len(LINEUP_SLOTS))})",
            (roster_id, *LINEUP_SLOTS),
        )
    ]
    projections = {}
    for player_id in player_ids:
        p = project_player(conn, stats_season, player_id, week_implied, season_avg_implied, playing, fallback)
        if p is not None:
            projections[player_id] = p
    lineup = best_points_lineup(projections, slot_counts)
    return (
        sum(projections[pid]["mean"] for _, pid in lineup),
        sum(projections[pid]["variance"] for _, pid in lineup),
    )


def simulate_season(
    conn: sqlite3.Connection,
    stats_season: int,
    vegas_season: int,
    from_week: int,
    n_simulations: int = 10000,
    seed: int | None = None,
    calibration_params: tuple[float, float] | None = None,
) -> dict:
    """Monte Carlo over the league's real remaining regular-season schedule.
    calibration_params None looks up the latest stored fit (like
    weekly.optimize_lineup); with none stored the raw heuristic is used and
    calibration_used says so. Raises ValueError when there is nothing real to
    simulate, never returns a guessed result."""
    league_row = conn.execute(
        "SELECT settings_json, roster_positions_json FROM league ORDER BY fetched_at DESC LIMIT 1"
    ).fetchone()
    if league_row is None:
        raise ValueError("No league data cached yet. Run `dynasty-agent refresh` first.")
    settings = json.loads(league_row["settings_json"])
    playoff_week_start = settings.get("playoff_week_start", 15)
    playoff_teams = settings.get("playoff_teams", 6)
    slot_counts = starting_slot_counts(json.loads(league_row["roster_positions_json"]))

    roster_rows = conn.execute("SELECT roster_id, wins, ties, fpts FROM rosters").fetchall()
    if not roster_rows:
        raise ValueError("No rosters synced yet. Run `dynasty-agent refresh` first.")
    current_wins = {r["roster_id"]: (r["wins"] or 0) + 0.5 * (r["ties"] or 0) for r in roster_rows}
    current_fpts = {r["roster_id"]: (r["fpts"] or 0.0) for r in roster_rows}
    all_roster_ids = list(current_wins)

    calibration_params = usable(current_calibration(conn) if calibration_params is None else calibration_params)
    fallback = position_variance_fallback(conn, stats_season)

    # One win probability per real remaining matchup, computed once, not per trial.
    week_matchups: list[tuple[int, int, float]] = []  # (roster_a, roster_b, win_probability_a)
    for week in range(from_week, playoff_week_start):
        rows = conn.execute(
            "SELECT roster_id, matchup_id FROM matchups WHERE week = ? AND matchup_id IS NOT NULL", (week,)
        ).fetchall()
        if not rows:
            continue  # not synced for this week, skip rather than guess

        by_matchup: dict[int, list[int]] = {}
        for r in rows:
            by_matchup.setdefault(r["matchup_id"], []).append(r["roster_id"])

        week_implied = team_week_implied_points(vegas_season, week)
        season_avg_implied = team_season_avg_implied_points(vegas_season, week)
        playing = teams_playing(vegas_season, week)

        for roster_ids in by_matchup.values():
            if len(roster_ids) != 2:
                continue  # an unpaired or malformed entry, skip rather than guess a result
            roster_a, roster_b = roster_ids
            mean_a, var_a = _team_projection(
                conn, stats_season, roster_a, slot_counts, week_implied, season_avg_implied, playing, fallback)
            mean_b, var_b = _team_projection(
                conn, stats_season, roster_b, slot_counts, week_implied, season_avg_implied, playing, fallback)
            raw_prob_a = matchup_win_probability(mean_a - mean_b, (var_a + var_b) ** 0.5)
            win_prob_a = platt_scale(raw_prob_a, *calibration_params) if calibration_params else raw_prob_a
            week_matchups.append((roster_a, roster_b, win_prob_a))

    if not week_matchups:
        raise ValueError(
            f"No real remaining-schedule matchups found from week {from_week} through "
            f"{playoff_week_start - 1}. Run `dynasty-agent refresh` first so those weeks' matchups are synced."
        )

    rng = random.Random(seed)
    wins_sum = dict.fromkeys(all_roster_ids, 0.0)
    rank_sum = dict.fromkeys(all_roster_ids, 0)
    playoff_count = dict.fromkeys(all_roster_ids, 0)
    last_place_count = dict.fromkeys(all_roster_ids, 0)
    n_teams = len(all_roster_ids)

    for _ in range(n_simulations):
        sim_wins = dict(current_wins)
        for roster_a, roster_b, win_prob_a in week_matchups:
            winner = roster_a if rng.random() < win_prob_a else roster_b
            sim_wins[winner] = sim_wins.get(winner, 0) + 1

        # Points scored so far as the tiebreak: a stated stand-in for Sleeper's
        # rule, static on purpose (simulating full point totals is out of scope).
        standings = sorted(all_roster_ids, key=lambda rid: (-sim_wins.get(rid, 0), -current_fpts.get(rid, 0.0)))
        for rank, rid in enumerate(standings, start=1):
            wins_sum[rid] += sim_wins.get(rid, 0)
            rank_sum[rid] += rank
            if rank <= playoff_teams:
                playoff_count[rid] += 1
            if rank == n_teams:
                last_place_count[rid] += 1

    teams = {
        rid: {
            "current_wins": current_wins[rid],
            "avg_final_wins": wins_sum[rid] / n_simulations,
            "playoff_odds": playoff_count[rid] / n_simulations,
            "avg_final_rank": rank_sum[rid] / n_simulations,
            "last_place_odds": last_place_count[rid] / n_simulations,
        }
        for rid in all_roster_ids
    }
    return {
        "stats_season": stats_season,
        "vegas_season": vegas_season,
        "from_week": from_week,
        "playoff_week_start": playoff_week_start,
        "playoff_teams": playoff_teams,
        "n_simulations": n_simulations,
        "calibration_used": calibration_params is not None,
        "real_matchups_simulated": len(week_matchups),
        "teams": teams,
    }
