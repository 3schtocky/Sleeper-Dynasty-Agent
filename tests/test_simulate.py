"""The season simulation against a tiny league: four teams, two remaining
weeks, so every property can be checked by reasoning, not by trusting output."""

import json

import pytest

from dynasty_agent import simulate, weekly
from tests.test_integration import add_league, add_player, add_roster

STRONG, MID, MID2, WEAK = 1, 2, 3, 4


@pytest.fixture
def schedule(monkeypatch):
    """No Vegas lines; KC and BUF play every week."""
    monkeypatch.setattr(simulate, "team_week_implied_points", lambda season, week: {})
    monkeypatch.setattr(simulate, "team_season_avg_implied_points", lambda season, week: {})
    monkeypatch.setattr(simulate, "teams_playing", lambda season, week: {"KC", "BUF"})


def league(conn, playoff_teams=2, positions=("WR",)):
    add_league(conn, {"playoff_week_start": 6, "playoff_teams": playoff_teams})
    conn.execute("UPDATE league SET roster_positions_json = ?", (json.dumps(list(positions) + ["BN"]),))
    for rid, fppg in ((STRONG, 24.0), (MID, 14.0), (MID2, 12.0), (WEAK, 4.0)):
        add_roster(conn, rid)
        add_player(conn, f"w{rid}", "WR", fppg, roster_id=rid, slot="starter")
    # Real week-to-week swings: with constant scores every variance is 0 and every probability is exactly 0 or 1.
    conn.execute("UPDATE weekly_stats SET fantasy_points = fantasy_points + CASE WHEN week % 2 = 1 THEN 6 ELSE -6 END")
    # Real remaining pairings: week 4 is 1v2 and 3v4, week 5 is 1v3 and 2v4.
    for week, pairs in ((4, [(1, 2), (3, 4)]), (5, [(1, 3), (2, 4)])):
        for matchup_id, (a, b) in enumerate(pairs, start=1):
            for rid in (a, b):
                conn.execute(
                    "INSERT INTO matchups (league_id, week, roster_id, matchup_id, starters_json, players_json, fetched_at) "
                    "VALUES ('L1', ?, ?, ?, '[]', '[]', 't')", (week, rid, matchup_id))


def run(conn, **kw):
    return simulate.simulate_season(conn, 2025, 2026, 4, n_simulations=2000, seed=1, **kw)


def test_the_odds_add_up_to_the_playoff_spots_and_the_games_played(conn, schedule):
    league(conn, playoff_teams=2)
    r = run(conn)
    teams = r["teams"].values()
    assert sum(t["playoff_odds"] for t in teams) == pytest.approx(2.0)  # exactly two make it, every trial
    assert sum(t["avg_final_wins"] for t in teams) == pytest.approx(4.0)  # four real games, one winner each
    assert sum(t["last_place_odds"] for t in teams) == pytest.approx(1.0)
    assert sum(t["avg_final_rank"] for t in teams) == pytest.approx(1 + 2 + 3 + 4)
    assert r["real_matchups_simulated"] == 4


def test_the_stronger_team_has_the_better_odds(conn, schedule):
    league(conn)
    t = run(conn)["teams"]
    assert t[STRONG]["playoff_odds"] > t[MID]["playoff_odds"] > t[WEAK]["playoff_odds"]
    # By hand: a 10-point gap on a 9.8 spread wins 84.7%, a 12-point gap 89%, so about 1.74 wins of 2.
    assert 1.6 < t[STRONG]["avg_final_wins"] < 1.9 and t[WEAK]["avg_final_wins"] < 0.5


def test_a_seed_repeats_the_run_and_another_seed_changes_it(conn, schedule):
    league(conn)
    assert run(conn) == run(conn)
    other = simulate.simulate_season(conn, 2025, 2026, 4, n_simulations=2000, seed=2)
    assert other["teams"] != run(conn)["teams"]


def test_wins_already_banked_carry_into_the_final_count(conn, schedule):
    league(conn)
    conn.execute("UPDATE rosters SET wins = 3 WHERE roster_id = ?", (WEAK,))
    t = run(conn)["teams"][WEAK]
    assert t["current_wins"] == 3 and t["avg_final_wins"] >= 3


def test_a_fit_pulls_probabilities_toward_even_odds(conn, schedule):
    league(conn)
    raw = run(conn)
    assert raw["calibration_used"] is False
    fitted = run(conn, calibration_params=(0.2, 0.0))
    assert fitted["calibration_used"] is True
    assert fitted["teams"][STRONG]["avg_final_wins"] < raw["teams"][STRONG]["avg_final_wins"]
    assert fitted["teams"][WEAK]["avg_final_wins"] > raw["teams"][WEAK]["avg_final_wins"]


def test_a_stored_fit_is_found_and_a_bad_slope_is_ignored(conn, schedule):
    league(conn)
    insert = ("INSERT INTO calibration_params (fitted_at, start_season, end_season, sample_size, platt_a, platt_b, "
              "brier_before, brier_after, log_loss_before, log_loss_after, accuracy_before, accuracy_after) "
              "VALUES ('t', 2018, 2025, 100, ?, 0, 0.3, 0.2, 0.9, 0.6, 0.6, 0.6)")
    conn.execute(insert, (0.2,))
    assert run(conn)["calibration_used"] is True
    conn.execute(insert, (-0.5,))  # the newest fit is unusable
    assert run(conn)["calibration_used"] is False


def test_no_synced_matchups_is_an_error_not_a_guess(conn, schedule):
    league(conn)
    conn.execute("DELETE FROM matchups")
    with pytest.raises(ValueError, match="No real remaining-schedule matchups"):
        run(conn)


def test_a_team_fields_its_best_healthy_lineup_not_a_starter_on_a_bye(conn):
    add_league(conn)
    conn.execute("UPDATE league SET roster_positions_json = ?", (json.dumps(["WR", "BN"]),))
    add_roster(conn, 1)
    add_player(conn, "star_on_bye", "WR", 25.0, roster_id=1, slot="starter", team="MIA")
    add_player(conn, "bench_plays", "WR", 8.0, roster_id=1, slot="bench", team="KC")
    add_player(conn, "on_ir", "WR", 30.0, roster_id=1, slot="reserve", team="KC")  # cannot start
    mean, variance = simulate._team_projection(
        conn, 2025, 1, {"WR": 1}, {}, {}, {"KC", "BUF"}, {})
    assert mean == pytest.approx(8.0)  # the bench WR starts; the bye star scores 0; the IR player is not considered
    assert weekly.project_player(conn, 2025, "bench_plays", {}, {}, {"KC"}, {})["mean"] == pytest.approx(8.0)
