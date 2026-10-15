import random

import pytest

from dynasty_agent import availability as av
from dynasty_agent.availability import PlayerSeason


def season(gsis, year, missed, games=17, pos="WR", age=26.0, injury=None, starter=True):
    """A player-season with `missed` injury-missed games (or `injury` if the
    total and the injury part differ)."""
    inj = missed if injury is None else injury
    return PlayerSeason(gsis, year, pos, age, games, games - missed, inj, missed - inj, starter)


# -- which missed games count as injuries


def test_run_is_injury_when_flagged_inside_or_the_week_before():
    weeks = list(range(1, 11))
    played = {1, 2, 3, 8, 9, 10}                       # missed 4-7
    assert av.classify_season(weeks, played, {4}) == (10, 4, 0)      # flag on the first missed week
    assert av.classify_season(weeks, played, {3}) == (10, 4, 0)      # flag the week before the run
    assert av.classify_season(weeks, played, {6}) == (10, 4, 0)      # IR run flagged mid-way
    assert av.classify_season(weeks, played, set()) == (10, 0, 4)    # no flag: a suspension or benching
    assert av.classify_season(weeks, played, {1}) == (10, 0, 4)      # a flag long before does not count


def test_bye_week_does_not_count_as_a_missed_game_and_flag_may_sit_on_it():
    weeks = [1, 2, 3, 5, 6]                             # bye in week 4
    assert av.classify_season(weeks, {1, 2, 3, 6}, {4}) == (5, 1, 0)  # miss week 5, flag on the bye week before it
    assert av.classify_season(weeks, {1, 2, 3, 5, 6}, {4}) == (5, 0, 0)


# -- the prediction formula


def test_no_history_returns_the_prior_and_history_pulls_toward_the_player():
    assert av.predict_share(0.08, [], 2025, 400, 0.85) == 0.08
    one_bad_year = [(2024, 17, 10)]
    small_k = av.predict_share(0.08, one_bad_year, 2025, 10, 1.0)
    big_k = av.predict_share(0.08, one_bad_year, 2025, 10_000, 1.0)
    assert small_k > 0.3 > 0.08 < big_k < 0.09           # kappa decides how much one year moves it


def test_older_seasons_weigh_less():
    recent_bad = av.predict_share(0.08, [(2023, 17, 0), (2024, 17, 10)], 2025, 20, 0.5)
    old_bad = av.predict_share(0.08, [(2023, 17, 10), (2024, 17, 0)], 2025, 20, 0.5)
    assert recent_bad > old_bad


# -- the backtest must be leak free


def world(persistent: bool, seed=1, players=300, years=range(2017, 2026)):
    rng = random.Random(seed)
    out = []
    for i in range(players):
        rate = rng.choice([0.0, 0.05, 0.4]) if persistent else 0.1
        for y in years:
            missed = sum(rng.random() < rate for _ in range(17))
            out.append(season(f"p{i}", y, missed, age=24.0 + (y - 2017) * 0.0))
    return out


def test_history_cannot_beat_the_prior_when_nothing_persists():
    """Every player-season is an independent draw, so a leak-free model cannot
    do better than the prior. If the season being predicted leaked into the
    history, the model would score near zero and this fails."""
    seasons = world(persistent=False)
    s = av.backtest(seasons, kappa=1, lam=1.0, targets=range(2020, 2026))
    assert s["model"]["all"]["rmse"] > 0.98 * s["prior"]["all"]["rmse"]


def test_history_beats_the_prior_when_players_really_differ():
    seasons = world(persistent=True)
    s = av.backtest(seasons, kappa=17, lam=1.0, targets=range(2020, 2026))
    assert s["model"]["all"]["rmse"] < 0.8 * s["prior"]["all"]["rmse"]


def test_fit_picks_a_heavy_prior_when_history_adds_nothing_and_a_light_one_when_it_does():
    flat = av.fit_model(world(persistent=False, players=200), 2025)
    real = av.fit_model(world(persistent=True, players=200), 2025)
    assert flat["kappa"] > real["kappa"]
    assert real["backtest"]["model"]["all"]["rmse"] < real["backtest"]["league"]["all"]["rmse"]


def test_prior_is_shrunk_for_a_thin_band():
    base = [season(f"a{i}", 2020, 1, age=26.0) for i in range(100)]       # position mean ~ 1/17
    thin = [season("old", 2020, 17, age=33.0, injury=17)]                 # one player, always out
    prior = av.build_prior(base + thin)
    assert prior[("WR", "b3")] < 0.2                                      # nowhere near 1.0
    assert prior[("WR", "b3")] > prior[("WR", None)]


# -- recent misses and report status


def player_with_run(flag_week):
    """Plays weeks 1-3, misses 4-8; the report first flags him in flag_week."""
    weeks = list(range(1, 13))
    return PlayerSeason("g", 2024, "WR", 26.0, 12, 7, 5, 0, True, weeks,
                        frozenset({1, 2, 3, 9, 10, 11, 12}), frozenset({flag_week}), {flag_week: "Out"})


def test_lookback_ignores_a_flag_posted_after_the_cut():
    """At the cut before game 6 he has missed 4 and 5. The first report is in
    week 7, so at the cut nothing says those misses were injuries."""
    p = player_with_run(flag_week=7)
    assert av._recent_at(p, 5) == 0          # before week 6: no flag yet, the run looks like a benching
    assert av._recent_at(p, 8) == 3          # by the cut before game 9 the run is flagged and known
    flagged_early = player_with_run(flag_week=4)
    assert av._recent_at(flagged_early, 5) == 2


def test_cases_report_the_injury_misses_after_the_cut():
    p = player_with_run(flag_week=4)
    cases = {(recent, status): missed for _s, recent, status, missed in av.in_season_cases([p])}
    # before game 6 (index 5) he has missed 4 and 5 with a flag known: recent 2, and the next 4 games hold 3 more misses
    five = next(m for _s, recent, status, m in av.in_season_cases([p]) if recent == 2 and m.get(4) is not None)
    assert five[4] == 3 and five[1] == 1
    assert sum(1 for _ in av.in_season_cases([p])) == 12 - av.RECENT_GAMES


def test_status_table_orders_risk_and_interpolates():
    cases = []
    rng = random.Random(0)
    for _ in range(400):
        cases.append((2020, 0, "none", {1: 0, 2: 0, 4: int(rng.random() < 0.1), 8: int(rng.random() < 0.3), 12: 1}))
        cases.append((2020, 3, "Out", {1: 1, 2: 2, 4: 4, 8: 6, 12: 8}))
    table = av.build_status_table(cases)
    healthy = av.expected_missed(table, 0, "none", 8)
    out = av.expected_missed(table, 3, "Out", 8)
    assert out > healthy
    two, three, four = (av.expected_missed(table, 3, "Out", g) for g in (2, 3, 4))
    assert three == pytest.approx((two + four) / 2)                       # linear between stored horizons
    assert av.expected_missed(table, 3, "Out", 0) == 0.0


def test_a_rare_cell_borrows_from_its_row_then_from_everyone():
    cases = [(2020, 0, "none", {R: 0 for R in av.STATUS_HORIZONS})] * 100
    cases += [(2020, 2, "Doubtful", {R: R for R in av.STATUS_HORIZONS})] * 3            # too few to trust
    table = av.build_status_table(cases)
    key, _ = av.cell_for(table, 2, "Doubtful")
    assert key == "*|*"                                                                   # row "2|*" has 3 cases too


# -- loading from the database


@pytest.fixture
def loaded(conn, monkeypatch):
    weeks = list(range(1, 13))
    monkeypatch.setattr(av, "team_game_weeks", lambda through: {(2024, "KC"): weeks, (2023, "KC"): weeks})
    conn.execute("INSERT INTO player_ids (row_key, gsis_id, name, birthdate, fetched_at) VALUES ('g1','g1','A','2000-09-01','t')")

    def stats(gsis, year, week):
        conn.execute(
            "INSERT INTO weekly_stats (player_id, gsis_id, season, week, team, position, targets, fantasy_points, fetched_at) "
            "VALUES (?, ?, ?, ?, 'KC', 'WR', 8, 10, 't')", (gsis, gsis, str(year), week))

    for w in range(1, 13):                       # 2023: healthy starter
        stats("g1", 2023, w)
    for w in (1, 2, 3, 9, 10, 11, 12):           # 2024: misses 4-8
        stats("g1", 2024, w)
    conn.execute("INSERT INTO nfl_injuries (season, week, gsis_id, team, position, report_status, fetched_at) "
                 "VALUES (2024, 4, 'g1', 'KC', 'WR', 'Out', 't')")
    conn.execute("INSERT INTO nfl_injuries (season, week, gsis_id, team, position, report_status, fetched_at) "
                 "VALUES (2024, 5, 'g1', 'KC', 'WR', 'Questionable', 't')")
    return conn


def test_load_counts_games_misses_age_and_starter(loaded):
    s = {p.season: p for p in av.load_player_seasons(loaded, 2024)}
    assert (s[2023].team_games, s[2023].played, s[2023].injury_missed) == (12, 12, 0)
    assert (s[2024].team_games, s[2024].played, s[2024].injury_missed, s[2024].other_missed) == (12, 7, 5, 0)
    assert s[2024].starter and s[2024].age == pytest.approx(24.0, abs=0.1)
    assert s[2024].statuses == {4: "Out", 5: "Questionable"}


def test_a_player_lost_for_the_year_is_kept_only_with_a_starter_year_before_and_real_flags(loaded, monkeypatch):
    for w in (1, 2, 3):
        loaded.execute("INSERT INTO nfl_injuries (season, week, gsis_id, team, position, report_status, fetched_at) "
                       "VALUES (2025, ?, 'g1', 'KC', 'WR', 'Out', 't')", (w,))
    loaded.commit()
    monkeypatch.setattr(av, "team_game_weeks", lambda through: {(y, "KC"): list(range(1, 13)) for y in (2023, 2024, 2025)})
    s = {p.season: p for p in av.load_player_seasons(loaded, 2025)}
    assert s[2025].played == 0 and s[2025].injury_missed == 12


# -- storage and the per-player answer


def test_fit_store_and_player_risk(loaded, monkeypatch):
    seasons = av.load_player_seasons(loaded, 2024)
    monkeypatch.setattr(av, "load_player_seasons", lambda c, through: seasons * 1)
    monkeypatch.setattr(av, "fit_model", lambda s, last: {
        "kappa": 400, "lambda": 0.85, "prior": {"None|None": 0.08, "WR|None": 0.09},
        "backtest": {"model": {"all": {"rmse": 0.1}}, "prior": {"all": {"rmse": 0.1}}, "league": {"all": {"rmse": 0.1}}},
        "grid": [], "persistence": {"n": 3, "r": 0.0}, "start_season": 2023, "end_season": last,
        "sample_size": len(s), "other_share": 0.1})
    cases = []
    for _ in range(100):
        cases.append((2024, 0, "none", {R: 0.05 * R for R in av.STATUS_HORIZONS}))
        cases.append((2024, 3, "Out", {R: 0.8 * R for R in av.STATUS_HORIZONS}))
    monkeypatch.setattr(av, "in_season_cases", lambda s: cases)
    monkeypatch.setattr(av, "status_holdout", lambda s, last: {4: {"n": 1, "rmse_model": 1, "rmse_baseline": 2},
                                                               8: {"n": 1, "rmse_model": 1, "rmse_baseline": 2}})
    assert av.current_model(loaded) is None
    av.fit_and_store(loaded, 2024)
    model = av.current_model(loaded)
    assert model["kappa"] == 400 and model["status_table"]["3|Out"][4] == pytest.approx(3.2)
    loaded.execute("INSERT INTO players (player_id, full_name, position, team, age, injury_status, fetched_at) "
                   "VALUES ('s1', 'A', 'WR', 'KC', 24, 'IR', 't')")
    loaded.execute("UPDATE player_ids SET sleeper_id = 's1' WHERE gsis_id = 'g1'")
    monkeypatch.setattr(av, "remaining_games", lambda season, team, week: 8)
    index = {}
    for p in seasons:
        index.setdefault(p.gsis_id, []).append(p)
    ir = av.player_risk(loaded, model, "s1", index, 2025, 10)
    assert ir["label"] == "high" and ir["status_is_proxy"] and ir["missed_per_17"] > 10
    assert any("no body part or timeline" in r for r in ir["reasons"])
    loaded.execute("UPDATE players SET injury_status = NULL WHERE player_id = 's1'")
    healthy = av.player_risk(loaded, model, "s1", index, 2025, 10)
    assert healthy["label"] == "typical" and healthy["a_now"] > ir["a_now"]
    assert av.player_risk(loaded, model, "nobody", index, 2025, 10) is None
    with av.assumptions({"s1": 6}):                       # the user's word: out 6 of the 8 games left
        said = av.player_risk(loaded, model, "s1", index, 2025, 10)
    assert said["assumed_games_out"] == 6 and said["label"] == "high" and said["missed_per_17"] > healthy["missed_per_17"]
    assert any("you told me he is out about 6 games" in r and "can't verify" in r for r in said["reasons"])
    assert av.player_risk(loaded, model, "s1", index, 2025, 10)["assumed_games_out"] is None   # never leaks past the call


# -- risk is shown beside the numbers, never in them


def risk(label="elevated", per_17=2.9, healthy=1.1, reasons=("missed 2 of his last 3 games with an injury",), r=0.06):
    return {"label": label, "missed_per_17": per_17, "healthy_per_17": healthy, "reasons": list(reasons), "history_r": r}


def test_risk_line_and_block():
    from dynasty_agent import formatters
    line = formatters.risk_line("Jayden Daniels", risk())
    assert line.startswith("Jayden Daniels: elevated injury risk, about 2.9 expected games missed per 17 (a healthy player at this point: 1.1).")
    assert "Missed 2 of his last 3 games with an injury." in line
    block = formatters.risk_block([{"name": "A", "risk": risk()}])
    assert "not in them" in block[-1] and "0.06" in block[-1] and "no body part or timeline" in block[-1]
    assert formatters.risk_block([]) == []


def test_trade_prints_risk_without_changing_the_numbers():
    from dynasty_agent import formatters
    base = {"discount_rate": 0.2, "warnings": [], "sent": {"players": [], "picks": [], "win_now_total": 0, "player_three_year_total": 0,
            "market_value_total": 0, "unpriced": []}, "received": {"players": [], "picks": [], "win_now_total": 0, "player_three_year_total": 0,
            "market_value_total": 0, "unpriced": []}, "win_now_delta": 3.4, "player_three_year_delta": 4.1, "market_value_delta": 1421,
            "posture_label": "X", "posture_confidence": "low", "fit": "f", "consolidation": None}
    without = formatters.format_trade({**base, "risk": []}, 2026)
    with_risk = formatters.format_trade({**base, "risk": [{"name": "Jayden Daniels", "risk": risk()}]}, 2026)
    assert with_risk.startswith(without) and "elevated injury risk" in with_risk and "Net market value" in without


def test_points_flag_only_elevated_or_high_and_say_it_is_not_in_the_numbers():
    from dynasty_agent import formatters, points
    elevated = formatters.risk_line("D", risk())
    typical = formatters.risk_line("S", risk(label="typical", per_17=1.1, reasons=()))
    out = points._risk_points([typical, elevated], "trade")
    assert out[0] == elevated and "not in the numbers" in out[-1] and typical not in out
    assert points._risk_points([typical], "trade") == [] and points._risk_points(None, "trade") == []


def test_no_fitted_model_means_no_risk_rows(conn):
    assert av.risk_rows(conn, [("1", "A")]) == []
