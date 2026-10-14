import pytest

from dynasty_agent import matchup
from dynasty_agent.metrics import matchup_win_probability, platt_scale


@pytest.fixture
def stubbed(monkeypatch):
    """predict_matchup with no schedule file or player data: side A projects
    well ahead of side B."""
    monkeypatch.setattr(matchup, "team_week_implied_points", lambda season, week: {"KC": 27.0})
    monkeypatch.setattr(matchup, "team_season_avg_implied_points", lambda season, week: {})
    monkeypatch.setattr(matchup, "teams_playing", lambda season, week: {"KC"})
    sides = {"a": {"mean": 120.0, "variance": 150.0}, "b": {"mean": 100.0, "variance": 150.0}}
    monkeypatch.setattr(matchup, "_value_matchup_side", lambda conn, season, names, *rest: sides[names[0]])


def test_without_a_fit_only_the_raw_probability_is_reported(conn, stubbed):
    r = matchup.predict_matchup(conn, 2025, 2026, 6, ["a"], ["b"])
    raw = matchup_win_probability(20.0, 300 ** 0.5)
    assert r["win_probability_a"] == pytest.approx(raw)
    assert r["win_probability_b"] == pytest.approx(1 - raw)
    assert r["win_probability_a_calibrated"] is None
    assert r["win_probability_b_calibrated"] is None


def test_a_fit_adds_calibrated_values_beside_the_raw_ones(conn, stubbed):
    params = (0.26, -0.01)
    r = matchup.predict_matchup(conn, 2025, 2026, 6, ["a"], ["b"], calibration_params=params)
    raw = r["win_probability_a"]
    assert raw == pytest.approx(matchup_win_probability(20.0, 300 ** 0.5))  # raw untouched
    assert r["win_probability_a_calibrated"] == pytest.approx(platt_scale(raw, *params))
    assert r["win_probability_a_calibrated"] < raw  # an overconfident favorite is pulled toward 50%
    assert r["win_probability_a_calibrated"] > 0.5  # but still the favorite
    assert r["win_probability_a_calibrated"] + r["win_probability_b_calibrated"] == pytest.approx(1.0)
