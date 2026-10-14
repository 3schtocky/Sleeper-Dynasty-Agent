"""The calibration math (logit, Platt scaling, Brier score, log-loss). Ported
from the worktree-phase4-college-data branch with its own tests; calibration.py
fits it and every win probability now uses it."""

import pytest

from dynasty_agent.metrics import brier_score, fit_platt_scaling, log_loss, logit, platt_scale, sigmoid



def test_logit_and_sigmoid_are_inverses():
    assert round(sigmoid(logit(0.7)), 6) == 0.7


def test_logit_clamps_extreme_probabilities_instead_of_raising():
    # An exact 0.0/1.0 is real and reachable (matchup_win_probability
    # returns it on a degenerate zero-variance side), never +/-inf.
    assert logit(0.0) < -10
    assert logit(1.0) > 10


def test_sigmoid_at_zero_is_one_half():
    assert sigmoid(0.0) == 0.5


def test_sigmoid_handles_large_magnitude_without_overflow():
    assert sigmoid(-1000.0) == 0.0
    assert round(sigmoid(1000.0), 6) == 1.0


def test_platt_scale_is_identity_at_a1_b0():
    assert round(platt_scale(0.7, 1.0, 0.0), 6) == 0.7


def test_fit_platt_scaling_returns_identity_on_empty_sample():
    assert fit_platt_scaling([]) == (1.0, 0.0)


def test_fit_platt_scaling_recovers_known_overconfidence():
    # Raw probabilities of 0.9/0.1 are overconfident here by construction:
    # the real empirical win rate at each extreme is only 70%/30%, not
    # 90%/10%. A correct fit should deflate that (a < 1.0) and measurably
    # improve Brier score on the same sample.
    pairs = [(0.9, 1.0)] * 7 + [(0.9, 0.0)] * 3 + [(0.1, 0.0)] * 7 + [(0.1, 1.0)] * 3
    a, b = fit_platt_scaling(pairs, iterations=1000, learning_rate=0.1)
    assert a < 1.0
    calibrated_pairs = [(platt_scale(p, a, b), y) for p, y in pairs]
    assert brier_score(calibrated_pairs) < brier_score(pairs)


def test_brier_score_perfect_predictions_score_zero():
    assert brier_score([(1.0, 1.0), (0.0, 0.0)]) == 0.0


def test_brier_score_always_predicting_half_scores_one_quarter():
    assert brier_score([(0.5, 1.0), (0.5, 0.0)]) == 0.25


def test_brier_score_empty_sample_is_zero():
    assert brier_score([]) == 0.0


def test_log_loss_confident_correct_prediction_scores_near_zero():
    assert log_loss([(0.99, 1.0)]) < 0.02


def test_log_loss_confident_wrong_prediction_scores_high():
    assert log_loss([(0.99, 0.0)]) > 4.0


def test_log_loss_empty_sample_is_zero():
    assert log_loss([]) == 0.0


def test_a_slope_only_fit_returns_no_intercept_and_treats_both_teams_alike():
    import random

    rng = random.Random(4)
    raws = [rng.uniform(0.05, 0.95) for _ in range(400)]
    # An overconfident heuristic whose results favor the first-named team a little: a free intercept would learn that.
    pairs = [(p, 1.0 if rng.random() < platt_scale(p, 0.4, 0.3) else 0.0) for p in raws]
    a, b = fit_platt_scaling(pairs, fit_intercept=False)
    assert b == 0.0 and 0.0 < a < 1.0
    for p in (0.1, 0.35, 0.5, 0.8):
        assert platt_scale(1 - p, a, b) == pytest.approx(1 - platt_scale(p, a, b))  # swap the teams, get the complement
        assert (platt_scale(p, a, b) >= 0.5) == (p >= 0.5)  # and the favorite never changes
    _, free_b = fit_platt_scaling(pairs)
    assert free_b > 0.1  # whereas the free fit learned the shift, and so breaks both properties
    assert platt_scale(0.45, 0.4, free_b) > 0.5 > 0.45

