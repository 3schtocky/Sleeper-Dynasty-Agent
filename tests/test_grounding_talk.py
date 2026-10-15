from dynasty_agent import grounding


def bad(text, *known):
    return grounding.unsupported_claims(text, grounding.allowed_numbers(*known))


def test_ages_counts_and_years_pass_in_talk():
    assert not bad("Backs usually decline around age 26 or 27, so look at the top 10 picks in the 2027 class.")
    assert not bad("A 28-year-old with 3 years left is risky.")


def test_stats_and_values_are_blocked_in_talk():
    assert bad("He scores 20 points a game.") == {"20"}
    assert bad("About 70% of them bust.") == {"70"}
    assert bad("He is worth 7786 in trade value.") == {"7786"}
    assert bad("He averages 16.5.") == {"16.5"}
    assert bad("He had 1,200 yards.") == {"1200"}


def test_a_stat_already_shown_is_allowed():
    assert not bad("That is 28.5 points a week.", "win-now 28.5")
