from dynasty_agent.tools import false_premise_players

ME = 1
OWNERS = {"Jonah Coleman": 4, "Trey Benson": 4, "Rome Odunze": ME, "Jaylen Waddle": ME, "Free Guy": None}


def check(send=(), receive=(), send_picks=(), receive_picks=()):
    return false_premise_players(list(send), list(receive), list(send_picks), list(receive_picks), OWNERS.get, ME)


def test_trading_away_a_player_you_dont_own_alongside_your_own_pick_is_a_false_premise():
    # "Should I trade Jonah Coleman and my 2028 2nd for a 2027 1st?"
    assert check(send=["Jonah Coleman"], send_picks=["2028 2nd"], receive_picks=["2027 1st"]) == ["Jonah Coleman"]


def test_the_model_reversing_a_clean_trade_is_not_a_false_premise():
    # "my 2027 round 1 pick for Trey Benson", model put Benson in send and the pick in receive
    assert check(send=["Trey Benson"], receive_picks=["2027 1st"]) == []
    # "a guy offered me his 2028 1st for Rome Odunze", model reversed it
    assert check(receive=["Rome Odunze"], send_picks=["2028 1st"]) == []
    # a player-for-player swap, both reversed
    assert check(send=["Jonah Coleman"], receive=["Rome Odunze"]) == []


def test_a_correct_trade_and_a_free_agent_pass():
    assert check(send=["Rome Odunze"], receive=["Jonah Coleman"]) == []
    assert check(send=["Rome Odunze"], receive=["Free Guy"]) == []


def test_receiving_a_player_you_already_own_is_left_to_fix_sides():
    # "a guy offered me his 2028 1st for Jonah Coleman" read as receiving both, here with the user's own player
    assert check(receive=["Rome Odunze"], receive_picks=["2028 1st"]) == []
