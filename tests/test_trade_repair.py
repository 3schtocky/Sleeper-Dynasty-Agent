"""The two guards that finish what fix_sides cannot: a free agent on the wrong side of a trade, and a trade
question the model refused to route. Every model output below was captured live from qwen3:4b-instruct."""

import pytest

from dynasty_agent import chat_eval, config
from dynasty_agent.tools import repair_trade_call, trade_from_text


def trade(**args):
    return {"name": "evaluate_trade", "arguments": {"send_players": [], "receive_players": [], "send_picks": [],
                                                    "receive_picks": [], **args}}


def test_a_free_agent_the_model_put_on_the_send_side_is_what_the_user_receives():
    q = "Would you trade my 2027 round 1 pick for Trey Benson?"
    fixed = repair_trade_call(q, trade(send_players=["Trey Benson"], send_picks=["2027 1st"]))
    assert fixed["arguments"]["send_players"] == []
    assert fixed["arguments"]["receive_players"] == ["Trey Benson"]
    assert fixed["arguments"]["send_picks"] == ["2027 1st"]


def test_a_player_and_pick_copied_to_both_sides_are_kept_once_where_the_wording_puts_them():
    q = "Would you trade my 2027 round 1 pick for Trey Benson?"
    both = trade(send_players=["Trey Benson"], send_picks=["2027 1st"], receive_players=["Trey Benson"],
                 receive_picks=["2027 1st"])
    args = repair_trade_call(q, both)["arguments"]
    assert args["send_players"] == [] and args["receive_players"] == ["Trey Benson"]
    assert args["send_picks"] == ["2027 1st"] and args["receive_picks"] == []


@pytest.mark.parametrize("q, call", [
    # the user sends the player, so a player named before "for" stays on the send side
    ("Should I trade Trey Benson for a 2028 second?", trade(send_players=["Trey Benson"], receive_picks=["2028 2nd"])),
    # no "my <pick>" before "for": the wording does not settle the sides
    ("I'm offered Bijan Robinson for Rashee Rice and my 2027 first. Take it?",
     trade(send_players=["Rashee Rice"], receive_players=["Bijan Robinson"], send_picks=["2027 1st"])),
    ("Someone wants my 2027 1st for their Puka Nacua. Good deal?",
     trade(receive_players=["Puka Nacua"], send_picks=["2027 1st"])),
    # not a trade call at all
    ("What should I do with my 2027 pick for Trey Benson?", {"name": "pick_advice", "arguments": {}}),
])
def test_a_trade_the_wording_does_not_prove_wrong_is_left_alone(q, call):
    assert repair_trade_call(q, call) == call


def test_no_call_passes_through():
    assert repair_trade_call("anything for anyone", None) is None


@pytest.fixture
def league(conn, monkeypatch):
    monkeypatch.setattr(config, "SLEEPER_USER_ID", "u1")
    conn.execute("INSERT INTO rosters (roster_id, league_id, owner_id, fetched_at) VALUES (8, 'L', 'u1', 'x'), (12, 'L', 'u2', 'x')")
    for pid, name, pos in [("1", "CeeDee Lamb", "WR"), ("2", "Puka Nacua", "WR"), ("3", "Trey Benson", "RB"),
                           ("4", "Rome Odunze", "WR")]:
        conn.execute("INSERT INTO players (player_id, full_name, position, fetched_at) VALUES (?, ?, ?, 'x')", (pid, name, pos))
    conn.execute("INSERT INTO roster_players VALUES (8, '1', 'starter', 'x'), (12, '2', 'starter', 'x'), (8, '4', 'bench', 'x')")
    return conn  # Benson is on no roster


def test_a_refused_trade_question_becomes_a_trade_from_the_rosters(league):
    call = trade_from_text(league, "Can I get Puka Nacua for CeeDee Lamb straight up?")
    assert call["name"] == "evaluate_trade"
    assert call["arguments"]["send_players"] == ["CeeDee Lamb"]
    assert call["arguments"]["receive_players"] == ["Puka Nacua"]


def test_the_asking_order_does_not_change_the_sides(league):
    a = trade_from_text(league, "Would you give CeeDee Lamb for Puka Nacua?")["arguments"]
    b = trade_from_text(league, "Trade Puka Nacua for CeeDee Lamb?")["arguments"]
    assert a["send_players"] == b["send_players"] == ["CeeDee Lamb"]


@pytest.mark.parametrize("q", [
    "Can I get Trey Benson for CeeDee Lamb?",        # a free agent has no owner, so the sides are not proven
    "Trade CeeDee Lamb for Rome Odunze?",            # both are the user's own
    "How is CeeDee Lamb doing?",                     # one player
    "Who is better, CeeDee Lamb or Puka Nacua?",     # two players, no trade wording
])
def test_it_never_guesses_a_side(league, q):
    assert trade_from_text(league, q) is None


def test_the_scorer_applies_both_guards():
    case = next(c for c in chat_eval.CASES if c.question.startswith("Would you trade my 2027 round 1 pick for Trey Benson"))
    raw = trade(send_players=["Trey Benson"], send_picks=["2027 1st"])
    assert chat_eval.score(case, raw)[0]
