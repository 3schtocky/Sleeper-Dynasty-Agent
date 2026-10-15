import json

import pytest

from dynasty_agent import context, config, market, player_check, tools, trade_finder
from tests.test_integration import add_league, add_player, add_roster, fake_pick_values


def price(conn, pid, value):
    conn.execute("INSERT INTO market_values (player_id, source, as_of_date, value, fetched_at) VALUES (?, 'fantasycalc', '2026-01-01', ?, 't')",
                 (pid, value))


@pytest.fixture
def world(conn, monkeypatch):
    """Me (roster 1): three starting WRs and a bench WR with a price. Roster 2: a WR star and a QB who would not start for me."""
    monkeypatch.setattr(market, "fetch_values", lambda conn: fake_pick_values({"2027 1st": 2800, "2028 1st": 2000}))
    add_league(conn)
    conn.execute("UPDATE league SET settings_json = ?", (json.dumps({"playoff_week_start": 15, "draft_rounds": 3}),))
    add_roster(conn, 1)
    add_roster(conn, 2)
    for pid, pos, fppg, slot in [("q1", "QB", 20.0, "starter"), ("r1", "RB", 14.0, "starter"), ("r2", "RB", 13.0, "starter"),
                                 ("w1", "WR", 12.0, "starter"), ("w2", "WR", 11.0, "starter"), ("w3", "WR", 10.0, "starter"),
                                 ("t1", "TE", 9.0, "starter"), ("b1", "WR", 8.0, "bench"), ("b2", "RB", 7.0, "bench")]:
        add_player(conn, pid, pos, fppg, roster_id=1, slot=slot)
        price(conn, pid, 1000 + fppg * 100)
    add_player(conn, "star", "WR", 25.0, roster_id=2, slot="starter")
    add_player(conn, "qb2", "QB", 15.0, roster_id=2, slot="starter")
    for i in range(10):  # peers, so the buy/sell ranks have a group to compare against
        add_player(conn, f"p{i}", "WR", 5.0 + i, roster_id=2, slot="bench")
        price(conn, f"p{i}", 600 + 100 * i)
    price(conn, "star", 3500)
    price(conn, "qb2", 3000)
    monkeypatch.setattr(config, "SLEEPER_USER_ID", "u1")
    conn.execute("UPDATE rosters SET owner_id = 'u1' WHERE roster_id = 1")
    conn.execute("UPDATE players SET full_name = 'Star Receiver' WHERE player_id = 'star'")
    conn.commit()
    return conn


def test_the_finder_offers_spare_assets_for_the_player_who_lifts_the_lineup(world):
    r = trade_finder.find(world, 2025, 1, "WR")
    assert r["ideas"], "star WR should be found"
    idea = r["ideas"][0]
    assert idea["candidate"]["id"] == "star" and idea["offer"]["net"] > trade_finder.MIN_GAIN
    sent = {a["id"] for a in idea["offer"]["assets"]}
    assert not sent & {"w1", "w2", "w3", "q1", "r1", "r2"}, "never offers a starter the lineup needs"
    assert idea["label"] in ("fair", "a stretch", "a lowball")
    assert idea["checked"]["warnings"] == []


def test_a_qb_who_would_not_start_for_me_is_not_suggested(world):
    assert trade_finder.find(world, 2025, 1, "QB")["ideas"] == []


def test_no_position_picks_the_spot_with_the_biggest_upgrade(world):
    assert trade_finder.find(world, 2025, 1, None)["position"] == "WR"
    assert trade_finder.find(world, 2025, 1, None)["chosen_by"] == "biggest upgrade"


def test_numbers_text_names_the_players_and_ranks_by_lineup_gain(world):
    text = trade_finder.format_ideas(trade_finder.find(world, 2025, 1, "WR"))
    assert "Get Star Receiver" in text and "Your lineup: +" in text and "Would they say yes?" in text


def test_detect_finds_trade_requests_and_the_position():
    d = trade_finder.detect
    assert d("help me find good trade ideas, I need some new WR's in my lineup") == {"position": "WR"}
    assert d("I need an RB") == {"position": "RB"}
    assert d("find me trade targets") == {"position": None}
    assert d("who should I target at tight end?") == {"position": "TE"}
    assert d("I need help with my team") is None
    assert d("Should I trade Rome Odunze for Puka Nacua?") is None


def test_the_tool_is_python_only_and_the_model_still_sees_six_tools(world):
    assert tools.TOOL_NAMES == ["set_lineup", "evaluate_trade", "my_team", "waiver_targets", "pick_advice", "taxi_plan"]
    r = tools.run_tool(world, "find_trades", {"position": "WR"})
    assert r.compact["ideas"][0]["get"].startswith("Star Receiver") and r.compact["facts"]
    assert tools.detect_action(world, "I need a WR")["name"] == "find_trades"


def test_player_check_detects_one_player_a_question_asks_about(world):
    assert player_check.detect(world, "Is Star Receiver a buy or a sell?") == {"player_id": "star"}
    assert player_check.detect(world, "Should I trade Star Receiver for Player w1?") is None  # a trade is the router's
    assert player_check.detect(world, "Who should I start, Star Receiver?") is None
    assert player_check.detect(world, "What's the weather like?") is None


def test_player_check_card_for_a_player_on_another_team(world):
    c = player_check.card(world, 2025, 1, "star")
    assert c["where"].startswith("on ") and not c["mine"] and c["lineup_gain"] > 0
    assert c["lean"] in ("buy", "sell", "fair")
    r = tools.run_tool(world, "player_check", {"player_id": "star"})
    assert "Star Receiver" in r.numbers and any("trade offer" in f for f in r.compact["facts"])


def test_player_check_on_my_own_player_has_no_lineup_gain(world):
    c = player_check.card(world, 2025, 1, "w1")
    assert c["mine"] and "lineup_gain" not in c and c["where"] == "on your roster"


def test_follow_ups_to_a_finder_or_player_answer_are_finished_by_python(world):
    assert tools.detect_follow_up(world, "what about running back?", "find_trades") == {"name": "find_trades", "arguments": {"position": "RB"}}
    assert tools.detect_follow_up(world, "what about Star Receiver?", "player_check") == {"name": "player_check", "arguments": {"player_id": "star"}}
    assert tools.detect_follow_up(world, "what about running back?", "my_team") is None
    assert tools.detect_follow_up(world, "what about pizza?", "find_trades") is None
