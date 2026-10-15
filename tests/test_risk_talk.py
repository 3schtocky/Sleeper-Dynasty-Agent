"""Phase 3: nicknames, a surname already under discussion, injury questions and
reports, and pushback that must not flip a verdict."""

import io
import json
import sqlite3

import pytest

from dynasty_agent import chat, conversation_eval as ce, player_check, risk_talk, tools, valuation
from dynasty_agent.config import MIGRATIONS_DIR
from dynasty_agent.db import apply_migrations
from dynasty_agent.valuation import AmbiguousPlayer, resolve_player

LINEUP = ["QB", "RB", "RB", "WR", "WR", "WR", "TE", "FLEX"] + ["BN"] * 10


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    apply_migrations(c, MIGRATIONS_DIR)
    c.execute("INSERT INTO league (league_id, season, scoring_settings_json, roster_positions_json, settings_json, fetched_at) "
              "VALUES ('L1', '2026', '{}', ?, '{}', 't')", (json.dumps(LINEUP),))
    for pid, name, pos, team in [
        ("421", "Matthew Stafford", "QB", "LAR"), ("9", "Jayden Daniels", "QB", "WAS"),
        ("10", "Marcus Daniels", "RB", "NYJ"), ("11", "Chris Godwin", "WR", "TB"),
        ("12", "Christopher Daniels", "WR", None),
    ]:
        c.execute("INSERT INTO players (player_id, full_name, position, team, fetched_at) VALUES (?, ?, ?, ?, 't')",
                  (pid, name, pos, team))
    return c


# -- nicknames


def test_a_short_first_name_finds_the_full_name_but_never_overrides_an_exact_match(conn):
    assert resolve_player(conn, "Matt Stafford")["player_id"] == "421"
    assert resolve_player(conn, "Chris Godwin")["player_id"] == "11"          # exact match wins over Christopher
    assert resolve_player(conn, "matthew stafford")["player_id"] == "421"
    assert valuation.name_variants("matt stafford")[:2] == ["matt stafford", "matthew stafford"]


def test_the_player_detectors_see_a_nickname(conn):
    assert player_check.detect(conn, "Is Matt Stafford a buy or a sell?") == {"player_id": "421"}
    assert player_check.named_player_ids(conn, "what about matt stafford") == ["421"]


# -- a surname that matches several players means the one under discussion


def test_an_ambiguous_surname_resolves_to_the_player_already_in_the_chat(conn):
    with pytest.raises(AmbiguousPlayer):
        resolve_player(conn, "Daniels")
    earlier = tools.ToolResult("evaluate_trade", data={"received": {"players": [{"player_id": "9", "full_name": "Jayden Daniels"}]}})
    fixed = tools._resolve_in_context(conn, {"receive_players": ["Daniels"], "send_players": ["Matthew Stafford"]}, [earlier])
    assert fixed == {"receive_players": ["9"], "send_players": ["Matthew Stafford"]}
    # two candidates under discussion stay ambiguous, and no context changes nothing
    both = tools.ToolResult("evaluate_trade", data={"p": [{"player_id": "9", "full_name": "Jayden Daniels"},
                                                          {"player_id": "10", "full_name": "Marcus Daniels"}]})
    assert tools._resolve_in_context(conn, {"player": "Daniels"}, [both]) == {"player": "Daniels"}
    assert tools._resolve_in_context(conn, {"player": "Daniels"}, []) == {"player": "Daniels"}


# -- what counts as an injury question, a report, or pushback


@pytest.mark.parametrize("text,risk,report", [
    ("what about his injury risk?", True, False),
    ("is Daniels injury prone", True, False),
    ("how durable is he", True, False),
    ("Daniels just got hurt", False, True),
    ("he's out 6 weeks", False, True),
    ("Daniels tore his ACL", False, True),
    ("did Daniels get hurt?", True, False),
    ("how does injury affect running backs", True, False),     # a risk-shaped question; no pronoun, so no player is found
    ("who should I start", False, False),
])
def test_risk_questions_and_reports_are_told_apart(text, risk, report):
    assert risk_talk.is_risk_question(text) is risk
    assert risk_talk.is_report(text) is report


@pytest.mark.parametrize("text,games", [("about 6 games", 6), ("maybe six weeks", 6), ("a month", 4),
                                        ("season-ending", risk_talk.SEASON_ENDING), ("out for the year", risk_talk.SEASON_ENDING),
                                        ("not sure", None)])
def test_games_out_is_read_from_plain_words(text, games):
    assert risk_talk.games_out_in(text) == games


def test_pushback_is_recognized_but_a_new_trade_ask_is_not():
    assert risk_talk.is_pushback("I disagree, Daniels is way better and I want him")
    assert risk_talk.is_pushback("are you sure?")
    assert not risk_talk.is_pushback("what if I add my 2028 2nd?")
    assert not risk_talk.is_pushback("should I trade Lamb for Nacua")


def test_players_for_prefers_a_name_then_a_surname_in_context_then_a_pronoun(conn):
    known = {"9": "Jayden Daniels", "421": "Matthew Stafford"}
    assert risk_talk.players_for(conn, "is Matt Stafford hurt", known) == [("421", "Matthew Stafford")]
    assert risk_talk.players_for(conn, "Daniels just got hurt", known) == [("9", "Jayden Daniels")]
    assert risk_talk.players_for(conn, "what about his injury risk?", known) == list(known.items())
    assert risk_talk.players_for(conn, "how does injury affect running backs", known) == []
    assert risk_talk.players_for(conn, "what about his injury risk?", {}) == []


def test_detect_action_routes_each_kind(conn):
    trade = tools.ToolResult("evaluate_trade", data={"received": {"players": [{"player_id": "9", "full_name": "Jayden Daniels"}]}})
    risk = tools.detect_action(conn, "what about his injury risk?", [trade], "evaluate_trade")
    assert risk == {"name": "risk_check", "arguments": {"player_ids": ["9"]}}
    report = tools.detect_action(conn, "Daniels just got hurt", [trade], "evaluate_trade")
    assert report["arguments"]["report"] and report["arguments"]["games_out"] is None
    assert tools.detect_action(conn, "I disagree, I want him", [trade], "evaluate_trade") == {"name": "pushback", "arguments": {}}
    assert tools.detect_action(conn, "I disagree, I want him", [trade], "set_lineup") is None
    assert tools.detect_action(conn, "how does injury affect running backs", [trade], "evaluate_trade") is None


# -- pushback never flips the verdict


def trade(value, win_now, risk=None):
    return {"market_value_delta": value, "win_now_delta": win_now, "risk": risk or [],
            "received": {"players": [{"player_id": "9", "full_name": "Jayden Daniels"}]}}


def risk_row(label="elevated"):
    return {"name": "Jayden Daniels", "risk": {"label": label, "missed_per_17": 2.9, "healthy_per_17": 1.1,
                                              "reasons": ["missed 2 of his last 3 games with an injury"], "history_r": 0.06}}


def test_the_stance_follows_the_numbers_not_the_pushback():
    for value, win_now, words in [(1421, 3.4, "favor you on both"), (-900, -2.0, "favor the other team on both"),
                                  (500, -1.0, "mixed: you win on trade value")]:
        text, _ = risk_talk.pushback_reply(trade(value, win_now))
        assert words in text and "my read stands" in text
        assert "you're right" not in text.lower() and "i was wrong" not in text.lower()


def test_the_question_is_about_the_incoming_players_games_missed_when_risk_is_elevated():
    text, asked = risk_talk.pushback_reply(trade(1421, 3.4, [risk_row()]))
    assert "How many games do you expect Jayden Daniels to miss" in text and "not in those numbers" in text
    assert asked == ["9"]
    typical, asked = risk_talk.pushback_reply(trade(1421, 3.4, [risk_row("typical")]))
    assert "What do you see that the numbers don't?" in typical and asked == []


# -- the chat keeps the user's answer for the question it asked


class NoModel:
    model = "none"

    def route(self, *a, **k):
        raise AssertionError("the router must not be reached")

    def stream(self, *a, **k):
        raise AssertionError("the topic check must not be reached")


def test_a_reported_injury_asks_for_games_then_uses_the_answer_as_an_assumption(monkeypatch):
    seen = []

    def run(conn, name, args, recent=None, overrides=None):
        seen.append((name, dict(args), dict(overrides or {})))
        if args.get("report") and args.get("games_out") is None:
            return tools.ToolResult(name, clarification="Roughly how many games?")
        return tools.ToolResult(name, numbers="Jayden Daniels: high injury risk", take=False)

    monkeypatch.setattr(tools, "run_tool", run)
    monkeypatch.setattr(tools, "detect_action", lambda conn, q, recent=None, last_tool=None:
                        {"name": "risk_check", "arguments": {"player_ids": ["9"], "report": True, "games_out": None}}
                        if "hurt" in q else None)
    s = chat.ChatSession(conn=None, client=NoModel(), out=io.StringIO(), width=80)
    s.ask("Daniels just got hurt")
    assert s.pending_games == ["9"]
    s.ask("about 6 games")
    assert seen[-1] == ("risk_check", {"player_ids": ["9"], "games_out": 6}, {"9": 6})
    assert s.pending_games == [] and s.overrides == {"9": 6}


def test_an_unrelated_reply_clears_the_pending_question(monkeypatch):
    monkeypatch.setattr(tools, "detect_action", lambda *a, **k: None)
    s = chat.ChatSession(conn=None, client=NoModel(), out=io.StringIO(), width=80)
    s.pending_games = ["9"]
    with pytest.raises(AssertionError):          # reaches the router, which is the point
        s.ask("who should I start")
    assert s.pending_games == []


# -- the eval rule that catches a swapped label


def test_agrees_with_facts_catches_elevated_called_high():
    t = chat.Turn("q", compact={"facts": ["Jayden Daniels: elevated injury risk, about 2.9 expected games missed per 17."]},
                  reply="Jayden Daniels has a high injury risk, about 2.9 games per 17.")
    assert "elevated injury risk, reply says high" in ce.agrees_with_facts()(t, "", None)
    ok = chat.Turn("q", compact=t.compact, reply="Jayden Daniels has elevated injury risk, about 2.9 games per 17.")
    assert ce.agrees_with_facts()(ok, "", None) is None


def test_run_tool_applies_the_context_resolution(conn, monkeypatch):
    """The call site, not just the helper: the runner must receive the resolved id."""
    from dynasty_agent import context
    monkeypatch.setattr(context, "my_roster_id", lambda c: 1)
    monkeypatch.setattr(context, "stats_season", lambda c: 2026)
    got = {}
    monkeypatch.setitem(tools._RUNNERS, "evaluate_trade", lambda c, me, season, args: got.update(args) or tools.ToolResult(
        "evaluate_trade", numbers="x", take=False))
    earlier = tools.ToolResult("evaluate_trade", data={"received": {"players": [{"player_id": "9", "full_name": "Jayden Daniels"}]}})
    tools.run_tool(conn, "evaluate_trade", {"receive_players": ["Daniels"]}, recent=[earlier])
    assert got["receive_players"] == ["9"]


def test_pushback_still_means_the_trade_after_a_risk_answer(conn):
    trade = tools.ToolResult("evaluate_trade", data={"received": {"players": [{"player_id": "9", "full_name": "Jayden Daniels"}]}})
    risk = tools.ToolResult("risk_check", data={"players": [{"player_id": "9", "full_name": "Jayden Daniels"}]})
    assert tools.detect_action(conn, "I disagree, I want him anyway", [trade, risk], "risk_check") == {"name": "pushback", "arguments": {}}
    assert tools.detect_action(conn, "I disagree, I want him anyway", [risk], "risk_check") is None     # no trade to argue about


def test_the_trade_take_gets_a_bottom_line_only_when_risk_is_flagged():
    from dynasty_agent import points
    base = {"net_market_value": "1421", "net_win_now": "+3.4"}
    flagged = ["Jayden Daniels: elevated injury risk, about 2.9 expected games missed per 17 (a healthy player at this point: 1.1)."]
    out = points._trade({**base, "injury_risk": flagged})
    assert any(x.startswith("Bottom line: on the numbers this favors you") and "Jayden Daniels" in x and "your call" in x for x in out)
    assert not any(x.startswith("Bottom line") for x in points._trade(base))
    mixed = points._trade({"net_market_value": "500", "net_win_now": "-1.0", "injury_risk": flagged})
    assert any(x.startswith("Bottom line: a close call") for x in mixed)


def test_agrees_with_facts_catches_make_the_trade_on_flagged_risk_and_dont_trade_on_a_favorable_trade():
    facts = ["The numbers favor you on both trade value and win-now points.",
             "Jayden Daniels: elevated injury risk, about 2.9 expected games missed per 17."]
    bad_make = chat.Turn("q", compact={"facts": facts}, reply="It favors you. Make the trade in the Sleeper app.")
    assert "flag injury risk" in ce.agrees_with_facts()(bad_make, "", None)
    bad_no = chat.Turn("q", compact={"facts": facts}, reply="No, don't trade Stafford for Daniels.")
    assert "reply says don't trade" in ce.agrees_with_facts()(bad_no, "", None)
    good = chat.Turn("q", compact={"facts": facts}, reply="The numbers favor you, but weigh Daniels' injury risk. It's your call.")
    assert ce.agrees_with_facts()(good, "", None) is None


FACTS = ("Jayden Daniels: elevated injury risk, about 2.9 expected games missed per 17. Matthew Stafford: typical injury risk. "
         "Bottom line: on the numbers this favors you, and the injury risk on Jayden Daniels is the thing to weigh.")


def test_a_sentence_that_swaps_the_label_or_directs_the_move_is_flagged():
    assert risk_talk.contradicts("Daniels has a high injury risk.", FACTS)
    assert risk_talk.contradicts("Stafford's risk is low and Daniels is elevated.", FACTS) == []        # "low" without "risk" after is fine
    assert risk_talk.contradicts("Stafford has low risk.", FACTS)
    assert risk_talk.contradicts("Make it in the Sleeper app.", FACTS)
    assert risk_talk.contradicts("You should make the trade.", FACTS)
    assert risk_talk.contradicts("Daniels has elevated injury risk and Stafford typical injury risk.", FACTS) == []
    assert risk_talk.contradicts("It is your call, so weigh the risk.", FACTS) == []
    assert risk_talk.contradicts("Make it in the Sleeper app.", "A plain trade fact with no risk.") == []     # only when risk is in play
    assert risk_talk.contradicts("Make it in the Sleeper app.", "Jayden Daniels: elevated injury risk.") == []  # no bottom line, no ban


def test_the_chat_leaves_such_a_sentence_out_and_keeps_the_rest(monkeypatch):
    class Model(NoModel):
        def stream(self, messages, options=None):
            yield "The numbers favor you, so weigh Daniels' elevated injury risk. Make it in the Sleeper app."

    result = tools.ToolResult("evaluate_trade", numbers="n", compact={"facts": [FACTS]})
    s = chat.ChatSession(conn=None, client=Model(), out=io.StringIO(), width=80)
    text = s._take("q", {"name": "evaluate_trade", "arguments": {}}, result, [])
    assert "weigh Daniels" in text and "Sleeper app" not in text
    assert any("contradicts the facts" in d and "Sleeper app" in d for d in s.last_dropped)
