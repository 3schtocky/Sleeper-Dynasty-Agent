import json
import sqlite3

from dynasty_agent import knowledge


def league_conn(slots, rec=1.0):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE league (name, num_teams, scoring_settings_json, roster_positions_json, settings_json, fetched_at)")
    conn.execute("INSERT INTO league VALUES ('L', 12, ?, ?, ?, '2026')", (
        json.dumps({"rec": rec, "pass_td": 4.0}), json.dumps(slots),
        json.dumps({"taxi_slots": 3, "reserve_slots": 1, "waiver_budget": 100, "trade_deadline": 9,
                    "playoff_teams": 6, "playoff_week_start": 15, "draft_rounds": 3, "pick_trading": 1})))
    return conn


def test_one_qb_full_ppr_league_is_described_as_not_superflex():
    facts = knowledge.league_facts(league_conn(["QB", "RB", "WR", "FLEX", "BN", "BN"]))
    assert "full PPR" in facts and "not superflex" in facts and "week 9" in facts and "$100" in facts


def test_a_superflex_slot_is_described_as_superflex():
    facts = knowledge.league_facts(league_conn(["QB", "SUPER_FLEX", "RB", "BN"]))
    assert "It is a superflex league" in facts and "not superflex" not in facts


def test_no_league_cached_still_gives_the_general_concepts():
    assert knowledge.league_facts(None) == "" and "Superflex" in knowledge.reference(None)
