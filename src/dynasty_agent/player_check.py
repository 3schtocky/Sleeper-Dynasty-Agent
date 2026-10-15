"""A player check: "is Puka Nacua a buy or a sell?", "what's Rashee Rice's trade value?".

The chat has no model-visible tool for one player's standing (a seventh tool
cost routing accuracy), and without one the model either invented facts or
sent the question to the waiver tool. Python detects the question, looks the
player up and writes the card; the model only phrases it.

The buy/sell lean compares two ranks among rostered players at the same
position: how good his win-now value is, and how highly the market prices
him. Production ranking well above price means the market undervalues him
(buy); price well above production means it overvalues him (sell). It is a
read on price against this season's production in this league, not a
forecast, and the card says so."""

from __future__ import annotations

import re
import sqlite3

from dynasty_agent import context, explain, market, valuation, weekly

LEAN_GAP = 0.15  # percentile-rank difference that counts as a real gap between production and price

_ASKS_ABOUT = re.compile(
    r"\b(buy|sell|worth|value|values|keep|hold|outlook|upside|tell me about|thoughts on|what do you think|how good|"
    r"good player|bust|breakout|stash|fall off)\b", re.I)
# Questions another tool owns: a swap between two assets, a start/sit, a waiver move.
_OTHER_TOOL = re.compile(
    r"\b(trade|swap|send|give)\b.*\bfor\b|\b(start|sit|bench|lineup|waiver|waivers|pick up|pickup|add|drop|bid|faab|taxi)\b",
    re.I)


def detect(conn: sqlite3.Connection | None, question: str, follow_up: bool = False) -> dict | None:
    """{'player_id': ...} when the question asks about exactly one named player in a way no other tool owns.
    A follow-up to an earlier player check ("what about Rashee Rice?") needs no asking-word."""
    if conn is None or _OTHER_TOOL.search(question) or not (follow_up or _ASKS_ABOUT.search(question)):
        return None
    rows = conn.execute(
        "SELECT DISTINCT player_id FROM players WHERE position IN ('QB','RB','WR','TE') AND length(full_name) > 6 "
        "AND ? LIKE '%' || lower(full_name) || '%'", (question.lower(),)).fetchall()
    return {"player_id": rows[0][0]} if len(rows) == 1 else None


def _rank(value: float, values: list[float]) -> float:
    """Share of the group at or below this value."""
    return sum(v <= value for v in values) / len(values) if values else 0.0


def card(conn: sqlite3.Connection, season: int, me: int, player_id: str) -> dict:
    valuations = valuation.player_valuations(conn, season)
    v = valuations.get(player_id)
    info = conn.execute("SELECT full_name, position, team, age FROM players WHERE player_id = ?", (player_id,)).fetchone()
    names = context.team_names(conn)
    held = conn.execute("SELECT roster_id, slot FROM roster_players WHERE player_id = ?", (player_id,)).fetchone()
    where = ("on your roster" if held["roster_id"] == me else f"on {names.get(held['roster_id'], 'another team')}") if held else "a free agent"
    out = {"player_id": player_id, "name": info["full_name"], "position": info["position"], "team": info["team"],
           "age": info["age"], "where": where, "mine": bool(held and held["roster_id"] == me),
           "slot": held["slot"] if held and held["roster_id"] == me else None,
           "market": market.latest_value(conn, player_id), "valued": v is not None}
    if v is None:
        return out
    out.update({"win_now": v["win_now_value"], "three_year": v["three_year_value"],
                "fppg": v["fantasy_points_per_game"], "games": v["games"]})
    peers = [(pid, pv) for pid, pv in valuations.items() if pv["position"] == info["position"] and market.latest_value(conn, pid)]
    if out["market"] and len(peers) >= 8:
        prod = _rank(v["win_now_value"], [pv["win_now_value"] for _, pv in peers])
        price = _rank(out["market"], [market.latest_value(conn, pid) for pid, _ in peers])
        out["production_rank"], out["price_rank"] = prod, price
        out["lean"] = "buy" if prod - price >= LEAN_GAP else "sell" if price - prod >= LEAN_GAP else "fair"
    if not out["mine"]:
        out["lineup_gain"] = weekly.lineup_gains(conn, valuations, me, [player_id])[player_id]
    return out


def format_card(c: dict) -> str:
    head = f"{c['name']} ({c['position']}, {c['team'] or 'no team'}, age {c['age']}), {c['where']}."
    if not c["valued"]:
        return head + "\nNo win-now value yet: he has no stats or projection in the database."
    lines = [head,
             f"Win-now value {c['win_now']:.1f} points per game, three-year value {c['three_year']:.1f} "
             f"({c['fppg']:.1f} points per game over {c['games']} games).",
             f"FantasyCalc market value: {c['market']:.0f} trade-value points." if c["market"] else "No FantasyCalc market value."]
    if "lean" in c:
        lines.append(f"Among {c['position']}s, his production is at the {explain.ordinal(round(c['production_rank'] * 100))} "
                     f"percentile and his price at the {explain.ordinal(round(c['price_rank'] * 100))}: {_LEAN_TEXT[c['lean']]}")
    if "lineup_gain" in c:
        lines.append(f"Adding him would lift your best lineup by {c['lineup_gain']:.1f} points per week." if c["lineup_gain"] > 0
                     else "He would not start for you, so adding him would not lift your best lineup.")
    elif c["slot"]:
        lines.append(f"He is in your {c['slot']} group.")
    lines.append("The lean compares price with this season's production in this league; it is not a forecast.")
    return "\n".join(lines)


_LEAN_TEXT = {
    "buy": "the market prices him below his production, so he leans buy.",
    "sell": "the market prices him above his production, so he leans sell.",
    "fair": "price and production line up, so he is priced about right.",
}


def compact(c: dict) -> dict:
    """The card as the model sees it: every number already formatted the way the card prints it."""
    out = {"name": c["name"], "position": c["position"], "team": c["team"] or "no team", "age": c["age"], "where": c["where"]}
    if c["market"]:
        out["market_value"] = f"{c['market']:.0f}"
    if c["valued"]:
        out.update({"win_now": f"{c['win_now']:.1f}", "three_year": f"{c['three_year']:.1f}", "points_per_game": f"{c['fppg']:.1f}",
                    "games": c["games"]})
    if "lean" in c:
        out["lean"] = c["lean"]
    if "lineup_gain" in c:
        out["lineup_gain"] = f"+{c['lineup_gain']:.1f}" if c["lineup_gain"] > 0 else "none, would not start"
    return out
