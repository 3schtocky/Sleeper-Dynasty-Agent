"""Talking points: the facts of a tool result as short plain sentences.

A 4B model turns a numbers block into advice unreliably (it reversed a SELL
call, garbled a percentile, listed numbers instead of saying why). So Python
writes the points, from the same values the tool already computed, and the
model only leads with the answer and phrases a few of them in a friendly
voice. Every number here is copied from the tool's own compact dict, so the
grounding check still passes. Each function returns the points in the order
the model should lean on them: the lean or warning first."""

from __future__ import annotations

import re


def _name(entry: str) -> str:
    """'Matthew Stafford QB 25.3 (Questionable)' -> 'Matthew Stafford'."""
    return re.split(r"\s+[A-Z]{2,3}\s", entry, maxsplit=1)[0]


def _trade(c: dict) -> list[str]:
    pts = []
    if c.get("headline"):
        pts.append(c["headline"])
    value, win_now = float(c["net_market_value"]), float(c["net_win_now"])
    if value > 0 and win_now > 0:
        lean = "The numbers favor you on both trade value and win-now points."
    elif value < 0 and win_now < 0:
        lean = "The numbers favor the other team on both trade value and win-now points."
    elif value > 0:
        lean = "The numbers are mixed: you win on trade value but lose on win-now points."
    elif value < 0:
        lean = "The numbers are mixed: you win on win-now points but lose on trade value."
    else:
        lean = "The numbers are even."
    pts.append(lean)
    pts.append(f"Net trade value is {c['net_market_value']} points and net win-now is {c['net_win_now']} points per game.")
    if c.get("consolidation"):
        pts.append(f"Note on roster fit: {c['consolidation']}")
    if c.get("fit"):
        pts.append(f"Fit with your team: {c['fit']}.")
    return pts


def _lineup(c: dict) -> list[str]:
    pts = [w for w in (c.get("warnings") or [])[:2]]
    if c.get("win_probability") not in (None, "n/a") and c.get("opponent_projected"):
        pts.append(f"The recommended lineup has a {c['win_probability']} chance to beat your opponent, who is projected "
                   f"at {c['opponent_projected']}.")
    if c.get("swaps_from_most_points"):
        pts.append("Playing for raw points alone would swap: " + "; ".join(c["swaps_from_most_points"]) + ".")
    if c.get("starters"):
        pts.append(f"Your top projected starter is {_name(c['starters'][0])}.")
    return pts


def _team(c: dict) -> list[str]:
    return [
        f"The call is {c['verdict']}: {c['reason']}",
        f"Confidence is {c['confidence']} after {c['games_played']} games.",
        f"Your win-now percentile is {c['win_now_percentile']} and your three-year percentile is "
        f"{c['three_year_percentile']}, against {c['compared_against_teams']} teams.",
        "Your most valuable players now: " + "; ".join(c["most_valuable_now"][:3]) + ".",
    ]


def _waiver(c: dict) -> list[str]:
    if "player" in c:
        pts = [f"{c['player']}: suggested bid {c['suggested_bid']}, lineup gain {c['lineup_gain']}, "
               f"budget left {c['remaining_budget']}."]
        if c.get("already_rostered"):
            pts.insert(0, "He is already on a roster, so he is not a waiver pickup.")
        if c.get("note"):
            pts.append(c["note"])
        return pts
    pts = [c["headline"][0].upper() + c["headline"][1:] + "."]
    pts += [t + "." for t in c["targets"][1:3]]
    return pts


def _picks(c: dict, rows: list[dict]) -> list[str]:
    meaning = {"BUY": "worth acquiring", "SELL": "worth trading away now", "HOLD": "worth keeping"}
    pts = [f"Your next rookie draft is {c['next_draft']}."]
    for r in rows:
        if r["projected_slot"] and r["ratio"] is not None:
            pts.append(f"Your {r['season']} {r['projected_slot']} pick is a {r['advice']} ({meaning.get(r['advice'], r['advice'])}): "
                       f"FantasyCalc prices it at {r['fantasycalc_price']:.0f} versus {r['comparable_value']:.0f} for last "
                       f"year's rookies at that slot.")
    if len(pts) == 1:
        pts.append("None of your picks is close enough to a draft slot to price.")
    pts.append(f"Your team's posture is {c['posture']}.")
    return pts


def _taxi(c: dict, data: dict) -> list[str]:
    st = data["settings"]
    pts = [f"You have {st['taxi_slots']} taxi slots and {len(data['taxi_now'])} are in use; "
           f"you have {st['reserve_slots']} IR slot and {len(data['ir_now'])} in use."]
    pts += [f"Recommended: {m}." for m in c["recommendation"]]
    if c.get("taxi_moves_locked_by_deadline"):
        pts.append("Taxi moves are locked by the deadline.")
    pts.append(f"Next season: {c['next_season']}.")
    return pts


def _ideas(c: dict) -> list[str]:
    if not c["ideas"]:
        return [f"No {c['position']} on another team would lift your best lineup enough for a package you can spare."]
    where = "your biggest upgrade spot" if c["chosen_by"] != "asked" else "the spot you asked about"
    pts = [f"I looked at {c['position']}, {where}, and found {len(c['ideas'])} ideas."]
    for i, idea in enumerate(c["ideas"], 1):
        pts.append(f"Idea {i}: get {idea['get']} for {idea['give']}. It lifts your lineup {idea['lineup_gain']} points per week, "
                   f"and the other side would likely say: {idea['they_would_say_yes']}.")
    pts.append("These are offers to send the other manager, not trades you can just make, and they may say no. A lowball "
               "is only an opening bid.")
    return pts


def _player(c: dict) -> list[str]:
    pts = [f"{c['name']} is a {c['position']} (age {c['age']}) {c['where']}."]
    if c["where"].startswith("on ") and c["where"] != "on your roster":
        pts.append("Getting him would take a trade offer to that manager, not a waiver pickup.")
    if "lean" in c:
        pts.append({"buy": "The market prices him below his production, so he leans buy.",
                    "sell": "The market prices him above his production, so he leans sell.",
                    "fair": "His price and production line up, so he is priced about right."}[c["lean"]])
    if "win_now" in c:
        pts.append(f"His win-now value is {c['win_now']} points per game."
                   + (f" His market value is {c['market_value']} trade-value points." if "market_value" in c else ""))
    else:
        pts.append("He has no win-now value yet because there are no stats or projection for him.")
    if "lineup_gain" in c:
        pts.append("He would not start for you." if c["lineup_gain"].startswith("none")
                   else f"Adding him would lift your best lineup {c['lineup_gain']} points per week.")
    pts.append("The buy or sell read compares price with this season's production, it is not a forecast.")
    return pts


def build(name: str, compact: dict, data) -> list[str]:
    """The talking points for a tool result, or [] for a tool with none."""
    if name == "evaluate_trade":
        return _trade(compact)
    if name == "set_lineup":
        return _lineup(compact)
    if name == "my_team":
        return _team(compact)
    if name == "waiver_targets":
        return _waiver(compact)
    if name == "pick_advice":
        return _picks(compact, data["rows"])
    if name == "find_trades":
        return _ideas(compact)
    if name == "player_check":
        return _player(compact)
    if name == "taxi_plan":
        return _taxi(compact, data)
    return []
