"""What the chat model may treat as true when it talks without a tool.

A 4B model gets football concepts wrong (it once said superflex makes
quarterbacks less valuable), and prompts cannot fix a wrong fact. So the
facts are written here, and the chat prompt carries them as a trusted
reference. Two parts: this league's real settings, read from the database
so they cannot drift, and a short set of general concepts. Keep it short:
a 4B model loses the thread of a long reference."""

from __future__ import annotations

import json
import sqlite3

# General concepts, vetted by hand. Nothing here is about a named player or the current week.
CONCEPTS = """General dynasty concepts:
- Dynasty: you keep your whole roster every year, so youth, long-term value and draft picks matter as much as this week's points. Picks can be traded.
- Superflex (or 2QB): an extra starting slot that can hold a second quarterback. It makes quarterbacks far more valuable. A league with one QB slot makes them worth much less.
- PPR: points per reception. Every catch scores, so it helps the players who catch the most: receivers first, then tight ends and running backs who catch many passes. It helps running backs who only run the ball the least. Full PPR is one point per catch.
- Flex: a starting slot any eligible position can fill, usually running back, wide receiver or tight end.
- Aging: running backs usually decline from about age 26 or 27 and lose value fastest. Wide receivers usually stay productive into their late twenties and often to about 30. Quarterbacks can play well into their late thirties. Tight ends often take two or three seasons to break out.
- Contender: has a roster that can win this season, so it trades picks and young prospects for current points. Rebuilder: trades aging veterans for picks and young players.
- Buy low means acquiring a good player whose price is temporarily down. Sell high means trading a player whose price is temporarily inflated.
- Rookie picks are lottery tickets: early first-round picks hit often, later ones rarely.
- FAAB: a blind-bid waiver budget in fake dollars. It is not real money, and unspent budget carries no cash value.
- Taxi squad: slots for young players you stash off the active roster. IR: slots for injured players.
- Trade value points (FantasyCalc) put players and picks on one scale so a trade can be totaled."""


def league_facts(conn: sqlite3.Connection | None) -> str:
    """This league's real format as a few plain lines; empty if no league is cached."""
    if conn is None:
        return ""
    row = conn.execute(
        "SELECT name, num_teams, scoring_settings_json, roster_positions_json, settings_json "
        "FROM league ORDER BY fetched_at DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return ""
    scoring = json.loads(row["scoring_settings_json"])
    slots = json.loads(row["roster_positions_json"])
    s = json.loads(row["settings_json"])
    starters = [p for p in slots if p not in ("BN", "IR", "TAXI")]
    qb_slots = starters.count("QB") + starters.count("SUPER_FLEX")
    rec = scoring.get("rec", 0)
    ppr = "full PPR" if rec == 1 else "half PPR" if rec == 0.5 else "standard (no PPR)" if rec == 0 else f"{rec} per reception"
    pass_td = scoring.get("pass_td", 4)
    lines = [
        f"This user's league has {row['num_teams']} teams and uses {ppr}, with {pass_td:g} points per passing touchdown.",
        f"Starting lineup: {', '.join(starters)}. "
        + ("It is a superflex league, so quarterbacks are very valuable here." if qb_slots > 1
           else "It has only one quarterback slot, so it is not superflex and quarterbacks are worth less here than in superflex."),
        f"Bench spots: {slots.count('BN')}. Taxi spots: {s.get('taxi_slots', 0)}"
        + (" (rookies only)." if not s.get("taxi_allow_vets") else ".")
        + f" IR spots: {s.get('reserve_slots', 0)}.",
        f"Waivers use a ${s.get('waiver_budget', 0)} FAAB budget. The trade deadline is week {s.get('trade_deadline')}. "
        f"{s.get('playoff_teams')} teams make the playoffs, starting week {s.get('playoff_week_start')}. "
        f"The rookie draft has {s.get('draft_rounds')} rounds."
        + (" Draft picks can be traded." if s.get("pick_trading") else ""),
    ]
    return "This user's league:\n" + "\n".join(f"- {line}" for line in lines)


def reference(conn: sqlite3.Connection | None) -> str:
    """The trusted block added to the chat prompt."""
    return "\n\n".join(part for part in (league_facts(conn), CONCEPTS) if part)
