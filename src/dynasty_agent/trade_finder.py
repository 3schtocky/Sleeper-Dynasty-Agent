"""Trade ideas: "I need a WR, find me trades".

Python does all the searching and every number; the chat model only phrases
the result. For one position (named by the user, else the spot where a
better player would lift the lineup most):

1. Candidates are players at that position on other teams who would raise my
   best possible lineup (a player who would not start for me gains 0).
2. For each, the offer is the cheapest-to-me package of one or two assets I
   can spare (my players and my picks) whose market value is at least
   MIN_OFFER_SHARE of his. "Cheapest" means the smallest drop in my own best
   lineup after those assets leave, so I never offer a starter I need.
3. Ideas are ranked by my net lineup gain. They are not filtered by whether
   the other side would say yes (the user asked for the best upgrades), but
   each is labeled fair, a stretch or a lowball from the market values.
4. The top ideas are run through valuation.evaluate_trade and dropped if it
   warns about anything, so no idea rests on a player I do not hold.

Win-now values come from the same valuations the lineup optimizer uses, so a
"lineup gain" here means points per week on this league's scale."""

from __future__ import annotations

import itertools
import json
import re
import sqlite3

from dynasty_agent import context, market, metrics, picks, valuation

POSITIONS = ("QB", "RB", "WR", "TE")
MIN_OFFER_SHARE = 0.7      # an offer below 70% of his market value is not an offer
FAIR_SHARE = 0.95          # their side receives at least this share of what it gives: fair
STRETCH_SHARE = 0.8        # between this and FAIR_SHARE: a stretch; below: a lowball
MIN_GAIN = 0.5             # points per week; a smaller lift is not worth a trade
TOP_CANDIDATES = 10
SPARE_PLAYERS = 8
SPARE_PICKS = 6
IDEAS = 3

_POSITION_WORDS = {
    "qb": "QB", "quarterback": "QB", "quarterbacks": "QB",
    "rb": "RB", "running back": "RB", "running backs": "RB",
    "wr": "WR", "receiver": "WR", "receivers": "WR", "wide receiver": "WR", "wide receivers": "WR",
    "te": "TE", "tight end": "TE", "tight ends": "TE",
}
_POS = r"(?:wr|rb|qb|te|(?:wide )?receivers?|running backs?|quarterbacks?|tight ends?)"
_FIND = re.compile(
    rf"\b(find|suggest|show|give|got any|any)\b[^.?!]{{0,40}}\btrade (ideas?|targets?|options?|partners?)\b"
    rf"|\btrade (targets?|ideas?)\b"
    rf"|\bi need\b[^.?!]{{0,25}}\b{_POS}'?s?\b"
    rf"|\b(upgrade|upgrades|target|targets)\b[^.?!]{{0,25}}\b{_POS}'?s?\b", re.I)


def detect(question: str) -> dict | None:
    """The arguments for a trade-idea question, or None. Python detects it, as with explain, so the model's
    six-tool routing stays as measured."""
    if not _FIND.search(question):
        return None
    return {"position": position_in(question)}


def position_in(text: str) -> str | None:
    """The position a sentence names ("some new WR's", "a running back"), or None."""
    m = re.search(rf"\b({_POS})'?s?\b", text, re.I)
    word = m.group(1).lower() if m else ""
    return _POSITION_WORDS.get(word) or _POSITION_WORDS.get(word.rstrip("s"))


class _Roster:
    """My roster as the lineup optimizer sees it: who counts toward my best lineup, and the value of each."""

    def __init__(self, conn: sqlite3.Connection, valuations: dict, me: int):
        row = conn.execute("SELECT roster_positions_json FROM league ORDER BY fetched_at DESC LIMIT 1").fetchone()
        self.slots = metrics.starting_slot_counts(json.loads(row["roster_positions_json"])) if row else {}
        self.valuations = valuations
        self.mine = {r["player_id"]: r["slot"] for r in conn.execute(
            "SELECT player_id, slot FROM roster_players WHERE roster_id = ?", (me,))}
        self.active = [p for p, slot in self.mine.items() if slot in ("starter", "bench") and p in valuations]
        self.base = self.total(self.active)

    def total(self, ids: list[str]) -> float:
        return metrics.best_lineup_total(
            [(self.valuations[p]["position"], self.valuations[p]["win_now_value"]) for p in ids], self.slots)

    def after(self, sent: list[str], received: list[str]) -> float:
        """Lineup gain after sending and receiving (received players join the bench; taxi and IR never start)."""
        ids = [p for p in self.active if p not in sent] + [p for p in received if p in self.valuations]
        return self.total(ids) - self.base


def _my_assets(conn, roster: _Roster, me: int, rate: float, valuations: dict) -> list[dict]:
    """What I could put in an offer: players with a market price that I can spare, and my picks, by value."""
    players = []
    for pid in roster.mine:
        mv = market.latest_value(conn, pid)
        v = valuations.get(pid)
        if mv and v:
            loss = roster.base - roster.total([p for p in roster.active if p != pid])
            players.append({"kind": "player", "id": pid, "name": v["full_name"], "market": mv, "loss": loss})
    players = sorted((p for p in players if p["loss"] < 3.0), key=lambda p: -p["market"])[:SPARE_PLAYERS]
    owned = []
    for pk in picks.inventory(conn):
        if pk["owner_roster_id"] != me:
            continue
        est = valuation.pick_value_estimate(conn, pk["season"], pk["round"], rate)
        if est["model_value"]:
            owned.append({"kind": "pick", "id": (pk["season"], pk["round"]), "name": f"{pk['season']} round {pk['round']}",
                          "market": est["model_value"], "loss": 0.0})
    return players + sorted(owned, key=lambda p: -p["market"])[:SPARE_PICKS]


def _candidates(conn, roster: _Roster, me: int, position: str) -> list[dict]:
    owner = {r["player_id"]: r["roster_id"] for r in conn.execute("SELECT player_id, roster_id FROM roster_players")}
    out = []
    for pid, rid in owner.items():
        v = roster.valuations.get(pid)
        if rid == me or not v or v["position"] != position or v["win_now_value"] <= 0:
            continue
        mv = market.latest_value(conn, pid)
        gain = roster.after([], [pid])
        if mv and gain >= MIN_GAIN:
            out.append({"id": pid, "name": v["full_name"], "position": position, "age": v["age"], "roster_id": rid,
                        "market": mv, "win_now": v["win_now_value"], "gain": gain})
    return sorted(out, key=lambda c: (-c["gain"], c["market"]))[:TOP_CANDIDATES]


def _best_offer(roster: _Roster, cand: dict, assets: list[dict]) -> dict | None:
    """The package of one or two assets worth at least MIN_OFFER_SHARE of the candidate that leaves my lineup best off."""
    best = None
    for n in (1, 2):
        for combo in itertools.combinations(assets, n):
            give = sum(a["market"] for a in combo)
            if give < MIN_OFFER_SHARE * cand["market"]:
                continue
            sent_players = [a["id"] for a in combo if a["kind"] == "player"]
            net = roster.after(sent_players, [cand["id"]])
            key = (-net, abs(give - cand["market"]))
            if best is None or key < best["key"]:
                best = {"key": key, "assets": list(combo), "net": net, "give": give}
    return best if best and best["net"] >= MIN_GAIN else None


def _label(give: float, get: float) -> str:
    share = give / get if get else 0.0
    return "fair" if share >= FAIR_SHARE else "a stretch" if share >= STRETCH_SHARE else "a lowball"


def _weakest_position(conn, roster: _Roster, me: int) -> str:
    """The position where the best few outside players would lift my lineup most."""
    best, best_pos = -1.0, "WR"
    for pos in POSITIONS:
        gains = sorted((c["gain"] for c in _candidates(conn, roster, me, pos)), reverse=True)[:3]
        score = sum(gains) / len(gains) if gains else 0.0
        if score > best:
            best, best_pos = score, pos
    return best_pos


def find(conn: sqlite3.Connection, season: int, me: int, position: str | None = None) -> dict:
    """Up to IDEAS trade ideas for a position. See the module docstring."""
    valuations = valuation.player_valuations(conn, season)
    roster = _Roster(conn, valuations, me)
    rate = picks.valuation_discount_rate()
    chosen_by = "asked" if position in POSITIONS else "biggest upgrade"
    position = position if position in POSITIONS else _weakest_position(conn, roster, me)
    assets = _my_assets(conn, roster, me, rate, valuations)
    ideas = []
    for cand in _candidates(conn, roster, me, position):
        offer = _best_offer(roster, cand, assets)
        if offer:
            ideas.append({"candidate": cand, "offer": offer})
    ideas.sort(key=lambda i: (-i["offer"]["net"], abs(i["offer"]["give"] - i["candidate"]["market"])))

    kept, owners_used = [], set()
    for idea in ideas:
        if len(kept) == IDEAS:
            break
        if idea["candidate"]["roster_id"] in owners_used and len(ideas) > IDEAS:
            continue  # spread the ideas across teams while there is a choice
        sent_players = [a["id"] for a in idea["offer"]["assets"] if a["kind"] == "player"]
        sent_picks = [a["id"] for a in idea["offer"]["assets"] if a["kind"] == "pick"]
        checked = valuation.evaluate_trade(conn, season, me, sent_players, sent_picks, [idea["candidate"]["id"]], [], rate)
        if checked["warnings"]:
            continue
        idea["label"] = _label(idea["offer"]["give"], idea["candidate"]["market"])
        idea["checked"] = checked
        kept.append(idea)
        owners_used.add(idea["candidate"]["roster_id"])
    return {"position": position, "chosen_by": chosen_by, "ideas": kept, "base_lineup": roster.base,
            "team_names": context.team_names(conn)}


def format_ideas(result: dict) -> str:
    pos, names = result["position"], result["team_names"]
    why = f"for {pos}" if result["chosen_by"] == "asked" else f"for {pos}, your biggest upgrade spot"
    if not result["ideas"]:
        return (f"Trade ideas {why}: none found. Nobody at {pos} on other teams would lift your best lineup by at least "
                f"{MIN_GAIN} points per week for a package you can spare.")
    lines = [f"Trade ideas {why} (ranked by how much each lifts your best lineup):", ""]
    for n, idea in enumerate(result["ideas"], 1):
        c, o, chk = idea["candidate"], idea["offer"], idea["checked"]
        give = " + ".join(a["name"] for a in o["assets"])
        lines += [
            f"{n}. Get {c['name']} ({c['position']}, age {c['age']}, {names.get(c['roster_id'], 'another team')}), give {give}",
            f"   Your lineup: +{o['net']:.1f} points per week. Market value you give {o['give']:.0f}, you get {c['market']:.0f} "
            f"(net {chk['market_value_delta']:+.0f}).",
            f"   Would they say yes? {idea['label'].capitalize()}: they receive {o['give']:.0f} for {c['market']:.0f} given.",
        ]
        if chk["consolidation"]:
            lines.append(f"   Note: {chk['consolidation']}")
        lines.append("")
    lines.append("Market values are FantasyCalc trade-value points, not dollars. Lineup gain is win-now points per week.")
    return "\n".join(lines)


def compact(result: dict) -> dict:
    names = result["team_names"]
    return {
        "position": result["position"],
        "chosen_by": result["chosen_by"],
        "ideas": [{
            "get": f"{i['candidate']['name']} ({names.get(i['candidate']['roster_id'], 'another team')})",
            "give": " + ".join(a["name"] for a in i["offer"]["assets"]),
            "lineup_gain": f"+{i['offer']['net']:.1f}",
            "net_market_value": f"{i['checked']['market_value_delta']:+.0f}",
            "they_would_say_yes": i["label"],
        } for i in result["ideas"]],
    }
