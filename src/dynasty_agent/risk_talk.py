"""Talking about injuries and disagreement, decided in Python.

The 4B model, left alone, answered "what about his injury risk?" by running
the trade again, called an "elevated" risk "high", said "don't trade" and then
"make the trade" on the same numbers, and caved when the user pushed back. So
Python owns the parts that must stay consistent:

- an injury question (`is_risk_question`) goes to the risk check for the
  named player, or the players in the last answer;
- an injury the user reports ("Daniels just got hurt") is a labeled
  assumption: Python asks how many games, then shows the risk with that
  number as the user's word, never as data;
- pushback on a trade verdict (`is_pushback`) gets a reply Python writes: the
  verdict stands unless there are new facts, the one counter-fact that matters
  (the incoming player's injury risk), and one pointed data question.

Nothing here changes a number or a verdict; it decides what is said about
them. See availability.py for why the risk is shown beside the numbers and
not subtracted from them.
"""

from __future__ import annotations

import re

from dynasty_agent import formatters

_RISK_WORDS = re.compile(
    r"\binjur(y|ies|ed)\b|\bhurt\b|\bdurab|\binjury[- ]prone\b|\bgames? (he )?(will )?miss|\bmisses? (a lot of )?games\b|"
    r"\bhealth(y|ier)\b|\bfragile\b", re.I)
# Words that point back at the player or players in the last answer.
_ANAPHORA = re.compile(r"\b(his|her|their|him|them|he|they|that (trade|deal|player)|the trade|those)\b", re.I)
# A report that something happened or will happen, not a question about risk.
_REPORT = re.compile(
    r"\b(just )?(got|gets|is|was|went|has been|been) (hurt|injured)\b|\bjust (got )?(hurt|injured)\b|"
    r"\b(tore|torn|broke|broken|sprained|strained|fractured)\b|\bacl\b|\bachilles\b|"
    r"\bseason[- ]ending\b|\bout for (the|this) (season|year)\b|\bis out\b.*\b(week|weeks|game|games)\b|"
    r"\bout (for )?\d+\s*(weeks?|games?)\b", re.I)
_QUESTION_START = re.compile(r"^\s*(did|does|do|is|was|has|have|will|how|what|why|should|could)\b", re.I)
_PUSHBACK = re.compile(
    r"\bi disagree\b|\bdisagree\b|\bno way\b|\bi (don'?t|do not) (think|agree|buy|believe)\b|\bnot convinced\b|"
    r"\byou('| a)?re wrong\b|\bthat'?s wrong\b|\bare you sure\b|\bcome on\b|\bi (still )?(want|like|love|prefer)\b|"
    r"\b(he|she)'?s (way |much |a lot )?better\b|\bi'?d rather\b|\bbut (he|she|they)\b", re.I)
_TRADE_ASK = re.compile(r"\b(trade|swap|send|give)\b.*\bfor\b|\bwhat if\b|\bwhat about\b", re.I)
_SEASON = re.compile(r"\bseason[- ]ending\b|\b(rest of|remainder of) the (season|year)\b|\bout for (the|this) (season|year)\b|"
                     r"\bfor the (season|year)\b|\bacl\b|\bachilles\b", re.I)
_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                 "ten": 10, "eleven": 11, "twelve": 12}
_GAMES = re.compile(r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\s*(?:more\s*)?(weeks?|games?)?\b", re.I)
SEASON_ENDING = 99  # "out for the season": capped at the games left by availability.player_risk


def is_report(text: str) -> bool:
    """The user says a player is hurt or out, as a statement ("Daniels just got hurt", "he's out 6 weeks")."""
    return bool(_REPORT.search(text)) and not (text.strip().endswith("?") and _QUESTION_START.match(text))


def is_risk_question(text: str) -> bool:
    return bool(_RISK_WORDS.search(text)) and not _REPORT.search(text)


def is_pushback(text: str) -> bool:
    return bool(_PUSHBACK.search(text)) and not _TRADE_ASK.search(text)


def games_out_in(text: str) -> int | None:
    """How many games the user says a player will miss: a number or number word, "a month" (4), or the whole season."""
    if _SEASON.search(text):
        return SEASON_ENDING
    if re.search(r"\ba month\b", text, re.I):
        return 4
    m = _GAMES.search(text)
    if not m:
        return None
    n = m.group(1).lower()
    return int(n) if n.isdigit() else _NUMBER_WORDS[n]


def players_for(conn, text: str, recent_players: dict[str, str]) -> list[tuple[str, str]]:
    """[(player_id, name)] the message is about: a full name (nicknames too), else a surname that matches a
    player already under discussion, else (when the message points back with a pronoun) every player in the last
    answer. Empty when none of those fit, so a general injury question stays general talk."""
    from dynasty_agent import player_check  # late: player_check imports formatters, this module is imported by tools

    ids = player_check.named_player_ids(conn, text)
    if ids:
        names = {r[0]: r[1] for r in conn.execute(
            f"SELECT player_id, full_name FROM players WHERE player_id IN ({','.join('?' * len(ids))})", ids)}
        return [(i, names[i]) for i in ids]
    words = set(re.findall(r"[a-z']+", text.lower()))
    surname = [(pid, name) for pid, name in recent_players.items() if name.split()[-1].lower() in words]
    if surname:
        return surname
    if _ANAPHORA.search(text):
        return list(recent_players.items())
    return []


def report_question(name: str) -> str:
    return (f"Roughly how many games do you expect {name} to miss? A rough guess is fine, or say it's a season-ending "
            f"injury. I'll use your number as an assumption, since Sleeper gives no timeline.")


def _lean_clause(value: float, win_now: float) -> str:
    if value > 0 and win_now > 0:
        return "they favor you on both trade value and win-now points"
    if value < 0 and win_now < 0:
        return "they favor the other team on both trade value and win-now points"
    if value > 0:
        return "they are mixed: you win on trade value but lose on win-now points"
    if value < 0:
        return "they are mixed: you win on win-now points but lose on trade value"
    return "they are even"


def pushback_reply(trade: dict) -> tuple[str, list[str]]:
    """The reply to a user who disagrees with a trade verdict. The verdict is
    restated from the same numbers, unchanged, because nothing new has been
    said. The counter-fact is the incoming player's injury risk when it is
    elevated or high, and the question asks for the one thing the user could
    know that the data does not. Returns the text and the ids of the player the
    question is about, so the chat can read the user's next message as the
    answer."""
    clause = _lean_clause(trade["market_value_delta"], trade["win_now_delta"])
    out = [f"I hear you, but nothing new has come in, so my read stands: on the numbers {clause}. It's your call either way."]
    incoming = {p["full_name"] for p in trade["received"]["players"]}
    worry = next((x for x in sorted(trade.get("risk", []), key=lambda x: -x["risk"]["missed_per_17"])
                  if x["name"] in incoming and x["risk"]["label"] != "typical"), None)
    if worry:
        out.append(f"The thing I would weigh against the move is injury risk, which is not in those numbers. "
                   f"{formatters.risk_line(worry['name'], worry['risk'])}")
        out.append(f"How many games do you expect {worry['name']} to miss from here? If you know something the data "
                   f"does not, tell me and I'll rerun it with your number.")
        asked = [p["player_id"] for p in trade["received"]["players"] if p["full_name"] == worry["name"]]
    else:
        asked = []
        out.append("What do you see that the numbers don't? If it is an injury or a role change, tell me and I'll rerun "
                   "it with that.")
    return " ".join(out), asked


_DIRECTS_THE_MOVE = re.compile(
    r"sleeper app|\bmake (it|the|this|that)\b|\bmaking (it|the|this|that)\b|\bdon'?t (make|do) (it|the|this|that)\b|"
    r"\bdo not (make|do) (it|the|this|that)\b|\bpass on (it|this|that|the)\b", re.I)


def contradicts(sentence: str, facts: str) -> list[str]:
    """Why a model sentence disagrees with the facts Python gave it, or []. Two
    things the 4B model got wrong live: calling an "elevated" risk "high" (or
    "typical" "low"), and telling the user to make or skip the trade after
    Python's bottom line had said it is their call. Checked per sentence like
    the number check, so a flagged sentence is left out and the rest is kept."""
    if "injury risk" not in facts:
        return []
    out = []
    for word in ("high", "elevated", "typical", "low", "moderate"):
        if re.search(rf"\b{word}\b (injury )?risk", sentence, re.I) and f"{word} injury risk" not in facts:
            out.append(f"says {word} risk, the facts do not")
    if "Bottom line" in facts and _DIRECTS_THE_MOVE.search(sentence):
        out.append("directs the move, the bottom line leaves it to the user")
    return out
