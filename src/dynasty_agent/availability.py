"""Availability: how many games a skill-position player is expected to miss.

**Why it exists:** nothing in the valuation discounted a player for missing
games, and per-game production even flatters one who missed half a season. A
trade for a player who missed 10 of 17 games last year read as a clear win.
This module turns 2017-2026 history into an expected-games-missed number that
Python computes and the model only retells.

**The data, checked before the definitions were written (live, 2026-10-14):**
- A missed week is an ABSENT weekly_stats row, not a zero-point row: of 2,732
  skill-position "Out" designations in nfl_injuries, 1 has a stat row that
  week. Zero-point rows are players who played and did nothing.
- nfl_injuries has report_status Out/Doubtful/Questionable, or NULL for a
  practice-only entry. Only Out and Doubtful mean "expected to miss".
- A player on IR stops appearing on the report after the first week out, so a
  missed week is not tied to a same-week flag. A run of consecutive missed
  weeks is injury-related when the report shows Out or Doubtful anywhere in
  the run or in the week before it. Other runs (a suspension, a late
  signing, a benching) are counted separately and reported, never mixed in.
- The 2026 season's injury reports were never ingested; refresh now does.

**Definitions** (regular season only; team byes do not count as games):
- G: games the player's team played; played: weeks with a stat row;
  m = injury-missed games / G.
- Only starter-level player-seasons count, so healthy backups and scratches
  do not look like durable players: at least half the games played had
  starter usage (QB 15+ pass attempts, RB weighted opportunity 10+, WR 4+
  targets, TE 3+ targets). A season with zero games counts only if the
  player was starter-level the year before and the report shows at least 3
  Out or Doubtful weeks, so a player lost for the year is not dropped
  (dropping him would flatter the whole position).

**Model:** see fit_model. Stated limitations are in the fit report, not here.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date

import duckdb

from dynasty_agent import nflverse
from dynasty_agent.calibration import STATS_TEAM
from dynasty_agent.db import utcnow

POSITIONS = ("QB", "RB", "WR", "TE")
INJURY_STATUSES = ("Out", "Doubtful")
# Games with starter-level usage, per position (see module docstring).
STARTER_USAGE = {
    "QB": ("pass_attempts", 15),
    "RB": ("weighted_opportunity", 10),
    "WR": ("targets", 4),
    "TE": ("targets", 3),
}
MIN_LOST_SEASON_WEEKS = 3


@dataclass
class PlayerSeason:
    gsis_id: str
    season: int
    position: str
    age: float | None
    team_games: int
    played: int
    injury_missed: int
    other_missed: int
    starter: bool
    # kept for the in-season table: which team games he played, which weeks
    # the report flagged him Out or Doubtful, and when his team played
    team_weeks: list[int] = field(default_factory=list, repr=False)
    played_weeks: frozenset[int] = field(default_factory=frozenset, repr=False)
    flagged_weeks: frozenset[int] = field(default_factory=frozenset, repr=False)
    statuses: dict[int, str] = field(default_factory=dict, repr=False)  # week -> Out/Doubtful/Questionable

    @property
    def m(self) -> float:
        return self.injury_missed / self.team_games


def age_band(position: str, age: float | None) -> str:
    """Quarterbacks age later than the other positions, so their bands sit
    higher. Unknown age gets its own band, which the model shrinks to the
    position mean."""
    if age is None:
        return "unknown"
    cuts = (26, 30, 34) if position == "QB" else (24, 27, 30)
    for i, cut in enumerate(cuts):
        if age <= cut:
            return f"b{i}"
    return f"b{len(cuts)}"


def team_game_weeks(through_season: int) -> dict[tuple[int, str], list[int]]:
    """(season, team) -> the regular-season weeks that team played, from the
    cached schedule. Team codes are mapped to the ones the stats files use
    (the schedule keeps OAK, SD, STL). Unplayed games (no score) are left out
    so the current season stops at the last finished week."""
    games_path = str(nflverse.ensure_games_cached())
    con = duckdb.connect()
    try:
        rows = con.execute(
            """
            SELECT season, week, home_team, away_team FROM read_parquet(?)
            WHERE game_type = 'REG' AND season <= ? AND home_score IS NOT NULL AND away_score IS NOT NULL
            """,
            [games_path, through_season],
        ).fetchall()
    finally:
        con.close()
    out: dict[tuple[int, str], set[int]] = defaultdict(set)
    for season, week, home, away in rows:
        for team in (home, away):
            out[(season, STATS_TEAM.get(team, team))].add(week)
    return {k: sorted(v) for k, v in out.items()}


def _runs(missed: list[int]) -> list[list[int]]:
    """Group missed team-game indexes into runs of consecutive games."""
    runs: list[list[int]] = []
    for idx in missed:
        if runs and idx == runs[-1][-1] + 1:
            runs[-1].append(idx)
        else:
            runs.append([idx])
    return runs


def injury_missed_weeks(team_weeks: list[int], played_weeks: set[int], flagged_weeks: set[int]) -> set[int]:
    """The missed team-game weeks that count as injury misses: a missed run
    of consecutive team games is injury-related when it contains a flagged
    week (Out or Doubtful) or starts the week after one (the report posts the
    designation ahead of the game, and the flag may sit on a bye week, so the
    check is the calendar week before the run)."""
    missed_idx = [i for i, w in enumerate(team_weeks) if w not in played_weeks]
    out: set[int] = set()
    for run in _runs(missed_idx):
        weeks = {team_weeks[i] for i in run}
        if weeks & flagged_weeks or (team_weeks[run[0]] - 1) in flagged_weeks:
            out |= weeks
    return out


def classify_season(
    team_weeks: list[int], played_weeks: set[int], flagged_weeks: set[int]
) -> tuple[int, int, int]:
    """(team_games, injury_missed, other_missed) for one player-season."""
    missed = sum(1 for w in team_weeks if w not in played_weeks)
    injury = len(injury_missed_weeks(team_weeks, played_weeks, flagged_weeks))
    return len(team_weeks), injury, missed - injury


def load_player_seasons(conn: sqlite3.Connection, through_season: int) -> list[PlayerSeason]:
    """Every skill-position player-season up to through_season with its
    games, injury misses and starter flag. Seasons are complete except the
    last one, which stops at the last finished week; fit_model uses only
    complete seasons."""
    weeks_of = team_game_weeks(through_season)

    birth = {
        r[0]: r[1] for r in conn.execute(
            "SELECT gsis_id, birthdate FROM player_ids WHERE gsis_id IS NOT NULL AND birthdate IS NOT NULL"
        )
    }
    # played weeks and usage, one pass
    played: dict[tuple[str, int], dict[int, tuple[str, str, float]]] = defaultdict(dict)
    pos_of: dict[tuple[str, int], str] = {}
    for gsis, season, week, team, pos, pa, wo, tg in conn.execute(
        """
        SELECT gsis_id, CAST(season AS INTEGER), week, team, position,
               pass_attempts, weighted_opportunity, targets
        FROM weekly_stats WHERE position IN ('QB','RB','WR','TE') AND gsis_id IS NOT NULL
        """
    ):
        if season > through_season:
            continue
        usage = {"pass_attempts": pa, "weighted_opportunity": wo, "targets": tg}[STARTER_USAGE[pos][0]] or 0
        played[(gsis, season)][week] = (STATS_TEAM.get(team, team), pos, usage)
        pos_of[(gsis, season)] = pos

    flagged: dict[tuple[str, int], set[int]] = defaultdict(set)
    statuses: dict[tuple[str, int], dict[int, str]] = defaultdict(dict)
    flag_team: dict[tuple[str, int], str] = {}
    for gsis, season, week, team, pos, status in conn.execute(
        "SELECT gsis_id, season, week, team, position, report_status FROM nfl_injuries "
        "WHERE report_status IN ('Out','Doubtful','Questionable') AND position IN ('QB','RB','WR','TE')"
    ):
        statuses[(gsis, season)][week] = status
        if status in INJURY_STATUSES:
            flagged[(gsis, season)].add(week)
            flag_team[(gsis, season)] = STATS_TEAM.get(team, team)
            pos_of.setdefault((gsis, season), pos)

    out: list[PlayerSeason] = []
    starter_before: set[tuple[str, int]] = set()
    for (gsis, season) in sorted(set(played) | set(flagged), key=lambda k: (k[1], k[0])):
        rows = played.get((gsis, season), {})
        pos = pos_of[(gsis, season)]
        if pos not in POSITIONS:
            continue
        flags = flagged.get((gsis, season), set())
        if rows:
            # one team per missed week: the team of the nearest game he played,
            # the earlier one on a tie (a mid-season trade is rare for starters)
            weeks = sorted(rows)
            team_weeks = []
            for w in range(1, max(weeks[-1], 18) + 1):
                nearest = min(weeks, key=lambda x: (abs(x - w), x))
                if w in weeks_of.get((season, rows[nearest][0]), ()):
                    team_weeks.append(w)
            if not team_weeks:
                continue
            threshold = STARTER_USAGE[pos][1]
            usage_games = sum(1 for v in rows.values() if v[2] >= threshold)
            starter = usage_games * 2 >= len(rows)
            games, injury, other = classify_season(team_weeks, set(rows), flags)
        else:
            if (gsis, season - 1) not in starter_before or len(flags) < MIN_LOST_SEASON_WEEKS:
                continue
            team = flag_team[(gsis, season)]
            team_weeks = weeks_of.get((season, team), [])
            if not team_weeks:
                continue
            games, injury, other = classify_season(team_weeks, set(), flags)
            starter = True
        if starter:
            starter_before.add((gsis, season))
        born = birth.get(gsis)
        age = None
        if born:
            try:
                b = date.fromisoformat(born[:10])
                age = (date(season, 9, 1) - b).days / 365.25
            except ValueError:
                age = None
        out.append(PlayerSeason(gsis, season, pos, age, games, len(rows), injury, other, starter,
                                team_weeks, frozenset(rows), frozenset(flags),
                                dict(statuses.get((gsis, season), {}))))
    return out


# ---------------------------------------------------------------- the model

# Games of pseudo-history that pull a thin age band toward its position mean.
BAND_SHRINK_GAMES = 600
# Horizons (games ahead) the status table is estimated for; between them the
# reader interpolates. A window needs that many games left after the cut and
# cuts start after game 3, so 12 is the longest a 17-game season supports.
STATUS_HORIZONS = (1, 2, 4, 8, 12)
# Statuses with no weekly history in nfl_injuries borrow the "out for at least
# the first four games" cases, the NFL's injured-reserve minimum. Labeled in
# every output that uses them.
PROXY_STATUSES = {"IR": "ir_proxy", "PUP": "ir_proxy", "NA": "ir_proxy", "Sus": "ir_proxy",
                  "COV": "Out", "DNR": "Out", "Out": "Out", "Doubtful": "Doubtful", "Questionable": "Questionable"}
KAPPA_GRID = (10, 25, 50, 100, 200, 400, 800, 1600, 1_000_000)
LAMBDA_GRID = (0.4, 0.55, 0.7, 0.85, 1.0)
FIRST_TARGET_SEASON = 2020


def build_prior(seasons: list[PlayerSeason]) -> dict:
    """Games-weighted mean injury-missed share by (position, age band), each
    band shrunk toward its position mean by BAND_SHRINK_GAMES, so a thin band
    cannot report a wild number. Keys: (pos, band), (pos, None) for the
    position mean and (None, None) for the league mean."""
    cell: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])  # missed, games
    for p in seasons:
        for key in ((p.position, age_band(p.position, p.age)), (p.position, None), (None, None)):
            cell[key][0] += p.injury_missed
            cell[key][1] += p.team_games
    prior = {k: v[0] / v[1] for k, v in cell.items() if v[1] > 0 and k[1] is None}
    for (pos, band), (missed, games) in cell.items():
        if band is not None and games > 0:
            pos_mean = prior[(pos, None)]
            prior[(pos, band)] = (missed + BAND_SHRINK_GAMES * pos_mean) / (games + BAND_SHRINK_GAMES)
    return prior


def prior_for(prior: dict, position: str, age: float | None) -> float:
    return prior.get((position, age_band(position, age))) or prior.get((position, None)) or prior[(None, None)]


def predict_share(mu: float, history: list[tuple[int, int, int]], target_season: int, kappa: float, lam: float) -> float:
    """Expected injury-missed share of a season: the prior mu pulled toward
    the player's own decayed record. history is (season, team_games,
    injury_missed) for seasons before the target. A season t ago weighs
    lam**(target - t); kappa is how many games of evidence the prior is worth.
    No history returns mu unchanged."""
    games = missed = 0.0
    for season, g, m in history:
        w = lam ** (target_season - season)
        games += w * g
        missed += w * m
    return (kappa * mu + missed) / (kappa + games)


def _histories(seasons: list[PlayerSeason]) -> dict[str, list[tuple[int, int, int]]]:
    out: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for p in seasons:
        out[p.gsis_id].append((p.season, p.team_games, p.injury_missed))
    return out


def backtest(
    seasons: list[PlayerSeason], kappa: float, lam: float, targets: range | tuple = (),
) -> dict:
    """Leak-free score. For each target season S, the prior and every
    player's history come only from seasons before S, then each starter-level
    player-season of S is predicted and compared with what happened.

    Scored on squared error (RMSE), not absolute error: the thing wanted is
    an expected number of missed games, and the best guess under absolute
    error is the median, which is near zero for almost everyone. Absolute
    error is reported too."""
    rows = {"model": [], "prior": [], "league": [], "has_history": []}
    for target in targets:
        train = [p for p in seasons if p.season < target and p.starter]
        if not train:
            continue
        prior = build_prior(train)
        hist = _histories(train)
        league_mean = prior[(None, None)]
        for p in seasons:
            if p.season != target or not p.starter:
                continue
            mu = prior_for(prior, p.position, p.age)
            h = hist.get(p.gsis_id, [])
            rows["model"].append((predict_share(mu, h, target, kappa, lam), p.m, p.team_games))
            rows["prior"].append((mu, p.m, p.team_games))
            rows["league"].append((league_mean, p.m, p.team_games))
            rows["has_history"].append(bool(h))
    return _score(rows)


def _score(rows: dict) -> dict:
    def stats(key: str, mask=None) -> dict:
        pts = [r for r, keep in zip(rows[key], rows["has_history"]) if mask is None or keep == mask]
        if not pts:
            return {"n": 0, "rmse": float("nan"), "mae": float("nan")}
        n = len(pts)
        return {"n": n, "rmse": (sum((p - a) ** 2 for p, a, _ in pts) / n) ** 0.5,
                "mae": sum(abs(p - a) for p, a, _ in pts) / n}
    return {k: {"all": stats(k), "with_history": stats(k, True)} for k in ("model", "prior", "league")}


def fit_model(seasons: list[PlayerSeason], last_complete: int) -> dict:
    """Grid-search (kappa, lambda) on the held-out backtest of
    FIRST_TARGET_SEASON..last_complete, then build the prior from every
    complete season. Returns the model dict that gets stored."""
    complete = [p for p in seasons if p.season <= last_complete]
    targets = range(FIRST_TARGET_SEASON, last_complete + 1)
    best = None
    grid = []
    for kappa in KAPPA_GRID:
        for lam in LAMBDA_GRID:
            s = backtest(complete, kappa, lam, targets)
            grid.append((kappa, lam, s["model"]["all"]["rmse"]))
            if best is None or s["model"]["all"]["rmse"] < best[2]["model"]["all"]["rmse"]:
                best = (kappa, lam, s)
    kappa, lam, score = best
    starters = [p for p in complete if p.starter]
    prior = build_prior(starters)
    persistence = _persistence(starters)
    return {
        "kappa": kappa, "lambda": lam, "prior": {f"{k[0]}|{k[1]}": v for k, v in prior.items()},
        "backtest": score, "grid": grid, "persistence": persistence,
        "start_season": min(p.season for p in complete), "end_season": last_complete,
        "sample_size": len(starters),
        "other_share": sum(p.other_missed for p in starters) / sum(p.team_games for p in starters),
    }


def _persistence(starters: list[PlayerSeason]) -> dict:
    """Correlation between a player's injury-missed share in one season and
    the next, over players with both: the plain-English answer to "does
    injury history predict anything"."""
    by = {(p.gsis_id, p.season): p.m for p in starters}
    pairs = [(m, by[(g, s + 1)]) for (g, s), m in by.items() if (g, s + 1) in by]
    n = len(pairs)
    if n < 3:
        return {"n": n, "r": None}
    mx, my = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    sx = sum((a - mx) ** 2 for a, _ in pairs) ** 0.5
    sy = sum((b - my) ** 2 for _, b in pairs) ** 0.5
    return {"n": n, "r": sum((a - mx) * (b - my) for a, b in pairs) / (sx * sy) if sx and sy else None}


# ------------------------------------------- in-season status and recent misses

RECENT_GAMES = 3
MIN_CELL_CASES = 30
# Labels compare a player with a healthy one at the same point in the season:
# this many extra expected missed games per 17. About 4 in 5 in-season cases
# are healthy, so quartile labels would mean nothing.
ELEVATED_EXCESS = 1.0
HIGH_EXCESS = 3.0
STATUS_KEYS = ("none", "Questionable", "Doubtful", "Out")
# What Sleeper calls it today -> the report status whose history stands in for
# it. IR, PUP and NA have no weekly history (a player on IR leaves the report),
# so they borrow the case "Out and already missed his last three games", the
# nearest data to the NFL's four-game IR minimum. Always labeled as a proxy.
SLEEPER_STATUS = {"Questionable": "Questionable", "Doubtful": "Doubtful", "Out": "Out", "COV": "Out", "DNR": "Out",
                  "IR": "Out", "PUP": "Out", "NA": "Out", "Sus": "Out"}
PROXY_SLEEPER = {"IR", "PUP", "NA", "Sus"}


def _status_key(status: str | None) -> str:
    return status if status in STATUS_KEYS else "none"


def _recent_at(p: PlayerSeason, c: int) -> int:
    """How many of his last RECENT_GAMES team games before game index c he
    missed with an injury, judged only on the games before c. The run of
    misses is cut off at c, so a flag posted after the cut cannot turn an
    unexplained absence into an injury after the fact."""
    known = injury_missed_weeks(p.team_weeks[:c], set(p.played_weeks), set(p.flagged_weeks))
    return sum(1 for w in p.team_weeks[c - RECENT_GAMES:c] if w in known)


def in_season_cases(seasons: list[PlayerSeason], horizons=STATUS_HORIZONS):
    """One case per starter-level player-season and cut point c (before his
    team's c-th game, c from RECENT_GAMES on): (season, recent, status,
    {R: injury-missed games among the next R team games}). recent is how many
    of his last RECENT_GAMES team games he missed with an injury; status is
    his report designation for the next game, known before it is played, so
    nothing in a case looks past its cut (the outcome windows do, by design).
    A horizon is included only when R
    full team games remain."""
    for p in seasons:
        if not p.starter or len(p.team_weeks) < RECENT_GAMES + 1:
            continue
        inj = injury_missed_weeks(p.team_weeks, set(p.played_weeks), set(p.flagged_weeks))
        for c in range(RECENT_GAMES, len(p.team_weeks)):
            recent = _recent_at(p, c)
            status = _status_key(p.statuses.get(p.team_weeks[c]))
            missed = {R: sum(1 for w in p.team_weeks[c:c + R] if w in inj)
                      for R in horizons if c + R <= len(p.team_weeks)}
            yield p.season, recent, status, missed


def build_status_table(cases: list[tuple]) -> dict:
    """Mean injury-missed games in the next R team games, by (recent, status)
    cell and horizon. Cells with fewer than MIN_CELL_CASES cases borrow the
    same `recent` row, then the all-cases row, so a rare cell never reports
    an average of three players. Returns {cell_key: {"n": int, R: mean}}
    with cell keys "recent|status", "recent|*" and "*|*"."""
    acc: dict[str, dict] = defaultdict(lambda: {"n": 0, **{R: [0.0, 0] for R in STATUS_HORIZONS}})
    for _season, recent, status, missed in cases:
        for key in (f"{recent}|{status}", f"{recent}|*", "*|*"):
            a = acc[key]
            a["n"] += 1
            for R, m in missed.items():
                a[R][0] += m
                a[R][1] += 1
    table = {}
    for key, a in acc.items():
        table[key] = {"n": a["n"], **{R: (a[R][0] / a[R][1] if a[R][1] else None) for R in STATUS_HORIZONS}}
    return table


def cell_for(table: dict, recent: int, status: str) -> tuple[str, dict]:
    for key in (f"{recent}|{status}", f"{recent}|*", "*|*"):
        cell = table.get(key)
        if cell and cell["n"] >= MIN_CELL_CASES:
            return key, cell
    return "*|*", table["*|*"]


def expected_missed(table: dict, recent: int, status: str, games_left: int) -> float:
    """Expected injury-missed games among the player's next games_left team
    games, interpolated between the stored horizons and held flat past the
    last one."""
    if games_left <= 0:
        return 0.0
    _key, cell = cell_for(table, recent, status)
    pts = [(0, 0.0)] + [(R, cell[R]) for R in STATUS_HORIZONS if cell.get(R) is not None]
    if games_left >= pts[-1][0]:
        return pts[-1][1]
    for (r0, v0), (r1, v1) in zip(pts, pts[1:]):
        if games_left <= r1:
            return v0 + (v1 - v0) * (games_left - r0) / (r1 - r0)
    return pts[-1][1]


def status_holdout(seasons: list[PlayerSeason], last_complete: int) -> dict:
    """Leave-one-season-out check of the status table: for each season, build
    the table from every other season and compare its predictions with what
    happened against the no-information baseline (the all-cases mean). RMSE
    over injury-missed games in the next 4 and 8 games."""
    cases = [c for c in in_season_cases([p for p in seasons if p.season <= last_complete])]
    out = {}
    for R in (4, 8):
        se_model = se_base = n = 0
        for held in sorted({c[0] for c in cases}):
            table = build_status_table([c for c in cases if c[0] != held])
            base = table["*|*"][R]
            for season, recent, status, missed in cases:
                if season != held or R not in missed:
                    continue
                _k, cell = cell_for(table, recent, status)
                se_model += (cell[R] - missed[R]) ** 2
                se_base += (base - missed[R]) ** 2
                n += 1
        out[R] = {"n": n, "rmse_model": (se_model / n) ** 0.5, "rmse_baseline": (se_base / n) ** 0.5}
    return out


# ------------------------------------------------------ fitting and storage

def fit_and_store(conn: sqlite3.Connection, last_complete: int) -> dict:
    """Fit both parts on every complete season through last_complete and store
    one new availability_model row: the history model (a position-and-age
    prior plus the player's own decayed record, grid-searched on a leak-free
    backtest) and the in-season table (recent injury misses and report status
    against the next games). Returns the report fit-availability prints."""
    seasons = load_player_seasons(conn, last_complete)
    if not any(p.starter for p in seasons):
        raise ValueError("No starter-level player-seasons found. Run `dynasty-agent backfill-history` first.")
    history = fit_model(seasons, last_complete)
    cases = list(in_season_cases([p for p in seasons if p.season <= last_complete]))
    table = build_status_table(cases)
    holdout = status_holdout(seasons, last_complete)

    model = {
        "kappa": history["kappa"], "lambda": history["lambda"], "prior": history["prior"],
        "status_table": {k: {"n": v["n"], **{str(R): v[R] for R in STATUS_HORIZONS}} for k, v in table.items()},
        "history_backtest": history["backtest"], "persistence": history["persistence"],
        "status_holdout": {str(R): v for R, v in holdout.items()}, "other_share": history["other_share"],
        "grid": history["grid"],
    }
    bt = history["backtest"]
    conn.execute(
        """
        INSERT INTO availability_model (
            fitted_at, start_season, end_season, sample_size, kappa, lambda,
            history_rmse, prior_rmse, league_rmse, model_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (utcnow(), history["start_season"], last_complete, history["sample_size"], history["kappa"],
         history["lambda"], bt["model"]["all"]["rmse"], bt["prior"]["all"]["rmse"], bt["league"]["all"]["rmse"],
         json.dumps(model)),
    )
    conn.commit()
    return {**model, "start_season": history["start_season"], "end_season": last_complete,
            "sample_size": history["sample_size"], "status_cases": len(cases)}


def current_model(conn: sqlite3.Connection) -> dict | None:
    """The latest stored fit, or None if fit-availability has never run.
    Callers apply no availability discount on None and never guess one."""
    row = conn.execute("SELECT * FROM availability_model ORDER BY id DESC LIMIT 1").fetchone()
    if row is None:
        return None
    model = json.loads(row["model_json"])
    model["status_table"] = {
        k: {"n": v["n"], **{int(R): x for R, x in v.items() if R != "n"}} for k, v in model["status_table"].items()
    }
    model["prior"] = {tuple(None if p == "None" else p for p in k.split("|")): v for k, v in model["prior"].items()}
    return {**model, "fitted_at": row["fitted_at"], "end_season": row["end_season"]}


# ---------------------------------------------------- one player, right now

# Sleeper's team code to the schedule's. Same one valuation.TEAM_ALIASES holds;
# repeated because valuation will import this module.
SLEEPER_TEAM = {"LAR": "LA"}


def remaining_games(season: int, team: str, from_week: int) -> int:
    """Regular-season games left for a team from from_week on, played or not
    yet scored, from the cached schedule."""
    con = duckdb.connect()
    try:
        row = con.execute(
            "SELECT count(*) FROM read_parquet(?) WHERE game_type = 'REG' AND season = ? AND week >= ? "
            "AND (home_team = ? OR away_team = ?)",
            [str(nflverse.ensure_games_cached()), season, from_week, SLEEPER_TEAM.get(team, team), SLEEPER_TEAM.get(team, team)],
        ).fetchone()
    finally:
        con.close()
    return int(row[0])


def expected_missed_games(model: dict, recent: int, status: str, games_left: int, base_share: float) -> float:
    """Expected injury-missed games in the next games_left team games: the
    status table up to its longest horizon, then the long-run share for any
    games beyond it."""
    last = STATUS_HORIZONS[-1]
    near = expected_missed(model["status_table"], recent, status, min(games_left, last))
    return near + base_share * max(0, games_left - last)


# Injuries the user reported in this chat: {sleeper_id: games out}. Set only
# for the duration of one tool call, by `assumptions`, so a stale one never
# leaks into a later command. Always shown as the user's word, never as data.
_ASSUMED: dict[str, int] = {}


@contextmanager
def assumptions(games_out: dict[str, int] | None):
    _ASSUMED.clear()
    _ASSUMED.update(games_out or {})
    try:
        yield
    finally:
        _ASSUMED.clear()


def risk_label(per_17: float, healthy_per_17: float) -> str:
    excess = per_17 - healthy_per_17
    if excess >= HIGH_EXCESS:
        return "high"
    if excess >= ELEVATED_EXCESS:
        return "elevated"
    return "typical"


def player_risk(
    conn: sqlite3.Connection,
    model: dict,
    sleeper_id: str,
    index: dict[str, list[PlayerSeason]],
    season: int,
    next_week: int,
    in_season: bool = True,
) -> dict | None:
    """Availability for one player right now, or None when he is not a QB, RB,
    WR or TE or has no id crosswalk row. `index` is gsis_id -> his
    PlayerSeasons (load_player_seasons grouped); `season` and `next_week` are
    the game about to be played. Out of season (`in_season` false) there is
    no recent record and the long-run share applies to a full 17.

    Returns expected games missed over the rest of the season and per 17, a
    label, a_now and a_3 (availability now and averaged over three
    seasons), and the reasons in plain words. Sleeper gives a status but no
    body part or timeline, which the output says."""
    row = conn.execute(
        "SELECT p.position, p.injury_status, p.team, p.age, i.gsis_id, i.birthdate FROM players p "
        "LEFT JOIN player_ids i ON i.sleeper_id = p.player_id WHERE p.player_id = ?", (sleeper_id,)
    ).fetchone()
    if row is None or row["position"] not in POSITIONS or not row["gsis_id"]:
        return None
    position, gsis = row["position"], row["gsis_id"]
    seasons = sorted(index.get(gsis, []), key=lambda p: p.season)
    complete = [p for p in seasons if p.season < season]
    age = row["age"]
    prior = prior_for(model["prior"], position, age)
    history = [(p.season, p.team_games, p.injury_missed) for p in complete if p.starter]
    base_share = predict_share(prior, history, season, model["kappa"], model["lambda"])

    games_left = remaining_games(season, row["team"], next_week) if in_season and row["team"] else 17
    sleeper_status = row["injury_status"]
    status = SLEEPER_STATUS.get(sleeper_status, "none") if sleeper_status else "none"
    proxy = sleeper_status in PROXY_SLEEPER
    recent = 0
    now = next((p for p in seasons if p.season == season), None) if in_season else None
    if now is not None and len(now.team_weeks) >= RECENT_GAMES:
        recent = _recent_at(now, len(now.team_weeks))
    if proxy:
        recent = RECENT_GAMES
    if not in_season:
        recent = 0

    said = _ASSUMED.get(sleeper_id)
    if said is not None:
        # the user's word: out for `said` games (capped at what is left), then the long-run share
        out_games = min(said, games_left)
        e = out_games + base_share * (games_left - out_games)
    elif in_season:
        e = expected_missed_games(model, recent, status, games_left, base_share)
    else:
        e = base_share * games_left
    rest = games_left or 1
    per_17 = e / rest * 17
    healthy_per_17 = (expected_missed_games(model, 0, "none", games_left, base_share) / rest * 17
                      if in_season else per_17)
    a_now = 1 - e / rest
    future = [1 - predict_share(prior_for(model["prior"], position, (age or 26) + t), history, season + t,
                                model["kappa"], model["lambda"]) for t in (1, 2)]
    a_3 = (a_now + sum(future)) / 3

    reasons = []
    if said is not None:
        reasons.append(f"you told me he is out about {said} games; I can't verify that" if said < games_left
                       else "you told me he is out for the season; I can't verify that")
    if sleeper_status:
        note = " (treated like a player who has already missed his last three games)" if proxy else ""
        reasons.append(f"Sleeper lists him {sleeper_status}{note}; it gives no body part or timeline")
    if recent:
        reasons.append(f"missed {recent} of his last {RECENT_GAMES} games with an injury")
    last = history[-1] if history else None
    if last and last[2] >= 4:
        reasons.append(f"missed {last[2]} of {last[1]} games with injuries in {last[0]}")
    if not history:
        reasons.append("no starter-level NFL history, so the position average applies")
    return {
        "gsis_id": gsis, "position": position, "age": age, "games_left": games_left,
        "status": sleeper_status, "status_is_proxy": proxy, "recent_misses": recent,
        "expected_missed": e, "missed_per_17": per_17, "label": risk_label(per_17, healthy_per_17),
        "a_now": a_now, "a_3": a_3, "long_run_per_17": base_share * 17, "healthy_per_17": healthy_per_17, "reasons": reasons,
        "history": history, "no_history": not history, "assumed_games_out": said,
    }


def risk_rows(conn: sqlite3.Connection, players: list[tuple[str, str]]) -> list[dict]:
    """Availability for (sleeper_id, name) pairs as [{"name", "risk"}], for the
    pairs that are QB/RB/WR/TE with an id match. Empty when fit-availability
    has never run, so every caller shows nothing rather than guessing. Shown
    beside the numbers, never folded into them (Ethan's call, 2026-10-14)."""
    model = current_model(conn)
    state = conn.execute("SELECT season, week, season_type FROM nfl_state ORDER BY fetched_at DESC LIMIT 1").fetchone()
    if model is None or state is None or state["season"] is None:
        return []
    season = int(state["season"])
    in_season = state["season_type"] == "regular"
    target = season + 1 if state["season_type"] == "off" else season
    index: dict[str, list[PlayerSeason]] = defaultdict(list)
    for p in load_player_seasons(conn, season):
        index[p.gsis_id].append(p)
    rows = []
    for sleeper_id, name in dict.fromkeys(players):
        r = player_risk(conn, model, sleeper_id, index, target, max(1, int(state["week"] or 1)), in_season)
        if r:
            r["history_r"] = model["persistence"]["r"]
            rows.append({"name": name, "risk": r})
    return rows
