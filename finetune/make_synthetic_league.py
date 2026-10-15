"""Build a synthetic league database from a real one, so a model can be trained on it and published.

The real database holds a private league: who owns whom, the managers' names, the league and draft ids and the
Sleeper responses cached from them. This keeps what is public (every NFL player, nflverse stats, FantasyCalc market
values, the league's format rules) and replaces what is not:

- manager and team names, user ids and league and draft ids become fake ones;
- players are shuffled among the roster rows that share their position and slot kind, so every roster keeps a
  legal shape (a WR starter slot still holds a WR) while nobody owns who they really own;
- win-loss records and points are redrawn;
- matchups, traded picks, transactions and every cached Sleeper response are deleted;
- the file is vacuumed so deleted rows leave no trace in its free pages.

It finishes by scanning every text column of every table for each identifier the real league had and fails if one
is left. Usage:  python finetune/make_synthetic_league.py <real.db> <synthetic.db> [--seed 7]
"""

from __future__ import annotations

import argparse
import random
import sqlite3
import sys
from pathlib import Path

FAKE_LEAGUE_ID = "100000000000000001"
FAKE_DRAFT_ID = "100000000000000002"


def private_strings(conn: sqlite3.Connection) -> list[str]:
    """Every identifier of the real league: ids, names, manager and team names."""
    found: set[str] = set()
    queries = [
        "SELECT league_id FROM league", "SELECT name FROM league", "SELECT draft_id FROM drafts",
        "SELECT user_id FROM users", "SELECT display_name FROM users", "SELECT team_name FROM users",
        "SELECT owner_id FROM rosters",
    ]
    for q in queries:
        try:
            found.update(str(r[0]) for r in conn.execute(q) if r[0] and len(str(r[0])) >= 5)
        except sqlite3.OperationalError:
            pass
    return sorted(found)


def build(src: Path, dst: Path, seed: int) -> None:
    if dst.exists():
        dst.unlink()
    source = sqlite3.connect(src)
    conn = sqlite3.connect(dst)
    source.backup(conn)
    source.close()
    conn.row_factory = sqlite3.Row
    rng = random.Random(seed)
    secrets = private_strings(conn)

    # 1. identities
    conn.execute("DELETE FROM api_cache WHERE cache_key LIKE 'sleeper:%'")
    for table in ("matchups", "traded_picks", "transactions"):
        conn.execute(f"DELETE FROM {table}")
    user_ids = {}
    for i, row in enumerate(conn.execute("SELECT user_id FROM users ORDER BY user_id").fetchall(), start=1):
        user_ids[row["user_id"]] = f"synthuser{i:02d}"
    for old, new in user_ids.items():
        n = new[-2:]
        conn.execute("UPDATE users SET user_id = ?, display_name = ?, team_name = ?, league_id = ? WHERE user_id = ?",
                     (new, f"Manager {n}", f"Team {n}", FAKE_LEAGUE_ID, old))
    conn.execute("UPDATE league SET league_id = ?, name = 'Synthetic Dynasty', previous_league_id = NULL", (FAKE_LEAGUE_ID,))
    conn.execute("UPDATE drafts SET draft_id = ?, league_id = ?", (FAKE_DRAFT_ID, FAKE_LEAGUE_ID))
    conn.execute("UPDATE draft_picks SET draft_id = ?", (FAKE_DRAFT_ID,))

    # 2. rosters: new owners, redrawn records, and players shuffled within position and slot kind
    roster_ids = [r[0] for r in conn.execute("SELECT roster_id FROM rosters ORDER BY roster_id")]
    owners = sorted(user_ids.values())
    rng.shuffle(owners)
    for rid, owner in zip(roster_ids, owners):
        games = conn.execute("SELECT coalesce(wins,0) + coalesce(losses,0) + coalesce(ties,0) FROM rosters WHERE roster_id = ?", (rid,)).fetchone()[0]
        wins = rng.randint(0, games)
        conn.execute("UPDATE rosters SET league_id = ?, owner_id = ?, wins = ?, losses = ?, ties = 0, fpts = ?, fpts_against = ? "
                     "WHERE roster_id = ?", (FAKE_LEAGUE_ID, owner, wins, games - wins,
                                             round(rng.uniform(85, 125) * games, 2), round(rng.uniform(85, 125) * games, 2), rid))
    rows = conn.execute("SELECT rp.roster_id, rp.player_id, rp.slot, p.position FROM roster_players rp "
                        "JOIN players p ON p.player_id = rp.player_id").fetchall()
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["position"], r["slot"] == "taxi"), []).append(r)
    conn.execute("DELETE FROM roster_players")
    for members in groups.values():
        ids = [m["player_id"] for m in members]
        rng.shuffle(ids)
        # a shuffle can put one player on two rosters of a group only if ids repeat, which the table forbids
        for m, pid in zip(members, ids):
            conn.execute("INSERT INTO roster_players (roster_id, player_id, slot, fetched_at) VALUES (?, ?, ?, ?)",
                         (m["roster_id"], pid, m["slot"], "2026-01-01T00:00:00+00:00"))
    # rookie draft results: same slots, shuffled players
    picks = conn.execute("SELECT rowid AS rid, player_id FROM draft_picks").fetchall()
    pids = [p["player_id"] for p in picks]
    rng.shuffle(pids)
    for p, pid in zip(picks, pids):
        conn.execute("UPDATE draft_picks SET player_id = ?, metadata_json = NULL WHERE rowid = ?", (pid, p["rid"]))

    my_user = owners[0]
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('synthetic_user_id', ?)", (my_user,))
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES ('synthetic_league_id', ?)", (FAKE_LEAGUE_ID,))
    conn.commit()
    conn.execute("VACUUM")
    conn.close()

    leaks = find_leaks(dst, secrets)
    if leaks:
        dst.unlink()
        sys.exit("LEAK: identifiers of the real league remain in " + ", ".join(sorted({f'{t}.{c}' for t, c in leaks})) +
                 ". The synthetic database was deleted.")
    print(f"synthetic league written to {dst}  ({len(secrets)} real identifiers checked, none left; your synthetic "
          f"user id is {my_user})")


def find_leaks(db: Path, secrets: list[str]) -> list[tuple[str, str]]:
    """(table, column) pairs where a real identifier still appears, scanning every text column."""
    conn = sqlite3.connect(db)
    leaks = []
    for (table,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").fetchall():
        for col in [r[1] for r in conn.execute(f"PRAGMA table_info({table})") if (r[2] or "").upper() in ("TEXT", "")]:
            for s in secrets:
                if conn.execute(f'SELECT 1 FROM "{table}" WHERE "{col}" LIKE ? LIMIT 1', (f"%{s}%",)).fetchone():
                    leaks.append((table, col))
                    break
    raw = db.read_bytes()  # also the file itself: free pages and indexes
    for s in secrets:
        if s.encode() in raw:
            leaks.append(("<database file bytes>", "-"))
            break
    conn.close()
    return leaks


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    build(a.src, a.dst, a.seed)
