"""`dynasty-agent setup`: the whole one-time load in one command, so a new user can go from a fresh clone to the chat.

It connects the league (`init`, when a username is given and there is no .env yet), then runs the same commands the
README used to list one by one, in an order that works on an empty database: league and market values, last
season's NFL stats, draft and combine data, college production, past seasons' stats and injury reports, then the
three fitted models (rookies, win-probability calibration, injury availability), then a refresh. Each step calls the
existing command function, so its output and behavior are exactly what running it by hand gives.

Safe to rerun: downloads are cached, backfilled seasons are skipped and a model that is already fitted is left alone
unless --refit is given. It stops at the first step that fails and says how to continue.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

from dynasty_agent import config, context
from dynasty_agent.db import get_db
from dynasty_agent.errors import AgentError

FIRST_COLLEGE_SEASON = 2008
FIRST_HISTORY_SEASON = 2017
FIRST_CALIBRATION_SEASON = 2018


@dataclass
class Step:
    name: str
    run: Callable[[], None]
    skip_if: Callable[[], str | None] = lambda: None  # a reason to skip, or None
    optional: bool = False  # a failure is a warning, not the end of setup


def _has_rows(table: str) -> bool:
    return get_db().execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None


def missing_pieces(conn) -> list[str]:
    """What a chat needs that setup has not loaded yet, as plain phrases. Empty when it is ready to use."""
    checks = [
        ("your league", "SELECT 1 FROM league LIMIT 1"),
        ("NFL stats", "SELECT 1 FROM weekly_stats LIMIT 1"),
        ("the draft and combine data", "SELECT 1 FROM nfl_draft_picks LIMIT 1"),
        ("the rookie model", "SELECT 1 FROM prospect_model LIMIT 1"),
        ("the win-probability calibration", "SELECT 1 FROM calibration_params LIMIT 1"),
        ("the injury availability model", "SELECT 1 FROM availability_model LIMIT 1"),
    ]
    return [label for label, query in checks if conn.execute(query).fetchone() is None]


def build_steps(commands, refit: bool) -> list[Step]:
    ns = argparse.Namespace

    def last_complete() -> int:
        return context.latest_complete_season(get_db())

    def this_season() -> int:
        return int(get_db().execute("SELECT season FROM nfl_state ORDER BY fetched_at DESC LIMIT 1").fetchone()["season"])

    def fitted(table: str, what: str) -> Callable[[], str | None]:
        return lambda: None if refit or not _has_rows(table) else f"{what} is already fitted (--refit to fit it again)"

    return [
        Step("Sync your league, rosters, picks and market values", lambda: commands.cmd_sync(ns())),
        Step("Download last season's NFL stats", lambda: commands.cmd_ingest_nflverse(ns(season=last_complete(), force=False))),
        Step("Download NFL draft picks and combine results", lambda: commands.cmd_ingest_draft_data(ns(force=False))),
        Step("Download college production (the longest step, cached after the first run)",
             lambda: commands.cmd_ingest_college(ns(season=FIRST_COLLEGE_SEASON, through=this_season(), force=False))),
        Step("Load past seasons' stats and injury reports",
             lambda: commands.cmd_backfill_history(ns(start_season=FIRST_HISTORY_SEASON, end_season=None, force=False))),
        Step("Fit the rookie model", lambda: commands.cmd_fit_prospect_model(ns()), fitted("prospect_model", "The rookie model")),
        Step("Calibrate the win-probability model against real games",
             lambda: commands.cmd_calibrate_matchup_model(ns(start_season=FIRST_CALIBRATION_SEASON, end_season=None)),
             fitted("calibration_params", "The win-probability calibration")),
        Step("Fit the injury availability model", lambda: commands.cmd_fit_availability(ns(end_season=None)),
             fitted("availability_model", "The injury model")),
        # Best effort: a data source being down (a 404 on this week's file) must not undo a finished setup, and the
        # chat refreshes again every time it starts.
        Step("Bring everything current for this week", lambda: commands.cmd_refresh(ns(verbose=False)), optional=True),
    ]


def run(args: argparse.Namespace, commands) -> None:
    """`commands` is the cli module (a stand-in in tests)."""
    if config.missing_config():
        if not getattr(args, "username", None):
            raise AgentError("No league is connected yet. Run `dynasty-agent setup --username <your Sleeper username>`.")
        print(f"Connecting your league for '{args.username}'...")
        commands.cmd_init(argparse.Namespace(username=args.username, league_id=getattr(args, "league_id", None)))
        config.reload()
        if config.missing_config():
            raise AgentError("Your league was not saved to .env. Run `dynasty-agent init --username <name>` and look for errors.")
        print()
    elif getattr(args, "username", None):
        print("A league is already connected (see .env), so --username is ignored. Delete .env to connect a different one.\n")

    steps = build_steps(commands, getattr(args, "refit", False))
    started = time.monotonic()
    for i, step in enumerate(steps, start=1):
        print(f"[{i}/{len(steps)}] {step.name}")
        reason = step.skip_if()
        if reason:
            print(f"    skipped: {reason}\n")
            continue
        t0 = time.monotonic()
        try:
            step.run()
        except KeyboardInterrupt:
            print(f"\nStopped at step {i}. Run `dynasty-agent setup` again to continue; finished work is kept.")
            raise SystemExit(130) from None
        except (AgentError, SystemExit, Exception) as e:  # noqa: BLE001 (any failure ends setup with the same advice)
            if isinstance(e, SystemExit) and e.code in (0, None):
                print(f"    done in {time.monotonic() - t0:.0f}s\n")
                continue
            detail = str(e) if not isinstance(e, SystemExit) else "the command reported an error above"
            if step.optional:
                print(f"    warning: {detail}. Setup is otherwise done; the chat tries this again when it starts.\n")
                continue
            print(f"\nStep {i} failed: {detail}")
            print("Fix that, then run `dynasty-agent setup` again. Steps that finished are cached or skipped, so it picks up here.")
            raise SystemExit(1) from None
        print(f"    done in {time.monotonic() - t0:.0f}s\n")
    print(f"Setup complete in {time.monotonic() - started:.0f}s. Start chatting with:  dynasty-agent chat")
