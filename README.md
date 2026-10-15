# Dynasty Agent

A CLI tool that manages a dynasty fantasy football team on [Sleeper](https://sleeper.com). It pulls your real roster and league settings from the Sleeper API, layers in play-by-play and weekly stats from [nflverse](https://github.com/nflverse), and prices players and picks with dynasty market data from [FantasyCalc](https://fantasycalc.com), to answer three questions: what's my roster worth, am I contending or rebuilding, and is this trade good.

It is built for **one specific league shape**: 12 teams, 1QB, full PPR, no TE premium, a 3-round rookie-only startup/rookie draft. It will run against a differently-shaped Sleeper dynasty league, since your real league settings and scoring get pulled live, however some of the tuning (the age curve, the QB-discount and WR-bump multipliers, the situation score weights) was calibrated for that shape specifically. Read `CLAUDE.md` for the full reasoning behind those numbers before trusting them blindly in, say, a superflex or half-PPR league.

**On Windows?** Everything below works identically, same Python, same `pip`, same commands, checked directly rather than assumed. See [`WINDOWS.md`](WINDOWS.md) for PowerShell-specific install steps and command syntax (line continuation is a backtick, not a backslash) instead of translating the Bash examples below yourself.

## What it does today

- **`sync`**: pulls your league, all rosters, users, traded picks, your league's drafts and their picks, current NFL week, and FantasyCalc dynasty values. Sleeper gives a renewed dynasty league a new league ID every season; when yours has been renewed, `sync` says so and prints the `init` command that switches to it.
- **`roster`**: your current roster with age, team, market value, and 30-day value trend.
- **`ingest-nflverse`**: derives per-player-per-week target share, air yards share, WOPR, weighted opportunity, red zone touches, team pass rate over expected, and fantasy points computed from *your* league's actual scoring settings, not a generic PPR assumption.
- **`valuate`**: win-now value and three-year value per player (kept separate, never blended into one number), plus a contend-or-rebuild verdict with its confidence stated, not implied.
- **`trade`**: evaluates a proposed trade: both sides on win-now and three-year value, future picks discounted by a tunable rate and checked against FantasyCalc for arbitrage, and consolidation opportunities flagged.
- **`predict-matchup`**: win probability between any two rosters, not necessarily your own league or even a dynasty league. A heuristic, shown as a calibrated number with the raw one beside it once `calibrate-matchup-model` has run. See "How the numbers work" below for exactly what that means.
- **`optimize-lineup`**: the starting lineup, out of your real roster, that maximizes win probability against your actual Sleeper opponent for a given week, not raw projected points. Reports the highest-raw-points lineup alongside for comparison, since they can differ.
- **`faab`**: a sized FAAB bid for one waiver target, against your real remaining budget and the weeks it has to last, scaled by how much the target would raise your best lineup against the biggest upgrade anyone on the wire offers.
- **`digest`**: the weekly brief: recommended lineup and win probability, real wind flags for your starters' games, top bench options, and sized suggestions for the best available waiver targets, all in one command.
- **`refresh`**: brings everything current in one run, by time of year: Sleeper and FantasyCalc always; this season's NFL stats, lines, and matchups in season; college stats August to January; draft and combine data February to May; the prospect model refit when a new draft class completes 3 NFL seasons. Ends with how old each source is. Run it first, or let `schedule` run it for you.
- **`schedule`** *(macOS)*: runs `refresh` every day at a set time (default 06:00) through macOS's own launchd, logging to `data/logs/refresh.log`. A run missed while the Mac slept happens on wake. Windows: see `WINDOWS.md`.
- **`ingest-draft-data`**: real NFL draft picks and combine testing (nflverse) plus a player id crosswalk (DynastyProcess and nflverse) linking Sleeper, ESPN, and Pro Football Reference ids.
- **`ingest-college`**: college production from sportsdataverse's free ESPN college football data, with each team's conference and opponent-adjusted strength.
- **`fit-prospect-model`**: fits the rookie model against how the 2018 through latest-complete draft classes actually scored in your league's format, and reports its held-out accuracy.
- **`prospect-board`**: a ranked rookie board. `--mode post-draft` ranks by draft capital (college production was tested and adds nothing once draft capital is known); `--mode pre-draft` ranks on college production and team context before the draft order exists, a weak signal labeled weak on every run.
- **`picks`**: your rookie picks for the next three drafts (`--all` for every team): the next draft's projected slot, what that slot's rookies have really produced since 2018, FantasyCalc's price, and buy, hold, or sell against what FantasyCalc now pays for the rookies that same slot bought last year.
- **`taxi`**: which players to move to taxi or IR right now to free bench spots, and whether next year's rookie draft forces a cut, every rule read from your league's settings.
- **`calibrate-blend`**: reruns the backtest behind how much last season counts this season (see below).
- **`backfill-history`**: loads past seasons' weekly stats and real injury reports (about 20 MB of play-by-play per season, a few seconds each), the history the next command needs. Skips seasons already loaded.
- **`fit-availability`**: fits how many games a QB, RB, WR or TE is expected to miss, from past seasons' real injury reports and games played. It found that a player's injury history predicts his next season barely at all (season-to-season r of about 0.06), whereas his recent misses and current report status predict the next few games well (23% lower error over the next 4 games on held-out seasons). Run it after `backfill-history`, and again after each season. Trade evaluations, player checks and trade ideas print each player's risk beside the numbers (label, expected games missed per 17, reasons). It is shown, never subtracted: no value, verdict or ranking changes because of it.
- **`calibrate-matchup-model`**: tests the win-probability heuristic against about 2,100 real NFL games and fits a correction for its overconfidence. Every win probability the tool prints (`optimize-lineup`, `predict-matchup`, `digest`, `simulate-season` and the chat) then shows the corrected number with the raw one beside it.
- **`simulate-season`**: simulates the rest of the regular season 10,000 times from your league's real remaining schedule: each team's playoff odds, average final wins and rank, and last-place odds.
- **`chat`** *(preview)*: ask about your league in plain English, answered by an open-source model running on your own machine. See "Talk to the agent" below.
- **`chat-eval`**: scores the local model's routing on 39 real questions, trade direction included.

## Requirements

- Python 3.12 or newer (`python3 --version` to check; get it from [python.org](https://www.python.org/downloads/) if needed).
- A Sleeper account that's a member of at least one dynasty league. You don't need an API key, Sleeper's read API is public.

## Install

```
git clone https://github.com/3schtocky/Sleeper-Dynasty-Agent.git
cd Sleeper-Dynasty-Agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

This creates a `.venv` and installs every dependency from `pyproject.toml`. Run `source .venv/bin/activate` in each new terminal window before using `dynasty-agent`.

## Point it at your own league

Nothing about your Sleeper account or league is hardcoded. Run:

```
dynasty-agent init --username <your-sleeper-username>
```

This looks up your Sleeper user ID, finds your current-season dynasty leagues, and writes a `.env` file (already in `.gitignore`, it never gets committed) with your username, user ID, league ID, and draft ID. If you're in more than one league this season, it lists them and asks you to re-run with `--league-id <id>` to pick one.

Prefer to do it by hand, or your league is for a season Sleeper doesn't return by default? Copy `.env.example` to `.env` and fill in the four values yourself. Your league ID and draft ID are both in your league's Sleeper URL and in the response from `https://api.sleeper.app/v1/user/<your-username>`.

If your league isn't 12-team/1QB/full-PPR, also set the `FANTASYCALC_NUM_QBS` / `FANTASYCALC_NUM_TEAMS` / `FANTASYCALC_PPR` variables shown (commented out) in `.env.example`, so the FantasyCalc market values you pull actually match your format.

## Run it

```
dynasty-agent refresh                        # everything current: league, values, this season's stats
dynasty-agent roster                         # sanity check: is this your team?
dynasty-agent valuate                        # win-now/three-year value + the verdict
dynasty-agent schedule --install             # optional, macOS: run refresh daily at 06:00 (--at to change)
```

First run on a new machine, before `refresh` has anything to build on: `sync`, then `ingest-nflverse --season <last completed season>`, `ingest-draft-data`, `ingest-college --season 2008 --through <current season>`, `fit-prospect-model`, `backfill-history`, `calibrate-matchup-model` and `fit-availability`. After that, `refresh` keeps it all current. Each ingest command still works on its own; `ingest-nflverse` re-downloads the current season automatically and takes `--force` for any other.

Evaluate a trade, players by name or Sleeper `player_id`, picks written any common way (`2027-1`, `"2027 1st"`, `"2027 early 1st"`, `"2027 1.05"`; a tier or slot gets FantasyCalc's tiered price):

```
dynasty-agent trade \
  --send "Rashee Rice" \
  --receive-pick 2027-1 --receive-pick 2027-3 \
  --discount-rate 0.20
```

`trade` reports three numbers per side, deliberately never summed together: **win-now** and **three-year** (this league's own formula, players only, a pick can't help you win this year so it doesn't appear there), and **market value** (FantasyCalc's real pricing for players plus a discount-adjusted model value for picks, the one number that's actually comparable across a player and a pick in the same trade).

Estimate a matchup for a specific week, any two rosters, kicker and defense included if you want, players by name or `player_id`:

```
dynasty-agent predict-matchup --week 1 \
  --team-a "Jalen Hurts" --team-a "Derrick Henry" --team-a "Puka Nacua" \
  --team-b "Jayden Daniels" --team-b "James Cook" --team-b "Drake London"
```

`--week` is required, it picks up real Vegas lines for that week from nflverse's own free schedules data (no paid odds API, no API key). The FPPG baseline season and the Vegas season are two different numbers on purpose, `--season` defaults to the most recently completed season (there's usually no current-season data yet) while `--vegas-season` defaults to whatever season is actually live right now, per your last `sync`. Override either if you're checking a specific past week.

Optimize your own lineup for a real week, against your real Sleeper opponent:

```
dynasty-agent optimize-lineup --week 1
```

Size a FAAB bid on a specific waiver target:

```
dynasty-agent faab --player "Marquise Brown"
```

Or just run the whole week in one shot, lineup, win probability, wind flags, and sized FAAB targets:

```
dynasty-agent digest --week 1
```

`digest` makes real live network calls per recommended starter (checking wind for their specific game), so it's the slowest command here, correctness kept over shaving that down.

Every command reruns fresh against whatever's cached in `data/` (git-ignored, local SQLite plus downloaded nflverse parquet files). Nothing here needs a server or an account beyond your own Sleeper login.

## Calibrated win probabilities and the season simulation

The win probability behind `optimize-lineup`, `digest` and `predict-matchup` started as an uncalibrated heuristic. Tested on 2,127 real NFL games from 2018 to 2025 it was badly overconfident: it needs its probabilities pulled most of the way back toward 50%. Run this once, and again after each season:

```
dynasty-agent backfill-history               # past seasons' stats and injury reports (skips seasons already loaded)
dynasty-agent calibrate-matchup-model        # tests the heuristic and stores the correction
dynasty-agent fit-availability               # injury availability: history, recent misses, report status
```

From then on every win probability reads like `Win probability: 48.8%  (raw heuristic 46.7%, ...)`: the corrected number, with the raw one beside it. Before the first fit, the raw number is shown with a note saying so. Then:

```
dynasty-agent simulate-season                # playoff odds for every team (--simulations, --seed)
```

`simulate-season` assumes each team fields its best healthy lineup every week and keeps its roster as it is today, so every team's odds are slightly flattered.

## Talk to the agent (preview)

`chat` answers plain-English questions about your league with a small open-source model running on your own machine through [Ollama](https://ollama.com). The model only picks which command to run and writes a sentence or two about the result. Every number comes from the same Python the commands above use, and the chat prints the command's full output before the model says anything.

One-time setup:

1. Install Ollama from [ollama.com/download](https://ollama.com/download) and open it (it runs in the menu bar).
2. Download the model, about 2.5 GB:
   ```
   ollama pull qwen3:4b-instruct
   ```

Then:

```
dynasty-agent chat
```

It refreshes your league data while the model loads, then takes questions:

```
you> Should I trade Jonah Coleman and my 2028 2nd for a 2027 1st?
  (the full trade evaluation, exactly as `trade` prints it)
No, you should not make this trade. The net market value is -221, and the net win-now is -6.1.
                                   qwen3:4b-instruct · 29.7 tok/s · first words 1.0s · tool: evaluate_trade
```

It can set your lineup, evaluate a trade, judge contend or rebuild, find waiver pickups and size a bid, advise on your picks and plan taxi and IR moves. Ask "who should I start?", "am I a contender?", "who should I pick up?", "how much should I bid on Trey Benson?", "should I sell my first?" or "anyone I should put on taxi?".

What keeps it honest:

- **Python computes every number.** The model's sentences are checked one at a time before they're shown: a sentence with a number that isn't in the command's output is dropped, with a note saying so.
- **Names it can't settle become questions.** "Trade Kenneth Walker for Puka Nacua?" asks which Kenneth Walker you mean.
- **Trade sides are checked against the rosters.** If the model puts a player you own on the side you'd receive, the evaluation moves them and says so.
- **It can't change your roster.** Sleeper's API is read-only: make every move in the Sleeper app.

Ask how it works and it shows the calculation with your own numbers, then explains it in plain language: "How did you get Trey Benson's win-now value?", "Why is his 3yr lower?", "What does situation score mean?", "Why does age matter for running backs?". The steps come from the same constants that produce the number and are checked against it before they are shown, so the model can retell them but not invent them. It works for a player's value, a trade ("why?" after the answer), the lineup's win probability, a FAAB bid, the contend or rebuild call, and a buy, hold or sell on a pick. Follow-ups like "why?" refer to the last answer.

Commands inside the chat: `/raw` shows what the model was given for the last answer, `/stats off` hides the stats line, `/help`, `/quit`. `--no-refresh` skips the start-up refresh for a quick restart.

**Watch it run** from a second terminal window: `ollama ps` shows the loaded model and its memory, and `tail -f ~/.ollama/logs/server.log` shows every request as it arrives.

**Settings**, in `.env`: `LLM_MODEL` (default `qwen3:4b-instruct`, chosen by a bake-off of eight small models, see `PLANNING.md`) and `OLLAMA_URL` (default `http://localhost:11434`). `dynasty-agent chat-eval --model <name>` scores any other Ollama model on the same 39 questions before you switch.

This is a preview: a small model runs at about 30 tokens per second on a MacBook Air M4, and it will sometimes phrase an answer awkwardly. The numbers above each answer are always the real ones.

## How the numbers work, briefly

- **This season and last, blended**: every per-game average starts from last season's (worth 4 games) and this season's real games add on top, so a new season takes over as it accumulates instead of all at once. A rookie starts from his draft-capital projection instead (worth 3 games). Both weights were chosen by backtest against real outcomes (`calibrate-blend`), not by feel.
- **Fantasy points** come from your league's real `scoring_settings`, pulled live from the Sleeper API, not a hardcoded PPR formula.
- **Age curve**: flat at full value through a position's peak, then an exponential decay past it, tuned per position (see `metrics.AGE_CURVES`).
- **Situation score**: a team's QB passing EPA, pass rate over expected, and sack rate allowed (inverted, a public proxy for offensive line quality), each percentile-ranked against all 32 NFL teams and averaged.
- **Pick values**: anchored to FantasyCalc's real current price for a `<round>` pick, discounted forward per year by `--discount-rate` (default 20%/year).
- **Matchup win probability** (`predict-matchup`) takes each side's players' real per-season mean and *sample* variance of weekly fantasy points (Bessel's correction, dividing by n-1, an earlier draft divided by n and understated it), scales the mean by a real Vegas-implied team total for the specific week versus that team's own season norm (nflverse's free schedules data, `spread_line`/`total_line`, no paid odds API), widens or collapses variance by current injury status (Questionable/Doubtful is closer to bimodal than a healthy player's normal week-to-week swing, so it widens; Out/IR collapses toward near-certain zero), and reads the win probability off the normal distribution of the resulting margin, the same idea Vegas spread-to-moneyline conversion uses. Its raw probability was tested against real outcomes (`calibrate-matchup-model`, below) and found badly overconfident, so the number shown is corrected. It still assumes players score independently of their teammates, which isn't quite true (a QB and his own WR1 correlate), and the projections under it are a heuristic grounded in real per-player, per-week data at every step.

- **Lineup optimizer** (`optimize-lineup`, `digest`) evaluates every valid lineup your real roster supports (respecting this league's own `roster_positions`, never hardcoded) and picks the one with the highest computed win probability against your real Sleeper opponent for that week, not the highest raw point total. They usually agree; when they don't, that's the flex slot earning its keep, trading a little mean for a lower-variance (or higher-ceiling, if you're the underdog) option.
- **Calibration** (`calibrate-matchup-model`) replays every completed regular-season game from 2018 on: both sides are projected the way the tool projects a week (last season blended with this season's earlier weeks only, then the injury report for that week and that week's Vegas lines), the raw win probability is recorded, and it is compared with who won. The correction is one number, `a`, in `sigmoid(a x logit(raw))`; it came out at 0.36. When the heuristic said about 95%, the favorite won 75% of the 413 such games (not 95%); when it said about 85%, 65% of 451. After the correction the predicted and actual rates agree to within a point or two in those ranges. It is a **slope only, with no intercept**: that makes it treat the two teams alike (swap them and you get the complement) and never change who is favored. It fixes how sure the heuristic is, not who it picks (accuracy stays at 63%). Which real team plays "team A" is a seeded coin flip, so home-field advantage, which the heuristic does not model, cannot leak into the correction. **Limits:** it is fitted on NFL team games with a roster rebuilt from who actually played, then applied to fantasy lineups; read it as right in direction and rough in size. Refit after each season.
- **Season simulation** (`simulate-season`) gives each real remaining matchup one calibrated win probability and flips a weighted coin for it 10,000 times, ranking teams by wins with points scored so far as the tiebreak. It reads only regular-season weeks, so "playoff odds" means odds of finishing in a qualifying seed, not of winning a bracket.
- **Opponent strength by position** is real EPA per play allowed by a defense, split by the offensive position that gained it (a QB's own scrambles count under QB, not RB), never raw fantasy points allowed, which is schedule-biased and noisy.
- **Weather** checks the real forecast (the National Weather Service's public API, free, no key) for a game's actual venue, not an assumed home stadium, international games are read from the real schedule data and either covered or explicitly flagged as not, never guessed. Domes and closed-roof games skip the check entirely, no network call.
- **FAAB sizing** (`faab`, `digest`) ranks free agents by how much each would raise your best possible lineup, so a backup QB behind your starter in a 1QB league ranks as depth only, however good. A bid is your remaining budget paced over the weeks left (to the playoffs, or over the playoff weeks once they start) and scaled by the target's lineup gain against the biggest upgrade on the wire.

Every one of these is a plain, readable function in `src/dynasty_agent/metrics.py` (or `weekly.py`/`weather.py` for the Phase 3 additions), unit tested where the logic is pure, verified live against real data everywhere else, not a black box.

## Project layout

```
src/dynasty_agent/   the CLI and every module (sleeper, nflverse, market, metrics, valuation, calibration,
                     simulate, and the chat: llm, tools, chat, grounding, formatters, explain)
migrations/          the SQLite schema, one file per migration, applied automatically
tests/                unit tests for the metric calculations, the tools and the chat (offline)
benchmarks/           the Phase 5 model bake-off harness
CLAUDE.md             environment, league settings, strategic reasoning, and working rules
PLANNING.md           phase-by-phase build status and open questions
WINDOWS.md            Windows-specific install steps and command syntax
```

## License

MIT, see `LICENSE`.
