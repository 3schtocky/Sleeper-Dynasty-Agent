# Dynasty Agent

**Ask your dynasty fantasy football team anything, in plain English.**

```
you> Should I trade Jonah Coleman and my 2028 2nd for a 2027 1st?
  (the full trade breakdown, straight from Python: both sides, market value, win-now, three-year, injury risk)
No, you should not make this trade. The net market value is -221, and the net win-now is -6.1.
                                       qwen3:4b-instruct · 29.7 tok/s · first words 1.0s · tool: evaluate_trade
```

Dynasty Agent connects to your [Sleeper](https://sleeper.com) league, adds NFL stats from [nflverse](https://github.com/nflverse) and dynasty trade values from [FantasyCalc](https://fantasycalc.com), and lets you **talk to it** about your team. A small open-source language model runs on your own computer, so it costs nothing per question and your league never leaves your machine.

You can ask it:

- **Trades:** "Should I trade Jonah Coleman and my 2028 2nd for a 2027 1st?" "Find me a running back to trade for."
- **Your team:** "Am I a contender or should I rebuild?" "Should I sell my first?"
- **Every week:** "Who should I start?" "Who should I pick up and how much should I bid?" "Anyone I should put on taxi?"
- **Injuries:** "What about his injury risk?" "Daniels just got hurt."
- **Why:** "How did you get his win-now value?" "Why does age matter so much for running backs?"
- **Strategy:** "When should a dynasty team switch from contending to rebuilding?"

> **The one rule: the language model never computes a number.** Python does all the math, from public data, in code you can read. The model only decides which analysis to run and puts the result into friendly words, and Python checks every sentence before you see it. See [How the chat stays honest](#how-the-chat-stays-honest).

## Contents

1. [Get chatting](#get-chatting) (about 20 minutes, once)
2. [What to ask](#what-to-ask)
3. [Chat tips](#chat-tips)
4. [What is happening under the hood](#what-is-happening-under-the-hood)
5. [The math, step by step](#the-math-step-by-step)
6. [The machine learning](#the-machine-learning)
7. [How the chat stays honest](#how-the-chat-stays-honest)
8. [Limits you should know about](#limits-you-should-know-about)
9. [Troubleshooting](#troubleshooting) and [Glossary](#glossary)
10. [Appendix: the commands behind the chat](#appendix-the-commands-behind-the-chat)

---

## Get chatting

There are five steps. The first four are one-time setup. After that, you start the chat and ask questions.

### 1. Install the basics

| You need | Why | Check it |
|---|---|---|
| Python 3.12 or newer | The tool is plain Python | `python3 --version` (Windows: `python --version`) |
| Git | To download the project | `git --version` |
| A Sleeper account in a dynasty league | It reads your league through Sleeper's public API. No API key and no password | Your Sleeper username |
| About 5 GB of free disk | 2 GB for NFL stats and the local database, 2.5 GB for the chat model | |
| 8 GB of RAM or more | The 4-billion-parameter model uses about 3 GB | |

You do not need a graphics card. The chat runs on a laptop at about 30 tokens per second on a MacBook Air M4, and about 85 on an RTX 3070.

### 2. Install the chat model

1. Install [Ollama](https://ollama.com/download) and open it. It runs quietly in the background.
2. Download the model (about 2.5 GB, once). In a terminal:

```
ollama pull qwen3:4b-instruct
```

### 3. Install the project

**macOS and Linux**

```
git clone https://github.com/3schtocky/Sleeper-Dynasty-Agent.git
cd Sleeper-Dynasty-Agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

**Windows (PowerShell)**

```
git clone https://github.com/3schtocky/Sleeper-Dynasty-Agent.git
cd Sleeper-Dynasty-Agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

A *virtual environment* (`.venv`) keeps this project's packages separate from the rest of your computer. Activate it in every new terminal window before you use `dynasty-agent`. If PowerShell says "running scripts is disabled", run `Set-ExecutionPolicy -Scope Process Bypass` once in that window. [`WINDOWS.md`](WINDOWS.md) covers installing Git and Python on Windows and the common errors.

### 4. Connect your league and load the data

Nothing about your account is written into the code. `init` finds your league and saves its IDs to a local `.env` file (git-ignored, never committed). Then paste the one-time data load. It downloads a lot (play-by-play stats are about 20 MB per season), so expect 10 to 15 minutes.

```
dynasty-agent init --username <your-sleeper-username>
```

If you are in more than one league, it lists them. Run it again with `--league-id <id>` to choose one.

```
dynasty-agent sync
dynasty-agent ingest-nflverse --season <last completed season>
dynasty-agent ingest-draft-data
dynasty-agent ingest-college --season 2008 --through <this year>
dynasty-agent fit-prospect-model
dynasty-agent backfill-history
dynasty-agent calibrate-matchup-model
dynasty-agent fit-availability
```

Those eight commands fetch your league, a few seasons of NFL history and the data the models are fitted on. You run them once. After that the chat refreshes your league every time it starts, and you can ask it about anything.

**Is your league a different shape?** The project was tuned on one league: 12 teams, 1 QB, full PPR, no TE premium and a 3-round rookie draft. Your scoring and roster rules are read live from Sleeper, so it runs on other shapes, but two things do not adapt on their own:

- Set `FANTASYCALC_NUM_QBS`, `FANTASYCALC_NUM_TEAMS` and `FANTASYCALC_PPR` in `.env` (they are listed in `.env.example`) so market prices match your format.
- The age curves, the QB discount and the receiver bump were calibrated for the league above. `CLAUDE.md` explains why each number is what it is. Read it before you trust them in a superflex or half-PPR league.

### 5. Start the chat

```
dynasty-agent chat
```

It refreshes your league while the model loads (the first reply can take about 20 seconds), then waits for you. Type a question and press Enter. Type `/quit` to leave.

That is the whole workflow. You do not need to remember any other command.

## What to ask

Phrase things the way you would to a friend. It understands names, nicknames ("Matt Stafford"), picks written any way ("my '27 first", "2027 round 1 pick") and follow-ups ("what about his injury risk?"). What it can do:

| Ask about | Examples | What runs |
|---|---|---|
| A trade | "Would you trade my 2027 1st for Trey Benson?" "I'm offered Bijan Robinson for Rashee Rice and my 2027 first. Take it?" | Both sides on win-now, three-year and market value, picks discounted, consolidation and roster fit, injury risk beside the numbers |
| Trade ideas | "Help me find a tight end." "What about running back?" | Realistic offers, scored, flagged when they are lowballs |
| One player | "Tell me about Jaxon Smith-Njigba." | Value, age, injury risk, market price, whether he fits your roster |
| Your team | "Am I a contender or should I rebuild?" "How does my roster look?" | Contend, rebuild or unclear, with its confidence |
| This week | "Who should I start?" "What are my chances this week?" | The lineup with the best win probability against your real opponent |
| Waivers | "Who should I pick up?" "How much should I bid on Trey Benson?" | Targets ranked by how much they raise your best lineup, and a sized bid |
| Draft picks | "Should I sell my first?" "Are my picks worth keeping?" | Buy, hold or sell for each pick, against what that slot has produced |
| Taxi and IR | "Anyone I should put on taxi?" | Which moves free the most roster spots |
| Injuries | "What about his injury risk?" "Daniels just got hurt." | Expected games missed. If you report an injury it asks how many games and uses your answer as a labeled assumption |
| How it works | "How did you get his win-now value?" "Why is his three-year lower?" "What is situation score?" | The calculation with your own numbers, step by step |
| Strategy | "How should I think about trading a 30 year old running back?" | General dynasty advice. It will not state a player's stats from memory |

**It pushes back, and you can push back too.** If you disagree with a trade verdict, say so. The verdict only changes when you give it something new, such as how many games a player will miss. It will not simply agree with you.

## Chat tips

- `/raw` shows exactly what the model was given for the last answer, and any sentence that was left out.
- `/stats off` hides the speed line under each answer (`/stats on` brings it back).
- `/help` lists commands and `/quit` leaves.
- `dynasty-agent chat --no-refresh` skips the start-up refresh for a quick restart.
- **Moves are yours to make.** Sleeper's API is read-only, so the agent can tell you what to do but cannot change your roster.
- **Watch it work** from a second terminal: `ollama ps` shows the loaded model and its memory use.
- **Settings** go in `.env`: `LLM_MODEL` (default `qwen3:4b-instruct`) and `OLLAMA_URL` (default `http://localhost:11434`). `dynasty-agent chat-eval --model <name>` scores another Ollama model on the same questions before you switch to it.

---

## What is happening under the hood

```
 public data                    local database                   you
 -----------                    --------------                   ---
 Sleeper API  ---+                                                 |  plain English
 (roster, rules) |                                                 v  (dynasty-agent chat)
 nflverse     ---+--> ingest --> SQLite + parquet files        +---------------+
 (stats, injuries,       (data/)        |                      | local model   |
  schedules, draft)                     v                      | (Ollama)      |
 FantasyCalc  ---+              +----------------+  chooses    +-------+-------+
 (market prices) |              |  Python math   | <---------- which   |
 sportsdataverse-+              |  (metrics,     |  function           |
 (college stats)                |  valuation,    |                     |
                                |  calibration,  |                     |
                                |  availability) |                     |
                                +-------+--------+                     |
                                        |                              v
                                        v                       a short take, checked
                                 numbers block  ------------>   by Python before shown
                                 (always shown first)
```

**Four layers, each with one job**

1. **Ingest.** Download public data and store it. Sleeper supplies your roster and rules. nflverse supplies weekly stats, injury reports, schedules and betting lines. FantasyCalc supplies dynasty trade values. College stats come from sportsdataverse. Each source is cached, so commands run offline after a `refresh`.
2. **Math.** Plain Python functions turn that data into values, probabilities and recommendations. These are in `src/dynasty_agent/` and covered by tests. They use no machine learning except where noted in [The machine learning](#the-machine-learning).
3. **Chat.** This is how you use it. A language model reads your plain-English question, chooses which analysis fits and writes a short, friendly take. It does not do arithmetic.
4. **Commands.** Every analysis is also a command that prints its working: the inputs, the result and the caveats. The chat runs the same functions, and the [appendix](#appendix-the-commands-behind-the-chat) lists them for scripting and debugging.

**Design rules the whole project follows**

- *Every recommendation shows its inputs.* The goal is an audit trail, not a bare verdict.
- *Unknowns are stated.* An estimate built from public data is labeled as one.
- *Backtest before you believe.* Constants such as the season-blend weight and the win-probability correction were fitted or tested against real outcomes, and the results (including the disappointing ones) are in `PLANNING.md`.
- *Prefer boring, readable code.* No framework, no black boxes. SQLite for storage, pure functions for math.

## The math, step by step

This section explains each calculation in the order the data flows. The code is in `src/dynasty_agent/metrics.py`, `valuation.py`, `blend.py`, `calibration.py`, `availability.py` and friends. `PLANNING.md` holds the build log, including every bug found and fixed.

### 1. Fantasy points from your scoring rules

Points per game come from your league's actual `scoring_settings`, pulled live from Sleeper (for example full PPR, 4-point passing touchdowns and 0.1 per yard). They are not a generic PPR formula. Change your league's rules and every number follows.

### 2. Last season, blended with this season

A player's per-game average should not jump to three noisy weeks the moment a new season starts. Instead, last season counts as a fixed number of "pseudo-games" and this season's real games add on top:

```
blended mean = (K x last season's mean + sum of this season's games) / (K + games played)
```

`K` is 4 for a veteran and 3 for a rookie (whose prior is a projection from draft capital). These weights were chosen by backtest (`calibrate-blend`): for many past players the tool predicted the rest of a season from weeks 1 to N under each candidate `K` and kept the one with the lowest error.

*Example:* a veteran averaged 12.0 last year, then scores 20 and 20 in two games. The blend is (4 x 12 + 40) / (4 + 2) = 14.7, not 20 and not 12.

### 3. Production score and position weights

The blended points per game are multiplied by a position weight that reflects what a point is worth in this league's format:

| Position | Weight | Why |
|---|---|---|
| QB | 0.70 | One starting QB per team and 4-point passing touchdowns make the position nearly replaceable |
| RB | 1.00 | PPR already rewards pass-catching backs in the points themselves |
| WR | 1.05 | Three starting receivers plus a flex make receiver the league's most valuable position |
| TE | 1.00 | No TE premium, so no bump |

### 4. The age curve

Value holds flat through a position's peak, then decays exponentially:

```
age multiplier = 1.0                           if age <= peak age
                 exp(-decay x (age - peak))    if age > peak age
```

| Position | Peak age | Decay per year | Multiplier at 28 | at 31 |
|---|---|---|---|---|
| RB | 25 | 0.32 | 0.38 | 0.15 |
| WR | 28 | 0.15 | 1.00 | 0.64 |
| TE | 30 | 0.15 | 1.00 | 1.00 |
| QB | 32 | 0.08 | 1.00 | 1.00 |

A 28-year-old running back keeps 38% of his value from the age curve alone. That one rule explains most of why dynasty managers sell backs early.

### 5. Situation score

A player's team affects his output. The situation score is the average of three public team numbers, each *percentile-ranked* against all 32 teams (0 to 100):

- the team's quarterback passing EPA (expected points added per dropback);
- its pass rate over expected;
- its sack rate allowed, inverted (a public proxy for offensive line quality).

It maps to a bounded multiplier: `0.85 + 0.30 x (score / 100)`. An average team (score 50) multiplies by 1.00. The best offense gives 1.15 and the worst 0.85, so context nudges value but can never overwhelm production.

### 6. Win-now and three-year value

```
win-now value    = production x age multiplier today        x situation multiplier
three-year value = production x average age multiplier over this year, next year and the year after x situation multiplier
```

*Example:* a 26-year-old receiver averaging 15.0 points per game on a strong offense (situation score 70) has win-now value 15.0 x 1.05 x 1.00 x 1.06 = 16.7. They are kept as two separate numbers and never blended, because a contender and a rebuilder should weigh them differently.

### 7. Contend or rebuild

Each team's best possible lineup (starters plus flex, from your league's own roster slots) is totaled twice, once on win-now value and once on three-year value. Your totals are then *percentile-ranked against the other teams* (your own team is left out so the best roster is not capped at the 96th percentile).

- **Contend:** win-now at or above the 60th percentile and three-year at or above the 40th.
- **Rebuild:** win-now below the 40th percentile and three-year at or above the 55th.
- **Unclear:** anything else. Many rosters genuinely are.

The cut-offs are round numbers chosen up front, not fitted, because a new league has no history to fit them to. The output states its confidence. With no games played, the verdict rests on roster construction alone and says so.

### 8. Draft picks

A pick's market value is anchored to FantasyCalc's current price for that round, then discounted for the wait:

```
pick value = base value x (1 - discount rate) ^ years until that draft
```

With a base of 1000 and the default 20% per year, a pick two drafts away is worth 1000 x 0.8^2 = 640. The discount rate is tunable (`--discount-rate`).

`picks` goes further. It projects your draft slot from reverse standings (blending roster strength toward the real record as games accumulate), looks up what rookies taken at that slot have produced since 2018, and compares FantasyCalc's price for the pick with what FantasyCalc now pays for the rookies that same slot bought last year. A price over 1.15 times that is a sell and under 0.87 times is a buy. Those bands are round and stated.

### 9. Evaluating a trade

For each side, `trade` totals win-now, three-year and market value for the players and picks involved, and reports the net. It also flags:

- *consolidation:* two good players for one great one changes your lineup in a way a raw sum hides;
- *roster fit:* whether you would need to drop someone;
- *warnings:* for example a player offered who is not actually on that roster, which the chat treats as a false premise instead of a trade.

Injury risk prints beside the numbers (see section 15), but it is not subtracted from them.

### 10. Win probability for a matchup

For a given week, each side is a sum of its starters. Each starter has a mean and a *sample variance* of weekly points (divide by n - 1, not n, because a season is a sample of a player's true spread). Then:

1. The mean is scaled by the betting market: this week's Vegas-implied team total divided by that team's own season average. A team expected to score more than usual lifts its players.
2. The injury report changes both mean and spread:

   | Status | Mean x | Variance x | Reasoning |
   |---|---|---|---|
   | Out, IR, Suspended | 0 | 0.1 | Near-certain zero |
   | Doubtful | 0.25 | 3.0 | Mostly out, but a wide outcome if he plays |
   | Questionable | 0.85 | 2.0 | Close to a coin flip between full and limited |
   | Healthy | 1 | 1 | |

   Widening the variance for "Questionable" is deliberate: scaling only the mean understates exactly the uncertainty the tag signals.
3. Team means and variances add (assuming players score independently), and the win probability is the normal CDF of the margin:

```
P(A beats B) = Phi( (mean A - mean B) / sqrt(variance A + variance B) )
```

This is the same idea that converts a Vegas point spread into a win probability. Its main simplification, independent players, is false in practice: a quarterback and his top receiver rise and fall together.

### 11. Calibrating that probability

The step-10 number is a heuristic, so it was *tested*. The tool replays 2,127 real NFL games from 2018 to 2025, projects both sides the way it would have projected them that week using only information available beforehand, records its raw probability and compares it with who won.

It was badly overconfident. When it said about 95%, the favorite won 75% of 413 such games. When it said about 85%, 65% of 451.

The fix is *Platt scaling* with a slope only:

```
calibrated = sigmoid( a x logit(raw) ),     fitted a = 0.36
```

`logit(p) = ln(p / (1 - p))` turns a probability into log-odds and `sigmoid` turns it back. A slope of 0.36 pulls every probability most of the way toward 50%: a raw 95% becomes 74% and a raw 85% becomes 65%, matching what really happened. The Brier score (average squared error of the probabilities, lower is better) improved from 0.2434 to 0.2223.

Two details matter:

- **No intercept, on purpose.** A free intercept made the answer depend on which team was labeled "A" and flipped a real underdog (46.7%) into a favorite (50.8%) while barely improving the score. A slope-only correction changes how *sure* the model is, never *who* it favors.
- **Home field cannot leak in.** Which real team plays "A" in the fit is a seeded coin flip, so the heuristic's missing home-field term does not contaminate the correction.

Every win probability the tool prints shows the calibrated number with the raw one beside it. Refit after each season.

### 12. Choosing a lineup

The optimizer evaluates every valid lineup your roster supports (respecting your own `roster_positions`) and picks the one with the highest *win probability* against your actual opponent, not the highest points. They usually agree. When they do not, the flex slot is trading a little expected score for a higher ceiling (if you are the underdog) or a lower spread (if you are the favorite). The highest-points lineup prints alongside for comparison.

### 13. The season simulation

`simulate-season` gives every remaining real matchup one calibrated win probability, flips a weighted coin for each, plays out the rest of the regular season 10,000 times and counts how often each team lands in a playoff seed (wins first, points scored as the tiebreak). It assumes every team fields its best healthy lineup, so every team's odds are slightly flattered.

### 14. FAAB bids

Free agents are ranked by how much each would raise your *best possible lineup*, so a backup QB behind your starter in a 1-QB league ranks as depth only, however good he is. A bid is your remaining budget paced over the weeks left, scaled by the player's lineup gain relative to the biggest upgrade anyone on the wire offers.

### 15. Injury availability

This is the one place the data contradicted the obvious plan, and the tool says so.

*The question:* how many games does a quarterback, running back, receiver or tight end miss?

*What was checked first:* a missed week is an absent stat row, not a zero-point row. Of 2,732 "Out" designations, 1 had a stat line that week. Misses are called injuries only when the injury report backs them (18% of misses are benchings or suspensions and are left out).

*What the data found:*

- **Injury history barely predicts next season.** Season-to-season correlation is r = 0.06 over 1,400 pairs. In a backtest that predicted each season using only earlier ones, a single league-wide average scored an error of 0.1556 and adding position, age and the player's own history scored 0.1560. They tie. Long-run risk therefore rests on the league average (about 1.3 to 1.5 games per 17), and the tool says so.
- **Recent misses and the current report status predict the next few games well.** A table of expected missed games, keyed by injury misses in the last three team games and the report status, cut held-out error by 23% over the next 4 games. A player who missed his last three with an injury misses 2.5 of the next 4. A healthy player misses 0.14.

*What the tool does with it:* it prints each player's risk (a label and expected games missed per 17) **beside** the numbers and does **not** subtract it from any value, verdict or ranking. History would not justify a discount. You decide what to do with the information, and in chat you can argue with the model about it.

### 16. Ranking rookies

Rookie values come from a *ridge regression* (linear regression with a small penalty that keeps about a dozen features stable on about 470 players) fitted to how drafted rookies actually scored, in your league's format, over their first three NFL seasons. It is scored by leaving one draft class out at a time (*leave-one-class-out cross-validation*), so every accuracy figure is on a class the model never saw.

The finding that shaped the board: **once draft capital is known, college production adds nothing.** Draft capital plus position scored a held-out error of 2.68 points per game. Adding dominator rating, breakout age, athleticism, conference and team strength scored 2.70. NFL teams already price college production into where they draft. So the *post-draft* board ranks on draft capital. The *pre-draft* board, before the order exists, uses college production and team context and is labeled weak on every run.

## The machine learning

Most of this project is statistics, not machine learning. These are the learned or model-driven parts, and what each one is for:

| Piece | Technique | Fitted on |
|---|---|---|
| Rookie values | Ridge regression, leave-one-class-out validation | Rookies drafted 2018 and later |
| Season blend weight `K` | Grid search on backtest error | Past veteran and rookie seasons |
| Win-probability correction | Platt scaling (logistic regression, one parameter) | 2,127 real NFL games |
| Injury availability | Empirical Bayes shrinkage plus a conditional-mean table, leak-free backtest | Skill-position seasons since 2017 |
| The chat model | A 4-billion-parameter open-source language model, `qwen3:4b-instruct` | Pretrained by Alibaba (Qwen team) |

Everything that is fitted is fitted on **past data only** and tested on data it did not see. Several tests exist to prove that: they remove a leak guard and confirm a test fails.

### The chat model

The model is [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), run through Ollama on your machine. It was chosen from a bake-off of eight small models. The faster ones swapped the sides of a trade or spent their tokens "thinking". It does three jobs, each with its own tightly written prompt:

1. **Route.** Given a question, call exactly one of six tools (`set_lineup`, `evaluate_trade`, `my_team`, `waiver_targets`, `pick_advice`, `taxi_plan`), or reply that the question is off topic. This runs at temperature 0 so the same question always routes the same way.
2. **Classify.** One word: `fantasy`, `smalltalk` or `other`.
3. **Take.** Given the facts a tool returned, write two to four friendly sentences. Or, with no tool, chat about fantasy strategy in at most five sentences.

A language model is the wrong tool for arithmetic and for remembering facts about specific players. So it does neither. Every number comes from a Python function, and the model is told it knows nothing reliable about any specific player.

### Measuring it

The chat is scored two ways, and both gate any change to the model or its prompts:

- `dynasty-agent chat-eval`: 39 real questions with the tool each should reach and what its arguments must say. A trade is scored on **direction**, not just on reaching `evaluate_trade`, because reversing the sides is the worst failure this tool can have.
- `dynasty-agent chat-eval --conversations --hard`: 57 steps of multi-turn conversations. Each step is checked for the right tool, grounded numbers, plain text, length and the absence of forbidden advice.

Current results for `qwen3:4b-instruct` with the Python guards below: **39 of 39** routed correctly and **57 of 57** conversation steps passed. A small model does miss sometimes, and the project's approach is to fix each miss with a deterministic Python guard, then add the miss as a failing eval step first.

### Fine-tuning

A fine-tuning recipe lives in [`finetune/`](finetune/README.md): QLoRA on a single consumer GPU, trained on a *synthetic* league so nothing private can end up in a published model. It is a work in progress. The default model stays `qwen3:4b-instruct` until a fine-tuned model passes every gate in that document.

## How the chat stays honest

A small model will sometimes invent a number, flip a verdict between turns or agree with whatever the user says. The design assumes it will, and puts a Python check in front of each failure:

| What can go wrong | The guard |
|---|---|
| The model quotes a number that is not in the tool output | Each sentence is checked before it is shown. A sentence with an ungrounded number is dropped (`/raw` shows what was dropped) |
| It swaps the sides of a trade | Sides are checked against the rosters. A player you own cannot be one you receive, and the evaluation moves them and says so |
| It puts a free agent on the wrong side ("my pick for Trey Benson") | The wording settles it: a player named only after "for", with your own pick offered before it, is what you receive |
| It refuses a plain trade question | If two named players are on different rosters, Python builds the trade itself |
| It says "make the trade" about an idea that is only a lowball offer | The trade finder's replies drop any sentence that tells you to make a trade |
| It writes a wall of text | Free talk stops after five sentences |
| It gives a verdict, then reverses it on the next turn | Python owns the stance. The verdict comes from the numbers, and pushback gets a Python-built reply that only changes if you supply a new fact |
| It states a player's injury risk from memory | A sentence that ties injury words to a named player or a number is left out. The only place a risk is stated is the Python risk check |
| An ambiguous name ("Kenneth Walker") | It asks which one |
| You mention an injury ("Daniels just got hurt") | It asks how many games, then reruns the numbers with your answer labeled as an unverified assumption |

Questions about how a number was built (`explain`), trade ideas, single-player checks, injury questions and pushback are recognized by Python directly, not by the model. Offering the model a seventh tool dropped its routing accuracy from 39 of 39 to 36 of 39 in testing, so it routes only the original six.

## Limits you should know about

- **Built for one league shape.** See step 4 of [Get chatting](#get-chatting).
- **Players are treated as independent** when estimating win probability, which they are not.
- **The calibration is fitted on NFL team games** and applied to fantasy lineups. Read it as right in direction and rough in size.
- **Injury data has no body part or timeline.** Sleeper gives a status and nothing else. IR, PUP and NA have no weekly history, so they are treated as a player who has already missed his last three games, and every output that does this says so.
- **Pre-draft rookie rankings are weak.** Only college production exists before the draft, and it adds nothing once draft capital is known.
- **FantasyCalc is a market, not the truth.** The tool shows where its own model disagrees with the market. It does not claim the market is wrong.
- **The model is small.** It can phrase an answer awkwardly. The numbers above each answer are always the real ones.
- **Sleeper's API is read-only.** The agent advises and you make every move in the Sleeper app.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `dynasty-agent: command not found` | Activate the virtual environment (`source .venv/bin/activate` or `.\.venv\Scripts\Activate.ps1`) |
| `No roster found for this user` | Run `dynasty-agent init --username <you>`, then `dynasty-agent refresh` |
| `Ollama isn't running` | Open the Ollama app (or run `ollama serve`) and try again |
| `The model qwen3:4b-instruct isn't installed` | `ollama pull qwen3:4b-instruct` |
| Your league was renewed and the data looks old | Sleeper gives a renewed dynasty league a new ID each season. `sync` says so and prints the `init` command |
| First answer from the chat is slow | The model is loading into memory (about 18 seconds on a laptop). The stats line says "loading model..." |
| Windows: scripts are disabled | `Set-ExecutionPolicy -Scope Process Bypass` in that window |
| `schedule` fails on Windows | It uses macOS launchd. See `WINDOWS.md` for Task Scheduler |
| A win probability shows only one number | The calibration has not been fitted yet. Run `backfill-history` and `calibrate-matchup-model` |
| Tests fail after you change a number | Intended. Many tests pin the math. Update the test in the same change and say why |

If something else breaks, run the command again with the full error text in hand and open an issue.

## Glossary

- **Dynasty:** a fantasy league where you keep your whole roster year after year, including rookies and draft picks.
- **PPR:** points per reception. Full PPR gives 1.0 per catch.
- **1QB:** one starting quarterback per team. Quarterbacks are then easy to replace, so they are worth less than in superflex.
- **FAAB:** free-agent acquisition budget. You bid fake dollars on waiver players.
- **Taxi squad:** extra roster spots (rookies only in this league) that do not count against your bench.
- **Win-now value / three-year value:** the project's two value numbers for a player. See section 6.
- **Market value:** FantasyCalc's dynasty trade value, in points that are not dollars.
- **Percentile rank:** where a value sits among all the others, 0 to 100.
- **EPA:** expected points added per play, a standard measure of how much a play helped.
- **Calibration:** checking that "70% likely" really happens about 70% of the time, and correcting the number if it does not.
- **Brier score:** the average squared error of probability forecasts. Lower is better; 0.25 is what always saying 50% scores.
- **Backtest:** replaying the past using only what was known at the time, to see how a method would have done.
- **Leak:** accidentally letting a prediction see information from its own future. Several tests exist to catch this.
- **Grounded:** every number in a sentence appears in the tool's output.
- **QLoRA:** a way to fine-tune a large model on a small GPU by training small add-on matrices on top of a compressed copy of it.

## Appendix: the commands behind the chat

You never need these to use the project, because the chat runs the same functions when you ask. They are here for scripting, for debugging a number and for people who prefer a terminal. Each command prints its working: the inputs, the result and the caveats. `dynasty-agent --help` and `dynasty-agent <command> --help` list every option.

| Command | What it does |
|---|---|
| `init --username <name>` | Finds your league and writes `.env` |
| `sync` | Pulls your league, rosters, picks, drafts and FantasyCalc values |
| `refresh` | Brings everything current, by time of year. The chat runs it when it starts |
| `roster` | Your players with age, team, market value and 30-day trend |
| `valuate` | Win-now and three-year value per player, plus contend or rebuild |
| `trade` | Evaluates a trade: both sides, picks discounted, consolidation flagged |
| `optimize-lineup --week N` | The lineup that maximizes win probability against your real opponent |
| `predict-matchup --week N` | Win probability between any two rosters |
| `faab --player "Name"` | A sized FAAB bid for one waiver target |
| `digest --week N` | The whole week in one command: lineup, wind flags, bench options and waiver targets |
| `picks` | Buy, hold or sell on your rookie picks (`--all` for every team) |
| `taxi` | Taxi and IR moves that free roster spots |
| `prospect-board` | A ranked rookie board (`--mode post-draft` or `--mode pre-draft`) |
| `simulate-season` | Playoff odds for every team from 10,000 simulated seasons |
| `ingest-nflverse`, `ingest-draft-data`, `ingest-college` | Download stats, draft and combine data, and college production |
| `fit-prospect-model`, `calibrate-blend`, `calibrate-matchup-model`, `fit-availability` | Fit or test the models. Refit after each season |
| `backfill-history` | Loads past seasons' stats and injury reports |
| `schedule` | Runs `refresh` daily (macOS launchd; see `WINDOWS.md` for Task Scheduler) |
| `chat`, `chat-eval` | The chat, and its evaluation |

A trade from the command line, written any way you like (players by name, picks as `2027-1`, `"2027 1st"` or `"2027 1.05"`):

```
dynasty-agent trade --send "Rashee Rice" --receive-pick 2027-1 --receive-pick 2027-3 --discount-rate 0.20
```

`trade` reports three numbers per side and deliberately never adds them together: **win-now** (points per game for this year's title), **three-year** (the same with age decline over three seasons) and **market value** (what FantasyCalc says the player or pick sells for). A pick cannot help you win this week, so it appears only in market value.

Everything is cached in `data/` (a local SQLite database plus downloaded stat files). It is git-ignored and safe to delete: `refresh` rebuilds it.

## Project layout

```
src/dynasty_agent/   the CLI and every module: sleeper, nflverse, market, metrics, valuation, blend,
                     calibration, availability, simulate, weekly, picks, prospect_model and the chat:
                     llm, tools, chat, grounding, formatters, explain, risk_talk, trade_finder
migrations/          the SQLite schema, one file per migration, applied automatically
tests/               offline unit tests for the math, the tools and the chat
benchmarks/          the model bake-off harness
finetune/            the fine-tuning recipe (synthetic league, dataset builder, training, probes)
CLAUDE.md            environment, league settings, strategic reasoning and working rules
PLANNING.md          the build log: every decision, bug found and live check, phase by phase
WINDOWS.md           Windows install steps and command syntax
```

Contributions are welcome. Read `CLAUDE.md` first for the working rules: tests for every metric, no black boxes, state what is unknown.

## License

MIT, see `LICENSE`. The default chat model, Qwen3-4B-Instruct-2507, is released by the Qwen team under Apache-2.0.
