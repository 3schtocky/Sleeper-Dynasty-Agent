# Windows install guide

Dynasty Agent runs the same on Windows as it does on macOS and Linux, same Python, same `pip`, same commands. Nothing in the code shells out to anything OS-specific (checked directly, not assumed: no `subprocess` calls, no hardcoded `/`-style paths, no POSIX-only system calls anywhere in `src/`). The only real differences are how you set up Python and the venv and how your terminal wants line breaks written. This guide covers both, in copy-pasteable PowerShell.

If you hit something this guide doesn't cover, the "How the numbers work" and command reference in the main [`README.md`](README.md) apply exactly as written, those aren't OS-specific either.

## Requirements

- Windows 10 or 11.
- **PowerShell**, not the old `cmd.exe`. Every command below is PowerShell syntax. Windows 11 opens PowerShell by default when you right-click in a folder and choose "Open in Terminal"; on Windows 10, search the Start menu for "PowerShell" (Windows Terminal, if you have it installed, is the nicer experience but not required).
- A Sleeper account that's a member of at least one dynasty league. No API key needed, Sleeper's read API is public.

## 1. Install Git, if you don't have it

Check first:

```powershell
git --version
```

If that errors, install it:

```powershell
winget install --id Git.Git -e
```

Close and reopen PowerShell afterward so it picks up the new PATH entry.

## 2. Install Python

You need Python 3.12 or newer. Check first:

```powershell
py --version
```

If that errors or shows something older than 3.12:

```powershell
winget install --id Python.Python.3.12 -e
```

Close and reopen PowerShell afterward so it picks up the new PATH entry.

## 3. Clone and set up the project

```powershell
git clone https://github.com/3schtocky/Sleeper-Dynasty-Agent.git
cd Sleeper-Dynasty-Agent
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

This creates a `.venv` folder and installs every dependency from `pyproject.toml`. Activate the venv (the `Activate.ps1` line) in each new PowerShell window before running `dynasty-agent`. If it fails with "running scripts is disabled", see Troubleshooting below.

## 4. Point it at your own league

```powershell
dynasty-agent init --username <your-sleeper-username>
```

This looks up your Sleeper user ID, finds your current-season dynasty leagues, and writes a `.env` file in the project folder (already in `.gitignore`, it never gets committed) with your username, user ID, league ID, and draft ID. If you're in more than one league this season, it lists them and asks you to re-run with `--league-id <id>` to pick one.

Prefer to do it by hand? Copy `.env.example` to `.env`:

```powershell
Copy-Item .env.example .env
```

then open `.env` in Notepad (or any editor) and fill in the four values. Your league ID and draft ID are both in your league's Sleeper URL and in the response from `https://api.sleeper.app/v1/user/<your-username>`.

## 5. Run it

```powershell
dynasty-agent refresh                        # everything current: league, values, this season's stats
dynasty-agent roster                         # sanity check: is this your team?
dynasty-agent valuate                        # win-now/three-year value + the verdict
```

First run on a new machine: `sync`, `ingest-nflverse --season <last completed season>`, `ingest-draft-data`, `ingest-college --season 2008 --through <current season>`, then `fit-prospect-model`; see the main README.

**Daily refresh.** `dynasty-agent schedule` uses macOS's launchd and tells you so on Windows. The Windows equivalent is one Task Scheduler command, run once from the project folder (written from Microsoft's documented `schtasks` syntax, not yet run on a Windows machine for this project):

```powershell
schtasks /Create /SC DAILY /ST 06:00 /TN "dynasty-agent refresh" /TR "cmd /c cd /d $PWD && .venv\Scripts\dynasty-agent.exe refresh >> data\logs\refresh.log 2>&1"
```

Remove it with `schtasks /Delete /TN "dynasty-agent refresh" /F`.

Evaluate a trade. PowerShell line continuation is a backtick (`` ` ``) at the very end of the line, not the backslash (`\`) you'd use in Bash, and it has to be the *last* character on the line, no trailing space after it or PowerShell won't recognize it:

```powershell
dynasty-agent trade `
  --send "Rashee Rice" `
  --receive-pick 2027-1 --receive-pick 2027-3 `
  --discount-rate 0.20
```

Or skip the continuation entirely and put it on one line, which sidesteps the trailing-space gotcha altogether:

```powershell
dynasty-agent trade --send "Rashee Rice" --receive-pick 2027-1 --receive-pick 2027-3 --discount-rate 0.20
```

Predict a matchup, same backtick rule:

```powershell
dynasty-agent predict-matchup --week 1 `
  --team-a "Jalen Hurts" --team-a "Derrick Henry" --team-a "Puka Nacua" `
  --team-b "Jayden Daniels" --team-b "James Cook" --team-b "Drake London"
```

Everything reruns fresh against whatever's cached in the `data\` folder (git-ignored, local SQLite plus downloaded nflverse files). Nothing here needs a server or an account beyond your own Sleeper login. What each command actually does, and how the underlying math works, is documented once in the main [`README.md`](README.md), not repeated here, it applies exactly as written regardless of OS.

**Calibrated win probabilities and the season simulation.** The raw win probability is badly overconfident (it was tested on 2,127 real NFL games), so fit the correction once, and again after each season. Same commands as on macOS, nothing OS-specific:

```powershell
dynasty-agent backfill-history               # past seasons' stats and injury reports, a few seconds a season
dynasty-agent calibrate-matchup-model        # tests the heuristic and stores the correction
dynasty-agent simulate-season                # playoff odds for every team
```

## 6. Optional: talk to the agent (preview)

Install Ollama from [ollama.com/download](https://ollama.com/download) (the Windows installer), open it, then in PowerShell:

```powershell
ollama pull qwen3:4b-instruct
dynasty-agent chat
```

Everything else about `chat` is the same as on macOS; see "Talk to the agent" in `README.md`. To watch requests arrive, Ollama's log is at `$env:LOCALAPPDATA\Ollama\server.log`.

## Troubleshooting

**"running scripts is disabled on this system"** when running `Activate.ps1`, or anything else PowerShell-script-based. PowerShell's default execution policy blocks unsigned scripts. Check your current policy and loosen it for your own account only (safer than a machine-wide change):

```powershell
Get-ExecutionPolicy
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**Windows Defender (or another antivirus) flags or quarantines `python.exe` or a downloaded Python installer.** This is a known false-positive pattern with newly-published binaries and freshly downloaded executables in general, not specific to this project. If it happens, add an exclusion for Python's install directory (typically `%LOCALAPPDATA%\Programs\Python`) and re-run the install.

**A path-length error deep inside `.venv`, something mentioning `MAX_PATH` or a path over 260 characters.** Rare on a current Windows 10/11 install (long paths are enabled by default in recent versions) but possible if yours is older or was upgraded from a much older install, or if you cloned into a deeply nested folder. Two ways out: clone somewhere shallower (like `C:\dev\` instead of a folder six levels deep in your Documents), or enable long path support once, as an administrator:

```powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

then restart.

**`ingest-nflverse` or `predict-matchup` fails to reach `github.com` or hangs on the first DuckDB query.** Every nflverse file, the schedules file included, downloads through `httpx` into `data/nflverse/` before DuckDB reads it locally, and on a locked-down corporate network or VPN that blocks outbound connections to unfamiliar hosts, that download can get blocked. Check whether a proxy or firewall is intercepting `github.com` and `objects.githubusercontent.com`; this isn't a Windows-specific issue, it'd behave the same way on any OS behind the same network policy.

**Everything works but output looks like `Ã¢â‚¬â€` instead of normal punctuation.** That's a console codepage issue, an old `cmd.exe` window not set to UTF-8. Switch to PowerShell (this guide assumes PowerShell throughout, not `cmd.exe`) or, if you must use `cmd.exe`, run `chcp 65001` first to switch its codepage to UTF-8.

## What's actually different from macOS/Linux, concretely

For the curious, not just "trust me": the whole Python application was audited for OS-specific assumptions while writing this guide, grepped for `subprocess`, hardcoded POSIX paths, `os.path.join` (this project uses `pathlib.Path` throughout instead, which produces the correct separator for whatever OS it runs on automatically), `chmod`/symlink calls, and shell scripts. None exist. Two small things *were* found and fixed as part of adding this guide: two places wrote or read a text file without an explicit `encoding="utf-8"`, which works by accident on most systems but isn't guaranteed on Windows, where the default text encoding often isn't UTF-8. Both now pin the encoding explicitly. Beyond that, the only real differences are the ones this guide walks through: how you set up Python and the venv, and how your shell wants multi-line commands written.
