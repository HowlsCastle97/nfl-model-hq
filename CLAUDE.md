# NFL Model HQ

Bayesian NFL margin prediction system with a Kalshi market comparison layer and a
public website. Built by Dave (HowlsCastle97), originally alongside NPS CS 4323
(Bayesian Methods for Neural Networks); now a personal project. A friend has
recently added an MLB portion that needs review.

Live site: https://howlscastle97.github.io/nfl-gambling-hq/ (GitHub Pages from
`docs/` on main).

## Architecture (data flows top to bottom)

- `features.py`: loads nflverse `games.csv` (franchise codes mapped: OAK->LV,
  SD->LAC, STL->LA), builds leakage-safe features chronologically: EWMA point
  differential, EPA aggregates from `team_game_stats.csv`, QB familiarity (share
  of team's last 16 starts by today's listed starter), rest, division, indoor.
  Rows with NaN `result` are future games: features are emitted, state updates
  are skipped.
- `prep_pbp.py`: one-time nflverse play-by-play download and per-team-game EPA
  aggregation (cached; ~300 MB, gitignored).
- `kalman.py`: joint Kalman filter over 32 team ratings plus home-field
  advantage. Season transition: revert 0.7 toward mean, inflate variance.
  Hyperparameters tuned by one-step predictive log-likelihood on seasons <=2023
  (empirical Bayes). Tuned params live in `rundown.py` as `KALMAN_PARAMS`.
- `models.py`: closed-form weighted ridge linear Gaussian model (lam=0 is MLE,
  lam>0 is MAP), `prob_margin_over` for strike probabilities.
- `ensemble.py`: deep ensemble of 5 heteroscedastic MLPs (mu and sigma heads),
  trained on beta-NLL (beta=0.5) to avoid the sigma-swallows-gradient pathology.
  Tuned: hidden=16, weight_decay=1e-2, epochs=200. `predict_split` returns
  (mu, aleatoric, epistemic); mixture variance = E[sigma^2] + Var[mu].
- `walkforward.py`: weekly-refit walk-forward harness; `model_factory` hook lets
  any fit/predict_dist model drop in. Season decay weights (half-life 2.0).
- `rundown.py`: deployment constants (RECAL_SCALE=1.039 variance recalibration,
  LIN_LAM=100), fits deployment models on all completed games, joins latest
  Kalshi prices from sqlite, computes fee-adjusted edges
  (fee = 0.07*p*(1-p)), Kalshi team aliases (LA<->LAR, etc.).
- `kalshi_logger.py`: polls Kalshi public API
  (https://external-api.kalshi.com/trade-api/v2), series KXNFLGAME, KXNFLSPREAD,
  KXNFLTOTAL (all three verified live 2026-09-07: SPREAD and TOTAL both carry a
  numeric `floor_strike` plus a subtitle naming the team or the total, and
  markets open weeks before kickoff), writes timestamped snapshots to
  `kalshi_prices.db` (gitignored; irreplaceable local data, never commit).
  Handles dollar-string and integer-cent price fields; schema self-migrates.
- `healthcheck.py`: watchdog over every layer that fails silently. Checks
  delivered data rather than configuration, which is the distinction that
  matters: all three failures of 2026-09-07 had a job that existed, was enabled,
  and was not delivering. Checks price history age per series (a Kalshi ticker
  rename would drop one series while the others look healthy), published feed
  age, site build age from the footer stamp, GitHub Actions results (parse
  failures and run failures told apart, grouped by workflow id so a pre-fix
  failure cannot stick forever), and the logger task's own exit code. Exit code
  is the number of failing checks. Run it by hand any time: `python
  healthcheck.py`. Scheduled every 30 minutes by `install_healthcheck_task.ps1`,
  Interactive and unelevated on purpose, because the alert is a window and a
  service account has no desktop to put one on. Alerts fire on the transition
  into failure and then at most once every 12 hours, so a known problem cannot
  train you to dismiss them unread. **Not a toast**: the WinRT toast call
  returned success and displayed nothing, since toasts need a registered
  AppUserModelID and can be suppressed by Focus Assist invisibly to the caller.
  An alert channel that fails silently is the exact bug this file exists to
  catch, so it uses a message box, which either appears or does not.
- `website.py`: builds the entire public site as one self-contained HTML file
  (`--out docs/index.html`). Tabs: This Week (cards), Parlay Lab (risk bands),
  Track Record (walk-forward 2021-2025, per-game), Bayesian 101.
- `run_v2.py` / `run_deliverable2.py`: reproduction scripts for the model
  comparison tables (2024 validation, 2025 test).

## Non-negotiable design principles

1. No leakage, ever: every feature and rating uses only games played before the
   game in question. Walk-forward evaluation only; never quote accuracy on data
   that influenced training or tuning (that is why Track Record starts at 2021).
2. Betting lines are never model inputs (only evaluation and market comparison).
3. Honesty in presentation: the site labels the Safe parlay band as near-zero
   edge, excludes ML legs where model exceeds market by >0.15 probability from
   the credible bands (they go to Moonshot with a warning), states the 52.4% ATS
   break-even, and shows calibration receipts. Keep this tone in any new copy.
4. Big model-market disagreements are treated as the model missing news, not
   free money ("square-3 rule"). Value verdicts require edge after fees. The
   rule is measured in units of the model's own predictive sd, not probability:
   `square3_gap` in `website.py` is the difference of probits, since
   p = Phi(mu/sigma) on both sides, thresholded at `SQUARE3_SIGMAS = 0.40`.
   Probability was the wrong yardstick because a fixed 0.15 gap is 0.39 sigma at
   an even price but 1.81 sigma at 0.90, so it barely applied to lopsided games.
   0.40 reproduces the old rule at a coin-flip price. The moneyline gap is taken
   against the Kalshi price, the spread gap against the Vegas line (a fair line
   implies a 50% cover, so that gap is just probit(p_cover)). Both the card
   caveat and the Parlay Lab wild-leg flag use this one function, and the page JS
   mirrors it so the caution survives a live price refresh.
5. Everything on the site is from the home team's perspective; gambler
   translations accompany technical numbers ("SEA -2", ML odds).
6. Dave's writing preference: no em dashes or hyphens as sentence punctuation in
   any written deliverable; use commas, colons, semicolons.

## Environment quirks (Windows, Anaconda base, Python 3.13)

- Anaconda + pip torch OpenMP clash: `os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"`
  must be set before torch imports (present at top of rundown.py and run_v2.py;
  preserve it).
- pip installs on this machine may need care; nflverse pbp parquet requires
  pyarrow.
- The site build takes minutes (walk-forward history plus ensemble training each
  run); caching history tables is an accepted future optimization.

## Weekly operating ritual (keep working)

1. `kalshi_logger.py` runs every 10 minutes from a Windows Task Scheduler entry
   named "Kalshi NFL price logger", which calls `run_kalshi_logger.cmd` (locates
   the repo via `%~dp0`, appends to `logs/`, gitignored). The same run then
   republishes the public price feed with `publish_prices.py --push`, because
   **this machine is the primary publisher, not GitHub Actions**: the "every 5
   minutes" workflow is best-effort cron and was measured delivering roughly
   every 30 minutes, with a 57 minute gap on 2026-09-07. The workflow stays
   enabled as a backstop for when this machine is off. Both force-push a single
   orphan commit to the `prices` branch, so whichever ran last wins and no
   history accumulates. The local push uses git plumbing (hash-object, mktree,
   commit-tree) rather than the workflow's orphan checkout, because a checkout
   racing a real edit on the development machine could lose work. Register or change it
   with `install_logger_task.ps1`, **from an elevated PowerShell**: without
   elevation S4U fails with "Access is denied" and it falls back to a task that
   only runs while you are signed in. Price history is unrecoverable, so nothing
   here should ever be left to a hand-run terminal: that is exactly how 11 days
   went missing in August 2026, silently, because nothing reports a dead logger.
   Check `Get-ScheduledTaskInfo -TaskName "Kalshi NFL price logger"` (a
   LastTaskResult of 0 is success) before suspecting the script.
2. Nothing needs watching by hand: `healthcheck.py` runs every 30 minutes and
   raises a message box on failure. To see the current state at any time, run it directly or
   read `logs/health.json`. If it ever reports "Weekly site rebuild last run
   FAILED", the fix is usually to press Run workflow on it in the GitHub UI
   (Actions tab, "Weekly site rebuild", Run workflow), since `workflow_dispatch`
   is declared for exactly that.
3. Sunday: refresh `games.csv` from nflverse raw GitHub URL, run
   `python website.py --out docs/index.html`, commit and push (Pages
   redeploys automatically).

## Current task list

1. Review the MLB addition (friend-contributed). Check it against the design
   principles above, especially leakage safety, team-code handling, walk-forward
   honesty, and that it does not break any NFL paths or the site build. MLB has
   different dynamics (162 games, heavy favorites rare, starting pitchers matter
   like QBs but more); flag silently-copied NFL assumptions that do not
   transfer.
2. Live prices ("tier 2" architecture): split slow/fast layers. Slow: weekly
   model rebuild as today. Fast: a scheduled GitHub Actions job (every 10-15
   min; free on public repos) that runs a lightweight Kalshi snapshot (no
   torch, no model) and publishes a small `prices.json` alongside the Pages
   site. The page's JS fetches prices.json on load (same-origin, no CORS
   risk) and re-renders market bars, gap text, and verdict badges client-side
   from model probabilities embedded in the page at build time. Parlays may
   stay build-time with a staleness note. The runner must never commit
   `kalshi_prices.db`; the local machine keeps the full history. Before
   building the JSON layer, test whether Kalshi's API allows browser CORS
   requests directly (fetch from a real browser); if yes, client-side direct
   fetch is even simpler and the Actions job is unnecessary for prices.
3. Automation via GitHub Actions: a scheduled workflow that takes a fresh Kalshi
   snapshot (runner-local, ephemeral; do NOT commit the db), rebuilds the site,
   and publishes to Pages. The local machine remains the keeper of price
   history; the Action only needs current prices for the build. Watch torch
   install time on runners (consider CPU-only wheel and pip caching). If task 2
   is built, this full rebuild only needs to run weekly.
4. This Week tab: edge filter buttons (like the Parlay Lab band buttons) to
   filter game cards by verdict tier: High value / Small edge / No value /
   No price.
5. Backlog: parse Kalshi spread-market strikes from logged subtitle/floor_strike
   once real KXNFLSPREAD rows accumulate and compute spread edges against real
   prices (currently graded against Vegas line at -110); totals model (target =
   total points, same pipeline; enables over/unders); backtest engine over
   logged prices; fractional Kelly sizing.

## Testing conventions

Validate end-to-end before delivering: synthetic data matching real schemas for
unit tests (see the mocked Kalshi pages pattern), and reproduce the paper
numbers (2025 walk-forward: linear+kalman+qb NLL ~3.980, RMSE ~12.94) when
touching the model path.
