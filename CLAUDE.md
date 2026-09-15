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
  are skipped. Opt-in, not in `V3_COLS`: a per-quarterback rating
  (`qb_lookup`, from `qb_game_stats.csv` via `prep_pbp.py --qb`), EPA per
  dropback following the player across teams, keyed on `passer_id` because
  `passer_player_id` is empty on every scramble. Shrunk toward an
  empirical-Bayes prior of -0.076, what a quarterback actually produces over
  his first 150 dropbacks (league average is +0.055). Tested by
  `experiment_qb.py` on 2026-09-14 and **not shipped**: it improved the linear
  model on selection and held-out seasons, but on the deployed ensemble the
  held-out difference was -0.0002 [-0.0131, +0.0123]. The likeliest reading is
  that the ensemble already extracts quarterback quality from team EPA, CPOE and
  `qb_fam_diff` together. Kept as the foundation for injury work, where "the
  listed starter is out, rate his backup" is exactly the case a whole-season
  average dilutes.
- `prep_pbp.py`: nflverse play-by-play download and per-team-game EPA
  aggregation (parquet cache ~300 MB, gitignored). Two modes. A full build walks
  every season and short-circuits on `team_game_stats.csv`, which is right once
  and wrong forever after: that short-circuit is why EPA sat at 2025 week 18
  while `games.csv` kept moving. `--refresh-latest` recomputes only the season in
  progress and splices it in, and is what the weekly workflow runs. The splice is
  done as **text**, not through a DataFrame: `read_csv` then `to_csv` drops a
  digit on every untouched row (0.09235475691101869 comes back as
  0.0923547569110186), which is both a silent data change and 7740 lines of noise
  in a weekly commit. Idempotent: re-running with no new games leaves the file
  byte identical. A full build's `last_season` defaults to the current calendar
  year (it was a hardcoded 2025); unpublished seasons 404 and are skipped. Note
  nflverse does revise history: a full rebuild on 2026-09-14 changed 476 rows of
  2020 EPA by up to 0.0043, and every pinned number still reproduced to four
  decimals, so check that rather than assume it after any full rebuild. Postseason still carries no EPA rows, so team EPA coasts
  through January on regular season values; pre-existing, and changing it is a
  model change needing walk-forward, not a staleness fix.
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
  LIN_LAM=100). `DeployedModel` is the single definition of the distribution
  the site publishes: the tuned ensemble plus RECAL_SCALE, via `predict_dist`.
  Use it for anything that needs published mu and sigma. Do not construct
  `DeepEnsemble()` directly for that: its defaults are not the tuned ones
  (hidden 32, 300 epochs) and its own `predict_dist` omits RECAL_SCALE, so it is
  3.9% more confident than the page. Fits deployment models on all completed games, joins latest
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
  Track Record, Bayesian 101. Track Record is walk-forward from
  `TRACK_FIRST_SEASON = 2021` through the latest season with a completed game,
  so a new season joins on its own the week its first result lands; seasons with
  nothing played are skipped, and only completed games are scored. A season that
  still has unplayed games is tagged "(live)". It is the only fully honest row
  on the tab, since every earlier season existed while the model was built.
  The tab grades `rd.DeployedModel`, the same ensemble that makes the This Week
  picks. It used to grade the ridge linear model, a different model from the one
  readers act on. So its numbers are NOT the pinned linear reproduction numbers
  below and should not be expected to match them.
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
5. Injury impact. Measured 2026-09-14 against the deployed ensemble's own
   2021-2025 walk-forward misses: each non-QB regular starter listed Out or
   Doubtful on the final report moves the team -0.50 points against the model's
   prediction (95% CI -1.04 to +0.03), monotone across 0, 1, 2 and 3+ starters
   out. A crude unweighted count, so suggestive rather than proven, and the
   first real unmodelled signal found. Data is free and backtestable: nflverse
   `injuries_{season}.parquet` (official reports from 2012, keyed by `gsis_id`)
   and `snap_counts_{season}.parquet`. Starter status must come from games
   strictly before the report week, because an injured player has no snap row
   in the week he misses. Two constraints: the final report lands Friday and
   inactives 90 minutes before kickoff, so a Tuesday or Wednesday rebuild cannot
   see them; and beat reporter or insider scraping is not a model input, since
   it has no archive aligned to prediction time and cannot be walk-forward
   tested. PFF grades are a paid licence and must not be scraped.
   First attempt, `experiment_injury.py` on 2026-09-14, **did not ship**.
   `prep_injuries.py` builds the rows (report designation, snap importance over
   the last eight games strictly before the report week, whether he played;
   crosswalk `gsis_id` to `pfr_id` via nflverse `players.parquet`, 100%). Absence
   by designation, 2012-2023: Out 1.00, Doubtful 0.99, Questionable 0.44. Six
   position-group weights fitted by OLS on the ensemble's 2021-2023 residuals
   were noise: every interval straddled zero and OL and LB came out with the
   wrong sign. Held out 2024-2025 on the deployed ensemble: +0.0038
   [-0.0074, +0.0151], RMSE 13.02 to 13.10, slightly worse. The design fault was
   six free parameters on 815 games; the remedy is shrinkage (one pooled weight,
   or group weights shrunk toward a common one). But 2024-2025 is now spent on
   this question, and the motivating count used 2021-2025, so any revised design
   must be pre-registered and confirmed prospectively on 2026, not re-tested on
   the same held-out seasons.
6. Backlog: parse Kalshi spread-market strikes from logged subtitle/floor_strike
   once real KXNFLSPREAD rows accumulate and compute spread edges against real
   prices (currently graded against Vegas line at -110); totals model (target =
   total points, same pipeline; enables over/unders); backtest engine over
   logged prices; fractional Kelly sizing.

## Testing conventions

Validate end-to-end before delivering: synthetic data matching real schemas for
unit tests (see the mocked Kalshi pages pattern), and reproduce the paper
numbers (2025 walk-forward: linear+kalman+qb NLL ~3.980, RMSE ~12.94) when
touching the model path. Those are the linear model's numbers, reproduced with
`walk_forward(..., lam=rd.LIN_LAM)` and no model factory; the Track Record tab
grades the deployed ensemble and shows different figures by design.
