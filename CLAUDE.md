# NFL Model HQ

Bayesian NFL margin prediction system with a Kalshi market comparison layer and a
public website. Built by Dave (HowlsCastle97), originally alongside NPS CS 4323
(Bayesian Methods for Neural Networks); now a personal project. A friend has
recently added an MLB portion that needs review.

Live site: https://howlscastle97.github.io/nfl-model-hq/ (GitHub Pages from
`docs/` on main). The repo is `nfl-model-hq`, so that is the path; the
`nfl-gambling-hq` URL written here until 2026-09-29 was never real and 404s.

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
  Also carries the totals side, which is sums where the margin side is
  differences: `TOTAL_COLS`. Pace and scoring volume (plays, drives, points per
  drive, for and against) ride the same 8 game EWMA as the EPA states, from the
  pace columns `prep_pbp.py` now writes. A team with no history starts on
  `PACE_PRIOR`, structural league numbers rather than zero, so no row ever
  divides by zero drives. Weather is the first input that is unknown at
  prediction time: `weather_columns` gives a covered roof neutral values (70F,
  no wind), uses a real reading outdoors, and otherwise falls back to the median
  of outdoor games played in the same calendar month, taken from completed games
  only. A game day forecast goes in through `build_features(weather_override=...)`,
  keyed by game_id.
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
- `totals.py`: the totals model, a separate engine from the margin one (design
  principle 7). `DeployedTotals` is what ships and is the single definition of the
  published total, the way `rd.DeployedModel` is for margins: ridge on
  `TOTAL_COLS`, no variance recalibration because its sigma came back honest
  (z sd 0.97 and 1.01). `fit_totals` fits it on every completed game with the
  usual season decay. The losing candidates stay in the file because a measured
  loss is worth keeping: `TotalsEnsemble` (the margin architecture on this
  target), `QuantileGBM` (LightGBM quantile regression on a 23 level grid),
  `HeteroLinear` (ridge mean, ridge log sigma), `TotalsTabPFN` (in-context
  inference, no per-week refit), `ConstantModel` as the floor, and
  `market_baseline` so a CRPS number has a scale. `python totals.py` prints the
  walk-forward table; `experiment_totals.py` is the pre-registered comparison.
  Full results under "Totals engine results" below. lightgbm and tabpfn are
  imported inside the two losing classes rather than at the top of the file and
  are deliberately **not** in requirements.txt: the site build never touches them,
  and putting them in the workflow's install would add minutes per run for code
  that only the experiment calls. `pip install lightgbm tabpfn` to rerun it.
- `walkforward.py`: weekly-refit walk-forward harness; `model_factory` hook lets
  any fit/predict_dist model drop in. Season decay weights (half-life 2.0).
  `target` picks the column to predict ("y" for margin, "y_total" for totals) and
  drops rows where it is NaN, so a season in progress grades only what has been
  played; `keep_cols` carries market columns through to the graded frame.
  `evaluate` reports CRPS alongside NLL, since CRPS is what selects the totals
  engine and NLL alone is too easily dominated by one 70 point game.
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
  `rec_records` adds the other half of the tab: not how good the predictions were
  but what a flat one unit bet on every published recommendation would have done,
  by week and by season, with ROI. Four strategies, deliberately paired as two
  unfiltered and two filtered, so a filter can only justify itself by beating the
  column beside it. Every ML pick is the model's outright favourite at that side's
  closing price, value or not, which is the "ML pick" row on every card; Dave acts
  on those, so they are graded. ML value picks are the subset where the model's
  number beat the price. Every spread pick is the model's side against the closing
  line in every game that had one, which is the accuracy table's ATS percentage
  expressed as money. Spread picks the card printed are the subset at
  `SPREAD_REC_P = 0.58`, the threshold the card uses to decide whether to show one.
  Spreads settle at -110 with pushes voided.
  As of 2026-09-29 the pairing already says something: the 58% spread filter earns
  its place (-3.4% against -6.8% unfiltered) and the moneyline value filter does
  not (-6.2% against -3.0% unfiltered). Do not quietly delete either unfiltered
  column; they are the only thing that makes those two statements possible.
  A paragraph under the season table computes, from the All seasons row, what the
  model claimed its picks would do, what they did, and what their prices implied
  (65.0%, 63.8%, 65.8%). Every judgement in it, including whether the model over
  or understated itself and which way the price gap ran, is derived rather than
  typed: an earlier draft hardcoded "honest to within half a point" and the
  ensemble's real gap was 1.3. Moneyline value picks are settled at the
  closing Vegas moneyline from games.csv, not at a Kalshi ask, because Kalshi
  prices have only been logged since 2026: the Vegas price carries the book's vig
  and the Kalshi 7% win fee is absent, so the table is the record of the value
  rule rather than a Kalshi statement, and the page says exactly that.
  Totals appear in five places, all fed by `tot.DeployedTotals`: `ou_rows` on each
  card, which returns two rows for the bold block rather than a line of grey note
  text ("Model total" sits with the other predictions, "O/U pick" with the other
  picks, and the value row keeps the last word), graded on the spread row's own
  0.58 and 0.545 thresholds, an over/under leg per game in the
  Parlay Lab at -110 under the same square-3 flag, `totals_history` as its own
  scorecard on the Track Record tab, and a Bayesian 101 passage on why one part of
  this site is deliberately not Bayesian. The O/U row borrows the spread row's
  vocabulary but not its standing: the totals model has never beaten 52.4% in
  backtest, best season 51.1%, so the This Week copy says in as many words that a
  green tag there is the model's strongest lean and not a measured edge. Do not
  quietly drop that sentence; it is the only thing keeping the shared vocabulary
  honest (principle 3). The fifth place is `totals_reason`, a second half to the
  Reasoning panel. Its arithmetic is exact, unlike the margin half: a linear fit
  is a sum, so `tot.contributions` returns terms that add to the published total
  with nothing left over, and the copy says so instead of carrying the margin
  panel's disclaimer.
  `plain_summary` quotes numbers rather than asserting conclusions. It used to say
  things like "X were simply the better side", which is a claim with the evidence
  deleted. Every clause now carries the figure it rests on, per team rather than
  as a gap (`{stat}_home` / `{stat}_away` out of `build_features`, display only),
  and `season_context` supplies records, points for and against, giveaways and the
  listed starters' own EPA per dropback and interception counts. Three rules hold
  it together and should survive any edit: tense still follows the evidence; a
  clause is dropped entirely when the two sides are within 0.03 EPA rather than
  calling a 0.02 gap "leakier"; and when this season's quarterback play contradicts
  the model's passing lean, the panel says so and names the eight game half-life
  as the reason, which is how a reader can tell a stale read from a wrong one.
  Turnovers are shown and never modelled: an interception is already priced inside
  the EPA columns at what the play cost, so a separate turnover input would count
  it twice. `prep_pbp` carries `giveaways` and `takeaways` per team-game and
  `interceptions` plus `passer` per quarterback-game for exactly this display.
- `weekly_report.py`: grades one week's published picks game by game, walk-forward,
  so it reproduces what the cards said rather than refitting with the answers
  known. Six ledgers, the same unfiltered/filtered pairs as the Track Record plus
  over/unders. `python weekly_report.py` takes the latest week with results.
  The moneyline column prints American odds **and** the model's probability in
  brackets, because an earlier version printed the payout alone in the position
  where the other columns print confidence, and a 12% price reads as a 12% belief.
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
7. Totals (over/unders) use a non-Bayesian engine selected by walk-forward CRPS;
   margin and moneyline models remain Bayesian. The two are separate models; a
   Gaussian copula joins them later for same-game combos.

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
3. Sunday: refresh `games.csv` from nflverse raw GitHub URL, `python prep_pbp.py
   --refresh-latest` and `--qb`, run `python website.py --out docs/index.html`,
   commit and push (Pages redeploys automatically). Then
   `python weekly_report.py` for the week just finished.
   The page header carries three separate facts and they are not interchangeable:
   when the page was built, when `games.csv` was last written (its mtime, which is
   the pull time in the normal flow but becomes checkout time if a build runs
   without refreshing), and how far the results actually run, taken from the data
   itself. The third is the one that cannot lie, which is why it is there. Both
   timestamps are published as instants with `data-stamp` and localised to the
   reader's clock by `localiseStamps`, with Eastern text as the no-script
   fallback, the same pattern the kickoff times use.

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
6. Totals and joint pricing. The first pass shipped on 2026-09-29: ridge on
   `TOTAL_COLS`, chosen by walk-forward CRPS against four other engines, live on
   the cards, in the Parlay Lab and with its own Track Record scorecard. What is
   left. Possession-based Monte Carlo simulation (pace times per-drive scoring from
   EPA matchups) as the future totals and joint margin-total engine; gives key
   number mass, which a Gaussian over a quantity that piles up on 41, 44 and 47
   cannot, and coherent same-game pricing. Gaussian copula with residual
   correlation estimated from history to join the margin and total models. Grade
   O/U against the logged `KXNFLTOTAL` ladder instead of the Vegas line, which is
   now possible: see the note under "Totals engine results". Any new engine is
   still chosen on its own merits by CRPS and is not required to be the margin
   model wearing a different target.
7. Backlog: parse Kalshi spread-market strikes from logged subtitle/floor_strike
   once real KXNFLSPREAD rows accumulate and compute spread edges against real
   prices (currently graded against Vegas line at -110); backtest engine over
   logged prices; fractional Kelly sizing.

## Week by week results

2026 week 4, graded 2026-10-05 on 15 games (ATL at NO still to play), walk-forward:

| ledger | record | ROI |
|---|---|---|
| every ML pick | 11-4 | +14.2% |
| of those, value picks | 4-1 | +50.1% |
| every spread pick | 9-4, 2 pushes | +32.2% |
| of those, the card printed | 5-0 | +90.9% |
| every O/U pick | 4-11 | -49.1% |
| of those, the card printed | 2-3 | -23.6% |

Fifteen bets a ledger, so none of this is a trend. Two things are worth watching
rather than acting on. The spread filter went 5-0, which is the third separate
piece of evidence that `SPREAD_REC_P` earns its place. And the totals model's two
most confident picks were its two biggest misses: GB at TB printed at 72% (model
46.1, actual 31) and KC at LV at 66% (model 42.1, actual 57). Confidence running
backwards on a model with no measured edge is the thing to watch. If that holds for
another two or three weeks, stop printing O/U picks and leave the row as
information, which is a copy change and not a model change.

## The deployed margin model is on probation

Chasing a reader's question about one card on 2026-09-29 turned up a comparison
nobody had run: the deployed deep ensemble against the ridge model it replaced,
walk-forward over every season the Track Record covers. Pooled over 1,407 games
the ensemble is **worse**, NLL +0.0088 [-0.0009, +0.0188] and RMSE 13.107 against
13.027, worse in four seasons of six, and identical where it counts (ATS 48.8%
against 48.9%, straight up 63.8% against 63.9%). Its one distinguishing feature, a
per-game sigma, is what NLL rewards, and NLL does not reward it.

Not grounds to swap, because the interval touches zero and because those seasons
include the ones the ensemble was tuned on. So the decision is pre-registered in
`experiment_margin_engine.py`, to be run when 2026 is complete, with the rule and
the tie-break fixed in advance (a tie swaps to ridge, the same rule already applied
to the totals engine). Read that file before touching the deployed model. It also
lists what a swap would cost, including the honest one: every card's plus-or-minus
becomes the same number, and the Bayesian 101 tab is built around the ensemble.

A related finding worth keeping: the ensemble's attribution for `div_game` averages
-0.16 points, which matches the data (division games run 0.5 points closer to the
home team than others), but its standard deviation is 2.46 with extremes past 5.6,
and the size of the swing correlates +0.52 with `kalman_var`. So it swings hardest
exactly when the filter knows least, which is early season. Its deficit against
ridge is 3.6 times larger on division games than elsewhere. Suggestive, not
established, and it disappears by construction if the swap happens.

## Totals engine results

**What shipped: ridge on `TOTAL_COLS`.** Four engines were tried against it on
walk-forward CRPS under a rule fixed before the numbers were read: an engine ships
only if its held-out CRPS beats ridge with a paired bootstrap interval excluding
zero. None did. The deep ensemble and the boosted quantile trees lost outright; the
heteroscedastic linear and TabPFN could not be separated from ridge. So the
candidate that is a closed form solve and fits in a second won on grounds that were
not accuracy.

Held out 2025, read once (n=272, 4000 resample paired bootstrap on per game CRPS):

| engine | CRPS | NLL | RMSE | O/U vs the line | minus ridge |
|---|---|---|---|---|---|
| closing Vegas total | 7.430 | 3.998 | 13.19 | n/a | -0.032 [-0.248, +0.184] |
| **ridge (shipped)** | **7.463** | **4.002** | **13.24** | **50.7%** | baseline |
| TabPFN v2, full context | 7.418 | 3.992 | 13.22 | 49.3% | -0.045 [-0.119, +0.030] |

Tuning season 2024, where the hyperparameters were chosen and nothing is held out:

| engine | CRPS | NLL | sd | O/U | minus ridge |
|---|---|---|---|---|---|
| ridge | 7.172 | 3.975 | 13.19 | 50.2% | baseline |
| deep ensemble (step 2) | 7.319 | 3.986 | 12.44 | 49.4% | +0.147 [-0.013, +0.303] |
| hetero linear, sigma_lam 4000 | 7.169 | 3.976 | 12.91 | 50.2% | -0.003 [-0.013, +0.006] |
| LightGBM q, 3 leaves 150 trees | 7.234 | 4.159 | 12.01 | 48.3% | +0.062 [-0.026, +0.155] |
| LightGBM q, 3 leaves 400 trees | 7.292 | 4.199 | 11.84 | 46.1% | +0.120 [+0.017, +0.224] |
| LightGBM q, 7 leaves | 7.429 | 4.336 | 11.01 | 47.2% | +0.257 [+0.115, +0.395] |
| LightGBM q, 15 leaves | 7.515 | 4.383 | 10.22 | 46.1% | +0.343 [+0.167, +0.524] |
| TabPFN v2, full context | 7.144 | 3.974 | 12.04 | 49.1% | -0.028 [-0.095, +0.041] |
| TabPFN v2, recent context only | 7.172 | 3.980 | 11.95 | 49.4% | +0.000 [-0.086, +0.089] |

What to take from it, and what not to repeat:

- **Capacity is the enemy on this target.** The boosted trees lose monotonically in
  the number of leaves: 3 leaves +0.06, 7 leaves +0.26, 15 leaves +0.34. The deep
  ensemble lost the same way. 272 games a season of a quantity with a 13 point
  spread does not support a flexible mean function. Do not bring a bigger model to
  this problem without a reason that is not "it worked on margins".
- **The spread really is constant.** `HeteroLinear` let sigma vary with the same
  features and moved CRPS by -0.003 with the interval through zero, at every
  regularisation tried. A windy December game is not measurably more predictable
  than a dome game once the mean model has used the weather.
- **TabPFN is the one that nearly worked.** Nominally best in both seasons, and
  never separable from ridge. Worth revisiting on 2.5 or 3.5 weights, which need a
  Prior Labs account: only v2 downloads anonymously, and `--tabpfn-version` takes
  the rest once that login exists. **CPU timing measured here** (16 threads, 3,900
  row context, 16 games): 173s per week on the default `n_estimators='auto'`, 25s
  at `n_estimators=1`, and the week's CRPS moves 0.006 between them, so one member
  is the default in `TotalsTabPFN`. Mean and the whole quantile grid come back from
  a single forward pass; asking separately doubled the bill. A season of
  walk-forward is 8 to 9 minutes, which is fine for research and too slow for the
  weekly build.
- **The quantile engines were scored on their own shape**, not on a Gaussian fitted
  to them: `crps_from_quantiles` (twice the pinball integral),
  `nll_from_quantiles` (piecewise linear density with exponential tails) and
  `moments_from_quantiles`. Note the last one truncates at the 1% and 99% levels,
  so any sd read off a quantile grid is about 5% low; that is measured in
  `test_totals.py` rather than corrected, because correcting it would mean
  inventing tails.
- **2025 is now spent on the totals question.** It was read once, for the winner
  and the baselines. Anything new here needs 2026 or later.
- **Kalshi totals are live and parseable.** `KXNFLTOTAL` has been logging since
  2026-08-27: `floor_strike` carries a clean half point strike, `yes_sub_title`
  reads "Over X points", and each game has a ladder of strikes, which is a
  market-implied distribution rather than a single number. Half point strikes
  cannot push. Grading O/U against those prices instead of the Vegas line is the
  next real step and needs no new plumbing.

The step 2 baseline table follows, for the record. Measured 2026-09-28,
walk-forward, weekly refit, season decay half-life 2.0, features `TOTAL_COLS`.
CRPS in points, lower better.

| season | engine | CRPS | NLL | RMSE | mean mu | mean sigma | z sd | O/U hit |
|---|---|---|---|---|---|---|---|---|
| 2024 | closing Vegas total | 6.994 | 3.952 | 12.55 | 44.3 | 13.3 fitted | 0.94 | n/a |
| 2024 | constant | 7.288 | 3.996 | 13.11 | 45.0 | 13.67 | 0.96 | 49.8% |
| 2024 | ridge linear | 7.172 | 3.975 | 12.86 | 44.4 | 13.19 | 0.97 | 50.2% |
| 2024 | deep ensemble | 7.319 | 3.986 | 13.12 | 44.2 | 12.44 | 1.04 | 49.4% |
| 2025 | closing Vegas total | 7.430 | 3.998 | 13.19 | 44.9 | 13.3 fitted | 1.00 | n/a |
| 2025 | constant | 7.807 | 4.045 | 13.81 | 45.4 | 13.67 | 1.01 | 46.0% |
| 2025 | ridge linear | 7.463 | 4.002 | 13.24 | 46.0 | 13.19 | 1.01 | 50.7% |
| 2025 | deep ensemble | 7.527 | 4.008 | 13.31 | 45.9 | 12.69 | 1.05 | 48.5% |

Paired bootstrap on per-game CRPS, 4000 resamples, pooled over both seasons
(n=544): linear minus market +0.105 [-0.034, +0.239]; ensemble minus linear
+0.106 [-0.013, +0.223]; linear minus constant -0.230 [-0.391, -0.067];
ensemble minus constant -0.125 [-0.320, +0.068].

What that says, and what to do with it:

- The features carry real information about totals. Linear beats the constant by
  an interval that excludes zero. That is the one clean positive result here.
- The deep ensemble, the engine that wins on margins, does not transfer. It is
  worse than the linear model in both seasons and cannot be separated from the
  constant. Do not assume the margin architecture is the right starting point for
  a new target; on 272 games a season, a 5 by 16 MLP ensemble has more capacity
  than a total will support.
- Nothing beats the closing line, as expected. The market is the scale, not the
  opponent: it was detectably better than the linear model in 2024 and
  indistinguishable in 2025.
- Sigma runs honest for the linear model (z sd 0.97 to 1.01) and 4 to 5% too
  confident for the ensemble (1.04 to 1.05), the same direction the margin model
  needed `RECAL_SCALE` for. `TotalsEnsemble(recal=...)` is where that would go,
  and no number has been baked in yet.
- Every O/U hit rate is below the 52.4% break-even, best case 50.7%. So an
  over/under row on the site is information, not a betting edge, and any green
  value badge on totals would be claiming something no backtest supports. That is
  a copy decision to make deliberately when step 5 lands.

## Testing conventions

Validate end-to-end before delivering: synthetic data matching real schemas for
unit tests (see the mocked Kalshi pages pattern), and reproduce the paper
numbers (2025 walk-forward: linear+kalman+qb NLL ~3.980, RMSE ~12.94) when
touching the model path. Those are the linear model's numbers, reproduced with
`walk_forward(..., lam=rd.LIN_LAM)` and no model factory; the Track Record tab
grades the deployed ensemble and shows different figures by design.
