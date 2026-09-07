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
  KXNFLTOTAL (TOTAL ticker unverified), writes timestamped snapshots to
  `kalshi_prices.db` (gitignored; irreplaceable local data, never commit).
  Handles dollar-string and integer-cent price fields; schema self-migrates.
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
   free money ("square-3 rule"). Value verdicts require edge after fees.
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

1. `kalshi_logger.py` runs continuously or via Task Scheduler `--once` every 10
   minutes (price history is unrecoverable; protect this).
2. Sunday: refresh `games.csv` from nflverse raw GitHub URL, run
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
