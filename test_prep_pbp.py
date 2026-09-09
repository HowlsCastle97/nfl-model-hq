"""The incremental EPA refresh: does it splice one season without touching the rest?

team_game_stats.csv holds sixteen years of aggregates that cost a 300 MB download
to produce. The weekly job rewrites part of it every Wednesday, so the property
that matters is not "the new rows are right" but "everything else is untouched,
to the byte". Reading the file through pandas and writing it back quietly drops a
digit on every row, which is both a data change and 7740 lines of noise in a
commit, so the splice is done as text and this proves it stayed that way.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

import prep_pbp

fail = []
def ok(c, m):
    if not c:
        fail.append(m)

HEADER = "game_id,team,off_epa_pass,off_epa_rush,cpoe,def_epa_pass,def_epa_rush"
# Deliberately 17 significant digits: this is what a pandas round trip mangles.
OLD = [
    "2024_01_AAA_BBB,AAA,0.09235475691101869,-0.6148186734009703,0.7908821851015091,-0.23547062664414098,-0.10200879649588994",
    "2024_01_AAA_BBB,BBB,-0.23547062664414098,-0.10200879649588994,-6.194479708318357,0.09235475691101869,-0.6148186734009703",
    "2025_01_CCC_DDD,CCC,0.11111111111111111,0.22222222222222221,0.33333333333333331,0.44444444444444442,0.55555555555555558",
]


def write_cache(path, lines):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + "\n")
        for line in lines:
            f.write(line + "\n")


def fake_season(rows):
    """Stand in for the parquet download with a frame of known rows."""
    def factory(season, pbp_dir="pbp_cache", wp_filter=None, force=False):
        if not rows:
            return None
        return pd.DataFrame(rows)
    return factory


with tempfile.TemporaryDirectory() as d:
    cache = os.path.join(d, "team_game_stats.csv")

    # 1. Refreshing 2025 must leave every 2024 line byte for byte.
    write_cache(cache, OLD)
    before = open(cache, encoding="utf-8").read().split("\n")
    prep_pbp.season_team_game_stats = fake_season([
        {"game_id": "2025_01_CCC_DDD", "team": "CCC", "off_epa_pass": 0.5,
         "off_epa_rush": 0.1, "cpoe": 1.0, "def_epa_pass": 0.2, "def_epa_rush": 0.3},
        {"game_id": "2025_02_EEE_FFF", "team": "EEE", "off_epa_pass": 0.6,
         "off_epa_rush": 0.2, "cpoe": 2.0, "def_epa_pass": 0.3, "def_epa_rush": 0.4},
    ])
    n = prep_pbp.refresh(2025, cache_path=cache, pbp_dir=os.path.join(d, "pbp"))
    after = open(cache, encoding="utf-8").read().split("\n")
    ok(n == 2, f"expected 2 refreshed rows, got {n}")
    ok(after[0] == HEADER, "header changed")
    kept = [l for l in after if l.startswith("2024_")]
    ok(kept == OLD[:2], "2024 rows were not preserved byte for byte")
    ok(all("0.09235475691101869" in l for l in kept[:1]),
       "precision was lost on an untouched row")
    ok(len([l for l in after if l.startswith("2025_")]) == 2,
       "the old 2025 row was not replaced")
    ok("2025_02_EEE_FFF,EEE" in "\n".join(after), "new game missing")
    print("2024 rows preserved exactly:", kept == OLD[:2])

    # 2. Running it again with the same data must produce the same file.
    once = open(cache, encoding="utf-8").read()
    prep_pbp.refresh(2025, cache_path=cache, pbp_dir=os.path.join(d, "pbp"))
    twice = open(cache, encoding="utf-8").read()
    ok(once == twice, "refresh is not idempotent")
    print("idempotent on a second run:", once == twice)

    # 3. A season nflverse has not published yet is a no-op, not a wipe.
    prep_pbp.season_team_game_stats = fake_season([])
    n = prep_pbp.refresh(2027, cache_path=cache, pbp_dir=os.path.join(d, "pbp"))
    ok(n == 0, "an unpublished season should refresh nothing")
    ok(open(cache, encoding="utf-8").read() == twice,
       "an unpublished season must leave the cache untouched")
    print("unpublished season leaves the file alone: True")

    # 4. Column order follows the existing header, not the frame's.
    write_cache(cache, OLD)
    prep_pbp.season_team_game_stats = fake_season([
        {"def_epa_rush": 0.3, "team": "CCC", "cpoe": 1.0, "game_id": "2025_01_CCC_DDD",
         "off_epa_rush": 0.1, "def_epa_pass": 0.2, "off_epa_pass": 0.5},
    ])
    prep_pbp.refresh(2025, cache_path=cache, pbp_dir=os.path.join(d, "pbp"))
    row = [l for l in open(cache, encoding="utf-8").read().split("\n")
           if l.startswith("2025_")][0]
    ok(row == "2025_01_CCC_DDD,CCC,0.5,0.1,1.0,0.2,0.3",
       f"columns were written out of header order: {row}")
    print("column order follows the header:", row)

print("\n" + ("FAIL:\n - " + "\n - ".join(fail) if fail else
              "PASS: the refresh splices one season and leaves the rest alone"))
sys.exit(1 if fail else 0)
