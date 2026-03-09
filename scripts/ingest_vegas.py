"""Convert a Vegas odds CSV into season parquet files for the feature builder.

Expected CSV columns (names flexible via --home-col etc.):
  - date (or game_date): YYYY-MM-DD
  - home_team, away_team: Retrosheet 3-letter codes (NYA, BOS, ...) or we try to normalize
  - open_ml_home, open_ml_away: American money line for home/away (e.g. -170, 150)
  - close_ml_home, close_ml_away (optional): closing lines for line-movement feature

Output: data/processed/vegas/odds_YYYY.parquet per season with
  game_date, home_team, away_team, vegas_implied_home_win [, vegas_line_movement].
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from mlb_predict.external.vegas import money_line_to_implied_prob


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Vegas odds CSV into parquet by season")
    ap.add_argument("--input-csv", type=Path, required=True, help="Path to CSV with money lines")
    ap.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--date-col", default="date", help="CSV column for game date")
    ap.add_argument("--home-col", default="home_team", help="CSV column for home team")
    ap.add_argument("--away-col", default="away_team", help="CSV column for away team")
    ap.add_argument("--open-ml-home", default="open_ml_home", help="CSV column for opening ML home")
    ap.add_argument("--open-ml-away", default="open_ml_away", help="CSV column for opening ML away")
    ap.add_argument(
        "--close-ml-home",
        default="close_ml_home",
        help="CSV column for closing ML home (optional)",
    )
    ap.add_argument(
        "--close-ml-away",
        default="close_ml_away",
        help="CSV column for closing ML away (optional)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    df[args.date_col] = pd.to_datetime(df[args.date_col])
    df["game_date"] = df[args.date_col].dt.date
    df["home_team"] = df[args.home_col].astype(str).str.strip().str.upper()
    df["away_team"] = df[args.away_col].astype(str).str.strip().str.upper()

    df["vegas_implied_home_win"] = money_line_to_implied_prob(
        pd.to_numeric(df[args.open_ml_home], errors="coerce")
    )
    if args.close_ml_home in df.columns and args.close_ml_away in df.columns:
        close_home = money_line_to_implied_prob(
            pd.to_numeric(df[args.close_ml_home], errors="coerce")
        )
        df["vegas_line_movement"] = close_home - df["vegas_implied_home_win"]
    else:
        df["vegas_line_movement"] = 0.0

    out_dir = args.processed_dir / "vegas"
    out_dir.mkdir(parents=True, exist_ok=True)
    for season, g in df.groupby(df["game_date"].apply(lambda d: d.year)):
        out = g[
            ["game_date", "home_team", "away_team", "vegas_implied_home_win", "vegas_line_movement"]
        ].drop_duplicates(subset=["game_date", "home_team", "away_team"])
        path = out_dir / f"odds_{season}.parquet"
        out.to_parquet(path, index=False)
        print(f"  {season}: {len(out)} rows → {path}")
    print(f"Done. Output directory: {out_dir}")


if __name__ == "__main__":
    main()
