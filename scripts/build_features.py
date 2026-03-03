"""Build the v2 feature matrix for one or more seasons.

Loads all gamelogs once (for cross-season Elo and rolling stats), then builds
per-season feature matrices that include Elo, multi-window rolling stats,
prior-year API pitcher stats, streak, rest days, and park factors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from winprob.features.builder import (
    build_feature_matrix,
    load_all_gamelogs,
    load_or_build_park_factors,
    _load_api_pitcher_map,
)
from winprob.statcast.fangraphs import load_fg_team_map


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="*", type=int, default=[])
    ap.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Root of the processed data directory",
    )
    args = ap.parse_args()

    seasons = args.seasons or list(range(2000, 2026))
    out_dir = args.processed_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    pitcher_stats_dir = args.processed_dir / "pitcher_stats"
    fg_dir = args.processed_dir / "fangraphs"
    statcast_cache_dir = args.processed_dir / "statcast_player"
    vegas_dir = args.processed_dir / "vegas"
    weather_dir = args.processed_dir / "weather"

    print("Loading all gamelogs for cross-season Elo and rolling stats…")
    gamelogs_all = load_all_gamelogs(args.processed_dir)
    print(
        f"  {len(gamelogs_all):,} game rows across {gamelogs_all['date'].dt.year.nunique()} seasons"
    )

    print("Computing park factors from all available gamelogs…")
    park_factors = load_or_build_park_factors(args.processed_dir)
    print(f"  {len(park_factors)} parks indexed")

    results = []
    for s in seasons:
        retro_path = args.processed_dir / "retrosheet" / f"gamelogs_{s}.parquet"
        cw_path = args.processed_dir / "crosswalk" / f"game_id_map_{s}.parquet"

        if not retro_path.exists():
            print(f"  {s}: gamelogs not found — skipping")
            results.append({"season": s, "status": "skipped", "n_games": 0})
            continue
        if not cw_path.exists():
            print(f"  {s}: crosswalk not found — skipping")
            results.append({"season": s, "status": "skipped", "n_games": 0})
            continue

        try:
            gl_season = pd.read_parquet(retro_path)
            cw = pd.read_parquet(cw_path)

            # Prior-season MLB API pitcher stats (season N-1 → prior for season N)
            prior_api_map = _load_api_pitcher_map(pitcher_stats_dir, s - 1)

            # Prior-season FanGraphs advanced team metrics
            fg_map_prior = load_fg_team_map(fg_dir, s - 1)
            if fg_map_prior:
                print(f"  {s}: {len(fg_map_prior)} FG team priors from {s - 1}")

            df = build_feature_matrix(
                season=s,
                gamelogs_season=gl_season,
                gamelogs_all=gamelogs_all,
                crosswalk=cw,
                park_factors=park_factors,
                prior_api_map=prior_api_map,
                fg_home_map=fg_map_prior,
                fg_away_map=fg_map_prior,
                statcast_cache_dir=statcast_cache_dir,
                vegas_dir=vegas_dir,
                weather_dir=weather_dir,
            )
            out_path = out_dir / f"features_{s}.parquet"
            df.to_parquet(out_path, index=False)
            n_outcome = int(df["home_win"].notna().sum())
            print(f"  {s}: {len(df)} games, {n_outcome} with outcome → {out_path}")
            results.append(
                {"season": s, "status": "ok", "n_games": len(df), "n_with_outcome": n_outcome}
            )
        except Exception as exc:
            import traceback

            print(f"  {s}: FAILED — {exc}")
            traceback.print_exc()
            results.append({"season": s, "status": "failed", "error": str(exc)})

    summary_path = out_dir / "build_features_summary.json"
    pd.DataFrame(results).to_json(summary_path, orient="records", indent=2)
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
