# DATA_SCHEMA.md

## MLB Prediction System — Data Contracts

This document defines on-disk schemas and invariants for all persisted datasets.

---

# 1. Conventions

## 1.1 Storage formats

Unless otherwise stated, each dataset is written in **both**:

- **Parquet** (authoritative, typed)
- **CSV** (human inspection)

Parquet files MUST be produced with stable column ordering and deterministic row ordering when feasible.

## 1.2 Paths

All paths are relative to repository root.

- Raw: `data/raw/**`
- Processed: `data/processed/**`

## 1.3 Identifiers

- `game_pk` (MLB Stats API) is the canonical game identifier wherever available.
- Crosswalk tables map Retrosheet game keys → MLB `game_pk`.

## 1.4 Hashing / provenance

Where a `*.checksum.json` file exists, it MUST include:

- Row counts
- SHA256 of output artifacts
- Source selection fields (when applicable)
- Any configuration that affects outputs

---

# 2. Raw Data Layout

## 2.1 MLB Stats API cache

Base: `data/raw/mlb_api/`

Layout:

- `data/raw/mlb_api/<endpoint>/<cache_key>.json`
- `data/raw/mlb_api/metadata.jsonl` (append-only)

### 2.1.1 `metadata.jsonl` schema

Each line is a JSON object:

| Field | Type | Required | Notes |
|---|---:|---:|---|
| `ts_unix` | float | yes | Unix timestamp |
| `url` | string | yes | Request URL |
| `params` | object | yes | Query params |
| `cache_key` | string | yes | SHA256 over endpoint+params |
| `endpoint` | string | yes | Endpoint name |
| `status` | int | yes | HTTP status |

Invariants:
- `metadata.jsonl` is append-only.
- cached payload filenames are deterministic from `(endpoint, params)`.

## 2.2 Retrosheet raw game logs

Base: `data/raw/retrosheet/gamelogs/`

- `GL<season>.TXT` (raw text)

---

# 3. Processed Data Schemas

## 3.1 Teams

Path:
- `data/processed/teams/teams_<season>.parquet`

Schema:

| Column | Type | Required | Notes |
|---|---|---:|---|
| `season` | int32 | yes | season requested |
| `mlb_team_id` | int32 | yes | Stats API team id |
| `abbrev` | string | yes | e.g., LAD |
| `name` | string | yes | team name |

Invariants:
- `mlb_team_id` unique per season.

## 3.2 Schedule

Paths:
- `data/processed/schedule/games_<season>.parquet`
- `data/processed/schedule/games_<season>.csv`
- `data/processed/schedule/games_<season>.checksum.json`

Schema:

| Column | Type | Required | Notes |
|---|---|---:|---|
| `game_pk` | int64 | yes | canonical game id |
| `season` | int32 | yes | season |
| `game_date_utc` | string | yes | ISO 8601 |
| `game_date_local` | string | no | ISO 8601, derived via venue tz |
| `home_mlb_id` | int32 | yes | team id |
| `away_mlb_id` | int32 | yes | team id |
| `home_abbrev` | string | yes | from teams endpoint |
| `away_abbrev` | string | yes | from teams endpoint |
| `venue_id` | int32 | no | venue id |
| `local_timezone` | string | no | IANA tz |
| `double_header` | string | no | Stats API field |
| `game_number` | int32? | no | DH number when present |
| `status` | string | no | e.g., Scheduled |
| `game_type` | string | yes | `R` = regular season, `S` = spring training |
| `home_score` | int32? | no | Final home team score (populated for completed games) |
| `away_score` | int32? | no | Final away team score (populated for completed games) |

Invariants:
- Unique `game_pk`.
- Dataset includes both `gameType=R` (regular season) and `gameType=S` (spring training) by default. Use `--no-preseason` to exclude spring training.

Checksum schema (minimum):

| Field | Type | Required |
|---|---|---:|
| `season` | int | yes |
| `row_count` | int | yes |
| `game_types` | list[str] | yes |
| `parquet_sha256` | string | yes |
| `csv_sha256` | string | yes |
| `raw_payloads_sha256` | string\|null | yes |
| `raw_file_count` | int | yes |
| `max_response_mb` | int | yes |
| `max_split_depth` | int | yes |
| `mlbapi_config` | object | yes |

## 3.3 Retrosheet game logs (GL)

Paths:
- `data/processed/retrosheet/gamelogs_<season>.parquet`
- `data/processed/retrosheet/gamelogs_<season>.csv`
- `data/processed/retrosheet/gamelogs_<season>.checksum.json`

Schema:
- Columns follow the Retrosheet GL format (see code for full list).
- The following normalized columns MUST exist and be parseable:

| Column | Type | Required |
|---|---|---:|
| `date` | date | yes |
| `game_num` | int32? | yes |
| `visiting_team` | string | yes |
| `home_team` | string | yes |
| `visiting_score` | int32? | no |
| `home_score` | int32? | no |
| `visiting_starting_pitcher_id` | string | no |
| `home_starting_pitcher_id` | string | no |

Checksum schema (minimum):

| Field | Type | Required |
|---|---|---:|
| `season` | int | yes |
| `row_count` | int | yes |
| `raw_sha256` | string | yes |
| `parquet_sha256` | string | yes |
| `csv_sha256` | string | yes |
| `raw_path` | string | yes |
| `source_used` | string | yes |
| `url_used` | string\|null | yes |
| `fallback_reason` | string\|null | yes |

## 3.4 Crosswalk (Retrosheet → MLB)

Paths:
- `data/processed/crosswalk/game_id_map_<season>.parquet`
- `data/processed/crosswalk/unresolved_<season>.parquet`
- `data/processed/crosswalk/crosswalk_coverage_report.parquet`
- `data/processed/crosswalk/crosswalk_coverage_report.csv`

Schema (`game_id_map_<season>.parquet`):

| Column | Type | Required |
|---|---|---:|
| `date` | date | yes |
| `home_mlb_id` | int32 | yes |
| `away_mlb_id` | int32 | yes |
| `home_retro` | string | yes |
| `away_retro` | string | yes |
| `dh_game_num` | int32? | no |
| `status` | string | yes | matched/missing/ambiguous |
| `mlb_game_pk` | int64? | no |
| `match_confidence` | float | yes |
| `notes` | string | yes |

Coverage invariants:
- Minimum required: **99.0% matched**.
- Any season below threshold MUST be flagged in `crosswalk_seasons_below_threshold.csv`.

## 3.5 Pitcher stats

Paths:

- `data/processed/pitcher_stats/pitchers_<season>.parquet`
- `data/processed/pitcher_stats/ingest_pitcher_stats_summary.json`

Schema:

| Column | Type | Notes |
|---|---|---|
| `player_id` | int64 | MLB Stats API player ID |
| `player_name` | string | Full name |
| `season` | int64 | Season |
| `era` | float64 | Earned Run Average |
| `k9` | float64 | Strikeouts per 9 innings |
| `bb9` | float64 | Walks per 9 innings |
| `fip_raw` | float64 | Raw FIP (before park adjustment) |
| `whip` | float64 | Walks + Hits per Inning Pitched |
| `ip` | float64 | Innings pitched |
| `games_started` | int64 | Games started |

Invariants:

- One row per pitcher-season; only starters with ≥1 GS are included.

## 3.6 FanGraphs team metrics

Paths:

- `data/processed/fangraphs/fangraphs_<season>.parquet`
- `data/processed/fangraphs/summary.json`

Schema:

| Column | Type | Notes |
|---|---|---|
| `team_fg` | string | FanGraphs team abbreviation |
| `season` | int64 | Season |
| `bat_woba` | float64 | Team weighted on-base average |
| `bat_iso` | float64 | Isolated power |
| `bat_babip` | float64 | Batting average on balls in play |
| `bat_hard_pct` | float64 | Hard Hit % |
| `bat_barrel_pct` | float64 | Barrel % |
| `bat_xwoba` | float64 | Expected wOBA |
| `pit_fip` | float64 | Fielding Independent Pitching |
| `pit_xfip` | float64 | Expected FIP |
| `pit_k_pct` | float64 | Team strikeout % |
| `pit_bb_pct` | float64 | Team walk % |
| `pit_hr_fb` | float64 | Home run / fly ball % |
| `pit_whip` | float64 | Team WHIP |

Invariants:

- One row per team-season; available from 2002 onward.

## 3.7 Features (119-feature matrix)

Paths:

- `data/processed/features/features_<season>.parquet`
- `data/processed/features/features_2026.parquet` (pre-season; `home_win = NaN`)
- `data/processed/features/features_spring_<season>.parquet` (spring training features; `home_win` populated from schedule scores)
- `data/processed/features/build_features_summary.json`

Schema:

| Column | Type | Notes |
|---|---|---|
| `game_pk` | int64 | Canonical game identifier |
| `is_spring` | float64 | 1.0 for spring training, 0.0 for regular season |
| `date` | object (`datetime.date`) | Game date (local) |
| `season` | int64 | Season |
| `game_type` | string | `R` = regular season, `S` = spring training |
| `home_mlb_id`, `away_mlb_id` | int64 | MLB team IDs |
| `home_retro`, `away_retro` | string | Retrosheet team codes |
| `home_win` | float64 | 1.0 / 0.0 / NaN (NaN for 2026 pre-season) |
| `home_elo`, `away_elo`, `elo_diff` | float64 | Elo ratings and differential |
| `home_win_pct_{15,30,60}` | float64 | Rolling win % |
| `away_win_pct_{15,30,60}` | float64 | Rolling win % |
| `home_run_diff_{15,30,60}` | float64 | Rolling run differential |
| `away_run_diff_{15,30,60}` | float64 | Rolling run differential |
| `home_pythag_{15,30,60}` | float64 | Rolling Pythagorean expectation |
| `away_pythag_{15,30,60}` | float64 | Rolling Pythagorean expectation |
| `home_win_pct_ewm`, `away_win_pct_ewm` | float64 | EWMA win % (span=20) |
| `home_run_diff_ewm`, `away_run_diff_ewm` | float64 | EWMA run differential |
| `home_pythag_ewm`, `away_pythag_ewm` | float64 | EWMA Pythagorean |
| `home_win_pct_home_only` | float64 | Win % in home games only (home team) |
| `home_pythag_home_only` | float64 | Pythagorean in home games (home team) |
| `away_win_pct_away_only` | float64 | Win % in road games only (away team) |
| `away_pythag_away_only` | float64 | Pythagorean in road games (away team) |
| `home_streak`, `away_streak` | float64 | Win/loss streak (+/−) |
| `home_rest_days`, `away_rest_days` | float64 | Days since last game (capped at 10) |
| `home_sp_era`, `away_sp_era` | float64 | Starter ERA (prior season) |
| `home_sp_k9`, `away_sp_k9` | float64 | Starter K/9 (prior season) |
| `home_sp_bb9`, `away_sp_bb9` | float64 | Starter BB/9 (prior season) |
| `home_bat_woba`, `away_bat_woba` | float64 | Team wOBA (prior season) |
| `home_bat_barrel_pct`, `away_bat_barrel_pct` | float64 | Team Barrel % |
| `home_bat_hard_pct`, `away_bat_hard_pct` | float64 | Team Hard Hit % |
| `home_pit_fip`, `away_pit_fip` | float64 | Team FIP |
| `home_pit_xfip`, `away_pit_xfip` | float64 | Team xFIP |
| `home_pit_k_pct`, `away_pit_k_pct` | float64 | Team K% |
| `pythag_diff_30` | float64 | `home_pythag_30 − away_pythag_30` |
| `pythag_diff_ewm` | float64 | `home_pythag_ewm − away_pythag_ewm` |
| `home_away_split_diff` | float64 | Home-only vs road-only win% split differential |
| `sp_era_diff` | float64 | `away_sp_era − home_sp_era` |
| `woba_diff` | float64 | `home_bat_woba − away_bat_woba` |
| `fip_diff` | float64 | `away_pit_fip − home_pit_fip` |
| `park_run_factor` | float64 | Park run factor (median over history) |
| `season_progress` | float64 | 0.0 (opener) → 1.0 (final day) |
| `feature_hash` | string | SHA256 of numeric feature values for this row |

Invariants:

- Total columns: ~126 (119 model features + identifiers + `home_win` + `feature_hash`)
- `date` column dtype is always `datetime.date` (never plain string)
- `game_type` is `R` for regular season, `S` for spring training. The `is_spring` binary feature is derived from `game_type`.
- 2026 rows have `home_win = NaN`; all 119 feature columns are populated from 2025 end-of-season team state
- Spring training games (`game_type=S`) use the same prior-season features; model predictions carry a caveat

## 3.8 Prediction snapshots

Path template:

- `data/processed/predictions/season=YYYY/snapshots/run_ts=<iso>_<model>.parquet`

Schema:

| Column | Type | Notes |
|---|---|---|
| `game_pk` | int64 | Canonical game identifier |
| `home_team` | string | Retrosheet home team code |
| `away_team` | string | Retrosheet away team code |
| `predicted_home_win_prob` | float64 | Model output probability |
| `run_ts_utc` | string | ISO 8601 UTC timestamp of the run |
| `model_version` | string | e.g. `xgboost_v3_train2025` |
| `schedule_hash` | string | SHA256 of the schedule Parquet |
| `feature_hash` | string | SHA256 of the feature Parquet |
| `lineup_param_hash` | string | Placeholder (reserved for lineup model) |
| `starter_param_hash` | string | Placeholder (reserved for pitcher model) |
| `git_commit` | string | Git HEAD SHA at run time |
| `tag` | string\|null | Optional human label |

Immutability:

- Snapshot files MUST never be overwritten.

## 3.9 Drift artifacts

Per-season metrics:

- `data/processed/drift/run_metrics_<season>.parquet`

Global (deduplicated by season + run_ts_utc):

- `data/processed/drift/global_run_metrics.parquet`

Schema (both files share the same columns):

| Column | Type | Notes |
|---|---|---|
| `run_ts_utc` | string | ISO 8601 UTC timestamp |
| `model_version` | string | e.g. `xgboost_v3_train2025` |
| `season` | int64 | Season evaluated |
| `n_games` | int64 | Games in diff |
| `mean_abs_delta` | float64 | Mean \|p_new − p_old\| |
| `p95_abs_delta` | float64 | 95th percentile \|delta\| |
| `max_abs_delta` | float64 | Maximum \|delta\| |
| `pct_gt_0p01` | float64 | % of games with \|delta\| > 0.01 |
| `pct_gt_0p02` | float64 | % of games with \|delta\| > 0.02 |
| `pct_gt_0p05` | float64 | % of games with \|delta\| > 0.05 |

## 3.10 Spring Training Features

Path:

- `data/processed/features/features_spring_<season>.parquet`

Schema:

Same columns as regular-season features (Section 3.7) with:

- `is_spring` always `1.0`
- `home_win` populated from schedule scores (not Retrosheet)
- Features built from prior-season team state (no in-season rolling stats)

Invariants:

- Built by `scripts/build_spring_features.py`
- Lenient: seasons with zero spring training games produce no file
- Uses MLB Stats API schedule scores (not Retrosheet gamelogs)

---

# 4. Manual Mapping File Contract

Path:
- `data/processed/team_id_map_retro_to_mlb.csv`

Schema:

| Column | Type | Required | Notes |
|---|---|---:|---|
| `retro_team_code` | string | yes | e.g., LAN |
| `mlb_team_id` | int32 | yes | Stats API team id |
| `mlb_abbrev` | string | no | convenience |
| `valid_from_season` | int32 | yes | inclusive |
| `valid_to_season` | int32 | yes | inclusive |

Invariants:
- For any `(retro_team_code, season)` there MUST be exactly one mapping row.
- Gaps or overlaps are errors.

---

# 5. Determinism Requirements

Any module generating derived data MUST:
- sort rows deterministically before writing
- avoid unordered set/dict iteration in output generation
- record config and hashes sufficient to reproduce
