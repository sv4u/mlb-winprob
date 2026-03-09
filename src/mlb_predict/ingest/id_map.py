from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass(frozen=True)
class RetroTeamMap:
    df: pd.DataFrame

    def retro_to_mlb_id(self, retro_code: str, season: int) -> int:
        m = self.df[
            (self.df["retro_team_code"] == retro_code)
            & (self.df["valid_from_season"] <= season)
            & (self.df["valid_to_season"] >= season)
        ]
        if len(m) != 1:
            raise KeyError(f"Ambiguous or missing mapping retro={retro_code} season={season}")
        return int(m.iloc[0]["mlb_team_id"])


def load_retro_team_map(
    path: Path = Path("data/processed/team_id_map_retro_to_mlb.csv"),
) -> RetroTeamMap:
    df = pd.read_csv(path)
    required = {"retro_team_code", "mlb_team_id", "valid_from_season", "valid_to_season"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in team map: {sorted(missing)}")
    return RetroTeamMap(df=df)
