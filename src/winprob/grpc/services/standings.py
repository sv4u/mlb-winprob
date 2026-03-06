"""StandingsService implementation: GetStandings, GetTeamStats."""

from __future__ import annotations

import logging
from typing import Any

import grpc
import pandas as pd

from winprob.app.data_cache import TEAM_NAMES, get_features, is_ready
from winprob.app.timing import timed_operation
from winprob.grpc.generated.winprob.v1 import common_pb2, standings_pb2, standings_pb2_grpc
from winprob.standings import (
    DIVISION_DISPLAY_ORDER,
    DIVISIONS,
    compute_league_leaders,
    compute_predicted_standings,
    merge_predicted_actual,
)

logger = logging.getLogger(__name__)


def _row_to_team_standing(row: Any) -> common_pb2.TeamStanding:
    """Build TeamStanding protobuf from a standings row."""
    ts = common_pb2.TeamStanding(
        retro_code=str(row.get("retro_code", "")),
        mlb_id=int(row.get("mlb_id", 0)),
        team_name=TEAM_NAMES.get(
            str(row.get("retro_code", "")), str(row.get("team_name", ""))
        ),
        pred_wins=float(row.get("pred_wins", 0)),
        pred_losses=float(row.get("pred_losses", 0)),
        pred_win_pct=float(row.get("pred_win_pct", 0)),
        pred_division_rank=int(row.get("pred_division_rank", 0)),
        pred_gb=str(row.get("pred_gb_str", row.get("pred_gb", "-"))),
        pred_total_games=int(row.get("pred_total_games", 0)),
    )
    if "actual_wins" in row and pd.notna(row.get("actual_wins")):
        ts.actual_wins = int(row["actual_wins"])
        ts.actual_losses = int(row["actual_losses"])
        ts.actual_win_pct = round(float(row["actual_win_pct"]), 3)
        ts.actual_gb = str(row.get("actual_gb", "-"))
        ts.actual_division_rank = int(row.get("actual_division_rank", 0))
        ts.runs_scored = int(row.get("runs_scored", 0))
        ts.runs_allowed = int(row.get("runs_allowed", 0))
        ts.run_diff = int(row.get("run_diff", 0))
        ts.wins_delta = round(float(row.get("wins_delta", 0)), 1)
        ts.pct_delta = round(float(row.get("pct_delta", 0)), 3)
        ts.rank_delta = int(row.get("rank_delta", 0))
    return ts


class StandingsServicer(standings_pb2_grpc.StandingsServiceServicer):
    """Implements StandingsService RPCs using standings and MLB API."""

    async def GetStandings(
        self,
        request: standings_pb2.GetStandingsRequest,
        context: grpc.aio.ServicerContext,
    ) -> standings_pb2.StandingsResponse:
        """Return predicted vs actual standings grouped by division."""
        if not is_ready():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("System initializing — data not loaded yet.")
            raise Exception("Not ready")
        season = request.season or 2026
        logger.debug("GetStandings season=%d", season)

        with timed_operation("predicted_standings"):
            df = get_features()
            pred_df = compute_predicted_standings(df, season=season)

        actual_df = pd.DataFrame()
        try:
            from winprob.mlbapi.client import MLBAPIClient
            from winprob.mlbapi.standings import fetch_standings

            async with timed_operation("mlb_api_standings"):
                async with MLBAPIClient() as client:
                    actual_df = await fetch_standings(client, season=season)
        except Exception as exc:
            logger.warning(
                "Could not fetch live standings for season=%d: %s",
                season,
                exc,
            )

        season_started = (
            not actual_df.empty
            and actual_df["wins"].sum() + actual_df["losses"].sum() > 0
        )

        if not pred_df.empty and season_started:
            standings = merge_predicted_actual(pred_df, actual_df)
        elif not pred_df.empty:
            standings = pred_df
        else:
            return standings_pb2.StandingsResponse(
                season=season,
                divisions=[],
                league_leaders=common_pb2.LeagueLeaders(),
            )

        league_leaders = compute_league_leaders(standings)

        divisions: list[common_pb2.DivisionStandings] = []
        for div_id in DIVISION_DISPLAY_ORDER:
            div_info = DIVISIONS.get(div_id, {})
            div_df = standings[standings["division_id"] == div_id].sort_values(
                "pred_win_pct", ascending=False
            )
            if div_df.empty:
                continue
            teams = [_row_to_team_standing(r) for _, r in div_df.iterrows()]
            divisions.append(
                common_pb2.DivisionStandings(
                    division_id=div_id,
                    division_name=div_info.get("name", ""),
                    league=div_info.get("league", ""),
                    teams=teams,
                )
            )

        ll = common_pb2.LeagueLeaders()
        for league in ["AL", "NL"]:
            entry: dict[str, Any] = league_leaders.get(league, {})
            if not entry:
                continue
            section = common_pb2.LeagueLeadersSection()
            pred = entry.get("predicted_leader", {})
            section.predicted_leader.CopyFrom(
                common_pb2.PredictedLeader(
                    team_name=pred.get("team_name", ""),
                    retro_code=pred.get("retro_code", ""),
                    pred_wins=pred.get("pred_wins", 0),
                    pred_losses=pred.get("pred_losses", 0),
                    pred_win_pct=pred.get("pred_win_pct", 0),
                )
            )
            if "actual_leader" in entry:
                act = entry["actual_leader"]
                section.actual_leader = common_pb2.ActualLeader(
                    team_name=act.get("team_name", ""),
                    retro_code=act.get("retro_code", ""),
                    actual_wins=act.get("actual_wins", 0),
                    actual_losses=act.get("actual_losses", 0),
                    actual_win_pct=act.get("actual_win_pct", 0),
                )
            if league == "AL":
                ll.AL.CopyFrom(section)
            else:
                ll.NL.CopyFrom(section)

        return standings_pb2.StandingsResponse(
            season=season,
            divisions=divisions,
            league_leaders=ll,
        )

    async def GetTeamStats(
        self,
        request: standings_pb2.GetTeamStatsRequest,
        context: grpc.aio.ServicerContext,
    ) -> standings_pb2.TeamStatsResponse:
        """Return batting and pitching stats for all teams in a season."""
        season = request.season or 2026
        try:
            from winprob.mlbapi.client import MLBAPIClient
            from winprob.mlbapi.standings import fetch_all_team_stats, fetch_standings

            async with timed_operation("mlb_api_team_stats"):
                async with MLBAPIClient() as client:
                    standings = await fetch_standings(client, season=season)
                    if standings.empty:
                        return standings_pb2.TeamStatsResponse(
                            season=season, teams=[]
                        )
                    total_games = standings["wins"].sum() + standings["losses"].sum()
                    if total_games == 0:
                        return standings_pb2.TeamStatsResponse(
                            season=season,
                            teams=[],
                            message="Season has not started yet.",
                        )
                    full = await fetch_all_team_stats(
                        client,
                        standings_df=standings,
                        season=season,
                    )
        except Exception as exc:
            logger.warning(
                "Could not fetch team stats for season=%d: %s",
                season,
                exc,
            )
            return standings_pb2.TeamStatsResponse(
                season=season, teams=[], error=str(exc)
            )

        teams: list[standings_pb2.TeamStatsEntry] = []
        for _, row in full.iterrows():
            teams.append(
                standings_pb2.TeamStatsEntry(
                    team_id=int(row.get("team_id", 0)),
                    team_name=str(row.get("team_name", "")),
                    division_name=str(row.get("division_name", "")),
                    league_name=str(row.get("league_name", "")),
                    record=f"{row.get('wins', 0)}-{row.get('losses', 0)}",
                    pct=round(float(row.get("pct", 0)), 3),
                    run_diff=int(row.get("run_diff", 0)),
                    batting=standings_pb2.BattingStats(
                        avg=round(float(row.get("bat_avg", 0)), 3),
                        obp=round(float(row.get("bat_obp", 0)), 3),
                        slg=round(float(row.get("bat_slg", 0)), 3),
                        ops=round(float(row.get("bat_ops", 0)), 3),
                        runs=int(row.get("bat_runs", 0)),
                        hits=int(row.get("bat_hits", 0)),
                        doubles=int(row.get("bat_doubles", 0)),
                        triples=int(row.get("bat_triples", 0)),
                        hr=int(row.get("bat_hr", 0)),
                        rbi=int(row.get("bat_rbi", 0)),
                        sb=int(row.get("bat_sb", 0)),
                        bb=int(row.get("bat_bb", 0)),
                        so=int(row.get("bat_so", 0)),
                    ),
                    pitching=standings_pb2.PitchingStats(
                        era=round(float(row.get("pit_era", 0)), 2),
                        wins=int(row.get("pit_wins", 0)),
                        losses=int(row.get("pit_losses", 0)),
                        saves=int(row.get("pit_saves", 0)),
                        ip=str(row.get("pit_ip", "")),
                        hits=int(row.get("pit_hits", 0)),
                        bb=int(row.get("pit_bb", 0)),
                        so=int(row.get("pit_so", 0)),
                        whip=round(float(row.get("pit_whip", 0)), 2),
                        hr=int(row.get("pit_hr", 0)),
                    ),
                )
            )
        return standings_pb2.TeamStatsResponse(season=season, teams=teams)
