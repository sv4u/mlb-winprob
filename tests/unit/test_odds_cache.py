"""Unit tests for winprob.app.odds_cache."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pandas as pd

from winprob.app.odds_cache import (
    american_to_decimal,
    american_to_implied,
    compute_ev_opportunities,
    match_odds_for_game,
    is_odds_configured,
    _pick_best_price,
)


# ---------------------------------------------------------------------------
# Sample event data matching The Odds API response shape
# ---------------------------------------------------------------------------


def _make_event(
    home: str = "New York Yankees",
    away: str = "Boston Red Sox",
    bookmakers: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a minimal Odds API event dict for testing."""
    if bookmakers is None:
        bookmakers = [
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": -150},
                            {"name": away, "price": 130},
                        ],
                    }
                ],
            },
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": home, "price": -140},
                            {"name": away, "price": 120},
                        ],
                    }
                ],
            },
        ]
    return {
        "id": "abc123",
        "sport_key": "baseball_mlb",
        "home_team": home,
        "away_team": away,
        "commence_time": "2026-06-15T23:05:00Z",
        "bookmakers": bookmakers,
    }


# ---------------------------------------------------------------------------
# american_to_implied
# ---------------------------------------------------------------------------


def test_implied_probability_negative_odds() -> None:
    """Negative American odds convert correctly to implied probability."""
    prob = american_to_implied(-150)
    assert abs(prob - 0.6) < 0.001


def test_implied_probability_positive_odds() -> None:
    """Positive American odds convert correctly to implied probability."""
    prob = american_to_implied(130)
    expected = 100.0 / 230.0
    assert abs(prob - expected) < 0.001


def test_implied_probability_even_money() -> None:
    """Even money (+100) yields 50% implied probability."""
    prob = american_to_implied(100)
    assert abs(prob - 0.5) < 0.001


def test_implied_probability_zero() -> None:
    """Zero odds return 0.5 as fallback."""
    prob = american_to_implied(0)
    assert abs(prob - 0.5) < 0.001


# ---------------------------------------------------------------------------
# _pick_best_price
# ---------------------------------------------------------------------------


def test_pick_best_price_negatives() -> None:
    """For negative prices, the least negative is the best."""
    assert _pick_best_price([-150, -140, -160]) == -140


def test_pick_best_price_positives() -> None:
    """For positive prices, the most positive is the best."""
    assert _pick_best_price([120, 130, 125]) == 130


def test_pick_best_price_mixed() -> None:
    """Mixed sign prices return the max (most favorable)."""
    assert _pick_best_price([-110, 100, 105]) == 105


def test_pick_best_price_empty() -> None:
    """Empty list returns 0."""
    assert _pick_best_price([]) == 0


# ---------------------------------------------------------------------------
# match_odds_for_game
# ---------------------------------------------------------------------------


def test_match_odds_found() -> None:
    """Matching event by retro codes returns a structured dict."""
    events = [_make_event()]
    result = match_odds_for_game(events, "NYA", "BOS")
    assert result is not None
    assert result["home_team"] == "New York Yankees"
    assert result["away_team"] == "Boston Red Sox"
    assert len(result["bookmakers"]) == 2
    assert result["commence_time"] == "2026-06-15T23:05:00Z"


def test_match_odds_no_match() -> None:
    """Non-matching retro codes return None."""
    events = [_make_event()]
    result = match_odds_for_game(events, "CHN", "CHA")
    assert result is None


def test_match_odds_best_price_selection() -> None:
    """Best price is the most favorable across bookmakers."""
    events = [_make_event()]
    result = match_odds_for_game(events, "NYA", "BOS")
    assert result is not None
    assert result["best_home_price"] == -140
    assert result["best_away_price"] == 130


def test_match_odds_implied_probability() -> None:
    """Implied probabilities are computed from best prices."""
    events = [_make_event()]
    result = match_odds_for_game(events, "NYA", "BOS")
    assert result is not None
    assert abs(result["home_implied_prob"] - american_to_implied(-140)) < 0.001
    assert abs(result["away_implied_prob"] - american_to_implied(130)) < 0.001


def test_match_odds_empty_events() -> None:
    """Empty events list returns None."""
    assert match_odds_for_game([], "NYA", "BOS") is None


def test_match_odds_empty_retro_codes() -> None:
    """Empty retro codes return None."""
    events = [_make_event()]
    assert match_odds_for_game(events, "", "BOS") is None
    assert match_odds_for_game(events, "NYA", "") is None


def test_match_odds_no_h2h_market() -> None:
    """Event with no h2h market produces empty bookmakers list."""
    ev = _make_event(
        bookmakers=[
            {
                "key": "test",
                "title": "Test",
                "markets": [{"key": "spreads", "outcomes": []}],
            }
        ]
    )
    result = match_odds_for_game([ev], "NYA", "BOS")
    assert result is not None
    assert result["bookmakers"] == []
    assert result["best_home_price"] == 0


# ---------------------------------------------------------------------------
# is_odds_configured
# ---------------------------------------------------------------------------


def test_is_odds_configured_true() -> None:
    """Returns True when config status indicates configured."""
    with patch(
        "winprob.app.odds_cache.get_odds_config_status",
        return_value={"configured": True, "source": "env"},
    ):
        assert is_odds_configured() is True


def test_is_odds_configured_false() -> None:
    """Returns False when config status indicates not configured."""
    with patch(
        "winprob.app.odds_cache.get_odds_config_status",
        return_value={"configured": False, "source": None},
    ):
        assert is_odds_configured() is False


# ---------------------------------------------------------------------------
# american_to_decimal
# ---------------------------------------------------------------------------


def test_american_to_decimal_positive() -> None:
    """Positive American odds convert to decimal correctly."""
    assert abs(american_to_decimal(130) - 2.30) < 0.001


def test_american_to_decimal_negative() -> None:
    """Negative American odds convert to decimal correctly."""
    assert abs(american_to_decimal(-150) - 1.6667) < 0.001


def test_american_to_decimal_even() -> None:
    """Even money (+100) converts to 2.0."""
    assert abs(american_to_decimal(100) - 2.0) < 0.001


# ---------------------------------------------------------------------------
# compute_ev_opportunities
# ---------------------------------------------------------------------------


def _make_features_df() -> pd.DataFrame:
    """Build a minimal features DataFrame matching one test event."""
    return pd.DataFrame(
        {
            "game_pk": [100001],
            "date": [pd.Timestamp("2026-06-15")],
            "season": [2026],
            "home_retro": ["NYA"],
            "away_retro": ["BOS"],
            "prob": [0.62],
            "home_win": [None],
        }
    )


_MOCK_NAMES = {"NYA": "Yankees", "BOS": "Red Sox"}


def test_compute_ev_finds_positive_edge() -> None:
    """Opportunities are returned when model probability exceeds implied."""
    events = [_make_event()]
    df = _make_features_df()
    with patch("winprob.app.data_cache.TEAM_NAMES", _MOCK_NAMES):
        opps = compute_ev_opportunities(events, df, min_edge=0.0)
    home_opps = [o for o in opps if o["selection"] == "home"]
    assert len(home_opps) >= 1
    assert home_opps[0]["edge"] > 0


def test_compute_ev_filters_by_min_edge() -> None:
    """Opportunities below min_edge are excluded."""
    events = [_make_event()]
    df = _make_features_df()
    with patch("winprob.app.data_cache.TEAM_NAMES", _MOCK_NAMES):
        opps = compute_ev_opportunities(events, df, min_edge=0.99)
    assert len(opps) == 0


def test_compute_ev_empty_events() -> None:
    """Empty events list returns no opportunities."""
    df = _make_features_df()
    opps = compute_ev_opportunities([], df, min_edge=0.0)
    assert opps == []


def test_compute_ev_empty_df() -> None:
    """Empty features DataFrame returns no opportunities."""
    events = [_make_event()]
    opps = compute_ev_opportunities(events, pd.DataFrame(), min_edge=0.0)
    assert opps == []


def test_compute_ev_sorted_by_edge() -> None:
    """Opportunities are sorted by edge descending."""
    events = [_make_event()]
    df = _make_features_df()
    with patch("winprob.app.data_cache.TEAM_NAMES", _MOCK_NAMES):
        opps = compute_ev_opportunities(events, df, min_edge=0.0)
    if len(opps) >= 2:
        assert opps[0]["edge"] >= opps[1]["edge"]
