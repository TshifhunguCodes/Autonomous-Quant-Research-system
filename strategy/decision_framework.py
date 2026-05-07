from __future__ import annotations

from collections.abc import Mapping
from typing import Any


TREND_STATES = {"TREND_UP", "TREND_DOWN", "TRENDING"}
CONTINUATION_REGIMES = {"TREND_HEALTHY", "TREND_START", "BREAKOUT", "BREAKOUT_EXPANSION"}


def _get(context: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        value = context.get(key, default)
    except AttributeError:
        value = getattr(context, key, default)
    return default if value is None else value


def _float(context: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(_get(context, key, default))
    except (TypeError, ValueError):
        return default


def _bool(context: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = _get(context, key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _direction(context: Mapping[str, Any]) -> str:
    signal = str(_get(context, "confirmed_signal", "")).upper()
    if signal == "BUY":
        return "LONG"
    if signal == "SELL":
        return "SHORT"
    return str(_get(context, "direction", "")).upper()


def _htf_aligned(context: Mapping[str, Any]) -> bool:
    direction = _direction(context)
    htf_bias = str(_get(context, "htf_bias", _get(context, "h1_bias", "NEUTRAL"))).upper()
    if direction == "LONG":
        return htf_bias in {"BULLISH", "LONG", "UP", "NEUTRAL"}
    if direction == "SHORT":
        return htf_bias in {"BEARISH", "SHORT", "DOWN", "NEUTRAL"}
    return False


def _counter_trend(context: Mapping[str, Any]) -> bool:
    state = str(_get(context, "market_regime", _get(context, "behavior_label", ""))).upper()
    direction = _direction(context)
    return (state == "TREND_UP" and direction == "SHORT") or (state == "TREND_DOWN" and direction == "LONG")


def should_enter_continuation_trade(context: Mapping[str, Any]) -> str:
    lifecycle = str(_get(context, "lifecycle_state", "TREND_HEALTHY")).upper()
    state = str(_get(context, "market_regime", _get(context, "behavior_label", ""))).upper()
    direction = _direction(context)
    retracement = _float(context, "fib_retracement_pct", 0.0)

    if lifecycle == "TREND_EXHAUSTING":
        return "BLOCKED_EXHAUSTION"
    if _bool(context, "fake_breakout") or str(_get(context, "liquidity_event", "")) in {"TRAP_BREAKOUT", "BREAKOUT_REJECTION"}:
        return "BLOCKED_LIQUIDITY_TRAP"
    if _float(context, "impulse_count", 0.0) >= 4 or _float(context, "continuation_strength", 0.0) >= 95:
        return "BLOCKED_OVEREXTENDED"
    if _float(context, "htf_level_distance_atr", 99.0) <= 0.5:
        return "BLOCKED_AT_POI"
    if _bool(context, "choch"):
        return "WAIT"

    structure_holding = (
        (direction == "LONG" and _bool(context, "higher_low_holding", True))
        or (direction == "SHORT" and _bool(context, "lower_high_holding", True))
    )
    nearby_opposition = _float(context, "opposing_liquidity_distance_atr", 99.0) <= 1.5
    overextended = _float(context, "impulse_extension", 1.0) > 2.5

    if lifecycle in CONTINUATION_REGIMES and state in TREND_STATES | {"BREAKOUT"} and structure_holding and _htf_aligned(context) and not nearby_opposition and not overextended:
        return "VALID"
    if retracement < 20 or retracement > 78.6:
        return "WAIT"
    return "WAIT"


def should_enter_retracement_trade(context: Mapping[str, Any]) -> str:
    retracement = _float(context, "fib_retracement_pct", 0.0)
    reaction = _bool(context, "wick_rejection") or _bool(context, "order_block") or _bool(context, "fvg_zone")
    zone_reaction = reaction and (_bool(context, "supply_zone") or _bool(context, "demand_zone") or _bool(context, "order_block") or _bool(context, "fvg_zone"))
    momentum_shift = _float(context, "continuation_strength", 0.0) >= 60 or _bool(context, "momentum_strength")

    if retracement > 100:
        return "REVERSAL_RISK"
    if retracement >= 78.6:
        return "REVERSAL_RISK"
    if _bool(context, "failed_continuation") and _bool(context, "choch"):
        return "REVERSAL_RISK"
    if retracement <= 38:
        return "ENTRY_OPPORTUNITY" if momentum_shift and _htf_aligned(context) else "NO_TRADE_ZONE"
    if retracement <= 61.8:
        return "ENTRY_OPPORTUNITY" if zone_reaction else "NO_TRADE_ZONE"
    if retracement <= 78.6:
        structure_holding = _bool(context, "higher_low_holding", True) or _bool(context, "lower_high_holding", True)
        return "ENTRY_OPPORTUNITY" if structure_holding and reaction else "NO_TRADE_ZONE"
    return "NO_TRADE_ZONE"


def should_enter_reversal_trade(context: Mapping[str, Any]) -> str:
    choch = _bool(context, "choch")
    bos = _bool(context, "bos")
    sweep = _bool(context, "liquidity_sweep") or str(_get(context, "liquidity_event", "")) == "CONFIRMED_SWEEP_REJECTION"
    retest = _bool(context, "wick_rejection") or _bool(context, "break_retest")
    retracement = _float(context, "fib_retracement_pct", 0.0)
    htf_lifecycle = str(_get(context, "htf_lifecycle", "UNKNOWN")).upper()
    htf_bias = str(_get(context, "htf_bias", "NEUTRAL")).upper()

    if choch and not bos and retracement >= 50 and retracement <= 78.6 and retest:
        return "EARLY_WARNING"
    if choch and bos and sweep and retest and (htf_lifecycle in {"REVERSAL_WATCH", "UNKNOWN"} or htf_bias in {"NEUTRAL", "RANGING"}):
        return "CONFIRMED_REVERSAL"
    return "NOT_YET"


def should_enter_counter_trend_trade(context: Mapping[str, Any]) -> str:
    lifecycle = str(_get(context, "lifecycle_state", "")).upper()
    retracement = _float(context, "fib_retracement_pct", 0.0)
    sweep = _bool(context, "liquidity_sweep") or str(_get(context, "liquidity_event", "")) == "CONFIRMED_SWEEP_REJECTION"
    htf_bias = str(_get(context, "htf_bias", "NEUTRAL")).upper()
    rejection = _bool(context, "wick_rejection") or _bool(context, "order_block") or _bool(context, "fvg_zone")
    flow_open = int(_float(context, "flow_open_trades", 0.0))

    if flow_open >= 2:
        return "BLOCKED"
    if lifecycle in {"TREND_HEALTHY", "TREND_START"}:
        return "BLOCKED"
    if htf_bias in {"BULLISH_STRONG", "BEARISH_STRONG"}:
        return "BLOCKED"
    if (lifecycle == "TREND_EXHAUSTING" or retracement > 78.6) and sweep and htf_bias in {"NEUTRAL", "RANGING", "REVERSAL_WATCH"} and rejection:
        return "ALLOWED"
    return "BLOCKED"


def select_strategy_mode(context: Mapping[str, Any]) -> str:
    state = str(_get(context, "market_regime", _get(context, "behavior_label", ""))).upper()
    lifecycle = str(_get(context, "lifecycle_state", "")).upper()
    spread_ratio = _float(context, "spread_ratio_to_average", 1.0)
    recent_losses = int(_float(context, "recent_losses_2h", 0.0))
    drawdown_locked = _bool(context, "daily_loss_locked")
    clear_liquidity = _bool(context, "liquidity_sweep") or _bool(context, "supply_zone") or _bool(context, "demand_zone")

    if drawdown_locked or spread_ratio > 2.5 or recent_losses >= 3:
        return "BOTH_PAUSED"
    if state == "CHOPPY" and not clear_liquidity:
        return "BOTH_PAUSED"
    if state in {"RANGE", "CHOPPY"}:
        return "FLOW_ONLY"
    if lifecycle == "TREND_HEALTHY" and _htf_aligned(context) and not _bool(context, "choch"):
        return "ALPHA_ONLY"
    if lifecycle in {"TREND_START", "BREAKOUT_EXPANSION", "REVERSAL_CONFIRMED"} or state in {"BREAKOUT", "REVERSAL"}:
        return "BOTH_ACTIVE"
    return "BOTH_ACTIVE"


def get_regime_behavior(context: Mapping[str, Any]) -> str:
    lifecycle = str(_get(context, "lifecycle_state", "")).upper()
    state = str(_get(context, "market_regime", _get(context, "behavior_label", ""))).upper()
    if lifecycle in {"TREND_START", "BREAKOUT_EXPANSION"}:
        return "TREND_START_BEHAVIOR"
    if lifecycle == "TREND_HEALTHY" or state in {"TREND_UP", "TREND_DOWN"}:
        return "TREND_HEALTHY_BEHAVIOR"
    if lifecycle == "TREND_EXHAUSTING":
        return "TREND_EXHAUSTING_BEHAVIOR"
    if state == "RANGE":
        return "RANGE_BEHAVIOR"
    if state == "BREAKOUT":
        return "BREAKOUT_BEHAVIOR"
    if lifecycle == "REVERSAL_CONFIRMED" or state == "REVERSAL":
        return "REVERSAL_PHASE_BEHAVIOR"
    return "NEUTRAL_BEHAVIOR"


def score_trade(context: Mapping[str, Any]) -> int:
    score = 0
    direction = _direction(context)
    state = str(_get(context, "market_regime", _get(context, "behavior_label", ""))).upper()
    lifecycle = str(_get(context, "lifecycle_state", "")).upper()

    if _htf_aligned(context):
        score += 25
    if state in {"TREND_UP", "TREND_DOWN", "RANGE", "BREAKOUT", "REVERSAL"} or lifecycle in CONTINUATION_REGIMES:
        score += 20
    if _bool(context, "bos") or _bool(context, "choch") or _bool(context, "higher_low_holding") or _bool(context, "lower_high_holding"):
        score += 20
    if _bool(context, "liquidity_sweep") or str(_get(context, "liquidity_event", "")) == "CONFIRMED_SWEEP_REJECTION":
        score += 15
    if _bool(context, "supply_zone") or _bool(context, "demand_zone") or _bool(context, "order_block") or _bool(context, "fvg_zone"):
        score += 10
    if _bool(context, "wick_rejection") or str(_get(context, "pattern", "")) in {"DOUBLE_TOP", "DOUBLE_BOTTOM", "BREAK_RETEST"}:
        score += 10
    if str(_get(context, "session", "")) in {"LONDON", "NEW_YORK"}:
        score += 5

    rsi = _float(context, "rsi14", 50.0)
    if (direction == "LONG" and rsi < 70) or (direction == "SHORT" and rsi > 30):
        score += 5
    if _float(context, "opposing_liquidity_distance_atr", 2.0) > 1.5:
        score += 5
    return int(score)
