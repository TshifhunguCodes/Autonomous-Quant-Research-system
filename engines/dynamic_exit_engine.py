from __future__ import annotations

from typing import Any


class DynamicExitEngine:
    """Lightweight exit planner for partials, runners, and trailing behavior."""

    @staticmethod
    def build_exit_plan(signal: dict[str, Any], unrealized_r: float, side: str) -> dict[str, Any]:
        lifecycle_state = str(signal.get("lifecycle_state", "TREND_HEALTHY"))
        exhaustion_score = float(signal.get("exhaustion_score", 0.0))
        continuation_strength = float(signal.get("continuation_strength", 0.0))
        liquidity_event = str(signal.get("liquidity_event", "NONE"))
        fake_breakout = bool(signal.get("fake_breakout", 0))
        trend_health_score = float(signal.get("trend_health_score", 50.0))
        htf_alignment = float(signal.get("multi_tf_alignment_score", signal.get("htf_alignment", 50.0)))
        atr_value = float(signal.get("atr14", signal.get("atr", 0.0)) or 0.0)

        exit_state = "OPEN"
        if lifecycle_state == "REVERSAL_CONFIRMED" or trend_health_score < 25:
            exit_state = "FORCE_EXIT"
        elif lifecycle_state == "REVERSAL_WATCH" or exhaustion_score >= 80 or fake_breakout:
            exit_state = "EXIT_WARNING"
        elif lifecycle_state in ["TREND_EXTENDED", "TREND_EXHAUSTING"] or exhaustion_score >= 60 or liquidity_event in [
            "FAKE_BREAKOUT",
            "BREAKOUT_REJECTION",
            "TRAP_BREAKOUT",
            "STOP_HUNT",
        ]:
            exit_state = "WEAKENING"
        elif unrealized_r >= 1.0:
            exit_state = "PROTECTED"
        elif lifecycle_state in ["TREND_HEALTHY", "BREAKOUT_EXPANSION"] and continuation_strength >= 70 and trend_health_score >= 65 and unrealized_r >= 0.5:
            exit_state = "SCALE_ALLOWED"

        partial_taken = unrealized_r >= 1.0 or exit_state in ["EXIT_WARNING", "FORCE_EXIT"]
        runner_active = (
            unrealized_r >= 1.0
            and continuation_strength >= 72
            and trend_health_score >= 60
            and htf_alignment >= 65
            and lifecycle_state in ["TREND_HEALTHY", "BREAKOUT_EXPANSION", "PROTECTED", "SCALE_ALLOWED"]
            and not fake_breakout
        )

        trailing_distance = atr_value * 1.2 if atr_value > 0 else 0.0
        if exit_state == "PROTECTED":
            trailing_distance = atr_value * 0.9 if atr_value > 0 else 0.0
        if exit_state == "SCALE_ALLOWED":
            trailing_distance = atr_value * 1.1 if atr_value > 0 else 0.0
        if exit_state == "WEAKENING":
            trailing_distance = atr_value * 0.7 if atr_value > 0 else 0.0
        if exit_state in ["EXIT_WARNING", "FORCE_EXIT"]:
            trailing_distance = atr_value * 0.45 if atr_value > 0 else 0.0

        exit_confidence = 40.0
        exit_confidence += exhaustion_score * 0.25
        exit_confidence += (100.0 - continuation_strength) * 0.20
        exit_confidence += (100.0 - trend_health_score) * 0.20
        exit_confidence += 15.0 if fake_breakout else 0.0
        exit_confidence += 10.0 if liquidity_event in ["FAKE_BREAKOUT", "BREAKOUT_REJECTION", "TRAP_BREAKOUT", "STOP_HUNT"] else 0.0
        exit_confidence += 10.0 if lifecycle_state == "REVERSAL_CONFIRMED" else 0.0
        exit_confidence = max(0.0, min(100.0, exit_confidence))

        return {
            "exit_state": exit_state,
            "partial_taken": bool(partial_taken),
            "runner_active": bool(runner_active),
            "dynamic_trailing_distance": float(max(0.0, trailing_distance)),
            "exit_confidence": float(exit_confidence),
        }
