#!/usr/bin/env python
"""AQRS V3 Institutional Validation Suite."""

from __future__ import annotations
from typing import Any
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import json

class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for numpy types to allow JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


@dataclass
class ValidationResult:
    """Holds validation test results."""
    test_name: str
    passed: bool
    score: float  # 0-100
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class InstitutionalValidator:
    """Comprehensive institutional validation for AQRS V3."""

    def __init__(self, df: pd.DataFrame, config: Any):
        self.df = df.copy()
        self.config = config

        # V3 compatibility: map alpha/flow scores to a unified confirm_score for validation logic
        if "confirm_score" not in self.df.columns:
            self.df["confirm_score"] = 0.0
            if "alpha_score" in self.df.columns:
                self.df.loc[self.df["signal"] == "ALPHA", "confirm_score"] = self.df["alpha_score"]
            if "flow_score" in self.df.columns:
                self.df.loc[self.df["signal"] == "FLOW", "confirm_score"] = self.df["flow_score"]

        self.results: list[ValidationResult] = []
        
        # Institutional Baseline: All entries are penalized by a standard 1.0x spread (5 points)
        # This ensures Comparison, Survival, and Walk-Forward tests are realistic.
        self.df["entry_price"] = self.df["entry_price"] + (5.0 / 2)
        self.starting_balance = float(config.backtest.starting_balance)

    def run_full_suite(self) -> dict[str, Any]:
        """Run all validation tests."""
        print("Starting AQRS V3 Institutional Validation Suite...")
        print("=" * 70)

        # Test 1: Strategy Comparison
        print("\n[1/8] Testing Alpha vs Flow vs Combined...")
        self.test_strategy_comparison()

        # Test 2: Spread Stress
        print("[2/8] Testing spread stress (1x, 1.5x, 2x)...")
        self.test_spread_stress()

        # Test 3: Slippage
        print("[3/8] Testing slippage simulation...")
        self.test_slippage()

        # Test 4: Walk-Forward
        print("[4/8] Testing walk-forward quarterly...")
        self.test_walkforward_quarterly()

        # Test 5: Monte Carlo
        print("[5/8] Testing Monte Carlo sequence reshuffle...")
        self.test_monte_carlo()

        # Test 6: Losing Streak
        print("[6/8] Testing max losing streak...")
        self.test_max_losing_streak()

        # Test 7: Account Survival
        print("[7/8] Testing account survival ($100/$500/$1000)...")
        self.test_account_survival()

        # Test 8: Session Profitability
        print("[8/8] Testing session profitability matrix...")
        self.test_session_profitability()

        # Compile readiness score
        readiness = self._compute_readiness_score()

        return {
            "results": [self._result_to_dict(r) for r in self.results],
            "readiness_score": readiness,
            "timestamp": datetime.now().isoformat(),
            "data_points": len(self.df),
            "date_range": f"{self.df['time'].iloc[0]} to {self.df['time'].iloc[-1]}",
        }

    def test_strategy_comparison(self):
        """Test Alpha only vs Flow only vs Combined strategies."""
        alpha_only = self.df[self.df["signal"] == "ALPHA"].copy()
        flow_only = self.df[self.df["signal"] == "FLOW"].copy()
        combined = self.df[self.df["signal"].isin(["ALPHA", "FLOW"])].copy()

        alpha_pnl = self._compute_pnl(alpha_only, seed=42)
        flow_pnl = self._compute_pnl(flow_only, seed=42)
        combined_pnl = self._compute_pnl(combined, seed=42)

        alpha_wr = len(alpha_pnl[alpha_pnl > 0]) / len(alpha_pnl) if len(alpha_pnl) > 0 else 0
        flow_wr = len(flow_pnl[flow_pnl > 0]) / len(flow_pnl) if len(flow_pnl) > 0 else 0
        combined_wr = len(combined_pnl[combined_pnl > 0]) / len(combined_pnl) if len(combined_pnl) > 0 else 0

        # Score based on which strategy is most consistent
        scores = {"alpha": alpha_wr, "flow": flow_wr, "combined": combined_wr}
        best_score = max(scores.values())
        consistency = best_score * 100

        result = ValidationResult(
            test_name="Strategy Comparison",
            passed=bool(combined_wr >= 0.45),
            score=consistency,
            details={
                "alpha_win_rate": float(alpha_wr),
                "flow_win_rate": float(flow_wr),
                "combined_win_rate": float(combined_wr),
                "alpha_trades": len(alpha_only),
                "flow_trades": len(flow_only),
                "combined_trades": len(combined),
                "best_strategy": max(scores, key=scores.get),
            },
            warnings=[
                "Alpha signals have low frequency" if len(alpha_only) < 100 else None,
                "Flow signals too broad" if flow_wr < 0.4 else None,
            ]
            if flow_wr < 0.45
            else [],
        )
        self.results.append(result)

    def test_spread_stress(self):
        """Test resilience to different spread multipliers."""
        base_spread = 5.0  # points
        spreads = [1.0, 1.5, 2.0]
        results_per_spread = {}

        for multiplier in spreads:
            spread_adjusted = base_spread * multiplier
            df_adjusted = self.df.copy()
            df_adjusted["entry_price"] = df_adjusted["close"] + spread_adjusted / 2
            pnl = self._compute_pnl(df_adjusted[df_adjusted["signal"].isin(["ALPHA", "FLOW"])], seed=42)
            results_per_spread[f"{multiplier}x"] = float(pnl.sum())

        # Score based on PnL degradation
        base_pnl = results_per_spread["1.0x"]
        if base_pnl <= 0:
            degradation = 1.0
            score = 0.0
            passed = False
        else:
            degradation = max(0, 1 - results_per_spread["2.0x"] / base_pnl)
            score = min(100.0, (1 - min(degradation, 0.5)) * 100)
            passed = bool(degradation < 0.3)

        result = ValidationResult(
            test_name="Spread Stress Test",
            passed=passed,
            score=score,
            details={
                "1x_spread_pnl": results_per_spread["1.0x"],
                "1_5x_spread_pnl": results_per_spread["1.5x"],
                "2x_spread_pnl": results_per_spread["2.0x"],
                "degradation": float(degradation),
            },
            warnings=(
                ["System degradation >30% with 2x spread"] if degradation > 0.3 else []
            ),
        )
        self.results.append(result)

    def test_slippage(self):
        """Test impact of slippage on profitability."""
        slippage_points = [0, 2, 5, 10]
        results_by_slip = {}

        for slip in slippage_points:
            df_slip = self.df[self.df["signal"].isin(["ALPHA", "FLOW"])].copy()
            df_slip["entry_price"] = df_slip["close"] + (5.0 / 2) + slip
            pnl = self._compute_pnl(df_slip, seed=42)
            results_by_slip[slip] = float(pnl.sum())

        # Score: how much can we handle
        base = results_by_slip[0]
        at_10 = results_by_slip[10]
        
        if base <= 0:
            resilience = 0.0
            score = 0.0
            passed = False
        else:
            resilience = max(0.0, (at_10 / base))
            score = min(100.0, resilience * 100)
            passed = bool(resilience > 0.7)

        result = ValidationResult(
            test_name="Slippage Simulation",
            passed=passed,
            score=score,
            details={
                "no_slippage_pnl": results_by_slip[0],
                "2pt_slippage_pnl": results_by_slip[2],
                "5pt_slippage_pnl": results_by_slip[5],
                "10pt_slippage_pnl": results_by_slip[10],
                "resilience": float(resilience),
            },
            warnings=["System fails with 10pt slippage"] if resilience < 0.5 else [],
        )
        self.results.append(result)

    def test_walkforward_quarterly(self):
        """Test walk-forward performance by quarter."""
        self.df["year_quarter"] = pd.to_datetime(self.df["time"]).dt.to_period("Q")

        all_signals = self.df[self.df["signal"].isin(["ALPHA", "FLOW"])]
        if len(all_signals) == 0:
            return
            
        # Compute PnL once for all signals to ensure RNG continuity across quarters
        all_pnls = self._compute_pnl(all_signals, seed=42)
        quarters = sorted(self.df["year_quarter"].unique())

        quarterly_pnls = []
        quarterly_wr = []

        for quarter in quarters:
            q_pnls = all_pnls[self.df.loc[all_pnls.index, "year_quarter"] == quarter]
            if len(q_pnls) > 0:
                wr = len(q_pnls[q_pnls > 0]) / len(q_pnls)
                quarterly_pnls.append(float(q_pnls.sum()))
                quarterly_wr.append(float(wr))

        # Score based on consistency across quarters
        if quarterly_wr:
            consistency = np.std(quarterly_wr) if len(quarterly_wr) > 1 else 0
            avg_wr = np.mean(quarterly_wr)
            score = (avg_wr - consistency) * 100
        else:
            score = 0

        result = ValidationResult(
            test_name="Walk-Forward Quarterly",
            passed=bool(len(quarterly_wr) > 2 and np.mean(quarterly_wr) > 0.45),
            score=max(0, score),
            details={
                "quarters_tested": len(quarterly_wr),
                "avg_win_rate": float(np.mean(quarterly_wr)) if quarterly_wr else 0,
                "wr_std_dev": float(np.std(quarterly_wr)) if len(quarterly_wr) > 1 else 0,
                "total_pnl": float(sum(quarterly_pnls)),
                "quarterly_pnls": quarterly_pnls,
            },
            warnings=(
                ["Win rate varies significantly across quarters"]
                if np.std(quarterly_wr) > 0.15 and len(quarterly_wr) > 1
                else []
            ),
        )
        self.results.append(result)

    def test_monte_carlo(self):
        """Test robustness via Monte Carlo trade sequence reshuffling."""
        signals = self.df[self.df["signal"].isin(["ALPHA", "FLOW"])].copy()
        original_pnl = self._compute_pnl(signals, seed=42).sum()

        mc_pnls = []
        rng = np.random.default_rng(42)
        for _ in range(25):  # Reduced from 100 for performance
            idx = rng.permutation(len(signals))
            shuffled = signals.iloc[idx].copy()
            mc_pnl = self._compute_pnl(shuffled, seed=42).sum()
            mc_pnls.append(mc_pnl)

        mc_pnls = np.array(mc_pnls)
        percentile_5 = np.percentile(mc_pnls, 5)
        percentile_95 = np.percentile(mc_pnls, 95)

        # Score: original PnL confidence
        if np.std(mc_pnls) > 0:
            z_score = (original_pnl - np.mean(mc_pnls)) / np.std(mc_pnls)
            confidence = 1 / (1 + np.exp(-z_score)) * 100
        else:
            confidence = 50

        result = ValidationResult(
            test_name="Monte Carlo Robustness",
            passed=bool(original_pnl > percentile_5),
            score=min(100, confidence),
            details={
                "original_pnl": float(original_pnl),
                "mc_mean_pnl": float(np.mean(mc_pnls)),
                "mc_std_pnl": float(np.std(mc_pnls)),
                "percentile_5": float(percentile_5),
                "percentile_95": float(percentile_95),
                "z_score": float(z_score) if np.std(mc_pnls) > 0 else 0,
            },
            warnings=(
                ["Original PnL below Monte Carlo 5th percentile (luck-based)"]
                if original_pnl < percentile_5
                else []
            ),
        )
        self.results.append(result)

    def test_max_losing_streak(self):
        """Test maximum consecutive losing trades."""
        signals = self.df[self.df["signal"].isin(["ALPHA", "FLOW"])].copy()
        pnl = self._compute_pnl(signals, seed=42)

        if len(pnl) == 0:
            result = ValidationResult(
                test_name="Max Losing Streak",
                passed=False,
                score=0,
                details={},
            )
            self.results.append(result)
            return

        # Find streaks
        losing_vals = (pnl <= 0).astype(int).values
        streaks = np.diff(np.concatenate(([0], losing_vals, [0]))).nonzero()[0].reshape(-1, 2)
        max_losing_streak = 0

        for start, end in streaks:
            if losing_vals[start] == 1:
                max_losing_streak = max(max_losing_streak, end - start)

        # Psychological tolerance: <5 is excellent, <10 is good
        if max_losing_streak < 5:
            score = 95
        elif max_losing_streak < 10:
            score = 80
        elif max_losing_streak < 15:
            score = 60
        else:
            score = 30

        result = ValidationResult(
            test_name="Max Losing Streak",
            passed=bool(max_losing_streak < 10),
            score=score,
            details={
                "max_consecutive_losses": int(max_losing_streak),
                "total_trades": len(pnl),
                "win_rate": float(len(pnl[pnl > 0]) / len(pnl)),
            },
            warnings=(
                [f"Max losing streak of {max_losing_streak} trades (psychological pressure)"]
                if max_losing_streak > 12
                else []
            ),
        )
        self.results.append(result)

    def test_account_survival(self):
        """Test account survival at $100, $500, $1000 starting balances."""
        account_sizes = [100, 500, 1000]
        survival_rates = {}
        signals = self.df[self.df["signal"].isin(["ALPHA", "FLOW"])].copy()
        pnls = self._compute_pnl(signals, seed=42).values

        for account_size in account_sizes:
            equity = account_size
            for pnl in pnls:
                equity += pnl
                if equity <= 0:
                    equity = 0
                    break
            survival_rates[account_size] = equity

        # Score based on $100 account survival (binary 100 or 0)
        score = 100.0 if survival_rates[100] > 0 else 0.0

        result = ValidationResult(
            test_name="Account Survival",
            passed=bool(all(sr > 0 for sr in survival_rates.values())),
            score=score,
            details={
                "final_equity_100": float(survival_rates[100]),
                "final_equity_500": float(survival_rates[500]),
                "final_equity_1000": float(survival_rates[1000]),
                "total_trades": len(signals),
            },
            warnings=(
                ["$100 account depleted"] if survival_rates[100] <= 0 else []
            ),
        )
        self.results.append(result)

    def test_session_profitability(self):
        """Test profitability matrix across trading sessions."""
        sessions = ["ASIA", "LONDON", "NEW_YORK"]
        session_profits = {}

        for session in sessions:
            session_data = self.df[
                (self.df["session"] == session)
                & (self.df["signal"].isin(["ALPHA", "FLOW"]))
            ]
            if len(session_data) > 0:
                pnl = self._compute_pnl(session_data, seed=42)
                wr = len(pnl[pnl > 0]) / len(pnl)
                session_profits[session] = {
                    "trades": len(session_data),
                    "win_rate": float(wr),
                    "pnl": float(pnl.sum()),
                }
            else:
                session_profits[session] = {
                    "trades": 0,
                    "win_rate": 0,
                    "pnl": 0,
                }

        # Score: best session win rate
        best_wr = max((s["win_rate"] for s in session_profits.values()), default=0)
        score = best_wr * 100

        result = ValidationResult(
            test_name="Session Profitability",
            passed=bool(best_wr > 0.45),
            score=score,
            details=session_profits,
            warnings=(
                ["Asia session underperforming"] if session_profits["ASIA"]["win_rate"] < 0.4 else []
            ),
        )
        self.results.append(result)

    def _compute_pnl(self, df: pd.DataFrame, seed: int | None = None) -> pd.Series:
        """Compute PnL for each trade (vectorized)."""
        if len(df) == 0:
            return pd.Series([], dtype=float)

        # Factor in price stressors (spread/slippage)
        # Original entry was 'close', new entry (stressed) is 'entry_price'
        price_stress = (df["entry_price"] - df["close"]).abs()
        stop_dist = df["stop_distance"].astype(float)
        pos_size = df["position_size"].astype(float)
        rr = self.config.risk.rr_ratio
        
        win_probs = (df["confirm_score"].astype(float) / 100.0) * 0.8 + 0.1
        win_probs = np.clip(win_probs, 0.35, 0.75)

        rng = np.random.default_rng(seed)
        outcomes = rng.binomial(1, win_probs)
        
        # Outcomes adjusted for stressors:
        # Win PnL = position_size * (target_points - stress_points)
        # Loss PnL = position_size * (stop_points + stress_points)
        pnl = np.where(
            outcomes == 1,
            pos_size * (stop_dist * rr - price_stress),
            -pos_size * (stop_dist + price_stress)
        )

        return pd.Series(pnl, index=df.index)

    def _result_to_dict(self, result: ValidationResult) -> dict[str, Any]:
        """Convert result to dict."""
        return {
            "test_name": result.test_name,
            "passed": bool(result.passed),
            "score": round(result.score, 1),
            "details": result.details,
            "warnings": result.warnings,
        }

    def _compute_readiness_score(self) -> dict[str, Any]:
        """Compute overall readiness score."""
        if not self.results:
            return {"score": 0, "tier": "Not Ready"}

        # Cap individual scores to 100.0
        scores = [min(100.0, r.score) for r in self.results]
        weights = [1.0] * len(scores)  # Equal weight for now

        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        weighted_score = min(100.0, weighted_score)

        # Determine tier
        if weighted_score >= 80:
            tier = "Strong Demo Candidate"
        elif weighted_score >= 50:
            tier = "Demo Ready"
        else:
            tier = "Not Ready"

        passed_tests = sum(1 for r in self.results if r.passed)

        return {
            "score": round(weighted_score, 1),
            "tier": tier,
            "passed_tests": passed_tests,
            "total_tests": len(self.results),
            "individual_scores": {r.test_name: round(r.score, 1) for r in self.results},
        }


def main():
    # Load pipeline data
    df = pd.read_csv("data/backtest/v3_research_output.csv", parse_dates=["time"])

    # Load config
    from core.config import load_config
    from config.v3_config import V3Config

    base_config = load_config()
    config = V3Config.load_from(base_config)

    # Run validation
    validator = InstitutionalValidator(df, config)
    results = validator.run_full_suite()

    # Print summary
    print("\n" + "=" * 70)
    print("AQRS V3 INSTITUTIONAL VALIDATION RESULTS")
    print("=" * 70)

    for result in results["results"]:
        status = "[PASS]" if result["passed"] else "[FAIL]"
        print(f"\n{status} | {result['test_name']}: {result['score']:.1f}/100")
        for key, value in result["details"].items():
            if isinstance(value, list):
                continue
            if isinstance(value, float):
                print(f"  └─ {key}: {value:.2f}")
            else:
                print(f"  └─ {key}: {value}")
        if result["warnings"]:
            for warning in result["warnings"]:
                if warning:
                    print(f"  [WARN] {warning}")

    readiness = results["readiness_score"]
    print("\n" + "=" * 70)
    print(f"READINESS SCORE: {readiness['score']:.1f}/100 → {readiness['tier']}")
    print(f"Tests Passed: {readiness['passed_tests']}/{readiness['total_tests']}")
    print("=" * 70)

    # Save results
    output_path = Path("data/backtest/v3_validation_report.json")
    output_path.write_text(json.dumps(results, indent=2, cls=NumpyEncoder))
    print(f"\n[OK] Full validation report saved to {output_path}")

    return results


if __name__ == "__main__":
    main()
