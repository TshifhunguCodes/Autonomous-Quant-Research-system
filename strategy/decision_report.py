import pandas as pd
from core.logging_utils import get_logger

logger = get_logger(__name__)

def generate_decision_report(config):
    """
    Evaluates system validity and provides recommendations based on regime performance data.
    """
    path = config.paths.backtest_dir / "system_regime_performance.csv"
    if not path.exists():
        return "Governance Report: system_regime_performance.csv not found. Run backtest first."

    try:
        df = pd.read_csv(path)
        if df.empty:
            return "Governance Report: Performance data is empty."

        session_df = df[df["regime_type"] == "session"].copy()
        
        governance_rows = []
        systems = ["ALPHA", "FLOW_EXPLORATORY"]
        sessions = ["LONDON", "ASIA", "NEW_YORK", "LATE_SESSION"]

        for sys in systems:
            for sess in sessions:
                row = session_df[(session_df["system"] == sys) & (session_df["regime_value"] == sess)]
                
                if row.empty:
                    status = "DISABLED (No Data)"
                    pf, pnl, trades = 0.0, 0.0, 0
                else:
                    pf = row.iloc[0]["profit_factor"]
                    pnl = row.iloc[0]["net_pnl"]
                    trades = row.iloc[0]["trades"]
                    
                    # GOVERNANCE LOGIC
                    if sys == "ALPHA":
                        if sess == "NEW_YORK":
                            status = "DISABLED (Strategy Override)"
                        elif trades < 10:
                            status = "RESEARCH ONLY (Low Sample)"
                        elif pf >= 1.2 and pnl > 0:
                            status = "ACTIVE"
                        else:
                            status = "RESEARCH ONLY (Sub-optimal)"
                    else: # FLOW_EXPLORATORY
                        if sess == "NEW_YORK" and pf < 1.0:
                            status = "DISABLED (Negative Expectancy)"
                        elif pf >= 1.1 and pnl > 0:
                            status = "RESEARCH ONLY (Collecting Alpha)"
                        else:
                            status = "RESEARCH ONLY (Baseline Sensor)"

                governance_rows.append({
                    "System": sys,
                    "Session": sess,
                    "Trades": trades,
                    "PF": pf,
                    "PnL": pnl,
                    "STATUS": status
                })

        gov_df = pd.DataFrame(governance_rows)

        # Operational Recommendation
        active_count = len(gov_df[gov_df["STATUS"] == "ACTIVE"])
        is_ready = "READY" if active_count >= 2 else "NOT READY (Insufficient Active Regimes)"

        report = [
            "\n" + "="*70,
            "FINAL STRATEGY GOVERNANCE REPORT",
            "="*70,
            gov_df.to_string(index=False),
            "-"*70,
            "OPERATIONAL CONFIGURATION SUMMARY:",
            f"  Deployment Readiness:  {is_ready}",
            f"  Alpha Core:            London / Asia (Sniper Mode)",
            f"  Flow Sensor:           Multi-Session (Reduced Risk)",
            f"  NY Protocol:           {gov_df[gov_df['Session']=='NEW_YORK']['STATUS'].iloc[0]}",
            "="*70 + "\n"
        ]
        return "\n".join(report)

    except Exception as e:
        logger.error("Error generating decision report: %s", e)
        return f"Governance Report Error: {str(e)}"

def generate_stress_validation_report(config):
    """
    Evaluates Out-of-Sample stress validation and provides a Readiness Score (0-100).
    """
    is_path = config.paths.backtest_summary.parent / "in_sample_summary.csv"
    oos_path = config.paths.backtest_summary.parent / "out_of_sample_summary.csv"
    stability_path = config.paths.backtest_dir / "consolidated_stability_report.csv"
    trades_path = config.paths.backtest_trades

    if not (is_path.exists() and oos_path.exists()):
        return "Stress Validation Report: IS/OOS artifacts missing. Run backtest with --in-sample-end and --oos-start."

    try:
        is_df = pd.read_csv(is_path)
        oos_df = pd.read_csv(oos_path)
        stab_df = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
        trades_df = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()

        is_sum = is_df.iloc[0]
        oos_sum = oos_df.iloc[0]

        # 0. Sample Adequacy Gates
        regime_perf_path = config.paths.backtest_dir / "system_regime_performance.csv"
        if not regime_perf_path.exists():
            return "Stress Validation Report: ⚠️ INSUFFICIENT DATA (not aborted)"

        r_df = pd.read_csv(regime_perf_path)
        
        # Requirement 1: Min 25 trades per system per regime (for all sessions/states traded)
        if r_df.empty or (r_df["trades"] < 25).any():
            return "Stress Validation Report: ⚠️ INSUFFICIENT DATA (not aborted)"

        # Requirement 2: Min 2 active regimes (Consistent with Governance Alpha Sniper logic)
        active_alpha = r_df[(r_df["system"] == "ALPHA") & (r_df["regime_type"] == "session") & 
                            (r_df["regime_value"] != "NEW_YORK") & (r_df["profit_factor"] >= 1.2) & (r_df["net_pnl"] > 0)]
        if len(active_alpha) < 2:
            return "Stress Validation Report: ⚠️ INSUFFICIENT DATA (not aborted)"

        # 1. Performance Degradation Calculation
        is_wr = is_sum["true_win_rate_pct"]
        oos_wr = oos_sum["true_win_rate_pct"]
        wr_delta = is_wr - oos_wr

        is_pf = is_sum["profit_factor"]
        oos_pf = oos_sum["profit_factor"]
        pf_delta = is_pf - oos_pf

        # 2. Rolling Stability & Sample Reliability
        total_trades = oos_sum["closed_trades"]
        wr_std_30d = 0.0
        pf_stability_30d = 0.0
        if not stab_df.empty:
            alpha_stab = stab_df[stab_df["System"] == "ALPHA"]
            if not alpha_stab.empty:
                wr_std_30d = alpha_stab.iloc[0]["WR_StdDev"]
                pf_stability_30d = alpha_stab.iloc[0]["PF_Stability"]

        # 3. Per-Regime Reliability (OOS focus)
        regime_reliability = "PASSED"
        unreliable_regimes = []
        if not trades_df.empty and "session" in trades_df.columns:
            # Filter trades for OOS period based on summary start/end if available, 
            # here we use the system_regime_performance logic as a proxy
            regime_perf_path = config.paths.backtest_dir / "system_regime_performance.csv"
            if regime_perf_path.exists():
                r_df = pd.read_csv(regime_perf_path)
                # Flag regimes with very low trade counts in the total backtest 
                # as they are unreliable for statistical inference
                low_volume = r_df[(r_df["system"] == "ALPHA") & (r_df["trades"] < 5)]
                if not low_volume.empty:
                    unreliable_regimes = low_volume["regime_value"].tolist()
                    regime_reliability = "CAUTION (Low Volume)"

        # 3. Readiness Scoring (0-100)
        score = 0
        
        # WR Degradation (Max 30)
        if wr_delta < 5: score += 30
        elif wr_delta < 15: score += 15
        
        # PF Degradation (Max 20)
        if pf_delta < 0.1: score += 20
        elif pf_delta < 0.3: score += 10
        
        # Stability Variance (Max 30) - Lower WR_StdDev is better
        # Using WR Std Dev as primary stability anchor
        if wr_std_30d < 10: score += 30
        elif wr_std_30d < 18: score += 15
        
        # Sample Volume Reliability (Max 20)
        if total_trades >= 50: score += 20
        elif total_trades >= 25: score += 10

        # 4. Classification
        if score >= 80:
            classification = "READY FOR DEMO LIVE"
            status_icon = "🟢"
        elif score >= 50:
            classification = "CONDITIONAL (Monitor Closely)"
            status_icon = "🟡"
        else:
            classification = "NOT READY (Re-evaluate Strategy)"
            status_icon = "🔴"

        report = [
            "\n" + "!"*70,
            "STRATEGY STRESS VALIDATION REPORT (OUT-OF-SAMPLE)",
            "!"*70,
            f"Win Rate Degradation:   {wr_delta:+.2f}% ({is_wr:.1f}% -> {oos_wr:.1f}%)",
            f"PF Stability Drift:     {pf_delta:+.2f} ({is_pf:.2f} -> {oos_pf:.2f})",
            f"Rolling 30d WR StdDev:  {wr_std_30d:.2f}%",
            f"Rolling 30d PF Stab:    {pf_stability_30d:.2f}",
            f"OOS Sample Size:        {total_trades} Trades",
            f"Regime Reliability:     {regime_reliability}",
            "-"*70,
            f"STRESS VALIDATION SCORE: {score}/100",
            f"DEPLOYMENT STATUS:       {status_icon} {classification}",
            "!"*70,
            "\nANALYSIS NOTES:",
            f"- System shows {'excellent' if wr_delta < 5 else 'acceptable' if wr_delta < 15 else 'poor'} regime consistency.",
            f"- Sample size is {'statistically robust' if total_trades >= 50 else 'marginal'} for the validation window.",
            f"- Performance {'held steady' if pf_delta < 0.1 else 'declined'} in unseen market data segments.",
            f"- Unreliable Regimes:    {', '.join(unreliable_regimes) if unreliable_regimes else 'None'}",
            "!"*70 + "\n"
        ]
        
        # Return as string for display
        return "\n".join(report)

    except Exception as e:
        logger.error("Error generating stress report: %s", e)
        return f"Stress Validation Report Error: {str(e)}"