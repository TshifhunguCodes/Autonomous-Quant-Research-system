import pandas as pd

from core.logging_utils import get_logger


logger = get_logger(__name__)


def _get_session_label(hour):
    """Classifies hour into market sessions for metadata tracking."""
    if 0 <= hour < 7: return "ASIA"
    if 7 <= hour < 13: return "LONDON"
    if 13 <= hour < 18: return "NEW_YORK"
    return "LATE_SESSION"


def _build_alpha_trade(row, config, equity):
    """System A: STRICT ALPHA logic"""
    if str(row.get("signal", "NO_TRADE")) != "ALPHA":
        return None
    hour = pd.to_datetime(row["time"]).hour
    quality = row.get("quality", "UNKNOWN")
    state = row.get("market_state", "UNKNOWN")
    score = float(row.get("confirm_score", 0))
    
    # Filter 1: Elite + structured high-conviction Alpha
    if quality != "ELITE":
        if not (
            quality == "HIGH"
            and state == "RANGING"
            and score >= config.regime.alpha_min_score_high_ranging
        ):
            return None
    if hour not in config.regime.alpha_session_hours or 13 <= hour < 18:
        return None

    # Filter 2: Regime-based Conviction (Stability Enhancement)
    if state == "TRENDING" and score < config.regime.alpha_min_score_elite_trending:
        return None

    if state in ["CHOPPY", "VOLATILE"]:
        return None
        
    trade = _build_trade_base(row, config, equity, risk_adj=1.0, is_exploratory=False)
    if trade:
        trade["system"] = "ALPHA"
        trade["data_type"] = "EXECUTION_RESULT"
    return trade

def _build_flow_trade(row, config, equity):
    """System B: filtered FLOW logic."""
    if str(row.get("signal", "NO_TRADE")) != "FLOW":
        return None
    state = row.get("market_state", "UNKNOWN")
    quality = row.get("quality", "UNKNOWN")
    score = float(row.get("confirm_score", 0))
    framework_score = float(row.get("institutional_trade_score", score))
    strategy_mode = str(row.get("strategy_mode", "BOTH_ACTIVE"))
    flow_trade_type = str(row.get("flow_trade_type", "NONE"))
    lifecycle_state = str(row.get("lifecycle_state", "TREND_HEALTHY"))
    breakout_quality = float(row.get("breakout_quality", 50.0))
    trap_probability = float(row.get("trap_probability", 0.0))
    multi_tf_alignment_score = float(row.get("multi_tf_alignment_score", 50.0))
    htf_exhaustion = float(row.get("htf_exhaustion", 50.0))
    htf_liquidity_alignment = int(row.get("htf_liquidity_alignment", 0))
    continuation_strength = float(row.get("continuation_strength", 0.0))
    fake_breakout = bool(row.get("fake_breakout", 0))
    confirmed_reversal = bool(row.get("confirmed_reversal", 0))
    liquidity_event = str(row.get("liquidity_event", "NONE"))

    if strategy_mode == "BOTH_PAUSED":
        return None
    if framework_score < 45:
        return None
    if quality not in ["HIGH", "ELITE"] and flow_trade_type == "NONE":
        return None
    if score < max(config.regime.flow_min_confirm_score, 45) and framework_score < 45:
        return None
    if fake_breakout or trap_probability >= 70:
        return None
    if lifecycle_state in ["TREND_EXHAUSTING", "REVERSAL_WATCH"]:
        return None
    if htf_exhaustion >= 60 or htf_liquidity_alignment < 0:
        return None
    if multi_tf_alignment_score < 65:
        return None
    if state == "BREAKOUT" and breakout_quality < 70:
        return None
    if state == "REVERSAL" and not confirmed_reversal:
        return None
    if state in ["TREND_UP", "TREND_DOWN"] and continuation_strength < 72:
        return None
    if liquidity_event in ["BREAKOUT_REJECTION", "TRAP_BREAKOUT", "STOP_HUNT"]:
        return None
    if flow_trade_type == "EARLY_REVERSAL_ENTRY" and not bool(row.get("choch", 0)):
        return None
    if flow_trade_type == "EXHAUSTION_FADE" and bool(row.get("bos", 0)):
        return None

    risk_adj = config.regime.flow_risk_multiplier * 0.25
    trade = _build_trade_base(row, config, equity, risk_adj=risk_adj, is_exploratory=True)
    if trade:
        trade["system"] = "FLOW_EXPLORATORY"
        trade["data_type"] = "EXPLORATORY_DATA"
    return trade

def _build_trade(row, config, equity):
    """Legacy compatibility wrapper for external modules like replay_engine."""
    # Priority routing: System A (Alpha) first, then System B (Flow)
    alpha_candidate = _build_alpha_trade(row, config, equity)
    if alpha_candidate:
        return alpha_candidate
    return _build_flow_trade(row, config, equity)

def _build_trade_base(row, config, equity, risk_adj=1.0, is_exploratory=False):
    """Common trade construction logic with adaptive filters"""
    rr_ratio = config.risk.rr_ratio
    if not is_exploratory and not bool(row.get("trade_allowed", True)):
        return None

    hour = pd.to_datetime(row["time"]).hour
    state = row.get("market_state", "UNKNOWN")
    quality = row.get("quality", "UNKNOWN")
    flow_trade_type = str(row.get("flow_trade_type", "NONE"))

    if float(row.get("confirm_score", 0)) > 100: return None

    if state == "CHOPPY":  # Low Volatility: Alpha stays strict, Flow is more permissive
        if not is_exploratory and quality != "ELITE":
            return None
        score_floor = 80 if not is_exploratory else config.regime.flow_min_confirm_score
        if is_exploratory:
            risk_adj *= config.regime.flow_choppy_risk_multiplier
    elif state == "VOLATILE":  # High Volatility: Alpha stays elite-only, Flow can still participate
        if not is_exploratory and (quality != "ELITE" or not (bool(row.get("major_support", 0)) or bool(row.get("major_resistance", 0)))):
            return None
        score_floor = 90 if not is_exploratory else config.regime.flow_min_confirm_score
        if is_exploratory:
            risk_adj *= config.regime.flow_volatility_risk_multiplier
        else:
            risk_adj *= 0.5
    else:  # Medium Volatility (RANGING, TRENDING)
        if state == "RANGING":
            if not is_exploratory and quality not in ["ELITE", "HIGH"]:
                return None
            score_floor = 65 if not is_exploratory else config.regime.flow_min_confirm_score
        else:  # Default Trending
            if not is_exploratory and quality != "ELITE":
                return None
            score_floor = (
                config.regime.alpha_min_score_elite_trending
                if not is_exploratory
                else config.regime.flow_min_confirm_score
            )

    if float(row.get("confirm_score", 0)) < score_floor:
        return None

    # 3. Session-Aware Adaptive Restrictions (Bypass for Exploratory System B)
    if not is_exploratory and config.regime.adaptive_ny_guard:
        is_new_york = 13 <= hour <= 20
        is_late_session = hour > 20 or hour < 2

        if is_new_york:
            score_floor += 10 
            if state in ["VOLATILE", "TRENDING"] and quality != "ELITE":
                return None
        elif is_late_session:
            if quality != "ELITE" or float(row.get("confirm_score", 0)) < 90:
                return None

    # 4. Setup-Level Weighting & Priority
    setup_key = f"{row.get('setup')}_{quality}_{state}"
    priority_mult = config.regime.setup_weights.get(setup_key, 1.0)
    
    if priority_mult == 0.0:
        return None
    
    risk_adj *= priority_mult

    # 5. Timeframe & Conviction Filters
    # (score_floor checked in caller)

    signal = row["confirmed_signal"]
    spread_price = float(row.get("spread", 0) or 0) * config.market.point_size
    slippage_price = config.backtest.slippage_points * config.market.point_size
    total_cost = spread_price + slippage_price
    entry_price = float(row["entry_price"])
    stop_loss = float(row["stop_loss"])
    take_profit = float(row["take_profit"])
    if pd.isna(entry_price) or pd.isna(stop_loss) or pd.isna(take_profit):
        return None

    if signal == "buy":
        if is_exploratory:
            stop_loss = min(stop_loss, entry_price - 0.01)
        executed_entry = entry_price + spread_price + slippage_price
        risk_distance = executed_entry - stop_loss
        if is_exploratory:
            take_profit = executed_entry + (risk_distance * rr_ratio)
        reward_distance = take_profit - executed_entry
    else:
        if is_exploratory:
            stop_loss = max(stop_loss, entry_price + 0.01)
        executed_entry = entry_price - spread_price - slippage_price
        risk_distance = stop_loss - executed_entry
        if is_exploratory:
            take_profit = executed_entry - (risk_distance * rr_ratio)
        reward_distance = executed_entry - take_profit

    if risk_distance <= 0 or reward_distance <= 0:
        return None

    # --- SPREAD & SLIPPAGE RESILIENCE GUARD ---
    cost_ratio = total_cost / risk_distance if risk_distance > 0 else 1.0
    max_allowed_cost_ratio = getattr(config.risk, 'max_cost_ratio', 0.20)
    if cost_ratio > max_allowed_cost_ratio:
        return None

    if hasattr(config.backtest, 'dynamic_risk_scaling') and config.backtest.dynamic_risk_scaling:
        if equity <= 100:
            risk_pct = 0.005
        elif equity <= 1000:
            risk_pct = 0.0075
        elif equity <= 10000:
            risk_pct = 0.01
        else:
            risk_pct = config.backtest.risk_per_trade
    else:
        risk_pct = config.backtest.risk_per_trade

    if hasattr(config.risk, 'use_atr_sizing') and config.risk.use_atr_sizing:
        atr_value = float(row.get("atr", 1.0))
        risk_amount = equity * config.risk.atr_risk_per_unit * atr_value * float(row.get("risk_multiplier", 1.0) or 1.0) * risk_adj
    else:
        risk_amount = equity * risk_pct * float(row.get("risk_multiplier", 1.0) or 1.0) * risk_adj

    return {
        "signal_time": row["time"],
        "side": signal,
        "setup": row.get("setup", "UNKNOWN"),
        "signal_label": row.get("signal", "UNKNOWN"),
        "quality": row.get("quality", "UNKNOWN"),
        "market_state": row.get("market_state", "UNKNOWN"),
        "market_regime": row.get("market_regime", "UNKNOWN"),
        "session": _get_session_label(hour),
        "h1_bias": row.get("h1_bias", "UNKNOWN"),
        "h1_alignment": row.get("h1_alignment", 0),
        "entry_price": entry_price,
        "executed_entry": executed_entry,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_distance": risk_distance,
        "reward_distance": reward_distance,
        "risk_multiplier": float(row.get("risk_multiplier", 1.0) or 1.0),
        "risk_amount": risk_amount,
        "confirm_score": row.get("confirm_score", 0),
        "is_first_breakout": bool(row.get("is_first_breakout", False)),
        "flow_trade_type": flow_trade_type,
    }

def _resolve_trade(trade, candle):
    """
    Resolves a trade and manages Break-Even logic.
    Returns (Outcome, ExitPrice, was_be_stop)
    """
    high = float(candle["high"])
    low = float(candle["low"])
    
    # 1. Update Break-Even status
    # HIGH volatility uses 1.2R for faster locking, MEDIUM/LOW uses 1.5R for breathing room.
    if not trade.get("is_be", False):
        be_threshold = 1.2 if trade.get("market_state") == "VOLATILE" else 1.5
        
        if trade["side"] == "buy" and high >= (trade["executed_entry"] + (trade["risk_distance"] * be_threshold)):
            trade["stop_loss"] = trade["executed_entry"]
            trade["is_be"] = True
        elif trade["side"] == "sell" and low <= (trade["executed_entry"] - (trade["risk_distance"] * be_threshold)):
            trade["stop_loss"] = trade["executed_entry"]
            trade["is_be"] = True

    if trade["side"] == "buy":
        if low <= trade["stop_loss"]:
            # If is_be is true, this is a BE exit, not a full LOSS
            outcome = "BE" if trade.get("is_be", False) else "LOSS"
            return outcome, trade["stop_loss"], trade.get("is_be", False)
        if high >= trade["take_profit"]:
            return "WIN", trade["take_profit"], False
    else:
        if high >= trade["stop_loss"]:
            outcome = "BE" if trade.get("is_be", False) else "LOSS"
            return outcome, trade["stop_loss"], trade.get("is_be", False)
        if low <= trade["take_profit"]:
            return "WIN", trade["take_profit"], False

    return None, None, False


def _empty_trades_frame():
    return pd.DataFrame(
        columns=[
            "signal_time",
            "side",
            "setup",
            "signal_label",
            "quality",
            "market_state",
            "market_regime",
            "h1_bias",
            "h1_alignment",
            "entry_price",
            "executed_entry",
            "stop_loss",
            "take_profit",
            "risk_distance",
            "reward_distance",
            "risk_multiplier",
            "risk_amount",
            "confirm_score",
            "exit_time",
            "exit_price",
            "result",
            "reentry_count",
            "pnl",
            "equity_after_trade",
            "is_first_breakout",
        ]
    )


def _build_summary(trades_df, config, ending_balance, max_drawdown_pct, skipped_overlap, label):
    closed_trades = (
        trades_df[trades_df["result"].isin(["WIN", "LOSS", "BE"])]
        if not trades_df.empty
        else pd.DataFrame()
    )
    wins = int((closed_trades["result"] == "WIN").sum()) if not closed_trades.empty else 0
    losses = int((closed_trades["result"] == "LOSS").sum()) if not closed_trades.empty else 0
    gross_profit = (
        float(closed_trades.loc[closed_trades["pnl"] > 0, "pnl"].sum())
        if not closed_trades.empty
        else 0.0
    )
    gross_loss = (
        float(closed_trades.loc[closed_trades["pnl"] < 0, "pnl"].sum())
        if not closed_trades.empty
        else 0.0
    )
    profit_factor = round(gross_profit / abs(gross_loss), 2) if gross_loss else 0.0
    win_rate = (
        round((wins / len(closed_trades)) * 100, 2)
        if len(closed_trades)
        else 0.0
    )
    # True Win Rate: Accuracy of the signal itself (excluding BE scratches)
    true_win_rate = (
        round((wins / (wins + losses)) * 100, 2)
        if (wins + losses) > 0
        else 0.0
    )

    # Calculate Sample Duration for Trade Frequency
    duration_days = 0
    if not closed_trades.empty:
        start_dt = pd.to_datetime(closed_trades["signal_time"]).min()
        end_dt = pd.to_datetime(closed_trades["exit_time"]).max()
        if pd.notna(start_dt) and pd.notna(end_dt):
            duration_days = max(1, (end_dt - start_dt).days)
    trades_per_day = round(len(closed_trades) / duration_days, 2) if duration_days > 0 else 0

    return pd.DataFrame(
        [
            {
                "label": label,
                "starting_balance": config.backtest.starting_balance,
                "ending_balance": round(ending_balance, 2),
                "net_pnl": round(ending_balance - config.backtest.starting_balance, 2),
                "closed_trades": len(closed_trades),
                "wins": wins,
                "losses": losses,
                "open_trades": int((trades_df["result"] == "OPEN").sum()) if not trades_df.empty else 0,
                "win_rate_pct": win_rate,
                "true_win_rate_pct": true_win_rate,
                "profit_factor": profit_factor,
                "max_drawdown_pct": round(max_drawdown_pct, 2),
                "skipped_overlap_signals": skipped_overlap,
                "trades_per_day": trades_per_day,
            }
        ]
    )


def _calculate_performance_score(history):
    """Calculates a selection score based on recent trade history."""
    if not history:
        return 1.0  # Neutral starting point
    
    wins = sum(1 for t in history if t['result'] == 'WIN')
    pfs = [t['pnl'] for t in history]
    gross_profit = sum(p for p in pfs if p > 0)
    gross_loss = abs(sum(p for p in pfs if p < 0))
    
    win_rate = wins / len(history)
    pf = (gross_profit / gross_loss) if gross_loss > 0 else (2.0 if gross_profit > 0 else 1.0)
    
    # Score = WinRate + normalized Profit Factor
    return win_rate + (min(pf, 3.0) / 3.0)

def run_backtest_frame(df, config, label="full_period", mode="COMBINED"):
    df = df.sort_values("time").reset_index(drop=True)

    equity = config.backtest.starting_balance
    peak_equity = equity
    max_drawdown_pct = 0.0
    active_trades = []
    trade_records = []
    skipped_overlap = 0
    reentry_candidates = [] 
    signal_entry_counts = {} # signal_time -> count
    alpha_history = []
    flow_history = []
    ny_trade_date = None
    ny_flow_count = 0

    for _, row in df.iterrows():
        current_time = row["time"]

        next_active_trades = []
        for active_trade in active_trades:
            outcome, exit_price, was_be_stop = _resolve_trade(active_trade, row)
            if outcome:
                if outcome == "WIN":
                    r_multiple = (
                        active_trade["reward_distance"]
                        / active_trade["risk_distance"]
                    )
                    pnl = (
                        active_trade["risk_amount"] * r_multiple
                        - config.backtest.commission_per_trade
                    )
                elif outcome == "BE":
                    pnl = -config.backtest.commission_per_trade # Preserve capital
                else:
                    pnl = (
                        -active_trade["risk_amount"]
                        - config.backtest.commission_per_trade
                    )

                equity += pnl
                peak_equity = max(peak_equity, equity)
                if peak_equity > 0:
                    drawdown_pct = ((peak_equity - equity) / peak_equity) * 100
                    max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

                # Only allow re-entry if the market isn't CHOPPY
                if was_be_stop and row["market_state"] != "CHOPPY":
                    reentry_candidates.append(active_trade)

                # Track history for dynamic priority
                if active_trade.get("system") == "ALPHA":
                    alpha_history = (alpha_history + [{"result": outcome, "pnl": pnl}])[-config.regime.priority_lookback:]
                elif active_trade.get("system") == "FLOW":
                    flow_history = (flow_history + [{"result": outcome, "pnl": pnl}])[-config.regime.priority_lookback:]

                trade_records.append(
                    {
                        **active_trade,
                        "exit_time": current_time,
                        "exit_price": exit_price,
                        "reentry_count": signal_entry_counts.get(active_trade["signal_time"], 1),
                        "result": outcome,
                        "pnl": round(pnl, 2),
                        "equity_after_trade": round(equity, 2),
                    }
                )
            else:
                next_active_trades.append(active_trade)

        active_trades = next_active_trades

        # --- RE-ENTRY LOGIC ---
        # Check if price has reclaimed original entry after a BE stop
        still_watching_reentry = []
        for candidate in reentry_candidates:
            reentry_triggered = False
            if candidate["side"] == "buy" and float(row["close"]) > candidate["executed_entry"]:
                reentry_triggered = True
            elif candidate["side"] == "sell" and float(row["close"]) < candidate["executed_entry"]:
                reentry_triggered = True
            
            # Only allow ONE re-entry per signal and ONLY for ELITE quality.
            # Re-entering MEDIUM trades often leads to the 30% win-rate drawdowns you saw.
            sig_id = candidate["signal_time"]
            if reentry_triggered and signal_entry_counts.get(sig_id, 1) < 2 and candidate["quality"] == "ELITE":
                # Create a fresh copy for the re-entry
                new_trade = candidate.copy()
                new_trade["is_be"] = False # Reset BE status
                new_trade["signal_label"] += "_REENTRY"
                signal_entry_counts[sig_id] = 2
                active_trades.append(new_trade)
            else:
                still_watching_reentry.append(candidate)
        reentry_candidates = still_watching_reentry

        if (
            active_trades
            and not config.backtest.allow_overlapping_positions
        ):
            if row["confirmed_signal"] in {"buy", "sell"}:
                skipped_overlap += 1
            continue

        if row["confirmed_signal"] not in {"buy", "sell"}:
            continue
            
        signal_entry_counts[row["time"]] = 1
        
        # Dual-System Decision Engine
        alpha_candidate = _build_alpha_trade(row, config, equity)
        flow_candidate = _build_flow_trade(row, config, equity)
        
        candidate_trade = None
        if mode == "ALPHA":
            candidate_trade = alpha_candidate
        elif mode == "FLOW":
            candidate_trade = flow_candidate
        else: # COMBINED: Alpha takes priority, Flow is exploratory secondary
            candidate_trade = alpha_candidate if alpha_candidate else flow_candidate

        if candidate_trade is None:
            continue

        if candidate_trade.get("system") != "ALPHA":
            flow_open = [t for t in active_trades if t.get("system") != "ALPHA"]
            alpha_open = [t for t in active_trades if t.get("system") == "ALPHA"]
            max_flow_open = 1 if alpha_open else int(row.get("flow_max_open_trades", 3) or 3)
            if len(flow_open) >= max_flow_open:
                skipped_overlap += 1
                continue
            if candidate_trade.get("flow_trade_type") == "EXHAUSTION_FADE" and any(t.get("flow_trade_type") == "EXHAUSTION_FADE" for t in flow_open):
                skipped_overlap += 1
                continue
            if candidate_trade.get("flow_trade_type") == "MICRO_RETRACEMENT_REENTRY" and any(t.get("flow_trade_type") == "MICRO_RETRACEMENT_REENTRY" for t in flow_open):
                skipped_overlap += 1
                continue

        # Session Capping for NY Flow Trades
        hour = pd.to_datetime(current_time).hour
        if not candidate_trade.get("system") == "ALPHA" and 13 <= hour < 18:
            trade_date = pd.to_datetime(current_time).date()
            if ny_trade_date != trade_date:
                ny_trade_date = trade_date
                ny_flow_count = 0
            
            if ny_flow_count >= getattr(config.regime, "max_ny_flow_trades", 1):
                continue
            ny_flow_count += 1

        candidate_trade["system_mode"] = mode
        active_trades.append(candidate_trade)

    for active_trade in active_trades:
        trade_records.append(
            {
                **active_trade,
                "exit_time": pd.NaT,
                "exit_price": None,
                "result": "OPEN",
                "pnl": 0.0,
                "equity_after_trade": round(equity, 2),
            }
        )

    trades_df = pd.DataFrame(trade_records) if trade_records else _empty_trades_frame()
    summary = _build_summary(
        trades_df=trades_df,
        config=config,
        ending_balance=equity,
        max_drawdown_pct=max_drawdown_pct,
        skipped_overlap=skipped_overlap,
        label=label,
    )
    return trades_df, summary


def run(config, in_sample_end: str | None = None, oos_start: str | None = None):
    # 1. Validate dates FIRST
    oos_valid = False
    if in_sample_end and oos_start:
        try:
            is_dt, oos_dt = pd.to_datetime(in_sample_end), pd.to_datetime(oos_start)
            if is_dt >= oos_dt:
                logger.warning("⚠️ Invalid OOS dates: IS end must be before OOS start. Skipping OOS mode.")
            else:
                oos_valid = True
        except (ValueError, TypeError):
            logger.error("❌ Date Parse Error: Could not parse in_sample_end='%s' or oos_start='%s'. "
                         "Please use YYYY-MM-DD format.", in_sample_end, oos_start)

    df = pd.read_csv(config.paths.trade_setups, parse_dates=["time"], low_memory=False)

    # 3. Prepare Multi-System Tasks
    system_modes = [
        ("ALPHA", config.paths.backtest_alpha_trades, config.paths.backtest_alpha_summary),
        ("FLOW", config.paths.backtest_flow_trades, config.paths.backtest_flow_summary),
        ("COMBINED", config.paths.backtest_trades, config.paths.backtest_summary)
    ]
    
    results = {}

    # 4. Execute backtests per system
    for mode, t_path, s_path in system_modes:
        trades_df, summary = run_backtest_frame(df, config, label=mode, mode=mode)
        trades_df.to_csv(t_path, index=False)
        summary.to_csv(s_path, index=False)
        results[mode] = {"summary": summary, "trades": trades_df}

    # Final comparison summary
    final_comparison = pd.concat([r["summary"] for r in results.values()])
    comparison_path = config.paths.backtest_summary.parent / "system_comparison.csv"
    final_comparison.to_csv(comparison_path, index=False)

    # --- CONSOLE SUMMARY DISPLAY ---
    print("\n" + "="*95)
    print(f"{'DUAL-SYSTEM PERFORMANCE COMPARISON':^95}")
    print("="*95)
    display_cols = ["label", "net_pnl", "win_rate_pct", "true_win_rate_pct", "profit_factor", "max_drawdown_pct", "closed_trades"]
    print(final_comparison[display_cols].to_string(index=False))

    if oos_valid:
        print("\n" + "-"*95)
        print(f"{'OVERFITTING VALIDATION: IN-SAMPLE VS OUT-OF-SAMPLE (COMBINED)':^95}")
        print("-"*95)
        
        _, is_sum = run_backtest_frame(df[df["time"] <= is_dt], config, label="IN_SAMPLE", mode="COMBINED")
        _, oos_sum = run_backtest_frame(df[df["time"] >= oos_dt], config, label="OUT_OF_SAMPLE", mode="COMBINED")
        
        is_oos_comp = pd.concat([is_sum, oos_sum])
        print(is_oos_comp[display_cols].to_string(index=False))
        
        # Stability Check: Calculate degradation
        is_wr = is_sum["true_win_rate_pct"].iloc[0]
        oos_wr = oos_sum["true_win_rate_pct"].iloc[0]
        wr_degradation = ((is_wr - oos_wr) / is_wr) * 100 if is_wr > 0 else 0
        
        print(f"\nStability Check: Win Rate Degradation (IS -> OOS): {wr_degradation:.2f}%")
        if wr_degradation > 20:
            print("🚨 WARNING: High degradation detected. Strategy may be overfitted.")
        else:
            print("✅ PASS: Strategy shows stable OOS performance.")

        is_sum.to_csv(config.paths.backtest_summary.parent / "in_sample_summary.csv", index=False)
        oos_sum.to_csv(config.paths.backtest_summary.parent / "out_of_sample_summary.csv", index=False)

    print("="*95 + "\n")
    logger.info("Dual-system backtest complete.")
