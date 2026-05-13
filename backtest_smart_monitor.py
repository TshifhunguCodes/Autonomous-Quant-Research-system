"""
Backtest Smart Monitor Impact
Simulates how the Smart Monitor would have affected historical trades
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from smart_monitor.quality_scorer import TradeQualityScorer
from smart_monitor.simple_learner import SimpleTradeLearner
from smart_monitor.adaptive_filter import AdaptiveFilter

def load_backtest_data():
    """Load historical backtest data"""
    flow_trades_path = Path("data/backtest/flow_trades.csv")
    flow_summary_path = Path("data/backtest/flow_summary.csv")
    
    if not flow_trades_path.exists():
        print("❌ Flow trades data not found")
        return None, None
    
    trades = pd.read_csv(flow_trades_path)
    summary = pd.read_csv(flow_summary_path) if flow_summary_path.exists() else None
    
    return trades, summary

def simulate_smart_monitor(trades_df):
    """
    Simulate Smart Monitor filtering on historical trades
    Returns filtered trades and comparison statistics
    """
    scorer = TradeQualityScorer()
    learner = SimpleTradeLearner()
    adaptive = AdaptiveFilter()
    
    # Initialize tracking
    original_trades = len(trades_df)
    allowed_trades = []
    blocked_trades = []
    blocked_reasons = []
    
    # Track cumulative PnL for both scenarios
    original_pnl = []
    filtered_pnl = []
    
    for idx, row in trades_df.iterrows():
        # Convert row to signal dict
        signal = row.to_dict()
        
        # Ensure required fields exist
        if 'time' not in signal:
            continue
        
        # Calculate quality score
        quality_score = scorer.score_trade(signal)
        signal['smart_quality_score'] = quality_score
        
        # Get ML prediction
        ml_allow, ml_prob, ml_reason = learner.should_allow_trade(signal)
        
        # Get adaptive filter decision
        adaptive_allow, adaptive_reason, adaptive_lot_mult = adaptive.get_adaptive_filters(signal)
        
        # Combine decisions (same logic as smart_monitor.py)
        allow = True
        reasons = []
        lot_multiplier = 1.0
        
        # Quality check
        if quality_score < 55:
            allow = False
            reasons.append(f"LOW_QUALITY ({quality_score:.0f})")
        elif quality_score < 65:
            lot_multiplier *= 0.5
        elif quality_score < 75:
            lot_multiplier *= 0.8
        
        # ML check
        if not ml_allow:
            allow = False
            reasons.append(ml_reason)
        
        # Adaptive check
        if not adaptive_allow:
            allow = False
            reasons.append(adaptive_reason)
        
        # Apply lot multipliers
        if quality_score >= 80:
            lot_multiplier *= 1.2
        elif quality_score >= 70:
            lot_multiplier *= 1.0
        elif quality_score >= 60:
            lot_multiplier *= 0.7
        else:
            lot_multiplier *= 0.4
        
        # Get PnL (estimate from available data)
        pnl = row.get('pnl', 0)
        if pnl == 0 and 'profit' in row:
            pnl = row.get('profit', 0)
        
        # Track original scenario
        original_pnl.append(pnl)
        
        # Track filtered scenario
        if allow:
            allowed_trades.append(row)
            filtered_pnl.append(pnl * lot_multiplier)
        else:
            blocked_trades.append(row)
            blocked_reasons.append("; ".join(reasons))
            filtered_pnl.append(0)  # Trade was blocked, no PnL
    
    # Create results DataFrame
    results = {
        'original': {
            'total_trades': original_trades,
            'total_pnl': sum(original_pnl),
            'avg_pnl': np.mean(original_pnl) if original_pnl else 0,
            'win_rate': sum(1 for p in original_pnl if p > 0) / len(original_pnl) * 100 if original_pnl else 0,
        },
        'filtered': {
            'total_trades': len(allowed_trades),
            'total_pnl': sum(filtered_pnl),
            'avg_pnl': np.mean(filtered_pnl) if filtered_pnl else 0,
            'win_rate': sum(1 for p in filtered_pnl if p > 0) / len(filtered_pnl) * 100 if filtered_pnl else 0,
        },
        'blocked': {
            'total_blocked': len(blocked_trades),
            'blocked_reasons': blocked_reasons[:10],  # First 10 reasons
        }
    }
    
    return results, allowed_trades, blocked_trades

def print_comparison(results):
    """Print comparison between original and filtered results"""
    print("\n" + "=" * 70)
    print("SMART MONITOR BACKTEST COMPARISON")
    print("=" * 70)
    
    orig = results['original']
    filt = results['filtered']
    block = results['blocked']
    
    print(f"\n{'Metric':<25} {'Original':>12} {'Filtered':>12} {'Change':>12}")
    print("-" * 65)
    print(f"{'Total Trades':<25} {orig['total_trades']:>12} {filt['total_trades']:>12} {filt['total_trades'] - orig['total_trades']:>+12}")
    print(f"{'Total PnL':<25} {orig['total_pnl']:>12.2f} {filt['total_pnl']:>12.2f} {filt['total_pnl'] - orig['total_pnl']:>+12.2f}")
    print(f"{'Average PnL per Trade':<25} {orig['avg_pnl']:>12.2f} {filt['avg_pnl']:>12.2f} {filt['avg_pnl'] - orig['avg_pnl']:>+12.2f}")
    print(f"{'Win Rate (%)':<25} {orig['win_rate']:>12.1f} {filt['win_rate']:>12.1f} {filt['win_rate'] - orig['win_rate']:>+12.1f}")
    
    print(f"\n📊 Trades Blocked: {block['total_blocked']}")
    print(f"📊 Trades Allowed: {filt['total_trades']}")
    print(f"📊 Block Rate: {block['total_blocked'] / (block['total_blocked'] + filt['total_trades']) * 100:.1f}%")
    
    if block['blocked_reasons']:
        print(f"\n🚫 Common Block Reasons:")
        for reason in block['blocked_reasons'][:5]:
            print(f"   - {reason}")
    
    # Calculate improvement metrics
    if orig['total_pnl'] != 0:
        pnl_improvement = (filt['total_pnl'] - orig['total_pnl']) / abs(orig['total_pnl']) * 100
        print(f"\n📈 PnL Improvement: {pnl_improvement:+.1f}%")
    
    if orig['win_rate'] != 0:
        win_rate_improvement = filt['win_rate'] - orig['win_rate']
        print(f"📈 Win Rate Improvement: {win_rate_improvement:+.1f} percentage points")

def main():
    """Main backtest function"""
    print("Loading backtest data...")
    trades_df, summary_df = load_backtest_data()
    
    if trades_df is None:
        print("\n⚠️  No backtest data available for Smart Monitor simulation.")
        print("   The system needs historical trade data to run backtests.")
        print("\n   To generate backtest data, run:")
        print("   python main_v3.py --mode backtest --run-days 30")
        return
    
    print(f"Loaded {len(trades_df)} historical FLOW trades")
    print(f"Date range: {trades_df['time'].min()} to {trades_df['time'].max()}")
    
    print("\nRunning Smart Monitor simulation...")
    results, allowed, blocked = simulate_smart_monitor(trades_df)
    
    print_comparison(results)
    
    # Save detailed results
    output_path = Path("data/backtest/smart_monitor_backtest.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create comparison report
    report_data = []
    for i, (_, row) in enumerate(trades_df.iterrows()):
        if i < len(allowed):
            status = "ALLOWED"
        else:
            status = "BLOCKED"
        report_data.append({
            'time': row.get('time', ''),
            'original_pnl': row.get('pnl', 0),
            'status': status,
            'quality_score': row.get('smart_quality_score', 'N/A')
        })
    
    report_df = pd.DataFrame(report_data)
    report_df.to_csv(output_path, index=False)
    print(f"\n📄 Detailed results saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    main()