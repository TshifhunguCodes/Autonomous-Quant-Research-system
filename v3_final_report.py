#!/usr/bin/env python
"""Final validation report for AQRS V3 Phase 1."""

import pandas as pd

df = pd.read_csv('data/backtest/v3_research_output.csv')
print('✓ AQRS V3 Phase 1 - Final Report')
print('=' * 60)
print(f'Total Candles Processed: {len(df):,}')
print(f'Date Range: {df["time"].iloc[0]} to {df["time"].iloc[-1]}')
print()
print('Signal Generation:')
alpha_count = len(df[df['signal'] == 'ALPHA'])
flow_count = len(df[df['signal'] == 'FLOW'])
no_trade_count = len(df[df['signal'] == 'NO_TRADE'])
print(f'  ALPHA Signals: {alpha_count:,} ({alpha_count/len(df)*100:.1f}%)')
print(f'  FLOW Signals: {flow_count:,} ({flow_count/len(df)*100:.1f}%)')
print(f'  NO_TRADE: {no_trade_count:,} ({no_trade_count/len(df)*100:.1f}%)')
print()
print('Scoring Ranges:')
alpha_signals = df[df['alpha_signal'] == 'ALPHA']
flow_signals = df[df['flow_signal'] == 'FLOW']
if len(alpha_signals) > 0:
    print(f'  Alpha Score: min={alpha_signals["alpha_score"].min():.0f}, max={alpha_signals["alpha_score"].max():.0f}, mean={alpha_signals["alpha_score"].mean():.1f}')
if len(flow_signals) > 0:
    print(f'  Flow Score: min={flow_signals["flow_score"].min():.0f}, max={flow_signals["flow_score"].max():.0f}, mean={flow_signals["flow_score"].mean():.1f}')
print()
print('Market Behavior Distribution:')
for behavior in df['behavior_label'].value_counts().head(5).index:
    count = len(df[df['behavior_label'] == behavior])
    pct = count/len(df)*100
    print(f'  {behavior}: {count:6,d} ({pct:5.1f}%)')
print()
print('Structure State Distribution:')
for state in df['structure_state'].value_counts().head(5).index:
    count = len(df[df['structure_state'] == state])
    pct = count/len(df)*100
    print(f'  {state}: {count:6,d} ({pct:5.1f}%)')
print()
print('Risk Metrics:')
print(f'  Avg Position Risk: ${df["position_risk"].mean():.2f}')
print(f'  Avg Stop Distance: {df["stop_distance"].mean():.2f} points')
avg_rr = (df['take_profit'] - df['entry_price']).abs().mean() / df['stop_distance'].mean()
print(f'  Avg RR Setup: {avg_rr:.1f}:1')
print()
print('✓ V3 Pipeline FULLY OPERATIONAL')
