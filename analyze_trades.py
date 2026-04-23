import pandas as pd

# Load the trades data
df = pd.read_csv('data/backtest/backtest_trades.csv', parse_dates=['signal_time'])

# Extract hour for session
df['hour'] = df['signal_time'].dt.hour

# Categorize sessions (rough forex sessions)
def get_session(hour):
    if 0 <= hour < 8:
        return 'Asian'
    elif 8 <= hour < 16:
        return 'European'
    else:
        return 'US'

df['session'] = df['hour'].apply(get_session)

# Filter to closed trades
closed = df[df['result'].isin(['WIN', 'LOSS'])].copy()

# Group by session, quality, market_state
breakdown = closed.groupby(['session', 'quality', 'market_state', 'result']).size().unstack(fill_value=0)

# Calculate win rate
breakdown['total'] = breakdown['WIN'] + breakdown['LOSS']
breakdown['win_rate'] = (breakdown['WIN'] / breakdown['total'] * 100).round(2)

# Sort by total trades descending
breakdown = breakdown.sort_values('total', ascending=False)

print("Breakdown of Wins/Losses by Session, Quality, and Market State:")
print(breakdown)

# Overall by each dimension
print("\nBy Session:")
session_summary = closed.groupby('session')['result'].value_counts().unstack().fillna(0)
session_summary['total'] = session_summary.sum(axis=1)
session_summary['win_rate'] = (session_summary['WIN'] / (session_summary['WIN'] + session_summary['LOSS']) * 100).round(2)
print(session_summary)

print("\nBy Quality:")
quality_summary = closed.groupby('quality')['result'].value_counts().unstack().fillna(0)
quality_summary['total'] = quality_summary.sum(axis=1)
quality_summary['win_rate'] = (quality_summary['WIN'] / (quality_summary['WIN'] + quality_summary['LOSS']) * 100).round(2)
print(quality_summary)

print("\nBy Market State:")
state_summary = closed.groupby('market_state')['result'].value_counts().unstack().fillna(0)
state_summary['total'] = state_summary.sum(axis=1)
state_summary['win_rate'] = (state_summary['WIN'] / (state_summary['WIN'] + state_summary['LOSS']) * 100).round(2)
print(state_summary)