import pandas as pd
from pathlib import Path
root = Path('c:/HTLM Clones/Autonomous-Quant-Research-system')
dec = pd.read_csv(root / 'data' / 'replay' / 'replay_decisions.csv', parse_dates=['time'])
trades = pd.read_csv(root / 'data' / 'replay' / 'replay_trades.csv', parse_dates=['signal_time', 'exit_time'])
dec_sel = dec[(dec['time'] >= '2026-04-01') & (dec['time'] <= '2026-04-24')]
print('decisions selected', len(dec_sel), 'confirmed', len(dec_sel[dec_sel['confirmed_signal'] != 'no_trade']))
print(dec_sel[['time', 'confirmed_signal', 'quality', 'confirm_score']].tail(20).to_dict('records'))
print('trades total', len(trades), 'trades in range', len(trades[(trades['signal_time'] >= '2026-04-01') & (trades['signal_time'] <= '2026-04-24')]))
print(trades[['signal_time', 'system', 'quality']].tail(20).to_dict('records'))
