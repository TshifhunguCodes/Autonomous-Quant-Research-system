import streamlit as st
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
from functools import lru_cache

st.set_page_config(page_title="AQRS V2 Dashboard", layout="wide", initial_sidebar_state="expanded")

API_URL = "http://127.0.0.1:8001"

# Data cache for better performance
@st.cache_data(ttl=2)
def fetch_data_cached(endpoint, params_str):
    """Cached data fetching with TTL of 2 seconds"""
    try:
        # Parse params_str back to dict for API call
        import json
        params = json.loads(params_str) if params_str else None
        response = requests.get(f"{API_URL}/{endpoint}", params=params, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.toast(f"⚠️ {endpoint} unavailable", icon="⚠️")
        if endpoint in ["signals", "chart_data"]:
            return []
        return {}

def fetch_data(endpoint, params=None):
    """Wrapper for cached data fetching"""
    import json
    params_str = json.dumps(params) if params else ""
    return fetch_data_cached(endpoint, params_str)

# Sidebar Navigation
st.sidebar.title("🧠 AQRS V2 Dashboard")
page = st.sidebar.selectbox("Navigate", ["Main Dashboard", "Live Trades", "System Comparison", "Audit Logs"])
refresh_rate = st.sidebar.slider("Refresh Rate (sec)", 1, 5, 2)

def initialize_session_state():
    if "mode" not in st.session_state:
        st.session_state.mode = "LIVE"
    if "symbol" not in st.session_state:
        st.session_state.symbol = "XAUUSD" # Default, could be fetched from config
    if "refresh_rate" not in st.session_state:
        st.session_state.refresh_rate = 2 # UI refresh rate for LIVE mode

    # Replay specific states
    if "replay_running" not in st.session_state:
        st.session_state.replay_running = False
    if "replay_index" not in st.session_state:
        st.session_state.replay_index = 0
    if "replay_speed" not in st.session_state:
        st.session_state.replay_speed = 1.0 # candles per second
    if "replay_start_date" not in st.session_state:
        st.session_state.replay_start_date = (datetime.now() - timedelta(days=7)).date()
    if "replay_end_date" not in st.session_state:
        st.session_state.replay_end_date = datetime.now().date()
    if "replay_total_candles" not in st.session_state:
        st.session_state.replay_total_candles = 0
    if "replay_data_loaded" not in st.session_state:
        st.session_state.replay_data_loaded = False

    # Backtest specific states
    if "backtest_running" not in st.session_state:
        st.session_state.backtest_running = False
    if "backtest_start_date" not in st.session_state:
        st.session_state.backtest_start_date = (datetime.now() - timedelta(days=365)).date()
    if "backtest_end_date" not in st.session_state:
        st.session_state.backtest_end_date = datetime.now().date()

# --- UI Components ---
def render_top_bar():
    st.sidebar.title("🧠 AQRS V2 Control Center")

    st.session_state.mode = st.sidebar.radio(
        "Select Mode", ["LIVE", "REPLAY", "BACKTEST"],
        index=["LIVE", "REPLAY", "BACKTEST"].index(st.session_state.mode)
    )
    st.session_state.symbol = st.sidebar.selectbox(
        "Symbol", ["XAUUSD"], # Hardcoded for now, can be dynamic
        index=["XAUUSD"].index(st.session_state.symbol)
    )

    if st.session_state.mode == "REPLAY":
        st.sidebar.subheader("Replay Controls")
        st.session_state.replay_speed = st.sidebar.slider(
            "Replay Speed (candles/sec)", 0.1, 5.0, st.session_state.replay_speed, 0.1
        )
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.session_state.replay_start_date = st.date_input(
                "Start Date", st.session_state.replay_start_date, key="replay_start_date_input"
            )
        with col2:
            st.session_state.replay_end_date = st.date_input(
                "End Date", st.session_state.replay_end_date, key="replay_end_date_input"
            )

        st.sidebar.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
        with col_btn1:
            if st.button("▶️ Start Replay", width='stretch', disabled=st.session_state.replay_running):
                st.session_state.replay_running = True
                st.session_state.replay_index = 0 # Reset index on start
                st.session_state.replay_data_loaded = False # Mark data as not loaded yet
                # Send control to backend to initialize replay
                response = fetch_data("replay_control", {
                    "action": "start",
                    "start_date": st.session_state.replay_start_date.isoformat(),
                    "end_date": st.session_state.replay_end_date.isoformat()
                })
                if response:
                    st.session_state.replay_total_candles = response.get("total_candles", 0)
                    st.session_state.replay_data_loaded = response.get("data_loaded", False)
        with col_btn2:
            if st.button("⏸️ Pause", width='stretch', disabled=not st.session_state.replay_running):
                st.session_state.replay_running = False
                fetch_data("replay_control", {"action": "pause"})
        with col_btn3:
            if st.button("🔄 Reset", width='stretch', disabled=st.session_state.replay_running and st.session_state.replay_index > 0):
                st.session_state.replay_running = False
                st.session_state.replay_index = 0
                st.session_state.replay_total_candles = 0
                st.session_state.replay_data_loaded = False
                fetch_data("replay_control", {"action": "reset"})
        
        st.sidebar.progress(st.session_state.replay_index / max(1, st.session_state.replay_total_candles), 
                            text=f"Candle {st.session_state.replay_index}/{st.session_state.replay_total_candles}")

    elif st.session_state.mode == "BACKTEST":
        st.sidebar.subheader("Backtest Controls")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.session_state.backtest_start_date = st.date_input(
                "Start Date", st.session_state.backtest_start_date, key="backtest_start_date_input"
            )
        with col2:
            st.session_state.backtest_end_date = st.date_input(
                "End Date", st.session_state.backtest_end_date, key="backtest_end_date_input"
            )
        if st.sidebar.button("📊 Run Backtest", width='stretch', disabled=st.session_state.backtest_running):
            st.session_state.backtest_running = True
            with st.spinner("Running full backtest... This may take a while."):
                response = fetch_data("backtest_control", {
                    "action": "run",
                    "start_date": st.session_state.backtest_start_date.isoformat(),
                    "end_date": st.session_state.backtest_end_date.isoformat()
                })
                if response and response.get("status") == "Backtest started":
                    st.success("Backtest initiated. Check performance panel for results once complete.")
                else:
                    st.error("Failed to initiate backtest.")
            st.session_state.backtest_running = False # Reset after completion

    st.sidebar.markdown("---")
    st.session_state.refresh_rate = st.sidebar.slider("UI Refresh Rate (sec)", 1, 5, st.session_state.refresh_rate)
    st.sidebar.info(f"Dashboard Refresh Rate: {st.session_state.refresh_rate} seconds")


def render_market_panel(market_state_data):
    st.subheader("🌐 Global Market Intelligence")
    if not market_state_data:
        st.info("🔄 Waiting for market data...")
        return

    m = market_state_data
    # Map internal session codes to full descriptive names
    session_map = {
        "LONDON": "🇪🇺 European (London)",
        "ASIA": "🇯🇵 Asian (Tokyo/Tokyo)",
        "NEW_YORK": "🇺🇸 American (New York)",
        "LATE_SESSION": "🌊 Pacific / Late Trading"
    }
    display_session = session_map.get(m.get("session"), m.get("session", "Unknown Session"))

    with st.container(border=True):
        cols = st.columns(5, gap="medium")
        cols[0].metric("Trading Session", display_session)
        cols[1].metric("Market Structure", m.get("state", "N/A"))
        cols[2].metric("System Regime", m.get("regime", "N/A"))
        cols[3].metric("H1 Trend Bias", m.get("h1_bias", "N/A").upper())
        cols[4].metric("Volatility State", m.get("volatility", "N/A"))

    st.markdown("### 📊 Market Context")
    c1, c2, c3, c4 = st.columns(4)
    price = m.get('current_price')
    price_display = f"{price:.5f}" if isinstance(price, (int, float)) else "N/A"
    
    c1.info(f"**Price**\n`{price_display}`")
    c2.info(f"**Zones**\n`{m.get('current_zone', 'N/A')}`")
    c3.info(f"**Setup**\n`{m.get('setup', 'NONE').upper()}`")
    signal = m.get('confirmed_signal', 'none').upper()
    sig_color = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "⚪"
    c4.info(f"**Signal**\n{sig_color} `{signal}`")

def render_dual_engine_panel(market_state_data):
    st.subheader("⚙️ Dual Engine Decision System")
    if market_state_data:
        m = market_state_data
        col_alpha, col_flow = st.columns(2)
        
        with col_alpha:
            with st.container(border=True):
                st.markdown("### 🎯 ALPHA (Sniper Strategy)")
                alpha_score = m.get("alpha_score", 0)
                score_color = "🟢" if alpha_score > 75 else "🟡" if alpha_score > 50 else "🔴"
                st.write(f"**Conviction Score:** {score_color} {alpha_score}/100")
                st.write(f"**Signal:** `{m.get('confirmed_signal', 'NONE').upper()}`")
                st.write(f"**Setup:** `{m.get('setup', 'N/A').upper()}`")
                st.write(f"**Current Price:** `{m.get('current_price', 'N/A')}`")
        
        with col_flow:
            with st.container(border=True):
                st.markdown("### 🌊 FLOW (Exploratory Strategy)")
                flow_score = m.get("flow_score", 0)
                score_color = "🟢" if flow_score > 75 else "🟡" if flow_score > 50 else "🔴"
                st.write(f"**Conviction Score:** {score_color} {flow_score}/100")
                st.write(f"**Signal:** `{m.get('confirmed_signal', 'NONE').upper()}`")
                st.write(f"**Regime:** `{m.get('regime', 'N/A').upper()}`")
                st.write(f"**Market State:** `{m.get('state', 'N/A').upper()}`")
    else:
        st.info("🔄 Waiting for engine decisions...")

def render_active_trades_panel(active_trades):
    st.subheader("🚀 Active Positions")
    if active_trades and len(active_trades) > 0:
        active_df = pd.DataFrame(active_trades)
        st.dataframe(
            active_df,
            hide_index=True,
            width="stretch",
            column_config={
                "ticket": st.column_config.TextColumn("ID"),
                "type": st.column_config.TextColumn("Type"),
                "price_open": st.column_config.NumberColumn("Entry", format="%.5f"),
                "sl": st.column_config.NumberColumn("Stop Loss", format="%.5f"),
                "tp": st.column_config.NumberColumn("Take Profit", format="%.5f"),
                "pnl": st.column_config.NumberColumn("Unrealized PnL", format="$ %.2f"),
                "comment": st.column_config.TextColumn("Strategy")
            }
        )
    else:
        st.info("⏸️ No active positions at this time.")

def render_trade_feed(trade_history):
    st.subheader("📜 Trade Feed")
    if trade_history and len(trade_history) > 0:
        feed_df = pd.DataFrame(trade_history)
        # Filter for relevant columns and ensure 'time' is present
        display_cols = ["time", "system", "side", "status", "price", "retcode"]
        available_cols = [col for col in display_cols if col in feed_df.columns]
        feed_df = feed_df[available_cols]
        
        # Show most recent trades
        st.dataframe(
            feed_df.tail(15),
            hide_index=True,
            width="stretch",
            column_config={
                "time": st.column_config.TextColumn("Time"),
                "system": st.column_config.TextColumn("System"),
                "side": st.column_config.TextColumn("Side"),
                "status": st.column_config.TextColumn("Status"),
                "price": st.column_config.NumberColumn("Price", format="%.5f"),
                "retcode": st.column_config.TextColumn("Result")
            }
        )
        st.caption(f"Showing last 15 of {len(feed_df)} events")
    else:
        st.info("🔄 No trade events yet")

def render_chart(chart_data_raw, trades_data):
    st.subheader("Candlestick Chart")
    
    # Ensure chart_data_raw is a non-empty list of dictionaries
    if not isinstance(chart_data_raw, list) or not chart_data_raw:
        st.info("Waiting for chart data...")
        return
    
    # Check if all items are dictionaries, which is what df.to_dict(orient="records") produces
    if not all(isinstance(item, dict) for item in chart_data_raw):
        st.error(f"Chart data received in an unexpected format. Expected list of dicts, got: {chart_data_raw[:5]}")
        return

    chart_data = pd.DataFrame(chart_data_raw)
    
    # Ensure essential columns exist for charting
    required_cols = ['time', 'open', 'high', 'low', 'close']
    if not all(col in chart_data.columns for col in required_cols):
        st.error(f"Chart data is missing required columns: {', '.join(required_cols)}. Available: {chart_data.columns.tolist()}")
        return

    fig = go.Figure(data=[go.Candlestick(
        x=chart_data['time'],
        open=chart_data['open'],
        high=chart_data['high'],
        low=chart_data['low'],
        close=chart_data['close'],
        name='Candles'
    )])

    # Add trades to chart
    if trades_data and trades_data.get("history"):
        trades_df = pd.DataFrame(trades_data["history"])
        
        # Defensive check for 'status' column to prevent KeyError
        if not trades_df.empty and "status" in trades_df.columns:
            # Filter for executed trades that have entry price
            price_col = "price" if "price" in trades_df.columns else None
            if price_col:
                executed_trades = trades_df[(trades_df["status"] == "EXECUTED") & (trades_df[price_col].notna())].copy()
            else:
                executed_trades = pd.DataFrame()
            
            if not executed_trades.empty and "system" in executed_trades.columns:
                alpha_trades = executed_trades[executed_trades["system"] == "ALPHA"]
                flow_trades = executed_trades[executed_trades["system"] == "FLOW_EXP"]

                # Entry Arrows (Defensive check for signal_time and side)
                if "signal_time" in executed_trades.columns and "side" in executed_trades.columns:
                    if not alpha_trades.empty:
                        fig.add_trace(go.Scatter(
                            x=alpha_trades['signal_time'], y=alpha_trades['price'],
                            mode='markers', marker_symbol=np.where(alpha_trades['side'] == 'BUY', 'triangle-up', 'triangle-down'),
                            marker_color='green', marker_size=10, name='Alpha Entry',
                            hoverinfo='text', text=[f"Alpha {r['side']} @ {r['price']:.5f}" for _, r in alpha_trades.iterrows()]
                        ))
                    if not flow_trades.empty:
                        fig.add_trace(go.Scatter(
                            x=flow_trades['signal_time'], y=flow_trades['price'],
                            mode='markers', marker_symbol=np.where(flow_trades['side'] == 'BUY', 'triangle-up', 'triangle-down'),
                            marker_color='blue', marker_size=10, name='Flow Entry',
                            hoverinfo='text', text=[f"Flow {r['side']} @ {r['price']:.5f}" for _, r in flow_trades.iterrows()]
                        ))
        
        # TP/SL lines for active trades
        if trades_data["active"]:
            active_df = pd.DataFrame(trades_data["active"])
            for _, trade in active_df.iterrows():
                if trade['tp'] and trade['tp'] != 0:
                    fig.add_hline(y=trade['tp'], line_dash="dot", line_color="cyan", annotation_text="TP", annotation_position="top right")
                if trade['sl'] and trade['sl'] != 0:
                    fig.add_hline(y=trade['sl'], line_dash="dot", line_color="orange", annotation_text="SL", annotation_position="bottom right")

    fig.update_layout(xaxis_rangeslider_visible=False, height=600, title="XAUUSD Candlestick Chart")
    st.plotly_chart(fig, use_container_width=True)

def render_performance_panel(performance_data, account_info, mode):
    """Render performance metrics with mode-specific details"""
    st.subheader("📈 Performance & Trading Statistics")
    if performance_data:
        if mode == "LIVE":
            # Account Section
            with st.container(border=True):
                st.markdown("### 💳 MetaTrader 5 Account Information")
                if account_info and account_info.get("connected"):
                    acc_col1, acc_col2, acc_col3, acc_col4 = st.columns(4)
                    with acc_col1:
                        st.write(f"**Account Balance:** ${account_info.get('balance', 0):,.2f}")
                    with acc_col2:
                        st.write(f"**Current Equity:** ${account_info.get('equity', 0):,.2f}")
                    with acc_col3:
                        account_type = "🔒 DEMO Account" if account_info.get('is_demo') else "⚠️ LIVE Account"
                        st.write(f"**Account Type:** {account_type}")
                    with acc_col4:
                        st.write(f"**Login ID:** {account_info.get('login', 'N/A')}")
                else:
                    st.warning("⚠️ MT5 Account Not Connected or Not in Demo Mode")
            
            # Alpha System
            with st.container(border=True):
                st.markdown("### 🎯 ALPHA System (Sniper Strategy) Performance")
                alpha_perf = performance_data.get("ALPHA", {})
                alpha_col1, alpha_col2, alpha_col3 = st.columns(3)
                with alpha_col1:
                    st.write(f"**Total Trades Executed:** {alpha_perf.get('trades', 0)}")
                with alpha_col2:
                    st.write(f"**Trades Blocked:** {alpha_perf.get('blocked', 0)}")
                with alpha_col3:
                    st.write(f"**Last Trade Status:** {alpha_perf.get('last_status', 'N/A')}")
            
            # Flow System
            with st.container(border=True):
                st.markdown("### 🌊 FLOW System (Exploratory Strategy) Performance")
                flow_perf = performance_data.get("FLOW_EXP", {})
                flow_col1, flow_col2, flow_col3 = st.columns(3)
                with flow_col1:
                    st.write(f"**Total Trades Executed:** {flow_perf.get('trades', 0)}")
                with flow_col2:
                    st.write(f"**Trades Blocked:** {flow_perf.get('blocked', 0)}")
                with flow_col3:
                    st.write(f"**Last Trade Status:** {flow_perf.get('last_status', 'N/A')}")
            
        elif mode == "BACKTEST":
            combined_perf = performance_data.get("COMBINED", {})
            if combined_perf:
                with st.container(border=True):
                    st.markdown("### 📊 Backtest Performance Results")
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    with perf_col1:
                        st.write(f"**Net Profit/Loss:** ${combined_perf.get('pnl', 0):,.2f}")
                    with perf_col2:
                        st.write(f"**Total Trades:** {combined_perf.get('trades', 0)}")
                    with perf_col3:
                        st.write(f"**Win Rate:** {combined_perf.get('win_rate', 0):.2f}%")
                    
                    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
                    with detail_col1:
                        st.write(f"**Profit Factor:** {combined_perf.get('profit_factor', 0):.2f}")
                    with detail_col2:
                        st.write(f"**Maximum Drawdown:** {combined_perf.get('max_drawdown', 0):.2f}%")
                    with detail_col3:
                        st.write(f"**Average Trade:** ${combined_perf.get('avg_trade', 0):,.2f}")
                    with detail_col4:
                        st.write(f"**Sharpe Ratio:** {combined_perf.get('sharpe_ratio', 0):.2f}")
            else:
                st.info("⏳ Run a backtest to see detailed performance results")
        
        elif mode == "REPLAY":
            replay_perf = performance_data.get("REPLAY", {})
            if replay_perf:
                with st.container(border=True):
                    st.markdown("### 💳 Simulated Replay Account")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Simulated Balance", f"${replay_perf.get('balance', 0):,.2f}")
                    c2.metric("Simulated Equity", f"${replay_perf.get('equity', 0):,.2f}")
                    pnl = replay_perf.get('pnl', 0)
                    pnl_color = "normal" if pnl >= 0 else "inverse"
                    c3.metric("Realized PnL", f"${pnl:,.2f}", delta=f"{pnl:,.2f}", delta_color=pnl_color)
                
                with st.container(border=True):
                    st.markdown("### 🔄 Replay Engine Performance Results")
                    rep_col1, rep_col2, rep_col3 = st.columns(3)
                    with rep_col1:
                        st.write(f"**Net Profit/Loss:** ${replay_perf.get('pnl', 0):,.2f}")
                    with rep_col2:
                        st.write(f"**Total Trades:** {replay_perf.get('trades', 0)}")
                    with rep_col3:
                        st.write(f"**Win Rate:** {replay_perf.get('win_rate', 0):.2f}%")
                    
                    detail_col1, detail_col2, detail_col3, detail_col4 = st.columns(4)
                    with detail_col1:
                        st.write(f"**Profit Factor:** {replay_perf.get('profit_factor', 0):.2f}")
                    with detail_col2:
                        st.write(f"**Maximum Drawdown:** {replay_perf.get('max_drawdown', 0):.2f}%")
                    with detail_col3:
                        st.write(f"**Average Trade:** ${replay_perf.get('avg_trade', 0):,.2f}")
                    with detail_col4:
                        st.write(f"**Sharpe Ratio:** {replay_perf.get('sharpe_ratio', 0):.2f}")
            else:
                st.info("⏳ Replay performance results will appear here once simulation completes")
    else:
        st.info("🔄 Waiting for performance data...")


# --- Main App Logic ---
def main():
    initialize_session_state()
    render_top_bar()

    # Fetch data based on current mode
    params = {"mode": st.session_state.mode}
    if st.session_state.mode == "REPLAY":
        params["replay_index"] = st.session_state.replay_index
        params["start_date"] = st.session_state.replay_start_date.isoformat()
        params["end_date"] = st.session_state.replay_end_date.isoformat()

    state_data = fetch_data("state", params)
    trades_data = fetch_data("trades", params)
    performance_data = fetch_data("performance", params)
    chart_data = fetch_data("chart_data", params)

    # Extract account info from state_data
    account_info = state_data.get("account") if state_data else None

    # Main title with mode indicator
    mode_emoji = {"LIVE": "🔴", "REPLAY": "🔄", "BACKTEST": "📊"}
    st.title(f"{mode_emoji.get(st.session_state.mode, '📡')} AQRS V2 Dashboard - {st.session_state.mode} Mode")

    # Full-width layout - everything stacked vertically
    # 1. Global Market Intelligence
    render_market_panel(state_data.get("market") if state_data else None)
    st.divider()
    
    # 2. Dual Engine Decision System
    render_dual_engine_panel(state_data.get("market") if state_data else None)
    st.divider()
    
    # 3. Candlestick Chart
    render_chart(chart_data, trades_data)
    st.divider()
    
    # 3.5 Active Trades Table
    render_active_trades_panel(trades_data.get("active") if trades_data else None)
    st.divider()
    
    # 4. Trade Feed
    render_trade_feed(trades_data.get("history") if trades_data else None)
    st.divider()
    
    # 5. Performance Metrics
    render_performance_panel(performance_data, account_info, st.session_state.mode)

    # Auto-refresh logic for LIVE and REPLAY
    if st.session_state.mode == "LIVE" or (st.session_state.mode == "REPLAY" and st.session_state.replay_running and st.session_state.replay_data_loaded):
        # Increment replay index for next iteration
        if st.session_state.mode == "REPLAY" and st.session_state.replay_running:
            st.session_state.replay_index += 1
            # Send updated index to backend for its internal state management
            fetch_data("replay_control", {"action": "step", "index": st.session_state.replay_index})
            if st.session_state.replay_index >= st.session_state.replay_total_candles:
                st.session_state.replay_running = False # Stop replay at end
                st.info("🏁 Replay finished!")

        time.sleep(1.0 / st.session_state.replay_speed if st.session_state.mode == "REPLAY" else st.session_state.refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()