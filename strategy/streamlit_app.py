import streamlit as st
import pandas as pd
import time
import sys
import os
import subprocess
import requests # New import
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import numpy as np

# Ensure project root is in path for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.state_manager import DashboardStateManager, get_csv_tail

st.set_page_config(page_title="AQRS V3 Dashboard", layout="wide", initial_sidebar_state="expanded")

def fetch_data(endpoint, params=None):
    """Fetches data from the FastAPI backend."""
    backend_url = st.session_state.backend_url
    try:
        response = requests.get(f"{backend_url}/{endpoint}", params=params)
        response.raise_for_status() # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("❌ Could not connect to FastAPI backend. Please ensure `python -m strategy.backend_api` is running.")
        st.stop() # Stop execution to prevent further errors
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error fetching data from backend ({endpoint}): {e}")
        return None

# Sidebar Navigation
st.sidebar.title("🧠 AQRS V3 Dashboard")

def initialize_session_state():
    if "state_manager" not in st.session_state:
        st.session_state.state_manager = DashboardStateManager()
    
    if "backend_url" not in st.session_state:
        st.session_state.backend_url = "http://127.0.0.1:8001"

    # Initialize internal replay state if not exists
    if "replay_internal_state" not in st.session_state:
        st.session_state.replay_internal_state = {
            "running": False,
            "index": 0,
            "total_candles": 0,
            "start_date": None,
            "end_date": None,
            "data_loaded": False
        }

    if "replay_generating" not in st.session_state:
        st.session_state.replay_generating = False
    if "replay_proc" not in st.session_state:
        st.session_state.replay_proc = None

    backend_state = st.session_state.replay_internal_state
    
    if "mode" not in st.session_state:
        st.session_state.mode = "LIVE"
    if "symbol" not in st.session_state:
        st.session_state.symbol = "XAUUSD" # Default, could be fetched from config
    if "refresh_rate" not in st.session_state:
        st.session_state.refresh_rate = 2 # UI refresh rate for LIVE mode

    # Replay specific states
    if "replay_running" not in st.session_state:
        st.session_state.replay_running = backend_state.get("running", False)
    if "replay_index" not in st.session_state:
        st.session_state.replay_index = backend_state.get("index", 0)
    
    # Restore dates from backend if available, otherwise use defaults
    if "replay_start_date" not in st.session_state:
        bounds = st.session_state.state_manager.get_data_bounds()
        if bounds["end"]:
            # Default to the last 3 days of data for speed
            st.session_state.replay_start_date = bounds["end"] - timedelta(days=3)
        else:
            st.session_state.replay_start_date = date(2025, 1, 1)
            
    if "replay_end_date" not in st.session_state:
        bounds = st.session_state.state_manager.get_data_bounds()
        if bounds["end"]:
            st.session_state.replay_end_date = bounds["end"]
        else:
            st.session_state.replay_end_date = date(2026, 4, 30)

    # Keep the UI aligned with backend replay range when a replay is already loaded
    if backend_state.get("start_date") and backend_state.get("end_date"):
        backend_start = datetime.strptime(backend_state["start_date"], "%Y-%m-%d").date()
        backend_end = datetime.strptime(backend_state["end_date"], "%Y-%m-%d").date()
        st.session_state.replay_start_date = backend_start
        st.session_state.replay_end_date = backend_end

    if "replay_speed" not in st.session_state:
        st.session_state.replay_speed = 1.0 # candles per second
    
    # Sync metadata
    st.session_state.replay_total_candles = backend_state.get("total_candles", 0)
    st.session_state.replay_data_loaded = backend_state.get("data_loaded", False)

    # Backtest specific states
    if "backtest_running" not in st.session_state:
        st.session_state.backtest_running = False
    if "backtest_start_date" not in st.session_state:
        st.session_state.backtest_start_date = (datetime.now() - timedelta(days=365)).date()
    if "backtest_end_date" not in st.session_state:
        st.session_state.backtest_end_date = datetime.now().date()

# --- UI Components ---
def render_top_bar():
    st.sidebar.title("🧠 AQRS V3 Control Center")

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

        # Show available data range so user doesn't pick empty dates
        bounds = st.session_state.state_manager.get_data_bounds()
        if bounds["start"]:
            st.sidebar.caption(f"📅 Data available: {bounds['start']} to {bounds['end']}")

        st.sidebar.markdown("---")
        col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
        with col_btn1:
            if st.button("▶️ Start Replay", use_container_width=True, disabled=st.session_state.replay_running):
                # Initiate async generation
                response = fetch_data("replay_control", {
                    "action": "start",
                    "start_date": st.session_state.replay_start_date.isoformat(),
                    "end_date": st.session_state.replay_end_date.isoformat()
                })
                if response and response.get("status") == "generation_started":
                    st.session_state.replay_generating = True
                    st.session_state.replay_running = False
                    st.session_state.replay_data_loaded = False
                    st.rerun()
                elif response and response.get("status") == "already_started":
                    st.warning("Replay generation is already in progress.")
                    st.session_state.replay_generating = True # Ensure UI reflects this
                    st.rerun()
                else:
                    st.error(f"Failed to start replay generation: {response.get('message', 'Unknown error')}")
                st.session_state.replay_generating = True
                st.session_state.replay_running = False
                st.session_state.replay_data_loaded = False
                st.rerun()

        with col_btn2:
            if st.button("⏸️ Pause", use_container_width=True, disabled=not st.session_state.replay_running):
                st.session_state.replay_running = False
                fetch_data("replay_control", {"action": "pause"})
        with col_btn3:
            if st.button("🔄 Reset", use_container_width=True, disabled=st.session_state.replay_running and st.session_state.replay_index > 0):
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
        if st.sidebar.button("📊 Run Backtest", use_container_width=True, disabled=st.session_state.backtest_running):
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


def render_market_panel(market_state_data, mode="LIVE", replay_index=0, total_candles=0):
    st.subheader("🌐 Global Market Intelligence")
    if not market_state_data:
        if mode == "REPLAY":
            st.info(f"🔄 Loading replay data... (Index: {replay_index}/{total_candles})")
        else:
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
        cols = st.columns(6, gap="medium")
        cols[0].metric("Trading Session", display_session)
        cols[1].metric("Behavior", m.get("behavior", "N/A"))
        cols[2].metric("Structure", m.get("structure_state", "N/A"))
        cols[3].metric("System Regime", m.get("regime", "N/A"))
        cols[4].metric("H1 Trend Bias", m.get("h1_bias", "N/A").upper())
        cols[5].metric("Volatility", m.get("volatility", "N/A"))

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
    st.info(f"**Current Trade Logic / Pattern:**\n`{m.get('pattern', 'NONE')} | {m.get('execution_reason', 'N/A')}`")

    # Task 6: Current Regime Historical Score
    st.markdown("---")
    st.subheader("🧠 Regime Memory Engine")
    regime_exp = m.get("current_regime_expectancy", 0.0)
    regime_pf = m.get("current_regime_pf", 0.0)
    regime_trades = m.get("current_regime_trades", 0)

    exp_color = "green" if regime_exp > 0 else "red" if regime_exp < 0 else "gray"
    pf_color = "green" if regime_pf > 1.0 else "red" if regime_pf < 1.0 else "gray"

    st.markdown(f"""
        <div style="background-color:#262730; padding: 10px; border-radius: 5px;">
            <p style="font-size:16px; font-weight:bold;">Current Regime: <span style="color:white;">{m.get('regime', 'N/A')}</span></p>
            <p style="font-size:14px;">Historical Expectancy: <span style="color:{exp_color}; font-weight:bold;">{regime_exp:.2f}</span> (per trade)</p>
            <p style="font-size:14px;">Historical Profit Factor: <span style="color:{pf_color}; font-weight:bold;">{regime_pf:.2f}</span></p>
            <p style="font-size:14px;">Sample Size: <span style="color:white;">{regime_trades}</span> trades</p>
        </div>
    """, unsafe_allow_html=True)


def render_signals_feed(signals_data):
    st.subheader("📡 Signals Intelligence Feed")
    if signals_data and len(signals_data) > 0:
        sig_df = pd.DataFrame(signals_data)
        display_cols = ["time", "confirmed_signal", "pattern", "alpha_score", "flow_score", "behavior_label", "structure_state", "execution_reason"]
        available_cols = [c for col in display_cols if (c := col) in sig_df.columns]
        sig_df = sig_df[available_cols].copy()
        
        st.dataframe(
            sig_df.tail(20),
            hide_index=True,
            width="stretch",
            column_config={
                "time": st.column_config.TextColumn("Time"),
                "confirmed_signal": st.column_config.TextColumn("Signal"),
                "pattern": st.column_config.TextColumn("Pattern"),
                "alpha_score": st.column_config.NumberColumn("Alpha"),
                "flow_score": st.column_config.NumberColumn("Flow"),
                "behavior_label": st.column_config.TextColumn("Behavior"),
                "structure_state": st.column_config.TextColumn("Structure"),
                "execution_reason": st.column_config.TextColumn("Logic")
            }
        )
    else:
        st.info("🔄 No signals detected in this range yet.")

def render_dual_engine_panel(market_state_data, performance_data, trades_data, mode, replay_index=0, total_candles=0):
    st.subheader("⚙️ Dual Engine Decision System")
    if market_state_data:
        m = market_state_data
        col_alpha, col_flow = st.columns(2)
        
        # Get performance stats based on mode
        if mode == "LIVE":
            alpha_perf = performance_data.get("ALPHA", {})
            flow_perf = performance_data.get("FLOW_EXP", {})
        elif mode == "REPLAY":
            alpha_perf = performance_data.get("ALPHA", {})
            flow_perf = performance_data.get("FLOW_EXP", {})
        elif mode == "BACKTEST":
            combined_perf = performance_data.get("COMBINED", {})
            alpha_perf = combined_perf
            flow_perf = combined_perf
        else:
            alpha_perf = {}
            flow_perf = {}
        
        # Get active trades count
        active_trades = trades_data.get("active", []) if trades_data else []
        alpha_active = len([t for t in active_trades if t.get("comment", "").startswith("ALPHA")])
        flow_active = len([t for t in active_trades if t.get("comment", "").startswith("FLOW")])
        
        with col_alpha:
            with st.container(border=True):
                st.markdown("### 🎯 ALPHA (Sniper Strategy)")
                st.metric("Conviction Score", f"{m.get('alpha_score', 0)}/100")
                st.write(f"**Signal:** `{m.get('confirmed_signal', 'NONE').upper()}`")
                st.write(f"**Structure:** `{m.get('state', 'N/A')}`")
                st.write(f"**Trades Open:** {alpha_active}")
                alpha_wins = int(alpha_perf.get('wins', 0))
                alpha_losses = int(alpha_perf.get('losses', 0))
                alpha_total = int(alpha_perf.get('trades', 0))
                # If losses not explicitly set but we have trades and wins, calculate losses
                if alpha_losses == 0 and alpha_total > 0 and alpha_wins < alpha_total:
                    alpha_losses = alpha_total - alpha_wins
                st.write(f"**Trades Won:** {alpha_wins}")
                st.write(f"**Trades Lost:** {alpha_losses}")
                st.write(f"**Net PnL:** ${alpha_perf.get('pnl', 0):,.2f}")
        
        with col_flow:
            with st.container(border=True):
                st.markdown("### 🌊 FLOW (Exploratory Strategy)")
                st.metric("Exploration Score", f"{m.get('flow_score', 0)}/100")
                st.write(f"**Signal:** `{m.get('confirmed_signal', 'NONE').upper()}`")
                st.write(f"**Regime:** `{m.get('regime', 'N/A')}`")
                st.write(f"**Trades Open:** {flow_active}")
                flow_wins = int(flow_perf.get('wins', 0))
                flow_losses = int(flow_perf.get('losses', 0))
                flow_total = int(flow_perf.get('trades', 0))
                # If losses not explicitly set but we have trades and wins, calculate losses
                if flow_losses == 0 and flow_total > 0 and flow_wins < flow_total:
                    flow_losses = flow_total - flow_wins
                st.write(f"**Trades Won:** {flow_wins}")
                st.write(f"**Trades Lost:** {flow_losses}")
                st.write(f"**Net PnL:** ${flow_perf.get('pnl', 0):,.2f}")
    else:
        if mode == "REPLAY":
            st.info(f"🔄 Loading engine decisions... (Index: {replay_index}/{total_candles})")
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
        # Filter for relevant columns and ensure 'time' is present, handling replay-specific columns
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
                "side": st.column_config.TextColumn("Side", help="Trade direction (buy/sell)"),
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
    
    fig.update_layout(xaxis_rangeslider_visible=False, height=600, title="XAUUSD Candlestick Chart") # Removed use_container_width
    st.plotly_chart(fig, width='stretch') # Replaced use_container_width=True with width='stretch'

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
                
                # Range Intelligence Summary (Mirroring console output)
                with st.container(border=True):
                    st.markdown("#### 📋 V3 REPLAY RANGE SUMMARY")
                    rs_col1, rs_col2, rs_col3 = st.columns(3)
                    rs_col1.write(f"**Total ALPHA Signals:** {replay_perf.get('alpha_signals', 0)}")
                    rs_col2.write(f"**Total FLOW Signals:** {replay_perf.get('flow_signals', 0)}")
                    total_ops = replay_perf.get('alpha_signals', 0) + replay_perf.get('flow_signals', 0)
                    rs_col3.write(f"**Total Opportunities:** {total_ops}")

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

def render_intelligence_tab():
    st.subheader("🧠 AQRS V3 Intelligence: What Works Now")

    params = {"mode": st.session_state.mode}
    if st.session_state.mode == "REPLAY":
        params["replay_index"] = st.session_state.replay_index
        
    intelligence = fetch_data("intelligence", params)
    
    if not intelligence:
        st.info("🔄 Intelligence Matrix is building. Log more trade outcomes to see adaptive insights.")
        return

    tab1, tab2, tab3, tab4 = st.tabs(["Expectancy Matrix", "By Regime", "Weekly Governance", "Adaptive Rules"])
    
    with tab1:
        st.subheader("📊 Performance by Market Context")
        col1, col2 = st.columns(2)
        col1.write("**By Market Behavior**")
        col1.dataframe(intelligence["behavior"], hide_index=True)
        col2.write("**By Trading Session**")
        col2.dataframe(intelligence["session"], hide_index=True)
        
        st.write("**By Setup Type**")
        st.dataframe(intelligence["setup"], hide_index=True)
    
    with tab2:
        st.subheader("🌍 Performance by Market Regime")
        st.dataframe(intelligence["market_regime"], hide_index=True)

    with tab3:
        st.subheader("📅 Recent Trade Outcomes (Forensic Audit)")
        st.dataframe(intelligence["weekly_report"], use_container_width=True)

    with tab4:
        st.success("✅ **Active Rule**: Promoting Behavior Labels with PF > 1.3 and 30+ samples.")
        st.warning("⚠️ **Active Rule**: Throttling Setup types with negative net expectancy.")
        st.info("🧠 **New Rule**: Risk adjusted by current regime's historical expectancy.")

# --- Main App Logic ---
def main():
    initialize_session_state()
    render_top_bar()

    # --- Async Polling Logic ---
    # Streamlit now polls the FastAPI backend for replay generation status
    if st.session_state.replay_generating:
        try:
            # Poll the backend for the status of the replay generation process
            status_data = fetch_data("replay_control", {"action": "check_generation_status"})

            if status_data and status_data["status"] == "in_progress":
                st.info("🚀 **AQRS Engine is generating replay artifacts in the background...**")
                st.caption("You can still explore the sidebar or switch modes. The replay will start automatically once ready.")
                time.sleep(2) # Prevent rapid-fire reruns
                st.rerun()
            elif status_data and status_data["status"] == "completed":
                st.session_state.replay_generating = False
                
                # The backend has already loaded the data into its state_manager.
                # Now, we need to sync Streamlit's session state with the backend's replay state.
                backend_replay_state = fetch_data("replay_control", {"action": "status"})
                
                if backend_replay_state and backend_replay_state.get("data_loaded"):
                    st.session_state.replay_index = 0
                    st.session_state.replay_total_candles = backend_replay_state.get("total_candles", 0)
                    st.session_state.replay_data_loaded = True
                    st.session_state.replay_running = True # Start playback automatically
                    st.success("✅ Replay artifacts generated successfully!")
                    time.sleep(1)
                else:
                    st.error("❌ Generation finished but no data was found for this range.")
                    st.session_state.replay_data_loaded = False
                    st.session_state.replay_running = False
                st.rerun()
            elif status_data and (status_data["status"] == "failed" or status_data["status"] == "failed_to_load_data"):
                st.session_state.replay_generating = False
                st.error(f"❌ Replay generation failed: {status_data.get('message', 'Unknown error')}")
                st.session_state.replay_data_loaded = False
                st.session_state.replay_running = False
                st.rerun()
            elif status_data and status_data["status"] == "not_started":
                # This might happen if the backend restarted or process was cleared
                st.session_state.replay_generating = False
                st.warning("Replay generation process not found on backend. Please try starting again.")
                st.rerun()
        except Exception as e:
            st.session_state.replay_generating = False
            st.error(f"❌ An unexpected error occurred while checking replay status: {e}")
            st.rerun()

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
    signals_data = fetch_data("signals", params)

    # Extract account info from state_data
    if not state_data:
        st.error("🚨 **API Backend Connection Lost**")
        st.info("Please ensure `python -m strategy.backend_api` is running in your terminal.")
        if st.button("🔄 Retry Connection"):
            st.rerun()
        return

    account_info = state_data.get("account") if state_data else None

    # Main title with mode indicator
    mode_emoji = {"LIVE": "🔴", "REPLAY": "🔄", "BACKTEST": "📊"}
    st.title(f"{mode_emoji.get(st.session_state.mode, '📡')} AQRS V3 Dashboard - {st.session_state.mode} Mode")

    # Add Tabs to Dashboard
    main_tabs = st.tabs(["Dashboard", "What Works Now"])
    
    with main_tabs[0]:
        # Full-width layout - everything stacked vertically
        # 1. Global Market Intelligence
        render_market_panel(state_data.get("market") if state_data else None, st.session_state.mode, st.session_state.replay_index if st.session_state.mode == "REPLAY" else 0, st.session_state.replay_total_candles if st.session_state.mode == "REPLAY" else 0)
        st.divider()
        
        # 2. Dual Engine Decision System
        render_dual_engine_panel(state_data.get("market") if state_data else None, performance_data, trades_data, st.session_state.mode, st.session_state.replay_index if st.session_state.mode == "REPLAY" else 0, st.session_state.replay_total_candles if st.session_state.mode == "REPLAY" else 0)
        st.divider()

        # 2.5 Signals Intelligence Feed
        render_signals_feed(signals_data)
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

    with main_tabs[1]:
        render_intelligence_tab()

    # Auto-refresh logic for LIVE and REPLAY
    if st.session_state.mode == "LIVE" or (st.session_state.mode == "REPLAY" and st.session_state.replay_running and st.session_state.replay_data_loaded):
        # Increment replay index for next iteration
        if st.session_state.mode == "REPLAY" and st.session_state.replay_running:
            # Check bounds before incrementing
            if st.session_state.replay_index < st.session_state.replay_total_candles:
                st.session_state.replay_index += 1
                # Send updated index to backend for its internal state management
                fetch_data("replay_control", {"action": "step", "index": st.session_state.replay_index})
            
            # Stop replay if we've reached the end
            if st.session_state.replay_index >= st.session_state.replay_total_candles:
                st.session_state.replay_running = False
                st.info("🏁 Replay finished!")
        
        # Dynamically adjust sleep to match selected speed
        # High speed (5.0) = 0.2s refresh; Low speed (0.1) = 10s refresh
        time.sleep(1.0 / st.session_state.replay_speed if st.session_state.mode == "REPLAY" else st.session_state.refresh_rate)
        st.rerun()

if __name__ == "__main__":
    main()