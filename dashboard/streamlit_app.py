import streamlit as st


def run_dashboard(config, data):
    st.set_page_config(page_title="AQRS V3 Trading Brain", layout="wide")
    st.title("AQRS V3 — Institutional Market Behavior Trading Engine")
    st.sidebar.header("Controls")
    st.sidebar.write("Mode and session filters coming soon.")

    st.markdown("## Market behavior overview")
    st.write("Dashboard skeleton for live/replay/backtest status.")

    st.markdown("## System status")
    st.write(data)

    return None
