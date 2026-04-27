import pandas as pd
import numpy as np

class VolumeProfileEngine:
    """Calculates Session Volume Profile and identifies Point of Control (POC)."""

    @staticmethod
    def calculate_session_poc(df: pd.DataFrame, bucket_size: float = 0.5) -> pd.DataFrame:
        """
        Identifies the Point of Control (POC) for each session.
        POC is the price bin with the highest tick_volume.
        """
        out = df.copy()
        if "tick_volume" not in out.columns:
            return out

        # Create price bins for volume aggregation
        out["price_bin"] = (out["close"] / bucket_size).round() * bucket_size

        # Identify sessions (Using the logic from your reporting module)
        def get_sess(h):
            if 0 <= h < 7: return "ASIA"
            if 7 <= h < 13: return "LONDON"
            if 13 <= h < 18: return "NEW_YORK"
            return "LATE"

        out["session_id"] = out["time"].dt.hour.apply(get_sess)
        out["date_group"] = out["time"].dt.date

        # Group by date and session to find POC
        session_groups = out.groupby(["date_group", "session_id"])
        
        poc_map = {}
        for name, group in session_groups:
            # Sum volume per price bin
            vol_profile = group.groupby("price_bin")["tick_volume"].sum()
            if not vol_profile.empty:
                poc_price = vol_profile.idxmax()
                poc_map[name] = poc_price

        # Map current session POC
        out["session_poc"] = out.set_index(["date_group", "session_id"]).index.map(poc_map)

        # Identify Previous Session POC (Institutional Magnet)
        # We shift the results within the date/session sequence
        unique_sessions = out[["date_group", "session_id"]].drop_duplicates().sort_values(["date_group", "session_id"])
        unique_sessions["poc_val"] = unique_sessions.set_index(["date_group", "session_id"]).index.map(poc_map)
        unique_sessions["prev_session_poc"] = unique_sessions["poc_val"].shift(1)
        
        prev_poc_map = unique_sessions.set_index(["date_group", "session_id"])["prev_session_poc"].to_dict()
        out["prev_session_poc"] = out.set_index(["date_group", "session_id"]).index.map(prev_poc_map)

        # Proximity check: Is price near the Previous Session POC?
        out["near_prev_poc"] = (abs(out["close"] - out["prev_session_poc"]) <= (bucket_size * 2)).astype(int)

        return out

    @classmethod
    def enrich_intelligence(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich the pipeline with Volume Profile data."""
        df = cls.calculate_session_poc(df)
        return df