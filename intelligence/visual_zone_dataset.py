from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SIDES = {"BUY", "SELL"}


@dataclass(frozen=True)
class ChartCalibration:
    screenshot_id: str
    plot_left_x: float
    plot_right_x: float
    price_top_y: float
    price_bottom_y: float
    price_top: float
    price_bottom: float
    visible_start: pd.Timestamp
    visible_end: pd.Timestamp
    symbol: str = "XAUUSD"
    timeframe: str = "H1"
    timezone: str = "broker"
    x_axis_mode: str = "calendar"

    @classmethod
    def from_json(cls, path: Path) -> "ChartCalibration":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            screenshot_id=str(payload["screenshot_id"]),
            plot_left_x=float(payload["plot_left_x"]),
            plot_right_x=float(payload["plot_right_x"]),
            price_top_y=float(payload["price_top_y"]),
            price_bottom_y=float(payload["price_bottom_y"]),
            price_top=float(payload["price_top"]),
            price_bottom=float(payload["price_bottom"]),
            visible_start=pd.to_datetime(payload["visible_start"]),
            visible_end=pd.to_datetime(payload["visible_end"]),
            symbol=str(payload.get("symbol", "XAUUSD")),
            timeframe=str(payload.get("timeframe", "H1")),
            timezone=str(payload.get("timezone", "broker")),
            x_axis_mode=str(payload.get("x_axis_mode", "calendar")).lower(),
        )

    def x_to_fraction(self, x_value: float) -> float:
        span = max(self.plot_right_x - self.plot_left_x, 1.0)
        return float(np.clip((x_value - self.plot_left_x) / span, 0.0, 1.0))

    def y_to_price(self, y_value: float) -> float:
        span = max(self.price_bottom_y - self.price_top_y, 1.0)
        fraction = np.clip((y_value - self.price_top_y) / span, 0.0, 1.0)
        return float(self.price_top + fraction * (self.price_bottom - self.price_top))

    def x_to_time(self, x_value: float) -> pd.Timestamp:
        fraction = self.x_to_fraction(x_value)
        start_ns = self.visible_start.value
        end_ns = self.visible_end.value
        return pd.Timestamp(start_ns + int((end_ns - start_ns) * fraction))


def load_candles(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], on_bad_lines="skip")
    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing candle columns: {sorted(missing)}")
    return df.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)


def visible_candles(candles: pd.DataFrame, calibration: ChartCalibration) -> pd.DataFrame:
    mask = candles["time"].between(calibration.visible_start, calibration.visible_end)
    out = candles.loc[mask].copy().reset_index(drop=True)
    if out.empty:
        raise ValueError(
            "No candles found inside calibration window "
            f"{calibration.visible_start} to {calibration.visible_end}"
        )
    return out


def candle_at_fraction(candles: pd.DataFrame, fraction: float) -> int:
    if len(candles) == 1:
        return 0
    return int(np.clip(round(fraction * (len(candles) - 1)), 0, len(candles) - 1))


def nearest_candle_index(candles: pd.DataFrame, timestamp: pd.Timestamp) -> int:
    deltas = (candles["time"] - timestamp).abs()
    return int(deltas.idxmin())


def build_zones(
    annotations: pd.DataFrame,
    calibration: ChartCalibration,
    candles: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, row in annotations.iterrows():
        side = str(row.get("side", "")).strip().upper()
        if side not in SIDES:
            raise ValueError(f"Annotation row {i + 2} has unsupported side: {side!r}")

        x_center = float(row["x_center"])
        y_center = float(row["y_center"])
        x_radius = abs(float(row.get("x_radius", 0.0)))
        y_radius = abs(float(row.get("y_radius", 0.0)))

        if calibration.x_axis_mode == "candle_index":
            start_idx = candle_at_fraction(candles, calibration.x_to_fraction(x_center - x_radius))
            end_idx = candle_at_fraction(candles, calibration.x_to_fraction(x_center + x_radius))
            center_idx = candle_at_fraction(candles, calibration.x_to_fraction(x_center))
            if start_idx > end_idx:
                start_idx, end_idx = end_idx, start_idx
            start_time = candles.loc[start_idx, "time"]
            end_time = candles.loc[end_idx, "time"]
            center_time = candles.loc[center_idx, "time"]
        else:
            start_time = calibration.x_to_time(x_center - x_radius)
            end_time = calibration.x_to_time(x_center + x_radius)
            center_time = calibration.x_to_time(x_center)
            if start_time > end_time:
                start_time, end_time = end_time, start_time
            start_idx = nearest_candle_index(candles, start_time)
            end_idx = nearest_candle_index(candles, end_time)
            center_idx = nearest_candle_index(candles, center_time)
            start_time = candles.loc[start_idx, "time"]
            end_time = candles.loc[end_idx, "time"]
            center_time = candles.loc[center_idx, "time"]

        price_a = calibration.y_to_price(y_center - y_radius)
        price_b = calibration.y_to_price(y_center + y_radius)
        price_low = min(price_a, price_b)
        price_high = max(price_a, price_b)

        rows.append(
            {
                "screenshot_id": row.get("screenshot_id", calibration.screenshot_id),
                "annotation_id": row.get("annotation_id", f"{calibration.screenshot_id}_{i + 1:03d}"),
                "symbol": calibration.symbol,
                "timeframe": calibration.timeframe,
                "side": side,
                "start_time": start_time,
                "end_time": end_time,
                "center_time": center_time,
                "price_low": round(price_low, 3),
                "price_high": round(price_high, 3),
                "center_price": round(calibration.y_to_price(y_center), 3),
                "confidence": float(row.get("confidence", 1.0)),
                "source_x_center": x_center,
                "source_y_center": y_center,
                "source_x_radius": x_radius,
                "source_y_radius": y_radius,
                "notes": row.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["range"] = (out["high"] - out["low"]).replace(0, np.nan)
    out["body"] = (out["close"] - out["open"]).abs()
    out["upper_wick_ratio"] = ((out["high"] - out[["open", "close"]].max(axis=1)) / out["range"]).fillna(0.0)
    out["lower_wick_ratio"] = ((out[["open", "close"]].min(axis=1) - out["low"]) / out["range"]).fillna(0.0)
    out["body_ratio"] = (out["body"] / out["range"]).fillna(0.0)
    out["momentum"] = out["close"].diff().fillna(0.0)
    out["hour"] = out["time"].dt.hour
    out["day_of_week"] = out["time"].dt.dayofweek
    out["rolling_high_20"] = out["high"].rolling(20, min_periods=1).max()
    out["rolling_low_20"] = out["low"].rolling(20, min_periods=1).min()
    band = (out["rolling_high_20"] - out["rolling_low_20"]).replace(0, np.nan)
    out["range_position_20"] = ((out["close"] - out["rolling_low_20"]) / band).fillna(0.5).clip(0, 1)
    previous_close = out["close"].shift(1)
    true_range = np.maximum(
        out["high"] - out["low"],
        np.maximum((out["high"] - previous_close).abs(), (out["low"] - previous_close).abs()),
    )
    out["atr14"] = true_range.fillna(out["high"] - out["low"]).rolling(14, min_periods=1).mean()
    return out


def label_candles(candles: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    labels = add_candle_features(candles)
    labels["visual_zone_label"] = "NONE"
    labels["visual_zone_score"] = 0.0
    labels["visual_zone_annotation_id"] = ""

    for _, zone in zones.iterrows():
        time_mask = labels["time"].between(pd.to_datetime(zone["start_time"]), pd.to_datetime(zone["end_time"]))
        price_mask = (labels["high"] >= float(zone["price_low"])) & (labels["low"] <= float(zone["price_high"]))
        mask = time_mask & price_mask
        confidence = float(zone.get("confidence", 1.0))
        score = round(100.0 * np.clip(confidence, 0.0, 1.0), 2)
        stronger = mask & (score >= labels["visual_zone_score"])
        labels.loc[stronger, "visual_zone_label"] = str(zone["side"]).upper()
        labels.loc[stronger, "visual_zone_score"] = score
        labels.loc[stronger, "visual_zone_annotation_id"] = str(zone["annotation_id"])

    return labels


def train_visual_zone_profile(labels: pd.DataFrame, zones: pd.DataFrame) -> dict[str, Any]:
    feature_cols = [
        "body_ratio",
        "upper_wick_ratio",
        "lower_wick_ratio",
        "momentum",
        "range_position_20",
        "atr14",
        "hour",
    ]
    payload: dict[str, Any] = {
        "version": "1.0",
        "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "samples": int(len(labels)),
        "labeled_samples": int(labels["visual_zone_label"].isin(SIDES).sum()),
        "zones": int(len(zones)),
        "side_profiles": {},
    }

    for side in sorted(SIDES):
        side_df = labels[labels["visual_zone_label"] == side]
        if side_df.empty:
            payload["side_profiles"][side] = {"samples": 0}
            continue
        profile = {
            "samples": int(len(side_df)),
            "preferred_hours": [int(v) for v in side_df["hour"].value_counts().head(6).index.tolist()],
            "feature_mean": {col: round(float(side_df[col].mean()), 6) for col in feature_cols},
            "feature_std": {col: round(float(side_df[col].std(ddof=0)), 6) for col in feature_cols},
            "price_zone_low_mean": round(float(zones.loc[zones["side"] == side, "price_low"].mean()), 3),
            "price_zone_high_mean": round(float(zones.loc[zones["side"] == side, "price_high"].mean()), 3),
        }
        payload["side_profiles"][side] = profile

    return payload


def run(
    calibration_path: Path,
    annotations_path: Path,
    candles_path: Path,
    out_zones_path: Path,
    out_labels_path: Path,
    model_out_path: Path,
) -> dict[str, Any]:
    calibration = ChartCalibration.from_json(calibration_path)
    annotations = pd.read_csv(annotations_path, on_bad_lines="skip")
    candles = visible_candles(load_candles(candles_path), calibration)

    zones = build_zones(annotations, calibration, candles)
    labels = label_candles(candles, zones)
    model = train_visual_zone_profile(labels, zones)

    out_zones_path.parent.mkdir(parents=True, exist_ok=True)
    out_labels_path.parent.mkdir(parents=True, exist_ok=True)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)

    zones.to_csv(out_zones_path, index=False)
    labels.to_csv(out_labels_path, index=False)
    model_out_path.write_text(json.dumps(model, indent=2), encoding="utf-8")

    return {
        "visible_candles": int(len(candles)),
        "zones": int(len(zones)),
        "labeled_candles": int(labels["visual_zone_label"].isin(SIDES).sum()),
        "out_zones": str(out_zones_path),
        "out_labels": str(out_labels_path),
        "model": str(model_out_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert annotated MT5 screenshot zones into candle labels and a visual-zone profile."
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--candles", type=Path, default=Path("data/raw/xauusd_h1.csv"))
    parser.add_argument("--out-zones", type=Path, default=Path("data/visual_labels/zones.csv"))
    parser.add_argument("--out-labels", type=Path, default=Path("data/visual_labels/candle_labels.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("data/ai/visual_zone_model.json"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    summary = run(
        calibration_path=args.calibration,
        annotations_path=args.annotations,
        candles_path=args.candles,
        out_zones_path=args.out_zones,
        out_labels_path=args.out_labels,
        model_out_path=args.model_out,
    )
    print(json.dumps(summary, indent=2))
