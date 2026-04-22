import random

import pandas as pd

from core.logging_utils import get_logger
from strategy.replay_engine import run_replay_frame
from strategy.pipeline_transforms import (
    build_regime_layer,
    build_market_state,
    build_m5_features,
    build_structure,
    build_zones,
    merge_h1_context_into_m5,
)


logger = get_logger(__name__)


def _filter_m5_window(df, start, end):
    return df[(df["time"] >= start) & (df["time"] <= end)].copy()


def _build_market_regime_windows(m5: pd.DataFrame) -> list[dict]:
    raise NotImplementedError("Use _build_market_regime_windows_from_pipeline instead.")


def _build_market_regime_windows_from_pipeline(m5: pd.DataFrame, h1: pd.DataFrame, config) -> list[dict]:
    enriched = merge_h1_context_into_m5(build_m5_features(m5), h1)
    regime_frame = build_regime_layer(
        build_market_state(build_zones(build_structure(enriched), config)),
        config,
    )
    regime_frame["day"] = regime_frame["time"].dt.normalize()

    daily_regimes = (
        regime_frame.groupby("day")["market_regime"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .sort_values("day")
        .reset_index(drop=True)
    )
    if daily_regimes.empty:
        return []

    daily_regimes["block_id"] = (
        daily_regimes["market_regime"] != daily_regimes["market_regime"].shift(1)
    ).cumsum()

    windows = []
    for _, group in daily_regimes.groupby("block_id"):
        regime = group["market_regime"].iloc[0]
        start = pd.Timestamp(group["day"].iloc[0])
        end = pd.Timestamp(group["day"].iloc[-1]) + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        windows.append(
            {
                "slice_kind": "regime",
                "slice_name": f"{regime}_{start.date()}_{group['day'].iloc[-1].date()}",
                "regime": regime,
                "start": start,
                "end": end,
            }
        )
    return windows


def _build_random_walk_forward_windows(
    m5: pd.DataFrame,
    num_runs: int,
    window_days: int,
    seed: int,
):
    df = m5.copy().sort_values("time").reset_index(drop=True)
    unique_days = sorted(df["time"].dt.normalize().unique().tolist())
    if len(unique_days) < window_days:
        return []

    rng = random.Random(seed)
    max_start_index = max(0, len(unique_days) - window_days)
    if max_start_index == 0:
        chosen_indices = [0]
    else:
        sample_size = min(num_runs, max_start_index + 1)
        chosen_indices = sorted(rng.sample(range(max_start_index + 1), sample_size))

    windows = []
    for i, day_idx in enumerate(chosen_indices, start=1):
        start = pd.Timestamp(unique_days[day_idx])
        end_day = pd.Timestamp(unique_days[day_idx + window_days - 1])
        end = end_day + pd.Timedelta(days=1) - pd.Timedelta(minutes=5)
        windows.append(
            {
                "slice_kind": "random_walk_forward",
                "slice_name": f"walk_forward_{i:02d}",
                "regime": "MIXED",
                "start": start,
                "end": end,
            }
        )
    return windows


def _summarize_slice(summary_df: pd.DataFrame, trades_df: pd.DataFrame, meta: dict) -> dict:
    row = summary_df.iloc[0].to_dict()
    closed = trades_df[trades_df["result"].isin(["WIN", "LOSS"])].copy()
    expectancy_values = closed["pnl"].tolist() if not closed.empty else []
    expectancy = round(float(closed["pnl"].mean()), 2) if not closed.empty else 0.0
    positive_expectancy = int(expectancy > 0)
    row.update(
        {
            "slice_kind": meta["slice_kind"],
            "slice_name": meta["slice_name"],
            "regime": meta["regime"],
            "slice_start": meta["start"],
            "slice_end": meta["end"],
            "expectancy": expectancy,
            "positive_expectancy": positive_expectancy,
            "expectancy_values": "|".join(f"{value:.2f}" for value in expectancy_values),
        }
    )
    return row


def _build_robustness_summary(stress_runs: pd.DataFrame) -> pd.DataFrame:
    if stress_runs.empty:
        return pd.DataFrame(
            [
                {
                    "slice_count": 0,
                    "profitable_slice_ratio": 0.0,
                    "positive_expectancy_ratio": 0.0,
                    "consistency_score": 0.0,
                    "avg_drawdown_pct": 0.0,
                    "drawdown_std_pct": 0.0,
                    "drawdown_stability_score": 0.0,
                    "expectancy_mean": 0.0,
                    "expectancy_std": 0.0,
                    "expectancy_p10": 0.0,
                    "expectancy_p50": 0.0,
                    "expectancy_p90": 0.0,
                }
            ]
        )

    profitable_slice_ratio = round(float((stress_runs["net_pnl"] > 0).mean()), 4)
    positive_expectancy_ratio = round(float(stress_runs["positive_expectancy"].mean()), 4)
    consistency_score = round((profitable_slice_ratio + positive_expectancy_ratio) / 2, 4)

    drawdown_mean = float(stress_runs["max_drawdown_pct"].mean()) if not stress_runs.empty else 0.0
    drawdown_std = float(stress_runs["max_drawdown_pct"].std(ddof=0)) if len(stress_runs) > 1 else 0.0
    if drawdown_mean <= 0:
        drawdown_stability_score = 1.0 if drawdown_std == 0 else 0.0
    else:
        drawdown_stability_score = max(0.0, round(1 - (drawdown_std / drawdown_mean), 4))

    expectancy_series = stress_runs["expectancy"].astype(float)
    return pd.DataFrame(
        [
            {
                "slice_count": len(stress_runs),
                "profitable_slice_ratio": profitable_slice_ratio,
                "positive_expectancy_ratio": positive_expectancy_ratio,
                "consistency_score": consistency_score,
                "avg_drawdown_pct": round(drawdown_mean, 2),
                "drawdown_std_pct": round(drawdown_std, 2),
                "drawdown_stability_score": drawdown_stability_score,
                "expectancy_mean": round(float(expectancy_series.mean()), 2),
                "expectancy_std": round(float(expectancy_series.std(ddof=0)) if len(expectancy_series) > 1 else 0.0, 2),
                "expectancy_p10": round(float(expectancy_series.quantile(0.10)), 2),
                "expectancy_p50": round(float(expectancy_series.quantile(0.50)), 2),
                "expectancy_p90": round(float(expectancy_series.quantile(0.90)), 2),
            }
        ]
    )


def _build_expectancy_distribution(stress_runs: pd.DataFrame) -> pd.DataFrame:
    if stress_runs.empty:
        return pd.DataFrame(
            columns=["slice_name", "slice_kind", "regime", "closed_trades", "expectancy", "net_pnl"]
        )
    return stress_runs[
        ["slice_name", "slice_kind", "regime", "closed_trades", "expectancy", "net_pnl"]
    ].sort_values("expectancy").reset_index(drop=True)


def run(config, random_runs: int = 6, window_days: int = 5, seed: int = 42):
    m5 = pd.read_csv(config.paths.clean_m5, parse_dates=["time"])
    h1 = pd.read_csv(config.paths.clean_h1, parse_dates=["time"])

    regime_windows = _build_market_regime_windows_from_pipeline(m5, h1, config)
    random_windows = _build_random_walk_forward_windows(
        m5=m5,
        num_runs=random_runs,
        window_days=window_days,
        seed=seed,
    )
    all_windows = regime_windows + random_windows

    run_rows = []
    for meta in all_windows:
        replay_m5 = _filter_m5_window(m5, meta["start"], meta["end"])
        if replay_m5.empty:
            continue
        result = run_replay_frame(
            replay_m5=replay_m5,
            h1=h1[h1["time"] <= meta["end"]].copy(),
            config=config,
            label=meta["slice_name"],
        )
        run_rows.append(_summarize_slice(result["summary"], result["trades"], meta))

    stress_runs = pd.DataFrame(run_rows)
    regime_df = stress_runs[stress_runs["slice_kind"] == "regime"].reset_index(drop=True)
    random_df = stress_runs[stress_runs["slice_kind"] == "random_walk_forward"].reset_index(drop=True)
    expectancy_distribution = _build_expectancy_distribution(stress_runs)
    robustness_summary = _build_robustness_summary(stress_runs)

    stress_runs.to_csv(config.paths.stress_runs, index=False)
    regime_df.to_csv(config.paths.stress_regimes, index=False)
    random_df.to_csv(config.paths.stress_random_slices, index=False)
    expectancy_distribution.to_csv(config.paths.stress_expectancy_distribution, index=False)
    robustness_summary.to_csv(config.paths.stress_robustness_summary, index=False)

    logger.info("Stress runs saved at %s", config.paths.stress_runs)
    logger.info("Stress regime summary saved at %s", config.paths.stress_regimes)
    logger.info("Stress random slices saved at %s", config.paths.stress_random_slices)
    logger.info(
        "Stress expectancy distribution saved at %s",
        config.paths.stress_expectancy_distribution,
    )
    logger.info(
        "Stress robustness summary saved at %s",
        config.paths.stress_robustness_summary,
    )
    return robustness_summary
