from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AppPaths:
    root: Path
    config_file: Path
    raw_dir: Path
    clean_dir: Path
    features_dir: Path
    backtest_dir: Path
    backtest_alpha_trades: Path
    backtest_alpha_summary: Path
    backtest_flow_trades: Path
    backtest_flow_summary: Path
    replay_dir: Path
    stress_dir: Path
    live_dir: Path
    logs_dir: Path
    raw_m5: Path
    raw_h1: Path
    clean_m5: Path
    clean_h1: Path
    m5_features: Path
    mtf_context: Path
    structure: Path
    zones: Path
    market_state: Path
    regime_context: Path
    signals: Path
    setups: Path
    confirmed_signals: Path
    trade_setups: Path
    backtest_trades: Path
    backtest_summary: Path
    rolling_backtest_summary: Path
    monthly_performance: Path
    equity_curve: Path
    losing_streaks: Path
    losing_streak_stats: Path
    session_performance: Path
    best_setup_types: Path
    replay_decisions: Path
    replay_events: Path
    replay_trades: Path
    replay_summary: Path
    stress_runs: Path
    stress_regimes: Path
    stress_random_slices: Path
    stress_expectancy_distribution: Path
    stress_robustness_summary: Path
    execution_log: Path
    app_log: Path

    @classmethod
    def from_root(cls, root: Path, config_file: Path) -> "AppPaths":
        data_dir = root / "data"
        raw_dir = data_dir / "raw"
        clean_dir = data_dir / "clean"
        features_dir = data_dir / "research"
        backtest_dir = data_dir / "backtest"
        replay_dir = data_dir / "replay"
        stress_dir = data_dir / "stress"
        live_dir = data_dir / "live"
        logs_dir = root / "logs"
        return cls(
            root=root,
            config_file=config_file,
            raw_dir=raw_dir,
            clean_dir=clean_dir,
            features_dir=features_dir,
            backtest_dir=backtest_dir,
            backtest_alpha_trades=backtest_dir / "alpha_trades.csv",
            backtest_alpha_summary=backtest_dir / "alpha_summary.csv",
            backtest_flow_trades=backtest_dir / "flow_trades.csv",
            backtest_flow_summary=backtest_dir / "flow_summary.csv",
            replay_dir=replay_dir,
            stress_dir=stress_dir,
            live_dir=live_dir,
            logs_dir=logs_dir,
            raw_m5=raw_dir / "xauusd_m5.csv",
            raw_h1=raw_dir / "xauusd_h1.csv",
            clean_m5=clean_dir / "xauusd_m5_clean.csv",
            clean_h1=clean_dir / "xauusd_h1_clean.csv",
            m5_features=features_dir / "xauusd_m5_features.csv",
            mtf_context=features_dir / "mtf_context.csv",
            structure=features_dir / "structure.csv",
            zones=features_dir / "zones.csv",
            market_state=features_dir / "market_state.csv",
            regime_context=features_dir / "regime_context.csv",
            signals=features_dir / "signals.csv",
            setups=features_dir / "setups.csv",
            confirmed_signals=features_dir / "confirmed_signals.csv",
            trade_setups=features_dir / "trade_setups.csv",
            backtest_trades=backtest_dir / "backtest_trades.csv",
            backtest_summary=backtest_dir / "backtest_summary.csv",
            rolling_backtest_summary=backtest_dir / "rolling_backtest_summary.csv",
            monthly_performance=backtest_dir / "monthly_performance.csv",
            equity_curve=backtest_dir / "equity_curve.csv",
            losing_streaks=backtest_dir / "losing_streaks.csv",
            losing_streak_stats=backtest_dir / "losing_streak_stats.csv",
            session_performance=backtest_dir / "session_performance.csv",
            best_setup_types=backtest_dir / "best_setup_types.csv",
            replay_decisions=replay_dir / "replay_decisions.csv",
            replay_events=replay_dir / "replay_events.csv",
            replay_trades=replay_dir / "replay_trades.csv",
            replay_summary=replay_dir / "replay_summary.csv",
            stress_runs=stress_dir / "stress_runs.csv",
            stress_regimes=stress_dir / "stress_regimes.csv",
            stress_random_slices=stress_dir / "stress_random_slices.csv",
            stress_expectancy_distribution=stress_dir / "stress_expectancy_distribution.csv",
            stress_robustness_summary=stress_dir / "stress_robustness_summary.csv",
            execution_log=live_dir / "execution_log.csv",
            app_log=logs_dir / "quant_system.log",
        )


@dataclass(frozen=True)
class MarketConfig:
    symbol: str = "XAUUSD"
    point_size: float = 0.01
    m5_bars: int = 5000
    h1_bars: int = 2000
    history_years: float = 2.0


@dataclass(frozen=True)
class ZoneConfig:
    near_threshold: float = 3.0
    major_threshold: float = 5.0


@dataclass(frozen=True)
class RiskConfig:
    rr_ratio: float = 2.0
    base_stop_buffer: float = 3.5
    use_atr_sizing: bool = False
    atr_period: int = 14
    atr_risk_per_unit: float = 0.01


@dataclass(frozen=True)
class RegimeConfig:
    aligned_range_risk_multiplier: float = 1.0
    aligned_trend_risk_multiplier: float = 0.9
    neutral_risk_multiplier: float = 0.75
    trend_mismatch_risk_multiplier: float = 0.55
    volatile_risk_multiplier: float = 0.6
    choppy_risk_multiplier: float = 0.25
    medium_quality_risk_multiplier: float = 0.85
    high_quality_risk_multiplier: float = 1.0
    elite_quality_risk_multiplier: float = 1.0
    block_choppy_non_elite: bool = True
    block_volatile_medium: bool = True
    adaptive_ny_guard: bool = True
    flow_risk_multiplier: float = 0.5  # System B risk dampening
    alpha_session_hours: list[int] = field(default_factory=lambda: list(range(2, 16))) # System A hours
    dynamic_priority: bool = True
    priority_lookback: int = 15 # Number of recent trades to evaluate
    setup_weights: dict[str, float] = field(
        default_factory=lambda: {
            "SELL_SETUP_ELITE_RANGING": 1.2,
            "BUY_SETUP_ELITE_RANGING": 1.1,
            "BUY_SETUP_ELITE_VOLATILE": 0.5,
            "SELL_SETUP_MEDIUM_TRENDING": 0.0,
        }
    )


@dataclass(frozen=True)
class BacktestConfig:
    starting_balance: float = 10_000.0
    risk_per_trade: float = 0.01
    slippage_points: float = 5.0
    commission_per_trade: float = 2.5
    allow_overlapping_positions: bool = False


@dataclass(frozen=True)
class LiveConfig:
    enabled: bool = False
    lot: float = 0.01
    min_confirm_score: int = 60
    approved_qualities: list[str] = field(
        default_factory=lambda: ["ELITE", "HIGH"]
    )
    allow_duplicate_candle: bool = False
    require_h1_alignment: bool = True
    disallowed_market_states: list[str] = field(
        default_factory=lambda: ["CHOPPY"]
    )
    max_signal_age_minutes: int = 20
    max_spread_points: float = 25.0


@dataclass(frozen=True)
class SessionFiltersConfig:
    disable_late_session: bool = False
    late_session_start_hour: int = 20
    disabled_sessions: list[str] = field(default_factory=lambda: [])
    disabled_market_states: list[str] = field(default_factory=lambda: [])


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    paths: AppPaths
    market: MarketConfig
    zones: ZoneConfig
    risk: RiskConfig
    regime: RegimeConfig
    backtest: BacktestConfig
    live: LiveConfig
    session_filters: SessionFiltersConfig
    logging: LoggingConfig


def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    return value if isinstance(value, dict) else {}


def load_config(config_path: str | None = None) -> AppConfig:
    root = Path(__file__).resolve().parents[1]
    resolved_config = Path(config_path) if config_path else root / "config" / "app_config.json"
    if not resolved_config.is_absolute():
        resolved_config = (root / resolved_config).resolve()

    payload: dict[str, Any] = {}
    if resolved_config.exists():
        payload = json.loads(resolved_config.read_text(encoding="utf-8"))

    paths = AppPaths.from_root(root=root, config_file=resolved_config)
    return AppConfig(
        paths=paths,
        market=MarketConfig(**_section(payload, "market")),
        zones=ZoneConfig(**_section(payload, "zones")),
        risk=RiskConfig(**_section(payload, "risk")),
        regime=RegimeConfig(**_section(payload, "regime")),
        backtest=BacktestConfig(**_section(payload, "backtest")),
        live=LiveConfig(**_section(payload, "live")),
        session_filters=SessionFiltersConfig(**_section(payload, "session_filters")),
        logging=LoggingConfig(**_section(payload, "logging")),
    )


def ensure_runtime_dirs(config: AppConfig) -> None:
    directories = [
        config.paths.raw_dir,
        config.paths.clean_dir,
        config.paths.features_dir,
        config.paths.backtest_dir,
        config.paths.replay_dir,
        config.paths.stress_dir,
        config.paths.live_dir,
        config.paths.logs_dir,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
