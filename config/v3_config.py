from __future__ import annotations
from pathlib import Path
from typing import Any


class V3Config:
    """Wrapper config for AQRS V3 that can reuse existing core config values."""

    def __init__(self, base_config: Any):
        self.base = base_config
        self.market = base_config.market
        self.backtest = base_config.backtest
        self.risk = base_config.risk
        self.regime = base_config.regime
        self.live = getattr(base_config, "live", None)
        self.session_filters = getattr(base_config, "session_filters", None)
        self.smc = getattr(base_config, "smc", None)
        self.telegram = getattr(base_config, "telegram", None)
        self.paths = self._build_paths(base_config)

    def _build_paths(self, base_config: Any) -> Any:
        class Paths:
            pass

        paths = Paths()
        for name, value in vars(base_config.paths).items():
            setattr(paths, name, Path(value) if isinstance(value, (str, Path)) else value)
        return paths

    @classmethod
    def load_from(cls, base_config: Any):
        return cls(base_config)
