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
        self.paths = self._build_paths(base_config)

    def _build_paths(self, base_config: Any) -> Any:
        class Paths:
            pass

        paths = Paths()
        paths.clean_m5 = Path(base_config.paths.clean_m5)
        paths.raw_m5 = Path(base_config.paths.raw_m5)
        return paths

    @classmethod
    def load_from(cls, base_config: Any):
        return cls(base_config)
