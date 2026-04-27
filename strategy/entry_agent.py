import numpy as np
import pandas as pd

from core.logging_utils import get_logger
from strategy.pipeline_transforms import build_trade_setups, build_confirmations
from strategy.smc_ict_engine import SMCEngine
from strategy.volume_profile_engine import VolumeProfileEngine

logger = get_logger(__name__)

def run(config):
    df = pd.read_csv(config.paths.confirmed_signals, parse_dates=["time"])
    
    # Build standard setups
    setups = build_trade_setups(df, config)
    
    # Step 1: Enrich with Volume Profile FIRST (POC/VAH/VAL needed for SMC scoring)
    enriched = VolumeProfileEngine.enrich_intelligence(setups)
    
    # Step 2: Enrich with SMC Intelligence (now has access to Volume nodes)
    enriched = SMCEngine.enrich_intelligence(enriched)

    # Step 3: Re-calculate confirmations to include SMC/Volume confluence points
    # This ensures confirm_score and Quality reflect the new intelligence.
    enriched = build_confirmations(enriched)

    # Final Save
    enriched.to_csv(config.paths.trade_setups, index=False)
    logger.info("Trade setups saved at %s", config.paths.trade_setups)
