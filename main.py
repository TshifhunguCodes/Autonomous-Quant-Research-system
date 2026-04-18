from agents.data_agent import run as data_run
from agents.cleaning_agent import run as clean_run
from agents.feature_agent import run as feature_run
from strategy.structure_agent import run as structure_run
from strategy.zone_agent import run as zone_run
from strategy.market_state_agent import run as market_state_run
from strategy.signal_agent import run as signal_run
from strategy.setup_engine import run as setup_run
from strategy.risk_engine import run as risk_run

data_run()
clean_run()
feature_run()
print("STEP 1 COMPLETE")

structure_run()
zone_run()
market_state_run()

signal_run()
setup_run()
risk_run()

print("STEP 2 COMPLETE")


  