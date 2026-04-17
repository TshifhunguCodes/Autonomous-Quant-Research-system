from agents.data_agent import run as data_run
from agents.cleaning_agent import run as clean_run
from agents.feature_agent import run as feature_run

data_run()
clean_run()
feature_run()

print("STEP 1 COMPLETE")