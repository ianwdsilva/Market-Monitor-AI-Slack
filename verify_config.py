
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.config import settings

print(f"DEBUG: PAPER_TRADING_STARTING_BALANCE from env/settings: {settings.paper_trading_starting_balance}")

# Check if paper trader initializes with it
# Note: we can't fully initialize paper trader without potentially creating files or needing other configs
# but we can inspect the class if we mock the singleton or just trust the config test.
# Let's try to inspect the logic by reading the source or just trusting the config print.
# Actually, let's try to instantiate it with a mock webhook to see the attribute.

from app.services.paper_trader import PaperTrader

pt = PaperTrader(
    initial_capital=settings.paper_trading_starting_balance,
    webhook_url=None,
    data_dir="temp_data_test"
)

print(f"DEBUG: PaperTrader.initial_capital: {pt.initial_capital}")

if pt.initial_capital == 100000.0:
    print("SUCCESS: Default value loaded correctly (or env value if set to 100000)")
else:
    print(f"FAILURE: Expected 100000.0, got {pt.initial_capital}")

# Clean up
import shutil
if os.path.exists("temp_data_test"):
    shutil.rmtree("temp_data_test")
