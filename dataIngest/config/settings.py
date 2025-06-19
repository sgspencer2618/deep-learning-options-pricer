import os
from dotenv import load_dotenv

load_dotenv()

class SchedulerConfig:
    # Stock symbols to fetch data for
    SYMBOLS = os.getenv("SYMBOLS", "AAPL,MSFT,GOOGL,TSLA").split(",")
    
    # Number of trading days to fetch (1 for daily incremental updates)
    TRADING_DAYS = int(os.getenv("TRADING_DAYS", "1"))
    
    # Schedule time (24-hour format)
    SCHEDULE_HOUR = int(os.getenv("SCHEDULE_HOUR", "9"))
    SCHEDULE_MINUTE = int(os.getenv("SCHEDULE_MINUTE", "30"))
    
    # Timezone
    TIMEZONE = os.getenv("TIMEZONE", "America/New_York")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "scheduler.log")