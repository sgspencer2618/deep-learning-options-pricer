import argparse
import sys
import os
import logging

# add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from scheduler.data_ingestion_scheduler import DataIngestionScheduler
from config.settings import SchedulerConfig

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Run the data ingestion scheduler')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to process', 
                       default=SchedulerConfig.SYMBOLS)
    parser.add_argument('--days', type=int, help='Number of trading days to fetch', 
                       default=SchedulerConfig.TRADING_DAYS)
    parser.add_argument('--hour', type=int, help='Hour to run daily job (0-23)', 
                       default=SchedulerConfig.SCHEDULE_HOUR)
    parser.add_argument('--minute', type=int, help='Minute to run daily job (0-59)', 
                       default=SchedulerConfig.SCHEDULE_MINUTE)
    parser.add_argument('--test', action='store_true', help='Run test job every minute')
    parser.add_argument('--run-once', action='store_true', help='Run job once and exit')
    
    args = parser.parse_args()

    # Initialize scheduler
    scheduler = DataIngestionScheduler(symbols=args.symbols, days=args.days)

    if args.run_once:
        # Run the job once and exit (no scheduler needed)
        logger.info("Running data ingestion job once...")
        scheduler.run_once()  # Use the new method
        logger.info("Job completed.")
        return

    # Add jobs BEFORE starting the scheduler
    if args.test:
        # Add test job that runs every minute
        scheduler.add_test_job()
        logger.info("Added test job (runs every minute)")
    else:
        # Add daily job
        scheduler.add_daily_job(hour=args.hour, minute=args.minute)
        logger.info(f"Added daily job at {args.hour:02d}:{args.minute:02d}")
    
    # List scheduled jobs BEFORE starting
    scheduler.list_jobs()
    
    # Start the scheduler (this will block until stopped)
    try:
        logger.info("Starting scheduler...")
        scheduler.start()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"Scheduler error: {str(e)}")
    finally:
        # Ensure proper cleanup
        scheduler.shutdown()
        logger.info("Scheduler shutdown complete")

if __name__ == "__main__":
    main()