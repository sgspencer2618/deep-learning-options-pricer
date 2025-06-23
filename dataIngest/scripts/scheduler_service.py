import argparse
import sys
import os
import logging
from datetime import datetime
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from scheduler.data_ingestion_scheduler import DataIngestionScheduler
from config.settings import SchedulerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, 'logs', 'service.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def log_system_info():
    """Log system information."""
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current time: {datetime.now()}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Script path: {os.path.abspath(__file__)}")

def main():
    # Log system information
    log_system_info()
    
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

    # Clean up symbols (remove any extra quotes)
    if args.symbols and isinstance(args.symbols, list):
        args.symbols = [symbol.strip('"\'') for symbol in args.symbols]
    
    logger.info(f"Starting scheduler with settings:")
    logger.info(f"Symbols: {args.symbols}")
    logger.info(f"Days: {args.days}")
    logger.info(f"Schedule time: {args.hour:02d}:{args.minute:02d}")
    logger.info(f"Test mode: {args.test}")
    logger.info(f"Run once: {args.run_once}")

    # Initialize scheduler with memory job store
    scheduler = DataIngestionScheduler(symbols=args.symbols, days=args.days)

    if args.run_once:
        # Run the job once and exit (no scheduler needed)
        logger.info("Running data ingestion job once...")
        scheduler.run_once()
        logger.info("Job completed.")
        return

    # Add jobs
    if args.test:
        # Add test job that runs every minute
        scheduler.add_test_job()
        logger.info("Added test job (runs every minute)")
        
        # Debug output
        jobs = scheduler.scheduler.get_jobs()
        for job in jobs:
            logger.info(f"Debug - Job in scheduler: {job.id}, trigger: {job.trigger}")
    else:
        # Add daily job
        scheduler.add_daily_job(hour=args.hour, minute=args.minute)
        logger.info(f"Added daily job at {args.hour:02d}:{args.minute:02d}")
    
    # Start the scheduler
    try:
        logger.info("Starting scheduler...")
        
        # Start the scheduler directly (no paused=True)
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