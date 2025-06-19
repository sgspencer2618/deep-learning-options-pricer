import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.getenv('LOG_FILE', 'scheduler.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import necessary action functions from src
from src.data_ingest import upload_options_data_to_s3

def daily_ingestion_job(symbols, days):
    """Job function that runs daily to ingest options data."""
    logger.info(f"Starting daily data ingestion at {datetime.now()}")
    
    for symbol in symbols:
        try:
            logger.info(f"Processing data for {symbol}")
            success = upload_options_data_to_s3(symbol, 25)

            if success:
                logger.info(f"Successfully ingested data for {symbol}")
            else:
                logger.error(f"Failed to ingest data for {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")

class DataIngestionScheduler:
    def __init__(self, symbols=None, days=os.getenv('TRADING_DAYS', 1), db_path='scheduler_jobs.sqlite'):
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        self.days = days
        
        # Configure with SQLAlchemy job store for persistence
        jobstores = {
            'default': SQLAlchemyJobStore(url=f'sqlite:///{db_path}')
        }
        
        self.scheduler = BlockingScheduler(jobstores=jobstores)
        logger.info(f"Scheduler configured with SQLite database: {db_path}")
    
    def add_daily_job(self, hour=9, minute=30):
        """Add a daily job."""
        try:
            self.scheduler.add_job(
                func=daily_ingestion_job,
                trigger='cron',
                args=[self.symbols, self.days],
                hour=hour,
                minute=minute,
                id='daily_job',
                replace_existing=True
            )
            logger.info(f"Scheduled daily job at {hour:02d}:{minute:02d}")
            return True
        except Exception as e:
            logger.error(f"Error adding daily job: {e}")
            return False
    
    def add_test_job(self):
        """Add a test job that runs every minute."""
        try:
            self.scheduler.add_job(
                func=daily_ingestion_job,
                trigger='cron',
                args=[self.symbols, self.days],
                minute='*',
                id='test_job',
                replace_existing=True
            )
            logger.info("Scheduled test job to run every minute")
            return True
        except Exception as e:
            logger.error(f"Error adding test job: {e}")
            return False
    
    def run_once(self):
        """Run the job once without scheduling."""
        daily_ingestion_job(self.symbols, self.days)
    
    def start(self):
        """Start the scheduler."""
        try:
            logger.info("Starting scheduler...")
            self.scheduler.start()
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
        except Exception as e:
            logger.error(f"Error starting the scheduler: {e}")
            raise
    
    def shutdown(self):
        """Shutdown the scheduler gracefully."""
        try:
            if self.scheduler.running:
                logger.info("Shutting down scheduler...")
                self.scheduler.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")
    
    def list_jobs(self):
        """List all scheduled jobs."""
        try:
            jobs = self.scheduler.get_jobs()
            if not jobs:
                logger.info("No jobs scheduled")
            else:
                for job in jobs:
                    logger.info(f"Job ID: {job.id} - Next run: {job.next_run_time}")
            return jobs
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return []

if __name__ == "__main__":
    # Example usage
    symbols = ['AAPL']
    scheduler = DataIngestionScheduler(symbols=symbols, days=1, db_path='test_scheduler.sqlite')
    
    # Schedule test job
    scheduler.add_test_job()
    
    # List scheduled jobs
    scheduler.list_jobs()
    
    # Start the scheduler
    scheduler.start()