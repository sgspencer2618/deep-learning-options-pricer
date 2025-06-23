import logging
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
import sys
import os
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from src.data_ingest import upload_options_data_to_s3

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


def daily_ingestion_job(symbols, days):
    """Job function that runs daily to ingest options data."""
    logger.info(f"Starting daily data ingestion at {datetime.now()}")
    
    for symbol in symbols:
        try:
            # Clean symbol if needed
            symbol = symbol.strip('"\'')
            logger.info(f"Processing data for {symbol}")
            success = None
            success = upload_options_data_to_s3(symbol, 25)

            if success is None:
                logger.warning(f"No success, testing.")
            elif success:
                logger.info(f"Successfully ingested data for {symbol}")
            else:
                logger.error(f"Failed to ingest data for {symbol}")
                
        except Exception as e:
            logger.error(f"Error processing {symbol}: {str(e)}")

def test_job():
    """Test job function that runs every minute."""
    logger.info(f"Test Job Success")
    

class DataIngestionScheduler:
    def __init__(self, symbols=None, days=1):
        self.symbols = symbols or ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        self.days = days
        
        # Use MemoryJobStore instead of SQLAlchemyJobStore
        jobstores = {
            'default': MemoryJobStore()
        }
        
        self.scheduler = BlockingScheduler(jobstores=jobstores)
        logger.info("Scheduler configured with in-memory job store (no persistence)")
        
        # Add event listeners to log job execution and next run time
        self.scheduler.add_listener(self._log_job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._log_job_error, EVENT_JOB_ERROR)

    def _log_job_executed(self, event):
        """Log when a job is successfully executed and its next run time."""
        job = self.scheduler.get_job(event.job_id)
        if job:
            logger.info(f"Job {job.id} executed successfully")
            if job.next_run_time:
                logger.info(f"Next run time for {job.id}: {job.next_run_time}")
            else:
                logger.info(f"Job {job.id} has no future runs scheduled")

    def _log_job_error(self, event):
        """Log when a job execution fails."""
        job = self.scheduler.get_job(event.job_id)
        job_id = event.job_id if event else "Unknown"
        exception = event.exception if event else "Unknown error"
        
        logger.error(f"Error executing job {job_id}: {exception}")
        
        if job and job.next_run_time:
            logger.info(f"Next run time for {job.id} despite error: {job.next_run_time}")
    
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
                func=test_job,
                trigger='cron',
                args=[],
                minute='*',
                id='test_job',  # Make sure this ID is consistent
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

        self.list_jobs()
    
    def shutdown(self):
        """Shutdown the scheduler gracefully."""
        try:
            if self.scheduler.running:
                logger.info("Shutting down scheduler...")
                self.scheduler.shutdown(wait=False)
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}")
    
    def list_jobs(self):
        """List all scheduled jobs with detailed information."""
        try:
            jobs = self.scheduler.get_jobs()
            if not jobs:
                logger.info("No jobs scheduled")
            else:
                logger.info(f"Found {len(jobs)} scheduled job(s):")
                for job in jobs:
                    # Get job details
                    job_id = job.id
                    
                    # Check if next_run_time exists (it won't before scheduler starts)
                    if hasattr(job, 'next_run_time') and job.next_run_time is not None:
                        next_run = job.next_run_time
                    else:
                        next_run = "Not calculated yet (scheduler not started)"
                    
                    # Get trigger details
                    trigger_type = job.trigger.__class__.__name__
                    trigger_details = ""
                    
                    # Extract specific trigger details based on type
                    if trigger_type == "CronTrigger":
                        fields = job.trigger.fields
                        field_values = {f.name: str(f) for f in fields}
                        
                        # Format for better readability
                        if "minute" in field_values and field_values["minute"] == "*":
                            trigger_details = "Runs every minute"
                        else:
                            hour = field_values.get("hour", "0")
                            minute = field_values.get("minute", "0")
                            trigger_details = f"Runs daily at {hour}:{minute}"
                    
                    # Log job information
                    logger.info(f"Job ID: {job_id}")
                    logger.info(f"  Next run: {next_run}")
                    logger.info(f"  Trigger: {trigger_type}")
                    logger.info(f"  Details: {trigger_details}")

                    if job_id == 'test_job':
                        logger.info("  Function: test_job")
                    else:
                        logger.info(f"  Function: {job.func.__name__}")
                    
                    logger.info(f"  Args: {job.args}")
            
            return jobs
        except Exception as e:
            logger.error(f"Error listing jobs: {e}")
            return []

def log_memory_usage():
    """Log current memory usage."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        logger.info(f"Memory usage: {memory_info.rss / 1024 / 1024:.2f} MB")
    except ImportError:
        logger.warning("psutil not installed, cannot log memory usage")
    except Exception as e:
        logger.error(f"Error logging memory usage: {e}")

# Example usage
if __name__ == "__main__":
    scheduler = DataIngestionScheduler(symbols=['AAPL'], days=1)
    
    # Schedule test job
    scheduler.add_test_job()

    logger.debug("this is running by the way lmao")
    
    # List all scheduled jobs with details
    scheduler.list_jobs()
    
    # Log memory usage
    log_memory_usage()
    
    # Start the scheduler
    scheduler.start()

    # Wait briefly for scheduler to initialize
    time.sleep(1)

    # Log all scheduled jobs with their next run times
    logger.info("Scheduled job executions:")
    for job in scheduler.scheduler.get_jobs():
        logger.info(f"Job {job.id} next run: {job.next_run_time}")