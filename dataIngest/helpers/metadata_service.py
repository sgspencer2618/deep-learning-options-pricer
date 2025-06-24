from datetime import datetime
import os
import logging

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

def update_ingestion_metadata(uploader, symbol, dates, success=True):
    """Update metadata about ingested data in S3."""
    try:
        # Metadata key path
        metadata_key = f"options-data/{symbol}/{symbol}_ingestion_metadata.json"
        
        # Try to get existing metadata
        try:
            existing_metadata = uploader.read_json_from_s3(metadata_key)
        except:
            # If file doesn't exist, create new metadata
            existing_metadata = {
                "earliest_date": None,
                "latest_date": None,
                "total_dates": 0,
                "ingestion_history": []
            }
        
        # Get current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Sort dates to find earliest and latest
        if dates:
            sorted_dates = sorted(dates)
            earliest_new_date = sorted_dates[0]
            latest_new_date = sorted_dates[-1]
            
            # Update earliest date if it's earlier than what we have
            if existing_metadata["earliest_date"] is None or earliest_new_date < existing_metadata["earliest_date"]:
                existing_metadata["earliest_date"] = earliest_new_date
                
            # Update latest date
            if existing_metadata["latest_date"] is None or latest_new_date > existing_metadata["latest_date"]:
                existing_metadata["latest_date"] = latest_new_date
                
            # Update total dates
            existing_metadata["total_dates"] += len(dates)
        
        # Add entry to ingestion history
        ingestion_record = {
            "timestamp": current_time,
            "dates_ingested": dates,
            "success": success,
            "count": len(dates)
        }
        
        existing_metadata["ingestion_history"].append(ingestion_record)
        
        # Keep only the last 10 ingestion records to avoid metadata file growing too large
        existing_metadata["ingestion_history"] = existing_metadata["ingestion_history"][-10:]
        
        # Upload updated metadata
        uploader.upload_json_to_s3(existing_metadata, metadata_key)
        logger.info(f"Updated ingestion metadata for {symbol}. Earliest date: {existing_metadata['earliest_date']}")
        
        return True
    except Exception as e:
        logger.error(f"Error updating ingestion metadata: {str(e)}")
        return False