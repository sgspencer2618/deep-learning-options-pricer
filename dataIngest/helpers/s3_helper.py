import boto3
import pandas as pd
import io
import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class S3Uploader:
    def __init__(self):
        """Initialize S3 client with credentials from environment variables."""
        self.access_key = os.getenv('ACCESS_KEY')
        self.secret_key = os.getenv('SECRET_KEY')
        self.region = os.getenv('AWS_REGION')
        self.bucket_name = os.getenv('BUCKET_NAME')
        
        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            region_name=self.region
        )
        logger.info(f"S3Uploader initialized with bucket: {self.bucket_name}")

    def create_bucket(self, bucket_name: str = None) -> bool:
        """
        Create an S3 bucket.
        
        Args:
            bucket_name (str): Name of the bucket to create. If None, uses the default bucket name.
            
        Returns:
            bool: True if bucket created successfully, False otherwise
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
            
        try:
            # Create bucket with location constraint for regions other than us-east-1
            if self.region != 'us-east-1':
                self.s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region}
                )
            else:
                self.s3_client.create_bucket(Bucket=bucket_name)
            
            print(f"Successfully created bucket: {bucket_name}")
            return True
            
        except Exception as e:
            print(f"Error creating bucket {bucket_name}: {str(e)}")
            return False


    def bucket_exists(self, bucket_name: str = None) -> bool:
        """
        Check if a bucket exists.
        
        Args:
            bucket_name (str): Name of the bucket to check. If None, uses the default bucket name.
            
        Returns:
            bool: True if bucket exists, False otherwise
        """
        if bucket_name is None:
            bucket_name = self.bucket_name
            
        try:
            self.s3_client.head_bucket(Bucket=bucket_name)
            return True
        except Exception:
            return False
    
    def upload_dataframe_as_csv(self, df: pd.DataFrame, s3_key: str) -> bool:
        """
        Upload a pandas DataFrame to S3 as CSV format.
        
        Args:
            df (pd.DataFrame): The DataFrame to upload
            s3_key (str): The S3 key (path) where the file will be stored
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            # Convert DataFrame to CSV string
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=csv_string,
                ContentType='text/csv'
            )
            
            print(f"Successfully uploaded DataFrame to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            print(f"Error uploading DataFrame to S3: {str(e)}")
            return False
    
    def upload_dataframe_as_parquet(self, df: pd.DataFrame, s3_key: str) -> bool:
        """
        Upload a pandas DataFrame to S3 as Parquet format.
        
        Args:
            df (pd.DataFrame): The DataFrame to upload
            s3_key (str): The S3 key (path) where the file will be stored
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            logger.info(f"Uploading DataFrame with {len(df)} rows to S3 key: {s3_key}")
            
            # Convert DataFrame to Parquet format in memory
            parquet_buffer = io.BytesIO()
            df.to_parquet(parquet_buffer, index=False)
            parquet_bytes = parquet_buffer.getvalue()
            
            # Upload to S3
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=parquet_bytes,
                ContentType='application/octet-stream'
            )
            
            print(f"Successfully uploaded DataFrame to s3://{self.bucket_name}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Error uploading DataFrame to S3: {str(e)}")
            return False
    
    def upload_local_file(self, local_file_path: str, s3_key: str) -> bool:
        """
        Upload a local file to S3.
        
        Args:
            local_file_path (str): Path to the local file
            s3_key (str): The S3 key (path) where the file will be stored
            
        Returns:
            bool: True if upload successful, False otherwise
        """
        try:
            self.s3_client.upload_file(local_file_path, self.bucket_name, s3_key)
            print(f"Successfully uploaded {local_file_path} to s3://{self.bucket_name}/{s3_key}")
            return True
            
        except Exception as e:
            print(f"Error uploading file to S3: {str(e)}")
            return False
        

    def read_parquet_from_s3(self, s3_key: str) -> Optional[pd.DataFrame]:
        """
        Read a Parquet file from S3 and return it as a DataFrame.
        
        Args:
            s3_key (str): The S3 key (path) of the Parquet file
            
        Returns:
            pd.DataFrame: The DataFrame containing the data, or None if an error occurs
        """
        
        try:
            logger.info(f"Reading Parquet file from S3: {s3_key}")
            
            # Get the object from S3
            response = self.s3_client.get_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            # Read Parquet from the response body
            parquet_buffer = io.BytesIO(response['Body'].read())
            df = pd.read_parquet(parquet_buffer)
            
            logger.info(f"Successfully read DataFrame with {len(df)} rows from S3")
            return df
        except Exception as e:
            logger.error(f"Error reading Parquet file from S3: {str(e)}")
            return None