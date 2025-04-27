"""
Cloud storage integration for Vedic Knowledge AI.
Handles synchronizing local files with cloud storage services.
"""
import os
import logging
from typing import List, Dict, Any, Optional, Union

from ..config import (
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION, S3_BUCKET,
    GCP_PROJECT_ID, GCP_BUCKET_NAME,
    AZURE_STORAGE_CONNECTION_STRING, AZURE_CONTAINER_NAME,
    DB_DIR
)

# Configure logging
logger = logging.getLogger(__name__)

class CloudSyncManager:
    """Manager for cloud storage synchronization."""
    
    def __init__(self, local_directory: str = DB_DIR):
        """Initialize the cloud sync manager."""
        self.local_directory = local_directory
        
        # Check which cloud providers are configured
        self.has_aws = bool(AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET)
        self.has_gcp = bool(GCP_PROJECT_ID and GCP_BUCKET_NAME)
        self.has_azure = bool(AZURE_STORAGE_CONNECTION_STRING and AZURE_CONTAINER_NAME)
        
        logger.info(f"Initialized cloud sync manager (AWS: {self.has_aws}, GCP: {self.has_gcp}, Azure: {self.has_azure})")
    
    def sync_to_s3(self) -> bool:
        """Sync local directory to AWS S3."""
        if not self.has_aws:
            logger.warning("AWS credentials not configured")
            return False
        
        try:
            import boto3
            
            # Initialize S3 client
            s3 = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
            
            # Walk through local directory
            for root, dirs, files in os.walk(self.local_directory):
                for file in files:
                    local_path = os.path.join(root, file)
                    
                    # Determine S3 path (relative to local directory)
                    relative_path = os.path.relpath(local_path, self.local_directory)
                    s3_path = relative_path.replace("\\", "/")  # Ensure Unix-style paths
                    
                    # Upload file
                    logger.debug(f"Uploading {local_path} to s3://{S3_BUCKET}/{s3_path}")
                    s3.upload_file(local_path, S3_BUCKET, s3_path)
            
            logger.info(f"Successfully synced {self.local_directory} to S3 bucket {S3_BUCKET}")
            return True
            
        except ImportError:
            logger.error("boto3 not installed. Run 'pip install boto3' to use AWS S3 sync")
            return False
        except Exception as e:
            logger.error(f"Error syncing to S3: {str(e)}")
            return False
    
    def sync_from_s3(self) -> bool:
        """Sync from AWS S3 to local directory."""
        if not self.has_aws:
            logger.warning("AWS credentials not configured")
            return False
        
        try:
            import boto3
            
            # Initialize S3 client
            s3 = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION
            )
            
            # List objects in bucket
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=S3_BUCKET)
            
            # Download each file
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    s3_path = obj['Key']
                    local_path = os.path.join(self.local_directory, s3_path)
                    
                    # Create directories if needed
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    
                    # Download file
                    logger.debug(f"Downloading s3://{S3_BUCKET}/{s3_path} to {local_path}")
                    s3.download_file(S3_BUCKET, s3_path, local_path)
            
            logger.info(f"Successfully synced from S3 bucket {S3_BUCKET} to {self.local_directory}")
            return True
            
        except ImportError:
            logger.error("boto3 not installed. Run 'pip install boto3' to use AWS S3 sync")
            return False
        except Exception as e:
            logger.error(f"Error syncing from S3: {str(e)}")
            return False
    
    def sync_to_gcp(self) -> bool:
        """Sync local directory to Google Cloud Storage."""
        if not self.has_gcp:
            logger.warning("GCP credentials not configured")
            return False
        
        try:
            from google.cloud import storage
            
            # Initialize GCS client
            client = storage.Client(project=GCP_PROJECT_ID)
            bucket = client.bucket(GCP_BUCKET_NAME)
            
            # Walk through local directory
            for root, dirs, files in os.walk(self.local_directory):
                for file in files:
                    local_path = os.path.join(root, file)
                    
                    # Determine GCS path (relative to local directory)
                    relative_path = os.path.relpath(local_path, self.local_directory)
                    gcs_path = relative_path.replace("\\", "/")  # Ensure Unix-style paths
                    
                    # Upload file
                    blob = bucket.blob(gcs_path)
                    logger.debug(f"Uploading {local_path} to gs://{GCP_BUCKET_NAME}/{gcs_path}")
                    blob.upload_from_filename(local_path)
            
            logger.info(f"Successfully synced {self.local_directory} to GCS bucket {GCP_BUCKET_NAME}")
            return True
            
        except ImportError:
            logger.error("google-cloud-storage not installed. Run 'pip install google-cloud-storage' to use GCP sync")
            return False
        except Exception as e:
            logger.error(f"Error syncing to GCP: {str(e)}")
            return False
    
    def sync_to_azure(self) -> bool:
        """Sync local directory to Azure Blob Storage."""
        if not self.has_azure:
            logger.warning("Azure credentials not configured")
            return False
        
        try:
            from azure.storage.blob import BlobServiceClient
            
            # Initialize Azure client
            blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
            container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
            
            # Walk through local directory
            for root, dirs, files in os.walk(self.local_directory):
                for file in files:
                    local_path = os.path.join(root, file)
                    
                    # Determine Azure path (relative to local directory)
                    relative_path = os.path.relpath(local_path, self.local_directory)
                    azure_path = relative_path.replace("\\", "/")  # Ensure Unix-style paths
                    
                    # Upload file
                    blob_client = container_client.get_blob_client(azure_path)
                    with open(local_path, "rb") as data:
                        logger.debug(f"Uploading {local_path} to Azure {AZURE_CONTAINER_NAME}/{azure_path}")
                        blob_client.upload_blob(data, overwrite=True)
            
            logger.info(f"Successfully synced {self.local_directory} to Azure container {AZURE_CONTAINER_NAME}")
            return True
            
        except ImportError:
            logger.error("azure-storage-blob not installed. Run 'pip install azure-storage-blob' to use Azure sync")
            return False
        except Exception as e:
            logger.error(f"Error syncing to Azure: {str(e)}")
            return False
    
    def sync_to_all(self) -> Dict[str, bool]:
        """Sync to all configured cloud providers."""
        results = {}
        
        if self.has_aws:
            results["aws"] = self.sync_to_s3()
        
        if self.has_gcp:
            results["gcp"] = self.sync_to_gcp()
        
        if self.has_azure:
            results["azure"] = self.sync_to_azure()
        
        return results