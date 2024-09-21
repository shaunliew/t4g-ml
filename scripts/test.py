from google.cloud import storage
from google.oauth2 import service_account
import os

def count_local_files(local_path):
    total_files = 0
    for root, dirs, files in os.walk(local_path):
        total_files += len(files)
    return total_files

def count_gcs_files(bucket_name, gcs_path):
    key_path = "gcp-credential.json"
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = storage.Client(credentials=credentials)
    
    bucket = client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=gcs_path)
    return sum(1 for _ in blobs)

def test_gcs_access(bucket_name):
    try:
        key_path = "gcp-credential.json"
        credentials = service_account.Credentials.from_service_account_file(key_path)
        client = storage.Client(credentials=credentials)
        
        buckets = list(client.list_buckets())
        print(f"Successfully listed {len(buckets)} buckets.")
        
        bucket = client.get_bucket(bucket_name)
        print(f"Successfully accessed bucket: {bucket.name}")
        
        return True
    except Exception as e:
        print(f"Error accessing GCS: {str(e)}")
        return False

# Replace these with your actual values
bucket_name = 't4g-ml'
local_folder_path = 'raw_data'
gcs_folder_path = 'data'

if test_gcs_access(bucket_name):
    print("GCS access test passed successfully!")
    
    # Count files in local folder
    local_file_count = count_local_files(local_folder_path)
    print(f"Number of files in local folder: {local_file_count}")
    
    # Count files in GCS bucket
    gcs_file_count = count_gcs_files(bucket_name, gcs_folder_path)
    print(f"Number of files in GCS bucket path: {gcs_file_count}")
    
else:
    print("GCS access test failed.")