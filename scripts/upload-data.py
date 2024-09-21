from google.cloud import storage
from google.oauth2 import service_account
import os

def upload_folder_to_gcs(local_path, bucket_name, gcs_path):
    key_path = "gcp-credential.json"
    credentials = service_account.Credentials.from_service_account_file(key_path)
    client = storage.Client(credentials=credentials)

    bucket = client.get_bucket(bucket_name)

    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_path)
            gcs_file_path = os.path.join(gcs_path, relative_path)

            blob = bucket.blob(gcs_file_path)
            blob.upload_from_filename(local_file_path)
            print(f"File {local_file_path} uploaded to {gcs_file_path}")

def test_gcs_access(bucket_name):
    try:
        key_path = "gcp-credential.json"
        credentials = service_account.Credentials.from_service_account_file(key_path)
        client = storage.Client(credentials=credentials)
        
        buckets = list(client.list_buckets())
        print(f"Successfully listed {len(buckets)} buckets.")
        
        bucket = client.get_bucket(bucket_name)
        print(f"Successfully accessed bucket: {bucket.name}")
        
        blobs = list(bucket.list_blobs(max_results=5))
        print(f"Listed {len(blobs)} blobs in the bucket.")
        for blob in blobs:
            print(f" - {blob.name}")
        
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
    upload_folder_to_gcs(local_folder_path, bucket_name, gcs_folder_path)
else:
    print("GCS access test failed.")