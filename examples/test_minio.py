import boto3
import os
from dotenv import load_dotenv
from botocore.client import Config

s3_client = boto3.client('s3',
                         endpoint_url=f"http://{os.getenv('MINIO_HOST')}:{os.getenv('MINIO_PORT')}",
                         aws_access_key_id=os.getenv('MINIO_ACCESS_KEY'),
                         aws_secret_access_key=os.getenv('MINIO_SECRET_ACCESS_KEY'),
                         config=Config(signature_version='s3v4'),
                         region_name='us-east-1')


print(s3_client.list_buckets()['Buckets'])