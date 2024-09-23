import mlflow
from dotenv import load_dotenv
import os
load_dotenv()
TRACKING_URL = f"http://{os.getenv('MLFLOW_HOST')}:{os.getenv('MLFLOW_PORT')}"
mlflow.set_tracking_uri(TRACKING_URL)
client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URL)