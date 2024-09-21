# T4G ML Diabetic Retinopathy Classification

## Dataset

[Diabetic Retinopathy Arranged Kaggle](https://www.kaggle.com/datasets/amanneo/diabetic-retinopathy-resized-arranged/data)

### Object Class Label

0 - No DR

1 - Mild

2 - Moderate

3 - Severe

4 - Proliferative DR

## Setup GCP Credentials

get service account JSON from GCP Project in order to use GCP Service. Make sure to store it into the repo and run the command below before runs the backend python code.

```bash
export GOOGLE_APPLICATION_CREDENTIALS="credential.json"
```

## Connect GCS to Vertex AI for model training (optional)

```bash
MY_BUCKET=t4g-ml
cd ~/
gcsfuse --implicit-dirs --rename-dir-limit=100 --max-conns-per-host=100 $MY_BUCKET "/home/jupyter/t4g/gcs"
```

## Start the FastAPI server for development

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Start the FastAPI server for deployment

```bash
gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```
