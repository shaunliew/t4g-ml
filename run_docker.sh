docker build -t diabetic-retinopathy-api .

docker run \
  -p 8000:8000 \
  --env-file .env \
  -v ~/.aws:/root/.aws \
  diabetic-retinopathy-api