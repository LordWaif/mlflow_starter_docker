#!/bin/sh

echo "Aguardando MinIO iniciar..."
sleep 10

# Configura o alias do MinIO Client
until mc alias set myminio http://${MINIO_HOST}:${MINIO_PORT} ${MINIO_ROOT_USER} ${MINIO_ROOT_PASSWORD}
do
    echo "MinIO ainda não está disponível. Tentando novamente em 5 segundos..."
    sleep 5
done

echo "MinIO está pronto. Criando bucket..."
mc mb myminio/${MLFLOW_BUCKET_NAME} --ignore-existing
echo "Bucket criado ou já existente."