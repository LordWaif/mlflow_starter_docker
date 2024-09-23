# Configuração do Docker Compose para mlflow_minio

Este arquivo `docker-compose.yaml` configura contêineres para o MLflow e MinIO, serviços essenciais para o gerenciamento de experimentos e armazenamento de objetos. Abaixo estão as variáveis de ambiente utilizadas e suas respectivas utilidades:

## Variáveis de Ambiente

- **MLFLOW_PORT**: Porta na qual o serviço MLflow estará disponível.
- **MINIO_ACCESS_KEY**: Chave de acesso para autenticação no MinIO.
- **MINIO_SECRET_ACCESS_KEY**: Chave secreta para autenticação no MinIO.
- **MINIO_HOST**: Endereço do host onde o MinIO está rodando.
- **MINIO_PORT**: Porta na qual o serviço MinIO estará disponível.
- **PG_USER**: Usuário do banco de dados PostgreSQL.
- **PG_PASSWORD**: Senha do usuário do banco de dados PostgreSQL.
- **PG_HOST**: Endereço do host onde o PostgreSQL está rodando.
- **PG_PORT**: Porta na qual o serviço PostgreSQL estará disponível.
- **PG_DATABASE**: Nome do banco de dados PostgreSQL.
- **MLFLOW_BUCKET_NAME**: Nome do bucket S3 onde os artefatos do MLflow serão armazenados.
- **MINIO_VOLUME_PATH**: Define o caminho no host onde os dados do MinIO serão armazenados.
- **MINIO_CONSOLE_ADDRESS**: Endereço e porta para o console de administração do MinIO.
- **MINIO_ROOT_USER**: Usuário root para autenticação no MinIO.
- **MINIO_ROOT_PASSWORD**: Senha do usuário root para autenticação no MinIO.
- **MINIO_STORAGE_USE_HTTPS**: Define se o MinIO deve usar HTTPS para comunicação.

## Descrição dos Serviços

### mlflow

- **Imagem**: `mlflow_server`
- **Ports**: Mapeia a porta especificada em `${MLFLOW_PORT}` para `5000` no contêiner.
- **Environment**: Define as variáveis de ambiente necessárias para o MLflow.
- **Networks**: Conecta o contêiner à rede `backend`.
- **Command**: Comando para iniciar o MLflow.
- **Healthcheck**: Verifica a saúde do serviço MLflow.

### s3

- **Imagem**: `minio/minio`
- **Container Name**: `mlflow_minio`
- **Volumes**: Monta o volume especificado em `${MINIO_VOLUME_PATH}` para `/data` no contêiner.
- **Ports**: Mapeia as portas especificadas para o host.
- **Networks**: Conecta o contêiner à rede `backend`.
- **Environment**: Define as variáveis de ambiente necessárias para o MinIO.
- **Command**: Comando para iniciar o MinIO.
- **Healthcheck**: Verifica a saúde do serviço MinIO.

## Volumes

- **minio_data**: Volume Docker para armazenar dados do MinIO.

## Networks

- **backend**: Rede Docker do tipo bridge para comunicação entre contêineres.