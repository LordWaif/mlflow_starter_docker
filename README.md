# MLflow

## Diagrama

[![](https://mermaid.ink/img/pako:eNqNVMty2jAU_RWNOuyAMcHm4UU7BAiQxJ1Mw3RR04Wwr0GpLbmyPA0l_Ey76KqrfgI_VlkWhDBNwRs9zjlX9-F71zjgIWAXVyrrGUPqo4xK1-xnWC4hgRl21XZOMrWrvkA-EkHJPIasoBhRCaeCJkSs-jzmotS_abbsTifYm3jBm8KjPORGUfRv4iUXIYiTZmPK4JBkBe2O3T0iSRCSHjk5D4Ju5Mxwydvsl02lMmNRzL8FSyIkmg5KpOdP2PZ3QDkKOboT_AEk_4xqtbfo0h9ApJIp0PAxBeU8MAWVqkvN6Pt3AlIiiEADEvLMgH0NDvypAMoU5qkCxTvlQIPDdXmJ7omkWUTk9o-g_J3xdlhwnt5vf_IndOX3HvJMKjNjqrxQr21_JCDF_rWr8rVD5T1NntDI_wALmklx7MFIC8bKPcIyuv2lnkFFEEX0Ya7PhjnWzIk_SdKYMHlsaKLha9_j6o_jOguQQZICW-4o15pys34GUC-Er7nK1i7Ym4Ngb5XP0mSN58hEbmzdHgZ68xyo53vKORAIkpchlMwsny8ESZfIuy2qj1TYwRfKFqaQpioml-UCLHxNrf-QQGam0if5OmGGPTJpO0-ETPlWphYmoeXi7W3gKk5AJISGagboBn6t5__T8ed0-1mdfn6Xn-zwGc4g4Cw88sqyWg3LOhxjJ2ZA0foqSSSX_H7FAuxKkUMVC54vltiNSJypU56GRMKAElWHZH-bEvaJ82QnUUfsrvEjdi-cum017Wa73ehaju20Lqp4hd2aY3XrTqfpWO1uy-rYlu1sqvi7NtGot62203WatsI7DcduVjGERd945fzWY3zzFx4my4c?type=png)](https://mermaid.live/edit#pako:eNqNVMty2jAU_RWNOuyAMcHm4UU7BAiQxJ1Mw3RR04Wwr0GpLbmyPA0l_Ey76KqrfgI_VlkWhDBNwRs9zjlX9-F71zjgIWAXVyrrGUPqo4xK1-xnWC4hgRl21XZOMrWrvkA-EkHJPIasoBhRCaeCJkSs-jzmotS_abbsTifYm3jBm8KjPORGUfRv4iUXIYiTZmPK4JBkBe2O3T0iSRCSHjk5D4Ju5Mxwydvsl02lMmNRzL8FSyIkmg5KpOdP2PZ3QDkKOboT_AEk_4xqtbfo0h9ApJIp0PAxBeU8MAWVqkvN6Pt3AlIiiEADEvLMgH0NDvypAMoU5qkCxTvlQIPDdXmJ7omkWUTk9o-g_J3xdlhwnt5vf_IndOX3HvJMKjNjqrxQr21_JCDF_rWr8rVD5T1NntDI_wALmklx7MFIC8bKPcIyuv2lnkFFEEX0Ya7PhjnWzIk_SdKYMHlsaKLha9_j6o_jOguQQZICW-4o15pys34GUC-Er7nK1i7Ym4Ngb5XP0mSN58hEbmzdHgZ68xyo53vKORAIkpchlMwsny8ESZfIuy2qj1TYwRfKFqaQpioml-UCLHxNrf-QQGam0if5OmGGPTJpO0-ETPlWphYmoeXi7W3gKk5AJISGagboBn6t5__T8ed0-1mdfn6Xn-zwGc4g4Cw88sqyWg3LOhxjJ2ZA0foqSSSX_H7FAuxKkUMVC54vltiNSJypU56GRMKAElWHZH-bEvaJ82QnUUfsrvEjdi-cum017Wa73ehaju20Lqp4hd2aY3XrTqfpWO1uy-rYlu1sqvi7NtGot62203WatsI7DcduVjGERd945fzWY3zzFx4my4c)

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
