import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow.sklearn
from mlflow_starter import client, mlflow
from matplotlib import pyplot as plt
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec
import os

# Função para avaliação das métricas
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    np.random.seed(40)

    # Carregando dataset
    data = pd.read_csv("examples/red-wine-quality.csv")
    
    # Salvando dataset em artefatos
    os.makedirs("data_input", exist_ok=True)
    data.to_csv("data_input/red-wine-quality.csv", index=False)
    
    # Dividindo o dataset em treino e teste
    train, test = train_test_split(data, test_size=0.25)
    train.to_csv("data_input/train_wine.csv", index=False)
    test.to_csv("data_input/test_wine.csv", index=False)
    
    # Separando features e rótulos
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = .7
    l1_ratio = .7

    # Iniciando experimento
    experiment_name = "wine_quality_prediction"
    mlflow.set_experiment(experiment_name)
    experiment = client.get_experiment_by_name(experiment_name)

    print(f"Experiment ID: {experiment.experiment_id}")
    print(f"Artifact Location: {experiment.artifact_location}")

    # Iniciando uma nova run
    with mlflow.start_run(run_name="set_your_own_signature_and_input_example") as run:
        print(f"Run ID: {run.info.run_id}")

        # Definindo tags da run
        mlflow.set_tags({
            "author": "Jasson Carvalho",
            "model_type": "ElasticNet",
            "framework": "sklearn"
        })

        # Habilitando autolog
        mlflow.autolog(
            log_input_examples=False,
            log_model_signatures=False,
            log_models=False)
        
        # Treinando o modelo
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)
        
        # Fazendo previsões
        predicted_qualities = model.predict(test_x)
        
        # Avaliando o modelo
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)
        print(f"ElasticNet (alpha={alpha}, l1_ratio={l1_ratio}):")
        print(f"RMSE: {rmse}, MAE: {mae}, R2: {r2}")

        input_data = [
            {"name": "fixed acidity", "type": "float"},
            {"name": "volatile acidity", "type": "float"},
            {"name": "citric acid", "type": "float"},
            {"name": "residual sugar", "type": "float"},
            {"name": "chlorides", "type": "float"},
            {"name": "free sulfur dioxide", "type": "float"},
            {"name": "total sulfur dioxide", "type": "float"},
            {"name": "density", "type": "float"},
            {"name": "pH", "type": "float"},
            {"name": "sulphates", "type": "float"},
            {"name": "alcohol", "type": "float"}
        ]

        output_data = [{"type": "float"}]

        input_example = {
            "fixed acidity": 7.4,
            "volatile acidity": 0.7,
            "citric acid": 0.0,
            "residual sugar": 1.9,
            "chlorides": 0.076,
            "free sulfur dioxide": 11.0,
            "total sulfur dioxide": 34.0,
            "density": 0.9978,
            "pH": 3.51,
            "sulphates": 0.56,
            "alcohol": 9.4
        }

        input_schema = Schema([ColSpec(c["type"],c["name"],) for c in input_data])
        output_schema = Schema([ColSpec(c["type"]) for c in output_data])

        signature = ModelSignature(inputs=input_schema, outputs=output_schema)

        # Logando artefatos
        mlflow.log_artifact("data_input/red-wine-quality.csv")
        mlflow.sklearn.log_model(model, "model",signature=signature,input_example=input_example)
        # Logando gráfico de importância das features
        feature_importance = np.abs(model.coef_)
        feature_names = train_x.columns
        importance_df = pd.DataFrame({"feature": feature_names, "importance": feature_importance})
        importance_df = importance_df.sort_values(by="importance", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.bar(importance_df["feature"], importance_df["importance"])
        plt.title("Feature Importance")
        plt.xlabel("Features")
        plt.ylabel("Importance")
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig("examples/feature_importance.png")
        mlflow.log_artifact("examples/feature_importance.png")

        # Exibindo a última run ativa
        print(f"Run finalizada: ID {run.info.run_id}")

    # Obter informações da última run
    last_run = client.get_run(run.info.run_id)
    print(f"Última run - ID: {last_run.info.run_id}, Status: {last_run.info.status}")
