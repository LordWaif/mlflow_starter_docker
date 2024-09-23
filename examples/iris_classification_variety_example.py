import os
import mlflow.sklearn
from mlflow_starter import client, mlflow
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt
import pandas as pd
import os

# Configuração do MLflow
experiment_name = "iris_classification_variety"
mlflow.set_experiment(experiment_name)

experiment = client.get_experiment_by_name(experiment_name)
print(f"Experiment ID: {experiment.experiment_id}")
print(f"Artifact Location: {experiment.artifact_location}")

# Carregar e preparar os dados
df = pd.read_csv("examples/iris.csv")  # Substitua pelo caminho correto do dataset
X = df.drop(columns=["variety"])
y = df["variety"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir hiperparâmetros e variáveis para early stopping
n_estimators = 100
max_depth = 5

# Iniciar uma nova run no MLflow
with mlflow.start_run(run_name="iris_classification_run") as run:
    print(f"Run ID: {run.info.run_id}")
    
    # Definir tags para a run
    mlflow.set_tag("author", "Jasson Carvalho")
    mlflow.set_tag("model_type", "RandomForestClassifier")
    mlflow.set_tag("framework", "sklearn")
    
    # Logar parâmetros
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    
    # Treinar o modelo
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=1)
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    
    # Logar métricas
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    mlflow.log_metrics(metrics)
    
    # Logar o modelo
    mlflow.sklearn.log_model(model, "random_forest_model")

    os.makedirs("examples/data_input", exist_ok=True)
    # Logar o dataset, dados de treino e teste como artefatos
    df.to_csv("examples/data_input/input_dataset.csv", index=False)
    train = pd.concat([X_train, y_train], axis=1)
    pd.DataFrame(train).to_csv("examples/data_input/train_data.csv", index=False)
    test = pd.concat([X_test, y_test], axis=1)
    pd.DataFrame(test).to_csv("examples/data_input/test_data.csv", index=False)
    
    mlflow.log_artifacts("examples/data_input")
    
    # Logar um gráfico de importância das features
    feature_importance = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({"feature": feature_names, "importance": feature_importance})
    importance_df = importance_df.sort_values("importance", ascending=False)
    
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df["feature"], importance_df["importance"])
    plt.title("Feature Importance")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salvar e logar o gráfico
    plt.savefig("examples/feature_importance.png")
    mlflow.log_artifact("examples/feature_importance.png")
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
# Exibir a última run ativa
last_run = client.get_run(run.info.run_id)
print(f"Last active run - ID: {last_run.info.run_id}, Status: {last_run.info.status}")

