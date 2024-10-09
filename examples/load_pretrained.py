from mlflow_starter import mlflow

loaded_model = mlflow.transformers.load_model(
    "[artifact_path]",
    return_type="components"
)
modelo = loaded_model["model"]
tokenizador = loaded_model['tokenizer']