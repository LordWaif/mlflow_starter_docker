from mlflow_starter import client, mlflow
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, load_metric
import numpy as np
import os

# Configurações do MLflow
experiment_name = "pytorch_test"
mlflow.set_experiment(experiment_name)

# Baixar e preparar o dataset (exemplo usando um dataset de sentimentos)
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

# Função de tokenização
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

# Selecionar subconjuntos do dataset
train_subset = dataset["train"].shuffle(seed=42).select(range(50))
test_subset = dataset["test"].shuffle(seed=42).select(range(50))

# Tokenizar os subconjuntos
tokenized_train_dataset = train_subset.map(tokenize_function, batched=True)
tokenized_test_dataset = test_subset.map(tokenize_function, batched=True)

# Carregar o modelo BERT
model = BertForSequenceClassification.from_pretrained("neuralmind/bert-base-portuguese-cased", num_labels=2)

# Definir argumentos de treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
)

# Função de avaliação
def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

from transformers import DataCollatorWithPadding

# Criar um DataCollator que padroniza as sequências
data_collator = DataCollatorWithPadding(tokenizer)

# Treinador
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,  # Adicione esta linha
    compute_metrics=compute_metrics,
)

# Iniciar uma nova run no MLflow
with mlflow.start_run(run_name="bert_classification_run_pretrained_2") as run:
    print(f"Run ID: {run.info.run_id}")
    
    # Definir tags para a run
    mlflow.set_tag("author", "Jasson Carvalho")
    mlflow.set_tag("model_type", "BERT")
    mlflow.set_tag("framework", "Transformers")
    
    # Logar hiperparâmetros
    mlflow.log_param("batch_size", training_args.per_device_train_batch_size)
    mlflow.log_param("num_epochs", training_args.num_train_epochs)
    
    # Treinar o modelo
    trainer.train()

    # Avaliar o modelo
    eval_results = trainer.evaluate()
    
    # Logar métricas
    mlflow.log_metrics({
        "eval_accuracy": eval_results["eval_accuracy"],
        "eval_loss": eval_results["eval_loss"]
    })

    components = {
        "model": model,
        "tokenizer": tokenizer,
    }

    mlflow.transformers.log_model(
        transformers_model=components,
        artifact_path="model",
    )