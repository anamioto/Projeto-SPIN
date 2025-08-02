import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.sklearn
import os
import logging
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def carregar_dados():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").values.ravel()
    y_test = pd.read_csv("y_test.csv").values.ravel()
    return X_train, X_test, y_train, y_test

def treinar_modelo_xgb(X_train, y_train, X_test, y_test, params=None):
    logging.info("Treinando modelo XGBoost")
    if params is None:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "use_label_encoder": False
        }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    logging.info(f"Score no conjunto de teste: {score:.4f}")
    return model, score

def registra_mlflow_azure(model, score, experiment_name="inadimplencia-xgb", tags=None):
    logging.info("Registrando modelo no MLflow com Azure")
    # Configure MLflow para Azure (aqui, espera-se que já exista configuração de URI)
    # Exemplo de configuração:
    # mlflow.set_tracking_uri("azureml://<your-azure-ml-tracking-uri>")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBoostClassifier")
        mlflow.log_metric("test_score", score)
        if tags:
            mlflow.set_tags(tags)
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    # Caso use Azure ML para autenticação (opcional, ajuste para seu ambiente)
    # credential = DefaultAzureCredential()
    # ml_client = MLClient(credential, "<subscription_id>", "<resource_group>", "<workspace_name>")
    
    X_train, X_test, y_train, y_test = carregar_dados()
    model, score = treinar_modelo_xgb(X_train, y_train, X_test, y_test)
    registra_mlflow_azure(model, score, experiment_name="inadimplencia-xgb")

    logging.info("Pipeline de modelagem e registro concluído!")
