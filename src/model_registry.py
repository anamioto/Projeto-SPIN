import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
import os
import logging
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, balanced_accuracy_score, recall_score, f1_score
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azureml.core import Workspace

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def carregar_dados():
    logging.info("Carregando os dados pré-processados")
    df_transformado = pd.read_csv("df_transformado.csv")
    return df_transformado

def split_dados(df: pd.DataFrame, target: str, test_size=0.2, random_state=42):
    logging.info("Dividindo em amostras de treino 80% e teste 20%")
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def treinar_modelo_xgb(X_train, y_train, X_test, y_test, params=None):
    logging.info("Treinando modelo XGBoost")
    if params is None:
        params = {
            "objective": "binary:logistic",
            "use_label_encoder": False
        }
    model = xgb.XGBClassifier(**params, enable_categorical=True)
    model.fit(X_train, y_train) #treinamento
    y_pred = model.predict(X_test) #teste

    # Métricas importantes
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }

    for k, v in metrics.items():
        logging.info(f"{k}: {v:.4f}")

    return model, metrics

def treinar_modelo_rf(X_train, y_train, X_test, y_test, params=None):
    logging.info("Treinando modelo RandomForestClassifier")
    if params is None:
        params = {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": 42,
        }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train) #treinamento
    y_pred = model.predict(X_test)  # predição

    # Métricas importantes
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
    }

    for k, v in metrics.items():
        logging.info(f"{k}: {v:.4f}")

    return model, metrics


def registra_mlflow_azure_2(model, metrics, experiment_name="inadimplencia-rfc", tags=None):
    logging.info("Registrando modelo 2 no MLflow com Azure")
    # Configure MLflow para Azure (aqui, espera-se que já exista configuração de URI)
    # Exemplo de configuração:
    mlflow.set_tracking_uri("azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/0b97f8d7-e740-4d8a-be3c-96eea4182bf8/resourceGroups/aulasalura/providers/Microsoft.MachineLearningServices/workspaces/ds-workspace")
    logging.info("Registrando modelo Random Forest no MLflow com Azure")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestClassifier")
        # Log de métricas
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        if tags:
            mlflow.set_tags(tags)

        mlflow.sklearn.log_model(model, "model_rfc")

def registra_mlflow_azure(model, metrics, experiment_name="inadimplencia-xgb", tags=None):
    logging.info("Registrando modelo 1 no MLflow com Azure")
    # Configure MLflow para Azure (aqui, espera-se que já exista configuração de URI)
    # Exemplo de configuração:
    mlflow.set_tracking_uri("azureml://eastus.api.azureml.ms/mlflow/v1.0/subscriptions/0b97f8d7-e740-4d8a-be3c-96eea4182bf8/resourceGroups/aulasalura/providers/Microsoft.MachineLearningServices/workspaces/ds-workspace")
    logging.info("Registrando modelo XGBoost no MLflow com Azure")
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param("model_type", "XGBoostClassifier")
        # Log de métricas
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        if tags:
            mlflow.set_tags(tags)

        mlflow.xgboost.log_model(model, "model_xgb")

if __name__ == "__main__":
    #Autenticação do Azure ML
    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id = "0b97f8d7-e740-4d8a-be3c-96eea4182bf8",
        resource_group_name = "AulasAlura",
        workspace_name = "DS-Workspace" 
    )
    
    df_transformado = carregar_dados()
    df_transformado = df_transformado.set_index('ID_Cliente', drop=True)
    X_train, X_test, y_train, y_test = split_dados(df_transformado, target = "Status_Pagamento")
    model, metrics = treinar_modelo_xgb(X_train, y_train, X_test, y_test)
    model2, metrics2 = treinar_modelo_rf(X_train, y_train, X_test, y_test)
    registra_mlflow_azure(model, metrics, experiment_name="inadimplencia-xgb")
    registra_mlflow_azure_2(model2, metrics2, tags={"versao": "1.0", "pipeline": "producao"})

    logging.info("Pipeline de modelagem e registro concluído!")

