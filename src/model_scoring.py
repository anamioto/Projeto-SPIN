import pandas as pd
import numpy as np
import mlflow
import logging
import sys
from preprocessamento import tratar_valores_nulos, codificar_variaveis_categoricas, escalar_variaveis

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def carregar_novos_dados(caminho: str) -> pd.DataFrame:
    logging.info(f'Carregando novos dados de {caminho}')
    return pd.read_csv(caminho)

def preprocessar_novos_dados(df: pd.DataFrame, drop_cols=[]):
    logging.info('Iniciando pré-processamento dos novos dados')
    df_proc = tratar_valores_nulos(df.copy())
    df_proc = codificar_variaveis_categoricas(df_proc, drop_cols=drop_cols)
    numeric_cols = [col for col in df_proc.select_dtypes(include=[np.number]).columns if col not in drop_cols]
    df_proc = escalar_variaveis(df_proc, numeric_cols)
    return df_proc

def carregar_modelo_mlflow(model_uri: str):
    logging.info(f'Carregando modelo do MLflow: {model_uri}')
    model = mlflow.sklearn.load_model(model_uri)
    return model

def gerar_predicoes(model, X):
    logging.info('Gerando predições')
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        return preds, proba
    return preds, None

def salvar_resultado(df, preds, probas, saida='predicoes.csv'):
    logging.info(f'Salvando resultado em {saida}')
    df_result = df.copy()
    df_result['predito_inadimplente'] = preds
    if probas is not None:
        df_result['score_inadimplencia'] = probas
    df_result.to_csv(saida, index=False)

if __name__ == "__main__":
    # Parâmetros de entrada
    if len(sys.argv) < 4:
        print("Uso: python model_scoring.py <caminho_novos_dados.csv> <mlflow_model_uri> <colunas_a_ignorar_separadas_por_virgula>")
        sys.exit(1)
    caminho_novos_dados = sys.argv[1]
    model_uri = sys.argv[2]
    drop_cols = sys.argv[3].split(",") if sys.argv[3] else []

    df_novos = carregar_novos_dados(caminho_novos_dados)
    df_proc = preprocessar_novos_dados(df_novos, drop_cols=drop_cols)
    model = carregar_modelo_mlflow(model_uri)
    preds, probas = gerar_predicoes(model, df_proc)
    salvar_resultado(df_novos, preds, probas, saida="predicoes.csv")

    logging.info("Scoring finalizado. Resultado salvo em predicoes.csv.")
