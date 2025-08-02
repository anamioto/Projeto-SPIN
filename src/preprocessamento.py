import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Configura logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def carregar_dados(caminho: str) -> pd.DataFrame:
    """Carrega dados de um arquivo CSV."""
    logging.info(f'Carregando dados de {caminho}')
    return pd.read_csv(caminho)

def tratar_valores_nulos(df: pd.DataFrame) -> pd.DataFrame:
    """Preenche valores nulos com estratégias apropriadas."""
    logging.info('Tratando valores nulos')
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=[object]).columns:
        df[col] = df[col].fillna('desconhecido')
    return df

def codificar_variaveis_categoricas(df: pd.DataFrame, drop_cols: list = []) -> pd.DataFrame:
    """Codifica variáveis categóricas com LabelEncoder ou One-Hot."""
    logging.info('Codificando variáveis categóricas')
    label_encoders = {}
    for col in df.select_dtypes(include=[object]).columns:
        if col in drop_cols:
            continue
        if df[col].nunique() <= 2:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            df = pd.get_dummies(df, columns=[col], prefix=col)
    return df

def escalar_variaveis(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Escala variáveis numéricas."""
    logging.info('Escalando variáveis numéricas')
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

def split_dados(df: pd.DataFrame, target: str, test_size=0.2, random_state=42):
    """Divide em treino e teste."""
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def pipeline_preprocessamento(caminho_csv, target, drop_cols=[]):
    """Pipeline completo de pré-processamento."""
    df = carregar_dados(caminho_csv)
    df = tratar_valores_nulos(df)
    df = codificar_variaveis_categoricas(df, drop_cols=drop_cols)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]
    df = escalar_variaveis(df, numeric_cols)
    X_train, X_test, y_train, y_test = split_dados(df, target)
    logging.info('Pré-processamento finalizado')
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Caminho para o CSV (direto do GitHub bruto)
    url = "https://raw.githubusercontent.com/anamioto/Projeto-SPIN/main/data/base_clientes_inadimplencia.csv"
    target = "inadimplente"  # Ajuste se necessário
    drop_cols = ["id_cliente"]  # Ajuste se necessário
    X_train, X_test, y_train, y_test = pipeline_preprocessamento(url, target, drop_cols=drop_cols)

    # Salva dados prontos para o próximo passo
    X_train.to_csv("X_train.csv", index=False)
    X_test.to_csv("X_test.csv", index=False)
    y_train.to_csv("y_train.csv", index=False)
    y_test.to_csv("y_test.csv", index=False)
