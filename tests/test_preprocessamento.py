import pytest
import pandas as pd
from preprocessamento import tratar_valores_nulos, codificar_variaveis_categoricas, escalar_variaveis, split_dados

@pytest.fixture
def df_exemplo():
    data = {
        'idade': [30, None, 45],
        'sexo': ['M', 'F', None],
        'renda': [3000, 4000, None],
        'inadimplente': [1, 0, 1]
    }
    return pd.DataFrame(data)

def test_tratar_valores_nulos(df_exemplo):
    df_tratado = tratar_valores_nulos(df_exemplo.copy())
    assert df_tratado.isnull().sum().sum() == 0
    assert (df_tratado['idade'] == df_tratado['idade'].median()).sum() >= 1
    assert (df_tratado['sexo'] == "desconhecido").sum() >= 1

def test_codificar_variaveis_categoricas(df_exemplo):
    df_tratado = tratar_valores_nulos(df_exemplo.copy())
    df_codificado = codificar_variaveis_categoricas(df_tratado, drop_cols=['inadimplente'])
    assert not any(df_codificado.dtypes == object)
    # Checa se colunas one-hot aparecem
    assert any(col.startswith("sexo_") for col in df_codificado.columns) or 'sexo' in df_codificado.columns

def test_escalar_variaveis(df_exemplo):
    df_tratado = tratar_valores_nulos(df_exemplo.copy())
    df_codificado = codificar_variaveis_categoricas(df_tratado, drop_cols=['inadimplente'])
    numeric_cols = [col for col in df_codificado.columns if col != 'inadimplente' and pd.api.types.is_numeric_dtype(df_codificado[col])]
    df_escalado = escalar_variaveis(df_codificado, numeric_cols)
    assert abs(df_escalado[numeric_cols].mean()).max() < 1  # média próxima de 0

def test_split_dados(df_exemplo):
    df_tratado = tratar_valores_nulos(df_exemplo.copy())
    X_train, X_test, y_train, y_test = split_dados(df_tratado, target='inadimplente', test_size=0.34, random_state=0)
    assert len(X_train) + len(X_test) == len(df_tratado)
    assert set(y_train).issubset({0,1})
