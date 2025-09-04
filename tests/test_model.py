# tests/test_model_scoring.py
# -*- coding: utf-8 -*-
"""
Testes unitários para model_scoring.py

Boas práticas aplicadas:
- pytest + fixtures reutilizáveis
- Isolamento de I/O com tmp_path
- Mocks de mlflow e do módulo 'preprocessamento'
- Verificações de parâmetros e efeitos mínimos
- Testes determinísticos e de caminho feliz/tristes
"""
from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pytest


# === Ajuste este nome caso o arquivo tenha outro caminho/nome ===
SCRIPT_FILENAME = "model_scoring.py"


# ---------- Fixtures utilitárias ----------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """Mini DataFrame representativo para predição."""
    return pd.DataFrame(
        {
            "ID_Cliente": [1, 2, 3],
            "Data_Nascimento": ["2000-01-10", "1990-06-01", "1985-12-31"],
            "Estado": ["SP", "RJ", "MG"],
            "Cidade": ["São Paulo", "Rio de Janeiro", "Belo Horizonte"],
            "Sexo": ["F", "M", "F"],
            "Renda_Mensal": [4000, 7000, 5500],
            "Status_Pagamento": [0, 1, 0],  # alvo (usado só para checagem de exclusão de cols)
        }
    )


@pytest.fixture()
def dataset_csv(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    """Grava um CSV temporário para simular entrada do usuário."""
    path = tmp_path / "novos_dados.csv"
    sample_df.to_csv(path, index=False)
    return path


@pytest.fixture()
def inject_fake_preprocessamento(monkeypatch):
    """
    Injeta um módulo 'preprocessamento' falso em sys.modules para permitir que
    model_scoring.py importe as funções sem depender do código real.
    Depois, dentro dos testes, vamos monkeypatchar as funções no próprio módulo carregado.
    """
    fake = types.ModuleType("preprocessamento")

    def _identity(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return df.copy()

    # Assinaturas compatíveis com o model_scoring.py
    fake.tratar_valores_nulos = _identity
    fake.tratar_data_nascimento = _identity
    fake.converter_colunas_data = _identity
    fake.codificar_variaveis_categoricas = _identity

    def _escalar(df: pd.DataFrame, cols) -> pd.DataFrame:
        # Não escala de fato; apenas retorna cópia (testes checam chamada)
        return df.copy()

    fake.escalar_variaveis = _escalar

    sys.modules["preprocessamento"] = fake
    yield
    sys.modules.pop("preprocessamento", None)


@pytest.fixture()
def inject_fake_mlflow(monkeypatch):
    """
    Injeta um 'mlflow' mínimo com submódulo sklearn.load_model.
    """
    mlflow = types.ModuleType("mlflow")
    mlflow.sklearn = types.ModuleType("mlflow.sklearn")

    class FakeModel:
        def __init__(self, with_proba: bool = True):
            self._with_proba = with_proba

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            # Gera 0/1 determinístico
            return (np.arange(len(X)) % 2).astype(int)

        def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
            if not self._with_proba:
                raise AttributeError("Sem predict_proba")
            # Duas colunas: prob classe 0, prob classe 1
            p1 = np.linspace(0.1, 0.9, len(X))
            p0 = 1 - p1
            return np.c_[p0, p1]

    def load_model(uri: str) -> Any:
        # Retorna um modelo com predict_proba
        return FakeModel(with_proba=True)

    mlflow.sklearn.load_model = load_model
    sys.modules["mlflow"] = mlflow
    yield
    sys.modules.pop("mlflow", None)


@pytest.fixture()
def mod(tmp_path: Path, inject_fake_preprocessamento, inject_fake_mlflow, monkeypatch):
    """
    Importa src/model_scoring.py e isola efeitos.
    """
    monkeypatch.chdir(tmp_path)

    SCRIPT_FILENAME = "model_scoring.py"
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "src" / SCRIPT_FILENAME

    if not script_path.exists():
        candidates = [
            Path(__file__).with_name(SCRIPT_FILENAME),
            repo_root / SCRIPT_FILENAME,
            Path.cwd().parent / "src" / SCRIPT_FILENAME,
            Path("/mnt/data") / SCRIPT_FILENAME,
        ]
        script_path = next((p for p in candidates if p.exists()), None)

    if script_path is None or not script_path.exists():
        raise FileNotFoundError(
            "Não encontrei 'src/model_scoring.py'. "
            "Garanta a estrutura: <repo>/src/model_scoring.py."
        )

    spec = importlib.util.spec_from_file_location("model_scoring_module", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["model_scoring_module"] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)

    if not hasattr(module, "target"):
        setattr(module, "target", "Status_Pagamento")
    return module


# ---------- Testes por função ----------

def test_carregar_novos_dados_ler_csv(mod, dataset_csv: Path):
    df = mod.carregar_novos_dados(str(dataset_csv))
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "ID_Cliente" in df.columns


def test_carregar_modelo_mlflow_retorna_modelo(mod):
    model = mod.carregar_modelo_mlflow("models:/inadimplencia/Production")
    assert hasattr(model, "predict")


def test_gerar_predicoes_com_proba(mod, sample_df):
    class ModelComProba:
        def predict(self, X):
            return np.ones(len(X), dtype=int)

        def predict_proba(self, X):
            return np.c_[np.zeros(len(X)), np.ones(len(X))]

    preds, probas = mod.gerar_predicoes(ModelComProba(), sample_df.drop(columns=["Status_Pagamento"]))
    assert (preds == 1).all()
    assert probas is not None and np.allclose(probas, 1.0)


def test_gerar_predicoes_sem_proba(mod, sample_df):
    class ModelSemProba:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    preds, probas = mod.gerar_predicoes(ModelSemProba(), sample_df.drop(columns=["Status_Pagamento"]))
    assert (preds == 0).all()
    assert probas is None


def test_salvar_resultado_cria_csv(mod, tmp_path: Path, sample_df):
    preds = np.array([0, 1, 0])
    probas = np.array([0.2, 0.8, 0.4])
    saida = tmp_path / "predicoes.csv"
    mod.salvar_resultado(sample_df, preds, probas, saida=str(saida))
    assert saida.exists()
    df_out = pd.read_csv(saida)
    assert "predito_inadimplente" in df_out.columns
    assert "score_inadimplencia" in df_out.columns


# ---------- Teste do pipeline de pré-processamento com monkeypatch ----------

def test_preprocessar_novos_dados_encadeia_funcoes_e_usa_id_como_indice(
    monkeypatch, mod, sample_df
):
    # ---- spies para garantir encadeamento ----
    calls = {"nulos": 0, "nasc": 0, "datas": 0, "cat": 0, "escala": 0, "cols_escala": None}

    def tratar_valores_nulos(df: pd.DataFrame) -> pd.DataFrame:
        calls["nulos"] += 1
        return df.fillna({"Cidade": "desconhecido"})

    def tratar_data_nascimento(df: pd.DataFrame) -> pd.DataFrame:
        calls["nasc"] += 1
        out = df.copy()
        out["Idade"] = [24, 34, 39]
        return out

    def converter_colunas_data(df: pd.DataFrame, cols, *_, **__):
        calls["datas"] += 1
        return df

    def codificar_variaveis_categoricas(df: pd.DataFrame) -> pd.DataFrame:
        calls["cat"] += 1
        out = df.copy()
        # simula one-hot de Estado e label em Sexo
        for uf in sorted(out["Estado"].unique()):
            out[f"Estado_{uf}"] = (out["Estado"] == uf).astype("uint8")
        out["Sexo"] = (out["Sexo"] == "F").astype("int8")
        return out

    def escalar_variaveis(df: pd.DataFrame, cols) -> pd.DataFrame:
        calls["escala"] += 1
        calls["cols_escala"] = list(cols)
        return df

    monkeypatch.setattr(mod, "tratar_valores_nulos", tratar_valores_nulos)
    monkeypatch.setattr(mod, "tratar_data_nascimento", tratar_data_nascimento)
    monkeypatch.setattr(mod, "converter_colunas_data", converter_colunas_data)
    monkeypatch.setattr(mod, "codificar_variaveis_categoricas", codificar_variaveis_categoricas)
    monkeypatch.setattr(mod, "escalar_variaveis", escalar_variaveis)

    # ---- parâmetros do preprocessamento ----
    colunas_data = ["Data_Nascimento"]

    # IMPORTANTE: não inclua "ID_Cliente" em drop_cols, pois agora ele será índice.
    drop_cols = ["Cidade", "Data_Nascimento", "Status_Pagamento"]

    # Executa
    out = mod.preprocessar_novos_dados(
        sample_df, colunas_data=colunas_data, drop_cols=drop_cols
    )

    # ---- Encadeamento ocorreu ----
    assert calls["nulos"] == 1
    assert calls["nasc"] == 1
    assert calls["datas"] == 1
    assert calls["cat"] == 1
    assert calls["escala"] == 1

    # ---- Agora o comportamento esperado: usar ID_Cliente como índice ----
    assert out.index.name == "ID_Cliente", "O índice deve ser 'ID_Cliente'."
    assert out.index.is_unique, "O índice 'ID_Cliente' deve ser único."
    assert "ID_Cliente" not in out.columns, "ID_Cliente não deve permanecer como coluna."

    # O target não pode estar nas features (e nem entre as colunas escaladas)
    assert "Status_Pagamento" not in out.columns
    assert "Status_Pagamento" not in (calls["cols_escala"] or [])


# ---------- Teste E2E (sem entrar no bloco __main__) ----------

def test_e2e_scoring_sem_main(mod, dataset_csv: Path, tmp_path: Path, monkeypatch):
    # Preprocessamento fake que retorna apenas features numéricas para simplificar
    def fake_preprocess(df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        out = df.copy()
        if "Status_Pagamento" in out:
            out = out.drop(columns=["Status_Pagamento"])
        # cria uma feature numérica sintética
        out["f1"] = np.arange(len(out), dtype=float)
        return out

    # Modelo fake carregado via carregar_modelo_mlflow()
    class FakeModel:
        def predict(self, X):
            return (X["f1"] > 0).astype(int).to_numpy()

        def predict_proba(self, X):
            p1 = (X["f1"] + 1) / (X["f1"].max() + 1)
            p0 = 1 - p1
            return np.c_[p0.to_numpy(), p1.to_numpy()]

    monkeypatch.setattr(mod, "preprocessar_novos_dados", fake_preprocess)
    monkeypatch.setattr(mod, "carregar_modelo_mlflow", lambda _: FakeModel())

    df_novos = mod.carregar_novos_dados(str(dataset_csv))
    df_proc = mod.preprocessar_novos_dados(df_novos, colunas_data=["Data_Nascimento"], drop_cols=["ID_Cliente"])
    preds, probas = mod.gerar_predicoes(mod.carregar_modelo_mlflow("dummy"), df_proc)

    saida = tmp_path / "predicoes.csv"
    mod.salvar_resultado(df_novos, preds, probas, saida=str(saida))

    assert saida.exists()
    out = pd.read_csv(saida)
    assert "predito_inadimplente" in out.columns
    assert "score_inadimplencia" in out.columns
    # Mantém colunas originais + colunas de saída
    for col in ["ID_Cliente", "Estado", "Sexo"]:
        assert col in out.columns
