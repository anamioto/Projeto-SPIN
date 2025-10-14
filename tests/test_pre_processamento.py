# tests/test_pre_processamento.py
# -*- coding: utf-8 -*-
"""
Testes unitários para o script de pré-processamento.

Boas práticas aplicadas:
- Uso de pytest + fixtures reutilizáveis
- Isolamento de I/O com tmp_path
- Mocks de dependências externas (Azure ML / Credenciais)
- Testes determinísticos (monkeypatch do datetime)
- Verificações de tipos, formas e efeitos colaterais mínimos
"""
from __future__ import annotations

import sys
import types
import importlib.util
from pathlib import Path
from datetime import datetime
import builtins
import pandas as pd
import numpy as np
import pytest


# === Configurações gerais ===
SCRIPT_FILENAME = "pre-processamento.py"


# ---------- Fixtures utilitárias ----------

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """DataFrame pequeno e representativo para os testes unitários."""
    data = {
        "ID_Cliente": [1, 2, 3, 4],
        "Data_Nascimento": ["2000-01-16", "1990-01-14", "1985-12-31", "2002-07-01"],
        "Estado": ["SP", "RJ", "SP", "MG"],
        "Cidade": ["São Paulo", "Rio de Janeiro", "Campinas", None],
        "Status_Pagamento": [0, 1, 0, 1],  # alvo
        "Data_Contratacao": ["2023-01-01"] * 4,
        "Data_Vencimento_Fatura": ["2023-02-01", "2023-02-15", "2023-03-01", "2023-01-31"],
        "Data_Ingestao": ["2023-02-05"] * 4,
        "Data_Atualizacao": ["2023-02-10"] * 4,
        # numéricas com nulos
        "Renda_Mensal": [4000, None, 8000, 6000],
        "Limite_Credito": [2000, 3000, None, 2500],
        # categóricas adicionais
        "Sexo": ["F", "M", "F", "F"],
        "CPF": ["xxx", "yyy", "zzz", "www"],
        # colunas a dropar
        "Telefone": ["t1", "t2", "t3", "t4"],
        "Nome": ["n1", "n2", "n3", "n4"],
        "Email": ["e1", "e2", "e3", "e4"],
    }
    return pd.DataFrame(data)


@pytest.fixture()
def user_csv_path(tmp_path: Path, sample_df: pd.DataFrame) -> Path:
    """Cria um CSV temporário representando o dataset do usuário."""
    csv_path = tmp_path / "dataset.csv"
    sample_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture()
def inject_fake_azure_modules(user_csv_path: Path, monkeypatch):
    """
    Injeta módulos 'azure', 'azure.identity' e 'azure.ai.ml' falsos em sys.modules
    para que o import do script não faça chamadas reais à Azure.
    """
    # azure.identity.DefaultAzureCredential (stub)
    mod_identity = types.ModuleType("azure.identity")

    class DefaultAzureCredential:  # noqa: N801 (manter assinatura)
        def __init__(self, *_, **__):
            pass

    mod_identity.DefaultAzureCredential = DefaultAzureCredential

    # azure.ai.ml.MLClient (stub)
    mod_ml = types.ModuleType("azure.ai.ml")

    class _FakeDataAsset:
        def __init__(self, path: str):
            self.path = path

    class _FakeDataNamespace:
        def __init__(self, path: str):
            self._path = path

        def get(self, *_args, **_kwargs):
            # Retorna um objeto com atributo .path apontando para o CSV temporário
            return _FakeDataAsset(str(self._path))

    class MLClient:  # noqa: N801
        def __init__(self, *_args, **_kwargs):
            # Namespace 'data' com get() retornando o CSV
            self.data = _FakeDataNamespace(user_csv_path)

    mod_ml.MLClient = MLClient

    # Pacote azure.ai "container"
    mod_ai = types.ModuleType("azure.ai")
    # Pacote raiz "azure"
    mod_azure = types.ModuleType("azure")

    # Registra módulos na árvore
    sys.modules["azure"] = mod_azure
    sys.modules["azure.identity"] = mod_identity
    sys.modules["azure.ai"] = mod_ai
    sys.modules["azure.ai.ml"] = mod_ml

    yield

    # Limpa após o teste
    for name in ["azure.ai.ml", "azure.ai", "azure.identity", "azure"]:
        sys.modules.pop(name, None)


@pytest.fixture()
def mod(tmp_path: Path, inject_fake_azure_modules, monkeypatch):
    """
    Importa o módulo do usuário a partir de src/pre-processamento.py, isolando efeitos:
    - Chdir para tmp_path (arquivos gerados ficam aqui)
    - Carrega via importlib (nome do arquivo tem hífen)
    """
    monkeypatch.chdir(tmp_path)

    SCRIPT_FILENAME = "pre-processamento.py"
    # raiz do repo = pai da pasta 'tests'
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "src" / SCRIPT_FILENAME

    # Fallbacks (opcionais) caso a estrutura esteja diferente em algum ambiente
    if not script_path.exists():
        candidates = [
            Path(__file__).with_name(SCRIPT_FILENAME),     # ao lado do teste
            repo_root / SCRIPT_FILENAME,                   # raiz
            Path.cwd().parent / "src" / SCRIPT_FILENAME,   # exec em subpastas
            Path("/mnt/data") / SCRIPT_FILENAME,           # anexo (ambientes especiais)
        ]
        script_path = next((p for p in candidates if p.exists()), None)

    if script_path is None or not script_path.exists():
        raise FileNotFoundError(
            "Não encontrei 'src/pre-processamento.py'. "
            "Garanta a estrutura: <repo>/src/pre-processamento.py e <repo>/tests/..."
        )

    # Carrega o módulo (nome interno sem hífen)
    spec = importlib.util.spec_from_file_location("pre_processamento", script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pre_processamento"] = module
    assert spec and spec.loader, "Falha ao preparar import do módulo."
    spec.loader.exec_module(module)
    return module


# ---------- Testes unitários por função ----------

def test_carregar_dados_ler_csv(tmp_path: Path, mod):
    csv = tmp_path / "tiny.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(csv, index=False)
    df = mod.carregar_dados(str(csv))
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_tratar_valores_nulos_preenchimento(mod):
    df = pd.DataFrame({"num": [1.0, np.nan, 3.0], "cat": ["a", None, "b"]})
    out = mod.tratar_valores_nulos(df.copy())
    assert out["num"].isna().sum() == 0
    assert out["cat"].isna().sum() == 0
    # num preenchido por mediana, cat por 'desconhecido'
    assert "desconhecido" in out["cat"].unique()


def test_tratar_data_nascimento_calcula_idade(monkeypatch, mod):
    # Congela "hoje" em 2024-01-15 para tornar determinístico
    class FixedDate(datetime):
        @classmethod
        def today(cls):
            return cls(2024, 1, 15)

    monkeypatch.setattr(mod, "datetime", FixedDate)

    df = pd.DataFrame({"Data_Nascimento": ["2000-01-16", "2000-01-14"]})
    out = mod.tratar_data_nascimento(df.copy())
    # 2000-01-16 -> aniversário ainda não chegou (23)
    # 2000-01-14 -> aniversário já passou (24)
    assert (out["Idade"].tolist()) == [23, 24]
    assert np.issubdtype(out["Data_Nascimento"].dtype, np.datetime64)


def test_converter_colunas_data_converte_e_trata_invalidos(capsys, mod):
    df = pd.DataFrame(
        {
            "d1": ["2024-01-02", "02/01/2024", "inválido"],
            "d2": ["2023-12-31", None, "2024-01-01"],
        }
    )
    out = mod.converter_colunas_data(df, ["d1", "d2", "inexistente"], formato=None, erros="coerce")
    assert np.issubdtype(out["d1"].dtype, np.datetime64)
    assert np.issubdtype(out["d2"].dtype, np.datetime64)
    # inválidos viram NaT quando errors="coerce"
    assert out["d1"].isna().sum() == 1
    # Mensagem impressa para coluna inexistente
    captured = capsys.readouterr().out
    assert "não existe no DataFrame" in captured


def test_codificar_variaveis_categoricas_estado_e_cidade(mod):
    df = pd.DataFrame(
        {
            "Estado": ["SP", "RJ", "SP"],
            "Cidade": ["São Paulo", "Rio", "Campinas"],
            "Sexo": ["F", "M", "F"],  # binária -> int8
            "Status_Pagamento": [0, 1, 0],
        }
    )
    out = mod.codificar_variaveis_categoricas(df.copy())
    # 'Cidade' removida, coluna de frequência adicionada
    assert "Cidade" not in out.columns
    assert ("cidade_freq_por_estado" in out.columns) or ("cidade_freq_global" in out.columns)

    # One-hot de Estado
    estado_cols = [c for c in out.columns if c.startswith("Estado_")]
    assert len(estado_cols) >= 2
    assert all(str(out[c].dtype) == "uint8" for c in estado_cols)

    # 'Sexo' virou códigos inteiros pequenos (int8) e não é mais 'object'
    assert str(out["Sexo"].dtype) == "int8"


def test_escalar_variaveis_media_zero_desvio_um(mod):
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    out = mod.escalar_variaveis(df.copy(), ["a", "b"])
    # Médias próximas de 0 e desvios ~1
    means = out[["a", "b"]].mean().abs().to_numpy()
    stds = out[["a", "b"]].std(ddof=0).to_numpy()  # ddof=0 para comparar com StandardScaler
    assert np.all(means < 1e-7)
    assert np.allclose(stds, np.ones_like(stds), atol=1e-6)


def test_split_dados_separa_tamanhos(mod):
    df = pd.DataFrame(
        {
            "x1": np.arange(10),
            "x2": np.arange(10, 20),
            "y": [0, 1] * 5,
        }
    )
    X_train, X_test, y_train, y_test = mod.split_dados(df, target="y", test_size=0.3, random_state=0)
    assert len(X_train) == len(y_train) == 7
    assert len(X_test) == len(y_test) == 3
    assert "y" not in X_train.columns and "y" not in X_test.columns


# ---------- Teste de pipeline ponta a ponta + efeitos de topo ----------

def test_pipeline_preprocessamento_e2e_cria_splits_e_salva_csvs(
    mod, user_csv_path: Path, monkeypatch
):
    target = "Status_Pagamento"
    colunas_data = ["Data_Contratacao", "Data_Vencimento_Fatura", "Data_Ingestao", "Data_Atualizacao"]
    drop_cols = [
        "Telefone", "Nome", "Email",
        "Data_Nascimento", "Data_Contratacao", "Data_Vencimento_Fatura", "Data_Ingestao", "Data_Atualizacao",
    ]

    # Executa pipeline diretamente no CSV do usuário
    X_train, X_test, y_train, y_test = mod.pipeline_preprocessamento(
        str(user_csv_path), target, colunas_data, drop_cols=drop_cols
    )

    # Tipos e integridades
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, (pd.Series, pd.DataFrame))
    assert isinstance(y_test, (pd.Series, pd.DataFrame))
    assert target not in X_train.columns

    # O código no topo do módulo salva X_train.csv etc.; como mudamos o CWD para tmp_path,
    # os arquivos devem existir aqui.
    for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
        assert Path(fname).exists(), f"Esperava {fname} salvo no diretório de teste."


# ---------- Testes de robustez ----------

def test_converter_colunas_data_ignora_quando_erros_ignore(mod):
    df = pd.DataFrame({"d": ["2024-01-01", "invalida"]})
    out = mod.converter_colunas_data(df.copy(), ["d"], formato="%Y-%m-%d", erros="ignore")
    # Mantém string inválida sem explodir
    assert out["d"].dtype == object
    assert "invalida" in out["d"].tolist()