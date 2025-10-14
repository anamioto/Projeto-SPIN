#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import tempfile
from datetime import datetime
from pathlib import Path
import logging
import pandas as pd
import numpy as np
import mlflow
from mlflow.exceptions import RestException

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("batch-scoring")

# -------------------- WORKSPACE (fixo) --------------------
def get_mlclient() -> MLClient:
    return MLClient(
        DefaultAzureCredential(),
        subscription_id="0b97f8d7-e740-4d8a-be3c-96eea4182bf8",
        resource_group_name="AulasAlura",
        workspace_name="DS-Workspace",
    )

def set_mlflow_to_workspace(ml_client: MLClient) -> None:
    ws = ml_client.workspaces.get(ml_client.workspace_name)
    mlflow.set_tracking_uri(ws.mlflow_tracking_uri)
    log.info(f"MLflow Tracking URI: {ws.mlflow_tracking_uri}")

# -------------------- LOAD DO MODELO --------------------
def _load_from_registry(model_name: str, stage_or_version: str):
    uri = f"models:/{model_name}/{stage_or_version}"
    log.info(f"Tentando MLflow Registry: {uri}")
    return mlflow.pyfunc.load_model(uri)

def _load_from_azureml_model_asset(ml_client: MLClient, model_name: str, version: str):
    tmp_dir = Path(tempfile.mkdtemp(prefix="aml-model-"))
    log.info(f"Baixando Model asset '{model_name}:{version}' para {tmp_dir} ...")
    ml_client.models.download(name=model_name, version=version, download_path=str(tmp_dir))
    candidates = list(tmp_dir.rglob("MLmodel"))
    if not candidates:
        raise FileNotFoundError("Não encontrei 'MLmodel' no asset baixado.")
    model_dir = candidates[0].parent
    log.info(f"Carregando MLflow model de: {model_dir}")
    return mlflow.pyfunc.load_model(model_dir.as_posix())

def load_model_resiliente(ml_client: MLClient, model_name: str, stage_or_version: str):
    try:
        return _load_from_registry(model_name, stage_or_version)
    except RestException as e:
        log.warning(f"Registry falhou ({e}). Tentando Model asset ...")
    except Exception as e:
        log.warning(f"Registry falhou ({e}). Tentando Model asset ...")

    if not stage_or_version.isdigit():
        raise ValueError("Para carregar como Model asset, forneça VERSÃO numérica (ex.: --model-version 1).")
    return _load_from_azureml_model_asset(ml_client, model_name, stage_or_version)

# -------------------- DADOS / SCORING --------------------
def load_dataframe_from_local_csv(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path}")
    log.info(f"Lendo CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Shape: {df.shape}")
    return df

def _expected_feature_names_from_signature(pyfunc_model) -> list[str] | None:
    try:
        sig = getattr(pyfunc_model, "metadata", None)
        if sig and getattr(sig, "signature", None) and sig.signature.inputs:
            return [inp.name for inp in sig.signature.inputs.inputs if inp.name]
    except Exception:
        pass
    return None

def _align_dataframe_to_features(df: pd.DataFrame, expected_cols: list[str], id_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids_out = pd.DataFrame(index=df.index)
    for c in id_cols:
        if c not in df.columns:
            raise KeyError(f"Coluna de ID '{c}' não encontrada.")
        ids_out[c] = df[c]

    work = df.drop(columns=id_cols, errors="ignore").copy()
    for col in ["target", "label", "inadimplente", "Status_Pagamento"]:
        if col in work.columns:
            work = work.drop(columns=[col])

    extras = [c for c in work.columns if c not in expected_cols]
    if extras:
        log.warning(f"Removendo colunas não vistas no treino: {extras}")
        work = work.drop(columns=extras)

    missing = [c for c in expected_cols if c not in work.columns]
    if missing:
        log.warning(f"Criando colunas faltantes com 0: {missing}")
        for c in missing:
            work[c] = 0

    work = work[expected_cols]

    # Tentativa leve de coerção numérica
    for c in work.columns:
        if work[c].dtype == "object":
            try:
                work[c] = pd.to_numeric(work[c], errors="raise")
            except Exception:
                work[c] = work[c].astype(str)
        if np.issubdtype(work[c].dtype, np.integer):
            work[c] = work[c].astype("float64")

    return work, ids_out

def score_dataframe(model, df: pd.DataFrame, id_cols: list[str]) -> pd.DataFrame:
    expected = _expected_feature_names_from_signature(model)
    if not expected:
        native_model = getattr(getattr(model, "_model_impl", None), "sklearn_model", None)
        if native_model is not None and hasattr(native_model, "feature_names_in_"):
            expected = list(native_model.feature_names_in_)
    if not expected:
        raise ValueError("Não foi possível identificar as features esperadas pelo modelo.")

    X, ids_out = _align_dataframe_to_features(df, expected_cols=expected, id_cols=id_cols)

    preds = model.predict(X)
    out = ids_out.reset_index(drop=True)

    if isinstance(preds, pd.DataFrame):
        out = pd.concat([out, preds.add_prefix("pred_").reset_index(drop=True)], axis=1)
    else:
        out["prediction"] = preds

    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            proba_df = pd.DataFrame(proba)
            proba_df.columns = [f"proba_class_{i}" for i in range(proba_df.shape[1])]
            out = pd.concat([out, proba_df], axis=1)
    except Exception:
        pass

    return out

# -------------------- SALVAR CSV --------------------
def save_predictions_csv(df_out: pd.DataFrame, input_csv_path: str, output_prefix: str) -> Path:
    """
    Salva o CSV de predições no MESMO diretório do CSV de entrada.
    Se não for possível escrever lá, salva em ./outputs/ do working dir.
    """
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    input_dir = Path(input_csv_path).resolve().parent
    candidate_path = input_dir / f"{output_prefix}_{ts}.csv"

    try:
        df_out.to_csv(candidate_path, index=False, encoding="utf-8")
        log.info(f"CSV salvo (mesmo ambiente): {candidate_path}")
        return candidate_path
    except Exception as e:
        log.warning(f"Não consegui salvar em {input_dir} ({e}). Usando ./outputs/ ...")
        Path("outputs").mkdir(exist_ok=True)
        fallback = Path("outputs") / f"{output_prefix}_{ts}.csv"
        df_out.to_csv(fallback, index=False, encoding="utf-8")
        log.info(f"CSV salvo em fallback: {fallback}")
        return fallback

# (Opcional) publicar no blob
def publish_csv_as_data_asset(
    ml_client: MLClient,
    csv_path: str,
    name_prefix: str = "predicoes_inadimplencia",
    datastore: str = "workspaceblobstore",
    description: str | None = None,
) -> Data:
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    asset_name = f"{name_prefix}_{ts}"
    data_asset = Data(
        name=asset_name,
        path=csv_path,
        type=AssetTypes.URI_FILE,
        datastore=datastore,
        description=description or f"Predições geradas em {ts}",
        version="1",
    )
    created = ml_client.data.create_or_update(data_asset)
    log.info(f"Publicado no blob: {created.name}:{created.version} | URI: {created.path}")
    return created

# -------------------- MAIN --------------------
def main():
    parser = argparse.ArgumentParser(description="Scoring batch salvando CSV no mesmo ambiente do input.")
    parser.add_argument("--model-name", type=str, default="ModelRFC1")
    parser.add_argument("--model-version", type=str, default="1")
    parser.add_argument("--registry-stage", type=str, default=None, help="Ex.: Production (opcional)")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--id-cols", nargs="*", default=["ID_Cliente"])
    parser.add_argument("--output-prefix", type=str, default="predicoes_inadimplencia")
    parser.add_argument("--upload-output", type=str, default="false", help="true/false (default false)")
    parser.add_argument("--datastore", type=str, default="workspaceblobstore")
    args = parser.parse_args()

    ml_client = get_mlclient()
    set_mlflow_to_workspace(ml_client)

    stage_or_version = args.registry_stage if args.registry_stage else args.model_version
    model = load_model_resiliente(ml_client, args.model_name, stage_or_version)

    df_in = load_dataframe_from_local_csv(args.input_csv)
    df_out = score_dataframe(model, df_in, id_cols=args.id_cols)

    # SALVA NO MESMO AMBIENTE DO INPUT
    saved_path = save_predictions_csv(df_out, args.input_csv, args.output_prefix)

    # Opcional: publicar no blob (apenas se quiser)
    if args.upload_output.lower() == "true":
        publish_csv_as_data_asset(
            ml_client,
            str(saved_path),
            name_prefix=args.output_prefix,
            datastore=args.datastore,
            description=f"Predições {args.model_name}:{args.model_version} sobre {Path(args.input_csv).name}",
        )

    log.info("Concluído.")

if __name__ == "__main__":
    main()
