import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
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

def tratar_data_nascimento(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a idade."""
    logging.info('Calculando a idade dos clientes.')
    df["Data_Nascimento"] = pd.to_datetime(df["Data_Nascimento"])
    ref = datetime.today() #data de referencia = hoje
    idade = ref.year - df["Data_Nascimento"].dt.year
    ajuste_aniversario = (
        (df["Data_Nascimento"].dt.month > ref.month) |
        ((df["Data_Nascimento"].dt.month == ref.month) & 
         (df["Data_Nascimento"].dt.day > ref.day))
    ).astype(int)
    df["Idade"] = idade - ajuste_aniversario

    return df

def converter_colunas_data(
    df: pd.DataFrame,
    colunas_data: list,
    formato: str = None,
    erros: str = "coerce"
) -> pd.DataFrame:
    """
    Converte colunas de datas em string para datetime no DataFrame.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com as colunas a serem convertidas.
    colunas_data : list
        Lista com os nomes das colunas de data a converter.
    formato : str, opcional
        Formato esperado da data, ex: "%Y-%m-%d".
        Se None, pandas tentará inferir o formato automaticamente.
    erros : {"raise", "coerce", "ignore"}, padrão="coerce"
        - "raise": gera erro se houver valor inválido
        - "coerce": converte valores inválidos em NaT
        - "ignore": deixa os valores inválidos como string

    Retorna
    -------
    pd.DataFrame
        DataFrame com as colunas convertidas para datetime.
    """
    df = df.copy()
    for coluna in colunas_data:
        if coluna in df.columns:
            df[coluna] = pd.to_datetime(
                df[coluna],
                format=formato,
                errors=erros,
            )
        else:
            print(f"A coluna '{coluna}' não existe no DataFrame.")
    return df


def calcular_tempo_assinatura(df, coluna_data="Data_Contratacao"):
    """
    Calcula o tempo de assinatura em formato numérico.

    Parâmetros
    ----------
    df : pd.DataFrame
        DataFrame com a coluna de datas de contratação.
    coluna_data : str
        Nome da coluna que contém a data de contratação.

    Retorna
    -------
    pd.DataFrame
        DataFrame com uma nova coluna:
        - Tempo_Assinatura_Meses_Totais : int (quantidade total de meses de assinatura)

    """
    df = df.copy()
    hoje = pd.to_datetime(datetime.today().date())

    # diferença em meses totais
    meses_totais = (hoje.year - df[coluna_data].dt.year) * 12 + (hoje.month - df[coluna_data].dt.month)

    # ajuste se ainda não chegou no "dia" da contratação neste mês
    ajuste = (hoje.day < df[coluna_data].dt.day).astype(int)
    meses_totais = meses_totais - ajuste

    df["Tempo_Assinatura_Meses_Totais"] = meses_totais.astype(int)

    return df

def calcular_tempo_atraso_fatura(
    df: pd.DataFrame,
    coluna_vencimento: str = "Data_Vencimento_Fatura",
    coluna_status: str = "Status_Pagamento",
    coluna_valor_aberto: str = "Valor_em_Aberto",
    valor_status_inadimplente: str = "Inadimplente"
) -> pd.DataFrame:
    """
    Calcula tempo de atraso da fatura em dias e meses (numéricos),
    considerando atraso quando:
      - Status_Pagamento == "Inadimplente" OU
      - Valor_em_Aberto > 0.

    Regras:
      - Se vencimento ainda não chegou -> atraso = 0.
      - Meses aproximados = dias/30 (1 casa decimal).
    """
    df = df.copy()

    hoje = pd.to_datetime(datetime.today().date())

    # Diferença em dias
    dias_diff = (hoje - df[coluna_vencimento]).dt.days

    # Pendência = inadimplente ou valor em aberto
    cond_inadimplente = (df[coluna_status] == valor_status_inadimplente)
    cond_valor_aberto = (df[coluna_valor_aberto] > 0)
    ha_pendencia = cond_inadimplente | cond_valor_aberto

    # Atraso só se já venceu e há pendência
    dias_atraso = np.where(ha_pendencia & (dias_diff > 0), dias_diff, 0).astype(int)

    # Colunas finais
    df["Dias_Atraso_Fatura"] = dias_atraso

    return df

def codificar_variaveis_categoricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica variáveis categóricas de forma otimizada:
      - 'estado': one-hot (baixa cardinalidade) com dtype uint8
      - 'cidade': frequency encoding por estado (alta cardinalidade)
      - Demais colunas object:
          * binárias -> 0/1
          * demais -> one-hot com dtype uint8
    """
    logging.info('Codificando variáveis categóricas com tratamento otimizado de estado/cidade')
    df = df.copy()

    # Padronização de colunas alvo se existirem
    tem_estado = 'Estado' in df.columns
    tem_cidade = 'Cidade' in df.columns

    if tem_estado:
        df['Estado'] = df['Estado'].astype('category')

    # --- CIDADE: frequency encoding por estado ---
    if tem_estado and tem_cidade:
        # Frequência da cidade dentro do estado
        freq = (
            df.groupby(['Estado', 'Cidade'], dropna=False)
              .size()
              .div(len(df))
              .rename('cidade_freq_por_estado')
        )
        df = df.join(freq, on=['Estado', 'Cidade'])
        # Nulos viram 0 (cidade ausente ou desconhecida)
        df['cidade_freq_por_estado'] = df['cidade_freq_por_estado'].fillna(0.0)
        # Remove a coluna textual 'cidade'
        df.drop(columns=['Cidade'], inplace=True)

    elif tem_cidade and not tem_estado:
        # Sem estado: frequência global da cidade
        freq = (
            df.groupby(['Cidade'], dropna=False)
              .size()
              .div(len(df))
              .rename('cidade_freq_global')
        )
        df = df.join(freq, on='Cidade')
        df['cidade_freq_global'] = df['cidade_freq_global'].fillna(0.0)
        df.drop(columns=['Cidade'], inplace=True)

    # --- ESTADO: one-hot encoding ---
    if tem_estado:
        dummies_estado = pd.get_dummies(df['Estado'], prefix='Estado', dtype='uint8')
        df = pd.concat([df.drop(columns=['Estado']), dummies_estado], axis=1)

    # --- Demais categóricas ---
    # Seleciona objetos restantes (já tiramos estado/cidade)
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 2:
            # Binária -> 0/1 (preserva NaN como -1 se existir)
            # Map automático e estável com category codes
            codes = df[col].astype('category').cat.codes
            # cat.codes usa -1 para NaN
            df[col] = codes.astype('int8')
        else:
            # One-hot nas demais com dtype compacto para ser mais leve o processamento
            dummies = pd.get_dummies(df[col], prefix=col, dtype='uint8')
            df.drop(columns=[col], inplace=True)
            df = pd.concat([df, dummies], axis=1)

    return df

def escalar_variaveis(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Escala variáveis numéricas."""
    logging.info('Escalando variáveis numéricas')
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def pipeline_preprocessamento(caminho_csv, target, colunas_data, drop_cols=[]):
    """Pipeline completo de pré-processamento."""
    df = carregar_dados(caminho_csv)
    #print(df.head())
    df = tratar_valores_nulos(df)
    df = tratar_data_nascimento(df)
    df = df.set_index('ID_Cliente', drop=True)
    df = converter_colunas_data(df, colunas_data)
    df = calcular_tempo_assinatura(df)
    df = calcular_tempo_atraso_fatura(df)
    df.drop(drop_cols, axis=1, inplace=True)
    print(" ")
    df = codificar_variaveis_categoricas(df)
    print(" ")
    #print(df.head())
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]
    df = escalar_variaveis(df, numeric_cols)
    logging.info('Pré-processamento finalizado')
    return df


#Autenticação do Azure ML
ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id = "0b97f8d7-e740-4d8a-be3c-96eea4182bf8",
    resource_group_name = "AulasAlura",
    workspace_name = "DS-Workspace" 
)


# Diretório dos dados

#teste - remover a # do input_csv quando testar
#input_csv = ml_client.data.get("base_inadimplencia_testes", version="1")

#treino - colocar a # no input_csv quando após registrar o modelo treinado
input_csv = ml_client.data.get("base_clientes_inadimplencia", version="1")
df = pd.read_csv(input_csv.path) 
target = "Status_Pagamento"
colunas_data = ["Data_Contratacao", "Data_Vencimento_Fatura", "Data_Ingestao", "Data_Atualizacao"] 
drop_cols = ["Telefone", "Nome", "Email", "Data_Nascimento", "Data_Contratacao", "Data_Vencimento_Fatura", "Data_Ingestao", "Data_Atualizacao"]  # Ajuste se necessário
df_transformado  = pipeline_preprocessamento(input_csv.path, target, colunas_data, drop_cols=drop_cols)

# Salva dados prontos para o próximo passo
df_transformado.to_csv("df_transformado.csv", index=True)

