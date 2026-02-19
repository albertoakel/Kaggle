import os
import joblib
from datetime import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler,FunctionTransformer
from sklearn.impute import SimpleImputer


# =============================================================================
# PREPROCESSADOR — DIABETES PIPELINE (Baseline v1.4)
# =============================================================================
#
# Este módulo constrói e salva um pipeline de pré-processamento básico para
# modelagem supervisionada de diagnóstico de diabetes. Ele representa a linha
# de base (baseline engineering) utilizada para comparação com versões mais
# avançadas contendo feature engineering supervisionado.
#
# O foco desta versão é:
#
#     ✔ Simplicidade
#     ✔ Robustez estatística
#     ✔ Ausência de data leakage
#     ✔ Compatibilidade com modelos baseados em árvores
#       (XGBoost / LightGBM / HistGB)
#
# -----------------------------------------------------------------------------
# FLUXO GERAL
# -----------------------------------------------------------------------------
#
# 1) Leitura e separação do target
# 2) Train/Test split ANTES de qualquer decisão estatística
# 3) Identificação automática dos tipos de variáveis
# 4) Construção do ColumnTransformer
# 5) Fit apenas em X_train
# 6) Serialização do artefato (joblib)
#
# -----------------------------------------------------------------------------
# 1) PREVENÇÃO DE DATA LEAKAGE
# -----------------------------------------------------------------------------
#
# O split ocorre antes de:
#
#     - imputação
#     - estatísticas descritivas
#     - definição de bounds
#     - qualquer transformação aprendida
#
# Isso garante que o pipeline não utilize informação do conjunto de teste.
#
# -----------------------------------------------------------------------------
# 2) FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------
#
# compute_iqr_bounds()
#     Calcula limites baseados em IQR para detecção de outliers:
#
#         lower = Q1 - k * IQR
#         upper = Q3 + k * IQR
#
#     Usado apenas em treino para filtragem opcional.
#
# filter_outliers_train()
#     Remove observações fora dos limites definidos.
#     Nunca aplicado ao conjunto de teste.
#
# to_category()
#     Conversão explícita para dtype "category".
#
# to_float32()
#     Redução de memória e otimização para treino.
#
# to_int8()
#     Compressão de variáveis binárias.
#
# -----------------------------------------------------------------------------
# 3) IDENTIFICAÇÃO DE FEATURES
# -----------------------------------------------------------------------------
#
# Binárias:
#     family_history_diabetes
#     hypertension_history
#     cardiovascular_history
#
# Categóricas:
#     Detectadas automaticamente via dtype object
#
# Numéricas:
#     Todas as restantes (exceto binárias)
#
# -----------------------------------------------------------------------------
# 4) PIPELINE DE TRANSFORMAÇÃO
# -----------------------------------------------------------------------------
#
# NUMÉRICAS
# ---------
#     • Imputação por mediana
#     • Conversão float32
#
#     Motivação:
#         - robusto a outliers
#         - reduz memória
#         - adequado para árvores
#
#
# CATEGÓRICAS NOMINAIS
# ---------------------
#     • Mantidas como dtype category
#
#     Motivação:
#         - compatível com XGBoost enable_categorical
#         - evita explosão dimensional do OneHot
#         - acelera treino
#
#
# BINÁRIAS
# --------
#     • Conversão para int8
#
#     Motivação:
#         - compressão de memória
#         - sem perda semântica
#
#
# -----------------------------------------------------------------------------
# 5) EVOLUÇÃO HISTÓRICA DO PIPELINE
# -----------------------------------------------------------------------------
#
# v1.2
#     Imputação + StandardScaler + OneHot
#
# v1.3
#     Remoção do scaling (árvores não precisam)
#
# v1.31
#     Casting explícito para controle de memória
#
# v1.4 (ATUAL)
#     Abandono do OneHot
#     Uso de categorical nativo
#     Pipeline mais leve e rápido
#
# -----------------------------------------------------------------------------
# 6) SAÍDA SERIALIZADA
# -----------------------------------------------------------------------------
#
# O artefato salvo contém:
#
#     {
#         preprocessador,
#         lista de features numéricas,
#         lista de categóricas,
#         metadados de rastreabilidade
#     }
#




def compute_iqr_bounds(df, cols, threshold=1.9):
    bounds = {}
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        bounds[col] = {
            'lower': q1 - threshold * iqr,
            'upper': q3 + threshold * iqr
        }
    return bounds


def filter_outliers_train(X, y, bounds):
    mask = pd.Series(True, index=X.index)
    for col, b in bounds.items():
        mask &= (X[col] >= b['lower']) & (X[col] <= b['upper'])
    return X.loc[mask], y.loc[mask]

def to_category(X):
    return X.astype("category")


def to_float32(X):
    return X.astype("float32")

def to_int8(X):
    return X.astype("int8")

def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")


    os.makedirs(DATA_DIR, exist_ok=True)
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    TARGET = "diagnosed_diabetes"
    # 1. Leitura dos dados
    dfo = pd.read_csv("/home/akel/PycharmProjects/Kaggle/Diabetes_Prediction_Challenge/data/raw/train.csv")
    df = dfo.drop(columns='id') #v.16 comentar( incluir Id columns)

    #

    # ✂️ 2. Split ANTES de qualquer decisão estatística
    X = df.drop(columns=TARGET)
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=TEST_SIZE,random_state=RANDOM_STATE,stratify=y)

    # Identificação das features
    binary_features = ['family_history_diabetes','hypertension_history','cardiovascular_history']
    cat_features = X_train.select_dtypes(include='object').columns
    num_features = (X_train.select_dtypes(include='number').columns.difference(binary_features))

    #Versões
    # V1.2 -------------------------------------------------
    # num_pipeline = Pipeline([
    #     ('imputer', SimpleImputer(strategy='median')),
    #     ('scaler', StandardScaler()) ])
    #
    # cat_pipeline = Pipeline([
    #     ('onehot', OneHotEncoder(
    #     handle_unknown='ignore',
    #     sparse_output=False))])
    #
    # preprocessador = ColumnTransformer(
    #     transformers=[
    #         ('num', num_pipeline, num_features),
    #         ('cat', cat_pipeline, cat_features),
    #         ('bin', 'passthrough', binary_features)],
    #     verbose_feature_names_out=False)

    # V1.3 -------------------------------------------------
    # num_pipeline = Pipeline([
    #     ('imputer', SimpleImputer(strategy='median')) ])
    #
    # cat_pipeline = Pipeline([
    #     ('onehot', OneHotEncoder(
    #     handle_unknown='ignore',
    #     sparse_output=False))])
    #
    # preprocessador = ColumnTransformer(
    #     transformers=[
    #         ('num', num_pipeline, num_features),
    #         ('cat', cat_pipeline, cat_features),
    #         ('bin', 'passthrough', binary_features)],
    #     verbose_feature_names_out=False)


    # V1.31 -------------------------------------------------
    # num_pipeline = Pipeline([
    #     ('imputer', SimpleImputer(strategy='median')),
    #     ('cast_float32', FunctionTransformer(
    #         to_float32,
    #         feature_names_out='one-to-one'))  ])
    #
    # cat_pipeline = Pipeline([
    #     ('onehot', OneHotEncoder(
    #     handle_unknown='ignore',
    #     sparse_output=False)),
    #     ('cast_int8', FunctionTransformer(
    #     to_int8,
    #      feature_names_out='one-to-one' ))  ])
    #
    # bin_pipeline = Pipeline([
    #     ('cast_int8', FunctionTransformer(
    #         to_int8,
    #         feature_names_out='one-to-one' )) ])
    #
    #
    # preprocessador = ColumnTransformer(
    #     transformers=[
    #         ('num', num_pipeline, num_features),
    #         ('cat', cat_pipeline, cat_features),
    #         ('bin', bin_pipeline, binary_features)
    #     ],
    #     verbose_feature_names_out=False
    # )
    # preprocessador.set_output(transform="pandas")

    # V1.4 -------------------------------------------------
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('cast_float32', FunctionTransformer(
            to_float32,
            feature_names_out='one-to-one'))])

    cat_pipeline = Pipeline([
        ('to_category', FunctionTransformer(
            to_category,
            feature_names_out='one-to-one'))])

    bin_pipeline = Pipeline([
        ('cast_int8', FunctionTransformer(
            to_int8,
            feature_names_out='one-to-one' )) ])

    preprocessador = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features),
            ('bin', bin_pipeline, binary_features)
        ],
        verbose_feature_names_out=False
    )
    preprocessador.set_output(transform="pandas")


    preprocessador.fit(X_train)

    artifact = {
        'preprocessador': preprocessador,
        'num_features': list(num_features),
        'cat_features': list(cat_features),
        'metadata': {
            'dataset': 'Health & Diabetes Dataset',
            'descricao': (
                'Pipeline sem data leakage: '
                'split prévio, outliers no treino, '
                'sem std scaler.'
                'category,bin e float transformers.'
            ),
            'target': TARGET,
            'fit_on': 'X_train only',
            'created_at': datetime.now().isoformat(),
            'author': 'Alberto Akel',
            'version': 'v1.4'
        }
    }



    filename='preprocess_diabetes_v1.4.joblib'
    joblib.dump(artifact, filename)



    X_train.to_csv(os.path.join(DATA_DIR, 'X_train_raw.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_DIR, 'X_test_raw.csv'), index=False)
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train_raw.csv'), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, 'y_test_raw.csv'), index=False)

    print(' Joblib '+filename+ ' foi salvo!')

if __name__ == "__main__":
    main()







