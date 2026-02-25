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
# PREPROCESSADOR — DIABETES PIPELINE (Baseline v1.2)
# -----------------------------------------------------------------------------
# 5) EVOLUÇÃO HISTÓRICA DO PIPELINE
# -----------------------------------------------------------------------------
#
# v1.2
#     Imputação + StandardScaler + OneHot
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
    #V1.2 -------------------------------------------------
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) ])

    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(
        handle_unknown='ignore',
        sparse_output=False))])

    preprocessador = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('cat', cat_pipeline, cat_features),
            ('bin', 'passthrough', binary_features)],
        verbose_feature_names_out=False)

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
            'version': 'v1.2'
        }
    }



    filename='preprocess_diabetes_v1.2.joblib'
    joblib.dump(artifact, filename)



    X_train.to_csv(os.path.join(DATA_DIR, 'X_train_raw.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_DIR, 'X_test_raw.csv'), index=False)
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train_raw.csv'), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, 'y_test_raw.csv'), index=False)

    print(' Joblib '+filename+ ' foi salvo!')

if __name__ == "__main__":
    main()







