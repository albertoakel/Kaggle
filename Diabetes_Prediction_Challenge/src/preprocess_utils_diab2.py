import os
import joblib
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.impute import SimpleImputer


# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

class RiskFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        # parâmetros aprendidos no treino (evita inconsistência)
        self.min_age_ = X['age'].min()
        self.max_age_ = X['age'].max()

        paw_tmp = 1 / (np.log1p(X['physical_activity_minutes_per_week']) + 1)
        self.paw_min_ = paw_tmp.min()
        self.paw_max_ = paw_tmp.max()

        return self

    def transform(self, X):

        X = X.copy()

        # -----------------------------
        # Grupos
        # -----------------------------
        X['Age_Group'] = pd.cut(
            X['age'],
            bins=[18, 29, 44, 59, 69, 79, np.inf],
            labels=[
                'Jovem Adulto (18–29)',
                'Adulto (30–44)',
                'Meia-idade (45–59)',
                'Idoso Jovem (60–69)',
                'Idoso (70–79)',
                'Idoso Muito Longevo (80+)'
            ],
            include_lowest=True
        )

        X['PAW_Group'] = pd.cut(
            X['physical_activity_minutes_per_week'],
            bins=[0, 29, 149, 299, np.inf],
            labels=[
                'Sedentário',
                'Baixo (<150)',
                'Recomendado (150-300)',
                'Alto (>300)'
            ],
            include_lowest=True
        )

        # -----------------------------
        # Probabilidades fixas
        # -----------------------------
        age_prob = {
            'Jovem Adulto (18–29)': 0.4247,
            'Adulto (30–44)': 0.5527,
            'Meia-idade (45–59)': 0.6308,
            'Idoso Jovem (60–69)': 0.7018,
            'Idoso (70–79)': 0.8122,
            'Idoso Muito Longevo (80+)': 0.8319
        }

        paw_prob = {
            'Sedentário': 0.7514,
            'Baixo (<150)': 0.6319,
            'Recomendado (150-300)': 0.4032,
            'Alto (>300)': 0.3505
        }

        fh_prob = {0: 0.5804, 1: 0.8673}

        X['risk_age_p'] = X['Age_Group'].map(age_prob).astype(float).fillna(0.5)
        X['risk_paw_p'] = X['PAW_Group'].map(paw_prob).astype(float).fillna(0.5)
        X['risk_fh_p'] = X['family_history_diabetes'].map(fh_prob).astype(float).fillna(0.5)

        X['global_prob_risk'] = (
            X['risk_age_p'] +
            X['risk_paw_p'] +
            X['risk_fh_p']
        ) / 3

        # -----------------------------
        # Scores contínuos estáveis
        # -----------------------------
        X['risk_age_cont'] = (
            (X['age'] - self.min_age_) /
            (self.max_age_ - self.min_age_ + 1e-9)
        )

        paw_tmp = 1 / (np.log1p(X['physical_activity_minutes_per_week']) + 1)

        X['risk_paw_cont'] = (
            (paw_tmp - self.paw_min_) /
            (self.paw_max_ - self.paw_min_ + 1e-9)
        )

        X['continuous_risk_score'] = (
            (X['risk_age_cont'] * 0.35) +
            (X['risk_paw_cont'] * 0.15) +
            (X['family_history_diabetes'].astype(float) * 0.50)
        )

        return X


# ==========================================================
# CAST HELPERS
# ==========================================================

def to_category(X):
    return X.astype("category")

def to_float32(X):
    return X.astype("float32")

def to_int8(X):
    return X.astype("int8")


# ==========================================================
# MAIN
# ==========================================================

def main():

    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    TARGET = "diagnosed_diabetes"

    # Caminhos
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(DATA_DIR, exist_ok=True)

    # -----------------------------
    # Leitura
    # -----------------------------
    df = pd.read_csv(
        "/home/akel/PycharmProjects/Kaggle/Diabetes_Prediction_Challenge/data/raw/train.csv"
    ).drop(columns='id')

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    # -----------------------------
    # Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    # -----------------------------
    # FEATURE ENGINEERING (preview)
    # -----------------------------
    fe = RiskFeatureEngineer()
    X_tmp = fe.fit_transform(X_train)

    binary_features = [
        'family_history_diabetes',
        'hypertension_history',
        'cardiovascular_history'
    ]

    cat_features = X_tmp.select_dtypes(
        include=['object', 'category']
    ).columns

    num_features = X_tmp.select_dtypes(
        include='number'
    ).columns.difference(binary_features)

    # -----------------------------
    # Pipelines
    # -----------------------------
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('float32', FunctionTransformer(to_float32))
    ])

    cat_pipe = Pipeline([
        ('category', FunctionTransformer(to_category))
    ])

    bin_pipe = Pipeline([
        ('int8', FunctionTransformer(to_int8))
    ])

    preprocess = ColumnTransformer(
        [
            ('num', num_pipe, num_features),
            ('cat', cat_pipe, cat_features),
            ('bin', bin_pipe, binary_features)
        ],
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    preprocessador_step2 = Pipeline([
        ('feature_engineering', RiskFeatureEngineer()),
        ('preprocess', preprocess)
    ])

    # -----------------------------
    # Fit
    # -----------------------------
    preprocessador_step2.fit(X_train)

    # -----------------------------
    # Save artifact
    # -----------------------------
    artifact = {
        'preprocessador': preprocessador_step2,
        'metadata': {
            'dataset': 'Health & Diabetes Dataset',
            'version': 'v2',
            'target': TARGET,
            'created_at': datetime.now().isoformat(),
            'author': 'Alberto Akel'
        }
    }

    joblib.dump(artifact, 'preprocess_diabetes_v2.joblib')

    # -----------------------------
    # Save raw splits
    # -----------------------------
    X_train.to_csv(os.path.join(DATA_DIR, 'X_train_raw.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_DIR, 'X_test_raw.csv'), index=False)
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train_raw.csv'), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, 'y_test_raw.csv'), index=False)

    print("✅ Pipeline salvo com sucesso!")


if __name__ == "__main__":
    main()
