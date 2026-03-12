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

# =============================================================================
# PREPROCESSADOR — DIABETES PIPELINE (v3a)*
# =============================================================================
# principal diferenca da versão 3 para versão "3a" é não utilização distribuição log
# da criação feature paw_continua para o risco probabilistico
#
# Este módulo constrói e salva um pipeline completo de pré-processamento para
# modelagem preditiva de diagnóstico de diabetes. O objetivo é padronizar a
# engenharia de atributos e a tipagem de dados antes do treinamento de modelos
# (ex.: XGBoost com suporte nativo a categorias).
#
#
#
# O pipeline executa as seguintes etapas:
#
# ---------------------------------------------------------------------------
# 1) Mapeamento Ordinal
# ---------------------------------------------------------------------------
# Converte variáveis categóricas ordinais para escala numérica preservando
# ordem semântica:
#
#   education_level → [0..3]
#   income_level    → [0..4]
#
#
# ---------------------------------------------------------------------------
# 2) Feature Engineering de Risco (RiskFeatureEngineer)
# ---------------------------------------------------------------------------
# Cria variáveis derivadas baseadas em conhecimento epidemiológico:
#
# (a) Discretização
#     - Age_Group → faixas etárias
#     - PAW_Group → nível de atividade física
#
# (b) Encoding probabilístico supervisionado
#     Calcula, usando apenas dados de treino:
#
#         P(diabetes | grupo)
#
#     Gerando:
#         risk_age_p
#         risk_paw_p
#         risk_fh_p
#
#     e um score agregado:
#         global_prob_risk
#
#
# (c) Scores contínuos normalizados*
#     - risk_age_cont
#     - risk_paw_cont
#
#
#     Baseados em min-max scaling aprendido no treino.
#
#(d) Score contínuo supervisionado
#
#continuous_risk_score =
#     w1 * idade_normalizada
#   + w2 * atividade_normalizada
#   + w3 * histórico_familiar
#
#     onde os pesos são estimados por mínimos quadrados
#     usando apenas o treino.
#
# ---------------------------------------------------------------------------
# 3) Separação por tipo de variável (ColumnTransformer)
# ---------------------------------------------------------------------------
#
# Numéricas:
#     • imputação mediana
#     • conversão float32
#
# Ordinais:
#     • conversão int8
#
# Binárias:
#     • conversão int8
#
# Categóricas nominais:
#     • mantidas como dtype "category"
#       (para uso com XGBoost enable_categorical=True)
#
# ---------------------------------------------------------------------------
# 4) Saída
# ---------------------------------------------------------------------------
# O pipeline final retorna DataFrame tipado e é serializado com joblib como
# artefato reutilizável contendo:
#
#     {
#         'pipeline': objeto sklearn,
#         'metadata': informações de versão e rastreabilidade
#     }
#
# Esse artefato garante:
#     ✔ Reprodutibilidade
#     ✔ Consistência entre treino e inferência
#     ✔ Ausência de data leakage
#
# =============================================================================

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
class RiskFeatureEngineer2(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.age_bins = [18, 29, 44, 59, 69, 79, np.inf]
        self.age_labels = [
            'Jovem Adulto (18–29)',
            'Adulto (30–44)',
            'Meia-idade (45–59)',
            'Idoso Jovem (60–69)',
            'Idoso (70–79)',
            'Idoso Muito Longevo (80+)'
        ]

        self.paw_bins = [0, 29, 149, 299, np.inf]
        self.paw_labels = [
            'Sedentário',
            'Baixo (<150)',
            'Recomendado (150-300)',
            'Alto (>300)'
        ]

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("RiskFeatureEngineer precisa do target.")

        X = X.copy()
        #cria novas features para idade e atividade física
        X['Age_Group'] = pd.cut(X['age'], self.age_bins, labels=self.age_labels)
        X['PAW_Group'] = pd.cut(
            X['physical_activity_minutes_per_week'],
            self.paw_bins,
            labels=self.paw_labels
        )

        tmp = X.copy()
        tmp['target'] = y

        # obtém os valores percentuais do grupo com target
        self.age_prob_ = tmp.groupby('Age_Group', observed=False)['target'].mean().to_dict()
        self.paw_prob_ = tmp.groupby('PAW_Group', observed=False)['target'].mean().to_dict()
        self.fh_prob_ = tmp.groupby('family_history_diabetes')['target'].mean().to_dict()

        self.min_age_ = X['age'].min()
        self.max_age_ = X['age'].max()

        self.paw_min_ = X['physical_activity_minutes_per_week'].min()
        self.paw_max_ = X['physical_activity_minutes_per_week'].max()

        # cria variáveis contínuas
        risk_age_cont = (X['age'] - self.min_age_) / (self.max_age_ - self.min_age_ + 1e-9)

        risk_paw_cont = 1 - (
                (X['physical_activity_minutes_per_week'] - self.paw_min_) /
                (self.paw_max_ - self.paw_min_ + 1e-9)
        )

        fh = X['family_history_diabetes'].astype(float)


        X_mat = np.c_[np.ones(len(X)), risk_age_cont, risk_paw_cont, fh]

        W, _, _, _ = np.linalg.lstsq(X_mat, y, rcond=None)

        # 3. Salva os pesos normalizados (pulando o intercepto W[0])
        self.W_norm_ = W[1:] / W[1:].sum()


        return self

    def transform(self, X):
        X = X.copy()

        # cria novas features para idade e atividade física
        X['Age_Group'] = pd.cut(X['age'], self.age_bins, labels=self.age_labels)
        X['PAW_Group'] = pd.cut(
            X['physical_activity_minutes_per_week'],
            self.paw_bins,
            labels=self.paw_labels
        )

        #cria risco probabilistico baseado no grupo
        X['risk_age_p'] = X['Age_Group'].map(self.age_prob_).astype(float).fillna(0.5)
        X['risk_paw_p'] = X['PAW_Group'].map(self.paw_prob_).astype(float).fillna(0.5)
        X['risk_fh_p'] = X['family_history_diabetes'].map(self.fh_prob_).astype(float).fillna(0.5)

        #cria risco probabilistico global
        X['global_prob_risk'] = (
            X['risk_age_p'] +
            X['risk_paw_p'] +
            X['risk_fh_p']
        ) / 3

        #cria risco probabilistico continuo
        X['risk_age_cont'] = (
            (X['age'] - self.min_age_) /
            (self.max_age_ - self.min_age_ + 1e-9)
        )


        X['risk_paw_cont'] = 1-(
                (X['physical_activity_minutes_per_week']-self.paw_min_) /
                (self.paw_max_ - self.paw_min_ + 1e-9)
        )

        X['continuous_risk_score'] = (
                X['risk_age_cont'] * self.W_norm_[0] +
                X['risk_paw_cont'] * self.W_norm_[1] +
                X['family_history_diabetes'].astype(float) * self.W_norm_[2]
        )

        return X

# ==========================================================
# ORDINAL MAPPER
# ==========================================================
class OrdinalMapper(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.map_edu = {
            'No formal': 0,
            'Highschool': 1,
            'Graduate': 2,
            'Postgraduate': 3
        }

        self.map_income = {
            'Low': 0,
            'Lower-Middle': 1,
            'Middle': 2,
            'Upper-Middle': 3,
            'High': 4
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['education_level'] = X['education_level'].map(self.map_edu)
        X['income_level'] = X['income_level'].map(self.map_income)

        return X


# ==========================================================
# CAST HELPERS
# ==========================================================
def to_category(X): return X.astype("category")
def to_float32(X): return X.astype("float32")
def to_int8(X): return X.astype("int8")


# ==========================================================
# MAIN
# ==========================================================
def main():

    TARGET="diagnosed_diabetes"

    df=pd.read_csv(
        "/home/akel/PycharmProjects/Kaggle/Diabetes_Prediction_Challenge/data/raw/train.csv"
    ).drop(columns='id')

    X=df.drop(columns=TARGET)
    y=df[TARGET]



    X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.3,stratify=y,random_state=42
    )

    # -------- Preview completo --------
    preview_pipe=Pipeline([
        ('ord',OrdinalMapper()),
        ('fe',RiskFeatureEngineer2())
    ])

    X_tmp=preview_pipe.fit_transform(X_train,y_train)

    ordinal_features=['education_level','income_level']

    binary_features=[
        'family_history_diabetes',
        'hypertension_history',
        'cardiovascular_history'
    ]

    cat_features=X_tmp.select_dtypes(
        include=['object','category']
    ).columns

    num_features=(
        X_tmp.select_dtypes(include='number')
        .columns
        .difference(ordinal_features)
        .difference(binary_features)
    )

    preprocess=ColumnTransformer([
        ('num',Pipeline([
            ('imp',SimpleImputer(strategy='median')),
            ('float32',FunctionTransformer(to_float32))
        ]),num_features),

        ('ord',Pipeline([
            ('int8',FunctionTransformer(to_int8))
        ]),ordinal_features),

        ('cat',Pipeline([
            ('cat',FunctionTransformer(to_category))
        ]),cat_features),

        ('bin',Pipeline([
            ('int8',FunctionTransformer(to_int8))
        ]),binary_features)

    ],verbose_feature_names_out=False).set_output(transform="pandas")

    preprocessador_step2=Pipeline([
        ('ordinal_map',OrdinalMapper()),
        ('feature_engineering',RiskFeatureEngineer2()),
        ('preprocess',preprocess)
    ])

    preprocessador_step2.fit(X_train,y_train)

    # -----------------------------
    # Save artifact
    # -----------------------------
    artifact = {
        'preprocessador': preprocessador_step2,
        'metadata': {
            'dataset': 'Health & Diabetes Dataset',
            'version': 'v3a',
            'target': TARGET,
            'created_at': datetime.now().isoformat(),
            'author': 'Alberto Akel'
        }
    }

    joblib.dump(artifact,'preprocess_diabetes_v3a.joblib')

    print("✅ Pipeline salvo com sucesso!")


if __name__=="__main__":
    main()
