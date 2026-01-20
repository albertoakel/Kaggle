#preprocess_utils_tic.py
import os
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
# Scikit-learn - Model selection e avaliação
from sklearn.model_selection import train_test_split
# Scikit-learn - Pré-processamento e pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class preprocessador_titanic(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.embarked_mode_ = None
        self.age_medians_ = {}
        self.global_age_median_ = None

    """
    FIT       : Load self variables
    TRANSFORM : Unlade self variables
    """

    def fit(self, X, y=None):
        X = X.copy()

        if 'Embarked' in X.columns:
            # Garante que pegamos o valor (string) e não a Series
            mode_series = X['Embarked'].mode()
            self.embarked_mode_ = mode_series[0] if not mode_series.empty else 'S'

        # 1. Criar HasCabin logo no início para consistência
        if 'Cabin' in X.columns:
            X['HasCabin'] = X['Cabin'].notnull().astype(int)

        # 2. Cálculos de média (sua lógica está correta)
        if 'Age' in X.columns:
            self.global_age_median_ = X['Age'].median()
            group_cols = ['Sex', 'Pclass', 'HasCabin']
            for i in range(len(group_cols)):
                cols = group_cols[:len(group_cols) - i]
                self.age_medians_[tuple(cols)] = X.groupby(cols)['Age'].median()

        # 3. Aplicar TODAS as transformações que o transform faria
        # Isso garante que dummy_columns_ aprenda a estrutura final real
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1

        if 'Cabin' in X.columns:
            X['Deck'] = X['Cabin'].apply(
                lambda x: 'U' if pd.isnull(x) or str(x)[0] == 'T' else str(x)[0]
            )
            X.drop(columns='Cabin', inplace=True)

        if 'Age' in X.columns:
            X['Age2'] = X['Age']
            X.drop(columns='Age', inplace=True)

        drop_cols = [c for c in ['Name', 'Ticket'] if c in X.columns]
        X.drop(columns=drop_cols, inplace=True)

        # 4. Agora sim captura as colunas do dummy
        X_dummy = pd.get_dummies(X, drop_first=False)
        self.dummy_columns_ = X_dummy.columns

        return self

    # =========================
    # TRANSFORM
    # =========================
    def transform(self, X):
        X = X.copy()

        # -----------------------
        # Cabin → HasCabin + Deck
        # -----------------------
        if 'Cabin' in X.columns:
            X['HasCabin'] = X['Cabin'].notnull().astype(int)
            X['Deck'] = X['Cabin'].apply(
                lambda x: 'U' if pd.isnull(x) or str(x)[0] == 'T' else str(x)[0]
            )
            X.drop(columns='Cabin', inplace=True)

        # -----------------------
        # Embarked
        # -----------------------
        if 'Embarked' in X.columns:
            X['Embarked'] = X['Embarked'].fillna(self.embarked_mode_)

        # -----------------------
        # Age → Age2 (hierarchical imputation)
        # -----------------------
        if 'Age' in X.columns:
            X['Age2'] = X['Age']

            for cols, medians in self.age_medians_.items():
                keys = X[list(cols)].apply(tuple, axis=1)

                X['Age2'] = X['Age2'].fillna(keys.map(medians))

            X['Age2'] = X['Age2'].fillna(self.global_age_median_)
            X.drop(columns='Age', inplace=True)

        # -----------------------
        # FamilySize
        # -----------------------
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1

        # -----------------------
        # Drop columns
        # -----------------------
        drop_cols = [c for c in ['Name', 'Ticket'] if c in X.columns]
        X.drop(columns=drop_cols, inplace=True)

        # -----------------------
        # One-hot encoding
        # -----------------------
        X = pd.get_dummies(X, drop_first=False)
        X = X.reindex(columns=self.dummy_columns_, fill_value=0)

        return X

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "data","processed")

    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    TARGET = 'Survived'

#   1. Leitura dos dados
    dfo = pd.read_csv("/home/akel/PycharmProjects/Kaggle/Titanic/data/raw/train.csv").drop(columns='PassengerId')

#   2. Split antes de qualquer decisão estatística
    X = dfo.drop(columns=TARGET)
    y = dfo[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE )

#
# #🧩 3. Pré-processadores
    PP = preprocessador_titanic()
    PP.fit(X_train)

    artifact = {'preprocessador': PP}

    artifact = {
        'preprocessador': PP,
        'metadata': {
            'dataset': 'Kaggle Titanic Survival',
            'descricao': (
                'Preprocessador customizado: Engenharia de Deck (Cabin), '
                'Criação de FamilySize, HasCabin, e Imputação Hierárquica de Idade '
                'baseada em Sexo, Pclass e HasCabin.'
            ),
            'target_transform': 'None (Binary Classification)',
            'fit_on': 'X_train only (30% test split)',
            'created_at': datetime.now().isoformat(),
            'author': 'Alberto Akel',
            'version': 'v1.0'
        }
    }

# # save files
    joblib.dump(artifact, 'preprocess_Titanic_v1.0.joblib')


    X_train.to_csv(DATA_DIR+'/X_train_final.csv', index=False)
    X_test.to_csv(DATA_DIR+'/X_test_final.csv', index=False)
    y_train.to_csv(DATA_DIR+'/y_train_final.csv', index=False)
    y_test.to_csv(DATA_DIR+'/y_test_final.csv', index=False)
    print("✅ artifact e bases de treino/teste salvos com sucesso!")

if __name__ == "__main__":
    main()


