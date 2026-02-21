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

    """
    Preprocessador customizado para o dataset Titanic (Kaggle).

    Principais transformações
    --------------------------
    - Imputação de 'Embarked' pela moda
    - Criação de 'HasCabin' e 'Deck' a partir de 'Cabin'
    - Imputação hierárquica de 'Age' baseada em:
      (Sex, Pclass, HasCabin)
    - Criação de 'FamilySize'
    - Extração e agrupamento de títulos ('Title') do nome
    - One-hot encoding com alinhamento de colunas
    """


    def __init__(self):
        self.embarked_mode_ = None
        self.age_medians_ = {}
        self.global_age_median_ = None


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

        if 'Fare' in X.columns:
            self.fare_median_ = X['Fare'].median()  #new

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

        # Novo transformaçao: pega pronome dos nomes (v1.2)    <-- aqui
        X['Title'] = X['Name'].str.split(', ').str[1].str.split('.', n=1).str[0]  # pegando titulos dos nomes
        X['Title'] = X['Title'].replace({"Mlle": "Miss", "Ms": "Miss"})
        X['Title'] = X['Title'].replace("Mme", "Mrs")
        common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
        X['Title'] = X['Title'].apply(lambda x: x if x in common_titles else 'Rare')

        drop_cols = [c for c in ['Name', 'Ticket'] if c in X.columns]
        X.drop(columns=drop_cols, inplace=True)

        # 4. Agora sim captura as colunas do dummy
        X_dummy = pd.get_dummies(X, drop_first=False)
        self.dummy_columns_ = X_dummy.columns

        #paracatboots
        #self.final_columns_ = X.columns  # Apenas para garantir a ordem


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

        if 'Fare' in X.columns:
            X['Fare'] = X['Fare'].fillna(self.fare_median_) #new
        # -----------------------
        # FamilySize
        # -----------------------
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1

        # Novo transformaçao: pega pronome dos nomes (v1.2)  <-- aqui
        X['Title'] = X['Name'].str.split(', ').str[1].str.split('.', n=1).str[0]  # pegando titulos dos nomes
        X['Title'] = X['Title'].replace({"Mlle": "Miss", "Ms": "Miss"})
        X['Title'] = X['Title'].replace("Mme", "Mrs")
        common_titles = ['Mr', 'Miss', 'Mrs', 'Master']
        X['Title'] = X['Title'].apply(lambda x: x if x in common_titles else 'Rare')

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
        #paracatboots
        #X = X.reindex(columns=self.final_columns_)  # Garante ordem das colunas

        return X

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "data","raw")

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
                'agrupamento de titulos dos nomes'
            ),
            'target_transform': 'None (Binary Classification)',
            'fit_on': 'X_train only (30% test split)',
            'created_at': datetime.now().isoformat(),
            'author': 'Alberto Akel',
            'version': 'v1.2'
        }
    }

# # save files
    joblib.dump(artifact, 'preprocess_Titanic_v1.2.joblib')


    X_train.to_csv(DATA_DIR+'/X_train_raw.csv', index=False)
    X_test.to_csv(DATA_DIR+'/X_test_raw.csv', index=False)
    y_train.to_csv(DATA_DIR+'/y_train_raw.csv', index=False)
    y_test.to_csv(DATA_DIR+'/y_test_raw.csv', index=False)
    print("✅ artifact e bases de treino/teste salvos com sucesso!")

if __name__ == "__main__":
    main()


