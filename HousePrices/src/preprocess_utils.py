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

'''
Modulo responsavel pela criação artefato(joblib) pre-processado e separação das bases de treino/teste
'''

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "data","raw")

    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    TARGET = 'SalePrice'
    DROP_THRESHOLD = 0.1

    #1. Leitura dos dados
    dfo = pd.read_csv("/home/akel/PycharmProjects/Kaggle/HousePrices/data/raw/train.csv")

# ✂️ 2. Split ANTES de qualquer decisão estatística
    X = dfo.drop(columns=['Id', TARGET])
    y_log = np.log1p(dfo[TARGET])

    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE )

#  🧹 3. Remoção de colunas com muitos nulos
    colnull_train = X_train.columns[(X_train.isnull().sum() / X_train.shape[0] > DROP_THRESHOLD)]

    X_train = X_train.drop(columns=colnull_train)
    X_test  = X_test.drop(columns=colnull_train)

# Identificação das features
    num_features = X_train.select_dtypes(include=['number']).columns
    cat_features = X_train.select_dtypes(include=['object']).columns

#🧩 3. Pré-processadores
# 1.NAN -> median
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

# 2.categoric -> binario onehotcode
    cat_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first',
                             sparse_output=False,
                             handle_unknown='ignore'))])

    preprocessador = ColumnTransformer(transformers=[
        ('cat', cat_transformer, cat_features),
        ('num', num_transformer, num_features)
    ],verbose_feature_names_out=False)

#se for usar dentro de uma pipeline, comentar abaixo.
    preprocessador.fit(X_train) #descomentar para salvar artifact

    artifact = {
        'preprocessador': preprocessador,
        'colnull_train': colnull_train,
        'num_features': num_features,
        'cat_features': cat_features,
        'metadata': {
            'dataset': 'Kaggle House Prices )',
            'descricao': (
            'Preprocessador com remoção de colunas >10% nulos, '
            'imputação por mediana, padronização e one-hot encoding.' ),
            'target_transform': 'log1p(SalePrice)',
            'fit_on': 'X_train only',
            'created_at': datetime.now().isoformat(),
            'author': 'Alberto Akel',
            'version': 'v1.0'}}

# save files
    joblib.dump(artifact, 'preprocess_HP_v1.joblib')

    X_train.to_csv(DATA_DIR+'/X_train_raw.csv', index=False)
    X_test.to_csv(DATA_DIR+'/X_test_raw.csv', index=False)
    y_train.to_csv(DATA_DIR+'/y_train_raw.csv', index=False)
    y_test.to_csv(DATA_DIR+'/y_test_raw.csv', index=False)
    print("✅ artifact e bases de treino/teste salvos com sucesso!")

if __name__ == "__main__":
    main()


