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

def main():

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")


    os.makedirs(DATA_DIR, exist_ok=True)

    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    TARGET = "diagnosed_diabetes"

    dfo = pd.read_csv("/home/akel/PycharmProjects/Kaggle/Diabetes_Prediction_Challenge/data/raw/train.csv")
    df = dfo.drop(columns='id')

    X = df.drop(columns=TARGET)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y)

    map_edu = {'No formal': 0,'Highschool': 1,'Graduate': 2,'Postgraduate': 3 }
    map_income = {'Low': 0,'Lower-Middle': 1,'Middle': 2,'Upper-Middle': 3,'High': 4}

    for df_ in [X_train, X_test]:
        df_['education_level'] = df_['education_level'].map(map_edu)
        df_['income_level'] = df_['income_level'].map(map_income)


    binary_features = ['family_history_diabetes','hypertension_history','cardiovascular_history']

    # num_cols = (X_train.select_dtypes(include='number').columns.difference(binary_features) )
    # iqr_bounds = compute_iqr_bounds(X_train, num_cols, threshold=10.0)
    # X_train, y_train = filter_outliers_train(X_train, y_train, iqr_bounds)


    ordinal_features = ['education_level', 'income_level']

    num_features = (X_train.select_dtypes(include='number').columns.difference(ordinal_features).difference(binary_features))

    cat_features = X_train.select_dtypes(include='object').columns

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()) ])

    ord_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('scaler', StandardScaler()) ])

    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(
            handle_unknown='ignore',
            sparse_output=False)) ])

    preprocessador = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_features),
            ('ord', ord_pipeline, ordinal_features),
            ('cat', cat_pipeline, cat_features),
            ('bin', 'passthrough', binary_features)  # Mantém 0 e 1 como estão
        ],
        verbose_feature_names_out=False)
    print(X_train.shape)
    preprocessador.fit(X_train)

    artifact = {
        'preprocessador': preprocessador,
        'num_features': list(num_features),
        'ordinal_features': list(ordinal_features),
        'cat_features': list(cat_features),
        'binary_features': list(binary_features),
        'metadata': {
            'dataset': 'Health & Diabetes Dataset',
            'descricao': (
                'Pipeline sem data leakage: '
                'split prévio '
                'ordinais manuais, imputação, '
                'padronização e one-hot encoding.'
            ),
            'target': TARGET,
            'fit_on': 'X_train only',
            'created_at': datetime.now().isoformat(),
            'author': 'Alberto Akel',
            'version': 'v1.0'
        }
    }

    filename='preprocess_diabetes_v0.97.joblib'
    joblib.dump(artifact, filename)


    X_train.to_csv(os.path.join(DATA_DIR, 'X_train_raw.csv'), index=False)
    X_test.to_csv(os.path.join(DATA_DIR, 'X_test_raw.csv'), index=False)
    y_train.to_csv(os.path.join(DATA_DIR, 'y_train_raw.csv'), index=False)
    y_test.to_csv(os.path.join(DATA_DIR, 'y_test_raw.csv'), index=False)

    print(' Joblib '+filename+ ' foi salvo')


if __name__ == "__main__":
    main()







