# =====================================================
# 🏠 House Prices - XGBoost
# =====================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning,
                       message='Found unknown categories in columns')
import time
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)



# Scikit-learn - Pré-processamento e pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =====================================================
# 📁 1. Leitura dos dados
# =====================================================
dfo = pd.read_csv("/home/akel/PycharmProjects/Kaggle/HousePrices/data/train.csv")

df_train=dfo.copy()
# =====================================================
# 🧹 2. Pré-processamento inicial
# =====================================================
# remoção de colunas com muitos nulos (> 10%)
colnull_train=df_train.columns[(df_train.isnull().sum()/df_train.shape[0]>0.1)] # 
df_train=df_train.drop(columns=colnull_train,axis=1)

id_train=df_train['Id']

# obtendo nome das variáveis categóricas e numéricas
num_features = df_train.select_dtypes(include=['number']).columns.drop(['Id', 'SalePrice'])
cat_features = df_train.select_dtypes(include=['object']).columns

# =====================================================
# 🧩 3. Pré-processadores
# =====================================================
# NAN -> median
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# categoric -> binario onehotcode 
cat_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(drop='first',
                             sparse_output=False,
                             handle_unknown='ignore'))
])

preprocessador = ColumnTransformer(transformers=[
    ('cat', cat_transformer, cat_features),
    ('num', num_transformer, num_features)   
],verbose_feature_names_out=False) 
