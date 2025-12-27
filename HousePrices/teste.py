# =====================================================
# 🏠 House Prices - Modelos Lineares
# =====================================================
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning,
                       message='Found unknown categories in columns')
import time
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

# Scikit-learn - Model selection e avaliação
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV

# Scikit-learn - Pré-processamento e pipelines
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Scikit-learn - Modelos lineares
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV

# Scikit-learn - Métricas de avaliação
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Distribuições para busca de hiperparâmetros
from scipy.stats import randint, uniform,loguniform

from setup_notebook import setup_path
setup_path()
from src.model_utils import *
from joblib import dump,load
import joblib


# =====================================================
# ⚙️ 0. carregamento dos preprocessador
# =====================================================
temp = joblib.load('preprocess_house_prices_v1.joblib')
preprocessador=temp['preprocessador']
colnull_train = temp['colnull_train']

# =====================================================
# 📁 1. Leitura dos dados & Separação das bases
# =====================================================
X_train=pd.read_csv("/home/akel/PycharmProjects/Kaggle/HousePrices/data/processed/X_train_final.csv")
X_test=pd.read_csv("/home/akel/PycharmProjects/Kaggle/HousePrices/data/processed/X_test_final.csv")
y_train=pd.read_csv("/home/akel/PycharmProjects/Kaggle/HousePrices/data/processed/y_train_final.csv")
y_test=pd.read_csv("/home/akel/PycharmProjects/Kaggle/HousePrices/data/processed/y_test_final.csv")

# =====================================================
#  🤖 3.Definição dos Modelos
# =====================================================
model_lr = LinearRegression()
LR       = pipe_models(model_lr, preprocessador)

model_rd0 = Ridge(alpha=0.01)
RD0       = pipe_models(model_rd0, preprocessador)

model_rd1 = Ridge(alpha=19.30, max_iter=1000, random_state=42, solver='sag',
                 tol=0.001)
RD1       = pipe_models(model_rd1, preprocessador)

model_ls0 = Lasso(alpha=0.01,)
LS0       = pipe_models(model_ls0, preprocessador)

model_ls1 = Lasso(alpha=0.002043, max_iter=2000, tol=0.001)
LS1       = pipe_models(model_ls1, preprocessador)

# 1. Valida a estabilidade
r20 = valida(X_train, y_train, model=LR, N=7,write='on')

# 2. Testa a performance
LR.fit(X_train, y_train)
y_pred0 = LR.predict(X_test)
res0 = metricas_model(y_test, y_pred0, 'Linear Regression',write='on')

# 1. Valida a estabilidade
r21=valida(X_train,y_train,model=RD0,N=7)

# 2. Testa a performance
RD0.fit(X_train, y_train)
y_pred1 = RD0.predict(X_test)
res1=metricas_model(y_test, y_pred1, 'Ridge 0')
# 1. Valida a estabilidade
r22=valida(X_train,y_train,model=RD1,N=7)

# 2. Testa a performance
RD1.fit(X_train, y_train)
y_pred2 = RD1.predict(X_test)
res2=metricas_model(y_test, y_pred2, 'Ridge 1')

# 1. Valida a estabilidade
r23=valida(X_train,y_train,model=LS0,N=7)

# 2. Testa a performance
LS0.fit(X_train, y_train)
y_pred3 = LS0.predict(X_test)
res3=metricas_model(y_test, y_pred3, 'LASSO 0')

# 1. Valida a estabilidade
r24=valida(X_train,y_train,model=LS1,N=7)

# 2. Testa a performance
LS1.fit(X_train, y_train)
y_pred4 = LS1.predict(X_test)
res4=metricas_model(y_test, y_pred4, 'LASSO 1')


import matplotlib.pyplot as plt

# =====================================================
# 1. Organizar resultados em DataFrame
# =====================================================

resultados_teste = pd.DataFrame([
    res0, res1,res2,res3,res4])

resultados_cv = pd.DataFrame({
    "Modelo": ["r20", "r21","r22","r23","r24"],
    "R2_Médio": [np.mean(r20), np.mean(r21),np.mean(r22),np.mean(r23),np.mean(r24)],
    "R2_Std":   [np.std(r20),np.std(r21),np.std(r22),np.std(r23),  np.std(r24)]
})

display(resultados_teste)
display(resultados_cv)

# =====================================================
# 2. Gráfico — R² no conjunto de teste
# =====================================================

plt.figure(figsize=(7,4))
plt.bar(resultados_teste['Modelo'], resultados_teste['R²'])
plt.title("R² dos Modelos XGBoost (Teste)")
plt.ylabel("R²")
plt.ylim(0.7, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()

# =====================================================
# 3. Gráfico — R² médio da validação cruzada
# =====================================================

plt.figure(figsize=(7,4))
plt.bar(resultados_cv['Modelo'], resultados_cv['R2_Médio'], yerr=resultados_cv['R2_Std'])
plt.title("R² Médio (K-Fold) — XGB0 vs XGB1 vs XGB2")
plt.ylabel("R² Médio")
plt.ylim(0.7, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()

# =====================================================
# 4. Gráfico de Dispersão — y_test vs y_pred para cada modelo
# =====================================================
a=LR.predict(X_test)
predicoes = {
    "LR":  y_pred0,
    "RD0": y_pred2,
    "RD1":y_pred3,
    "LS0": y_pred3,
    "LS1": y_pred4
}

plt.figure(figsize=(15,4))



for i, (nome, y_p) in enumerate(predicoes.items(), 1):
    plt.subplot(1, 5, i)
    plt.scatter(y_test, y_p, alpha=0.4)
    plt.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()],
             'r--', linewidth=2)
    plt.ylim([9.5,14])
    plt.title(f"{nome} — Teste")
    plt.xlabel("y_test (Real)")
    plt.ylabel("y_pred (Modelo)")

plt.tight_layout()
plt.show()