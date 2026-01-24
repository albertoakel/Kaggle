#!/usr/bin/env python
# coding: utf-8

# ## 🛳️ Titanic - Random Forest
# Os modelos Random Forest Classifier são utilizados para capturar relações não lineares e interações complexas entre as variáveis. Por serem modelos baseados em árvores não assumem linearidade e são robustos a outliers;
# 
# ### Destaques do Notebook  
# 
# * **Pré-processamento inicial:**.
# 
# * **Transformação das variáveis:** a
# 
# * **Treinamento do modelo:** construção de um pipeline com o RandomForest,sem ajustes hiperparâmetros e com ajuste. 

# ## 1. Bibliotecas

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
from pathlib import Path
import time

from scipy.stats import ttest_rel

# sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score

from sklearn.base import BaseEstimator, TransformerMixin, clone

# Importações locais
from setup_notebook import setup_path
setup_path()
from src.model_utils import *
from src.preprocess_utils_tic import preprocessador_titanic
from src.plot_metrica_class import *


# ## 2. Dataload & Preprocessamento com joblib

BASE = Path.cwd().parent   
# =====================================================
# ⚙️ 0. carregamento dos preprocessador 
# =====================================================
temp = joblib.load(BASE /'src'/'preprocess_Titanic_v1.2.joblib')
PP2=temp['preprocessador']

# # =====================================================
# # 📁 1. Leitura dos dados & Separação das bases
# # =====================================================

DATA_DIR = BASE / "data" / "raw"
X_train = pd.read_csv(DATA_DIR / "X_train_raw.csv").reset_index(drop=True)
X_test  = pd.read_csv(DATA_DIR / "X_test_raw.csv")
y_train = pd.read_csv(DATA_DIR / "y_train_raw.csv").values.ravel()
y_test  = pd.read_csv(DATA_DIR / "y_test_raw.csv")


# # =====================================================
# #  🤖 3.Definição dos Modelos
# # =====================================================
model_RF0 = RandomForestClassifier(random_state=42, n_jobs=-1)  #Baseline
pipe_RF0      = pipe_models(model_RF0,PP2)


DATA_MODELS= BASE /"models"
pipe_RF1 = joblib.load(DATA_MODELS / 'modelo_RF_final_randsearch.roc_auc_v12.joblib')
pipe_RF2 = joblib.load(DATA_MODELS / 'modelo_RF_final_refine.roc_auc_v12.joblib')
pipe_RF3 = joblib.load(DATA_MODELS / 'modelo_RF_final_bayes.roc_auc_v12.joblib')


# ## 3.Treinamento
# ### Baseline

# Baseline
s0 = cross_val_score(pipe_RF0 , X_train, y_train, cv=10,scoring='roc_auc')

# 2. Testa a performance 
pipe_RF0.fit(X_train, y_train)
# 3.Otimização de Threshold
best_t_rf0, score_rf0 = best_threshold(pipe_RF0, X_test, y_test)

y_pred=pipe_RF0.predict(X_test)
print(f"{'='*70}")
print(f"🎯 Random Forest (Baseline) | cvscores : {s0.mean():.4f} ± {s0.std():.4f}")
print(f"{'='*70}")
print(f"📊 **Acurácia no Teste**: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n📋 **Relatório de Classificação**:")
print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test, y_pred)
print(f"🎯 **Matriz de Confusão**:")
print(f"               Previsto 0   Previsto 1")
print(f"Real 0         {cm[0,0]:<11} {cm[0,1]:<11}")
print(f"Real 1         {cm[1,0]:<11} {cm[1,1]:<11}")
print(f"{'─'*70}")


# RF1
s1= cross_val_score(pipe_RF1 , X_train, y_train, cv=10,scoring='roc_auc')

# 2. Testa a performance 
pipe_RF1.fit(X_train, y_train)

# 3.Otimização de Threshold
best_t_rf1, score_rf1 = best_threshold(pipe_RF1, X_test, y_test)

y_pred=pipe_RF1.predict(X_test)
print(f"{'='*70}")
print(f"🎯 Random Forest 1 | cvscores : {s1.mean():.4f} ± {s1.std():.4f}")
print(f"{'='*70}")
print(f"📊 **Acurácia no Teste**: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n📋 **Relatório de Classificação**:")
print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test, y_pred)
print(f"🎯 **Matriz de Confusão**:")
print(f"               Previsto 0   Previsto 1")
print(f"Real 0         {cm[0,0]:<11} {cm[0,1]:<11}")
print(f"Real 1         {cm[1,0]:<11} {cm[1,1]:<11}")
print(f"{'─'*70}")


#RF 2
s2= cross_val_score(pipe_RF2 , X_train, y_train, cv=10,scoring='roc_auc')

# 2. Testa a performance 
pipe_RF2.fit(X_train, y_train)

# 3.Otimização de Threshold
best_t_rf2, score_rf2 = best_threshold(pipe_RF2, X_test, y_test)

y_pred=pipe_RF2.predict(X_test)
print(f"{'='*70}")
print(f"🎯 Random Forest 2 | cvscores : {s2.mean():.4f} ± {s2.std():.4f}")
print(f"{'='*70}")
print(f"📊 **Acurácia no Teste**: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n📋 **Relatório de Classificação**:")
print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test, y_pred)
print(f"🎯 **Matriz de Confusão**:")
print(f"               Previsto 0   Previsto 1")
print(f"Real 0         {cm[0,0]:<11} {cm[0,1]:<11}")
print(f"Real 1         {cm[1,0]:<11} {cm[1,1]:<11}")
print(f"{'─'*70}")


#RF 3
s3= cross_val_score(pipe_RF3 , X_train, y_train, cv=10,scoring='roc_auc')

# 2. Testa a performance 
pipe_RF3.fit(X_train, y_train)

# 3.Otimização de Threshold
best_t_rf3, score_rf3 = best_threshold(pipe_RF3, X_test, y_test)

y_pred=pipe_RF3.predict(X_test)
print(f"{'='*70}")
print(f"🎯 Random Forest 3 | cvscores : {s3.mean():.4f} ± {s3.std():.4f}")
print(f"{'='*70}")
print(f"📊 **Acurácia no Teste**: {accuracy_score(y_test, y_pred):.4f}")
print(f"\n📋 **Relatório de Classificação**:")
print(classification_report(y_test, y_pred))
cm=confusion_matrix(y_test, y_pred)
print(f"🎯 **Matriz de Confusão**:")
print(f"               Previsto 0   Previsto 1")
print(f"Real 0         {cm[0,0]:<11} {cm[0,1]:<11}")
print(f"Real 1         {cm[1,0]:<11} {cm[1,1]:<11}")
print(f"{'─'*70}")


# relatório automatico
# Calcula os scores de validação cruzada para cada modelo(acc)
s0_acc = cross_val_score(pipe_RF0, X_train, y_train, cv=10)
s1_acc = cross_val_score(pipe_RF1, X_train, y_train, cv=10)
s2_acc = cross_val_score(pipe_RF2, X_train, y_train, cv=10)
s3_acc = cross_val_score(pipe_RF3, X_train, y_train, cv=10)

score0 = pipe_RF0.score(X_test, y_test)
score1 = pipe_RF1.score(X_test, y_test)
score2 = pipe_RF2.score(X_test, y_test)
score3 = pipe_RF3.score(X_test, y_test)

y_prob0 = pipe_RF0.predict_proba(X_test)[:, 1]
y_prob1 = pipe_RF1.predict_proba(X_test)[:, 1]
y_prob2 = pipe_RF2.predict_proba(X_test)[:, 1]
y_prob3 = pipe_RF3.predict_proba(X_test)[:, 1]


# # 1. Preparação dos Dados de Performance
models_list = [
    ('Modelo 0 ( Base )', pipe_RF0, s0, s0_acc, y_prob0, best_t_rf0),
    ('Modelo 1 (Random)', pipe_RF1, s1, s1_acc, y_prob1, best_t_rf1),
    ('Modelo 2 (Refine)', pipe_RF2, s2, s2_acc, y_prob2, best_t_rf2),
    ('Modelo 3 (Bayes )', pipe_RF3, s3, s3_acc, y_prob3, best_t_rf3,)
]

df_results,W = gerar_relatorio_estatistico(models_list,X_train, y_train,X_test, y_test)


#plotagem dos resultados
model_evaluation_grid(
    models_list=models_list,
    X_test=X_test,
    y_test=y_test,
    best_model_pipeline=W[1],
    best_model_name=W[0]
)

