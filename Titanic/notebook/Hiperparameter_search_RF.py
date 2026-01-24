#!/usr/bin/env python
# coding: utf-8

# ## fluxo de busca de hiperparametros
# 
# 1. Etapa Incial: `RandomizedSearchCV` Explorar região inicial de hiperparametros.
# 
# 2. Refinamento: `RandomizedSearchCV` Explorar região promissora
# 
# 3. Busca final: `BayesSearchCV`Refinamento inteligente aa região promissora

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import time

#Random Forest
from sklearn.ensemble import RandomForestClassifier
# sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin, clone

from scipy.stats import ttest_rel


#hiperparamentros search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from scipy.stats import randint, uniform


# Importações locais
from setup_notebook import setup_path
setup_path()
from src.model_utils import *
from src.preprocess_utils_tic import preprocessador_titanic


# ## 1. Load Data & Pipeline

BASE = Path.cwd().parent

PP2 = joblib.load(BASE/'src'/'preprocess_Titanic_v1.2.joblib')['preprocessador']

DATA_DIR = BASE/"data"/"raw"
X_train = pd.read_csv(DATA_DIR/"X_train_raw.csv")
X_test  = pd.read_csv(DATA_DIR/"X_test_raw.csv")
y_train = pd.read_csv(DATA_DIR/"y_train_raw.csv").values.ravel()
y_test  = pd.read_csv(DATA_DIR/"y_test_raw.csv").values.ravel()


# ## 2. Load Data & Pipeline

# Baseline
model_base = RandomForestClassifier(random_state=42, n_jobs=-1)
pipe_base  = pipe_models(model_base, PP2) # Pipiline base ( PP2 é o preprocessador)

pipe_base.fit(X_train, y_train)
y_probs0 = pipe_base.predict_proba(X_test)[:, 1]
# 2. Testar vários thresholds
thresholds = np.linspace(0.3, 0.7, 41)
best_threshold0 = 0.5
max_acc = 0

for t in thresholds:
    acc = accuracy_score(y_test, y_probs0 > t)
    if acc > max_acc:
        max_acc = acc
        best_threshold0 = t
print(f"{'='*70}")
print(f"🎯 Melhor Threshold: {best_threshold0:.3f}")
print(f"📈 Melhor Acurácia de Teste: {max_acc:.4f}")
print(f"{'='*70}")

baseline_scores = cross_val_score(pipe_base, X_train, y_train, cv=10)
print(f"{'='*70}")
print(f"Baseline: {baseline_scores.mean():.4f} ± {baseline_scores.std():.4f}")
print(f"Average CV Accuracy: {np.mean(baseline_scores)*100:.2f}%") 

mtd_scoring='accuracy'
#accuracy


# ## 3.Buscas por hiperparamentros
# ### 3.1.Random Search (Exploratória)

param_dist_1 = {
    'model__n_estimators': randint(50, 1000),      # Ampliado para capturar mais variações
    'model__max_depth': [None, 3, 5, 8, 10, 15, 20, 30, 50],  # Mais opções
    'model__min_samples_split': randint(2, 30),    # Ampliado
    'model__min_samples_leaf': randint(1, 30),     # Ampliado
    'model__max_features': ['sqrt', 'log2', None, 0.3, 0.5, 0.7],  # Adicionado valores numéricos
    'model__bootstrap': [True, False],
    'model__criterion': ['gini', 'entropy'],       # Adicionado critério
    'model__max_samples': [None, 0.5, 0.7, 0.8, 0.9] if True else [None],  # Se bootstrap=True
    'model__min_weight_fraction_leaf': uniform(0, 0.5),  # Novo parâmetro
    'model__max_leaf_nodes': [None, 50, 100, 500, 1000]  # Controle de crescimento
}

search_1 = RandomizedSearchCV(
    pipe_base, param_dist_1,
    n_iter=50, cv=10,
    scoring=mtd_scoring,
    random_state=42, n_jobs=-1, verbose=1
)

start = time.time()
search_1.fit(X_train, y_train)
end = time.time()

best_1 = search_1.best_estimator_


# 2. Testar vários thresholds
y_probs1 = best_1.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0.3, 0.7, 41)
best_threshold1 = 0.5
max_acc = 0

for t in thresholds:
    acc = accuracy_score(y_test, y_probs1 > t)
    if acc > max_acc:
        max_acc = acc
        best_threshold1 = t
print(f"{'='*70}")
print(f"🎯 Melhor Threshold: {best_threshold1:.3f}")
print(f"📈 Melhor Acurácia de Teste: {max_acc:.4f}")
print(f"{'='*70}")

#ACCURACY
scores1 = cross_val_score(best_1, X_train, y_train, cv=10)
print(f"{'='*70}")
print(f"Optimized: {scores1.mean():.4f} ± {scores1.std():.4f}")
print(f"Average CV Accuracy: {np.mean(scores1)*100:.2f}%")
print(f"Tempo total: {end-start:.2f} segundos")
print(f"Tempo por iteração: {(end-start)/50:.2f} segundos")
print("📌 Melhores Parâmetros:")
print(search_1.best_params_)
print("\n#Processo finalizado em:", time.strftime("%H:%M:%S"))


# ### 3.2 Random Search Refinado

param_dist_2 = {
    'model__n_estimators': randint(200, 350), 
    'model__max_depth': [None,1,4,6,8,9,10,11,12],
    'model__min_samples_split': randint(5,15),
    'model__min_samples_leaf': randint(1, 4),
    'model__max_features': ['sqrt',None],
    'model__criterion': ['gini','entropy'],
    'model__bootstrap': [False,True],
    'model__max_leaf_nodes': [None, 20, 50, 100], 
    'model__min_weight_fraction_leaf': [0.0, 0.1, 0.2] 

}

search_2 = RandomizedSearchCV(
    pipe_base, param_dist_2,
    n_iter=80, cv=10,
    scoring=mtd_scoring,
    random_state=42, n_jobs=-1, verbose=1

)
start = time.time()
search_2.fit(X_train, y_train)
end = time.time()

best_2 = search_2.best_estimator_
# 2. Testar vários thresholds
y_probs2 = best_2.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0.3, 0.7, 41)
best_threshold2 = 0.5
max_acc = 0

for t in thresholds:
    acc = accuracy_score(y_test, y_probs2 > t)
    if acc > max_acc:
        max_acc = acc
        best_threshold2 = t
print(f"{'='*70}")
print(f"🎯 Melhor Threshold: {best_threshold2:.3f}")
print(f"📈 Melhor Acurácia de Teste: {max_acc:.4f}")
print(f"{'='*70}")

#ACCURACY
scores2 = cross_val_score(best_2, X_train, y_train, cv=10)
print(f"{'='*70}")
print(f"Optimized: {scores2.mean():.4f} ± {scores2.std():.4f}")
print(f"Average CV Accuracy: {np.mean(scores2)*100:.2f}%")
print(f"Tempo total: {end-start:.2f} segundos")
print(f"Tempo por iteração: {(end-start)/80:.2f} segundos")
print("📌 Melhores Parâmetros:")
print(search_2.best_params_)
print("\n#Processo finalizado em:", time.strftime("%H:%M:%S"))


# ### 3.3 Bayesian Optimization (Optuna / skopt)

##  Definição do Espaço Bayesiano
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

param_dist_3 = {
    'model__n_estimators': Integer(250, 400),          # 312 ± 50
    'model__max_depth': Integer(8, 15),                # 11 ± 3
    'model__min_samples_split': Integer(8, 16),        # 12 ± 4
    'model__min_samples_leaf': Integer(1, 4),          # 2 ± 2
    'model__max_features': Categorical(['sqrt', None]),
    'model__criterion': Categorical(['gini', 'entropy']),
    'model__bootstrap': Categorical([True]),           # Confirmado como melhor
    'model__max_leaf_nodes': Integer(30, 100),         # 50 ± 20
    'model__min_weight_fraction_leaf': Real(0.0, 0.1)  # 0.0 até 0.1
}

#Configuração do BayesSearch
bayes_search = BayesSearchCV(
    estimator=pipe_base,
    search_spaces=param_dist_3,
    n_iter=60,
    cv=10,
    scoring=mtd_scoring,
    random_state=42,
    n_jobs=-1,
verbose=0)

start = time.time()
## Execução
print("🔍 Iniciando Bayesian Optimization...")
bayes_search.fit(X_train, y_train)
end = time.time()


best_3 = bayes_search.best_estimator_
# 2. Testar vários thresholds
y_probs3 = best_3.predict_proba(X_test)[:, 1]
thresholds = np.linspace(0.3, 0.7, 41)
best_threshold3 = 0.5
max_acc = 0

for t in thresholds:
    acc = accuracy_score(y_test, y_probs3 > t)
    if acc > max_acc:
        max_acc = acc
        best_threshold3 = t
print(f"{'='*70}")
print(f"🎯 Melhor Threshold: {best_threshold3:.3f}")
print(f"📈 Melhor Acurácia de Teste: {max_acc:.4f}")
print(f"{'='*70}")

#ACCURACY
scores3 = cross_val_score(best_3, X_train, y_train, cv=10)
print(f"{'='*70}")
print(f"Optimized: {scores3.mean():.4f} ± {scores3.std():.4f}")
print(f"Average CV Accuracy: {np.mean(scores2)*100:.2f}%")
print(f"Tempo total: {end-start:.2f} segundos")
print(f"Tempo por iteração: {(end-start)/60:.2f} segundos")
print("📌 Melhores Parâmetros:")
print(bayes_search.best_params_)
print("\n#Processo finalizado em:", time.strftime("%H:%M:%S"))


# ## 4. Comparação Estatística

# Calcula os scores de validação cruzada para cada modelo(roc_auc)
s1 = cross_val_score(best_1, X_train, y_train, cv=10,scoring='roc_auc')
s2 = cross_val_score(best_2, X_train, y_train, cv=10,scoring='roc_auc')
s3 = cross_val_score(best_3, X_train, y_train, cv=10,scoring='roc_auc')

# Calcula os scores de validação cruzada para cada modelo(acc)
s1_acc = cross_val_score(best_1, X_train, y_train, cv=10)
s2_acc= cross_val_score(best_2, X_train, y_train, cv=10)
s3_acc = cross_val_score(best_3, X_train, y_train, cv=10)

best_1.fit(X_train, y_train)
best_2.fit(X_train, y_train)
best_3.fit(X_train, y_train)

score1 = best_1.score(X_test, y_test)
score2 = best_2.score(X_test, y_test)
score3 = best_3.score(X_test, y_test)

y_prob1 = best_1.predict_proba(X_test)[:, 1]
y_prob2 = best_2.predict_proba(X_test)[:, 1]
y_prob3 = best_3.predict_proba(X_test)[:, 1]



# 1. Preparação dos Dados de Performance
models_list = [
    ('Modelo 1 (Randon)', best_1, s1, s1_acc, y_prob1, best_threshold1),
    ('Modelo 2 (Refine)', best_2, s2, s2_acc, y_prob2, best_threshold2),
    ('Modelo 3 ( Bayes )',  best_3, s3, s3_acc, y_prob3, best_threshold3)
]

print(f"{'='*80}")
print(f"{'RELATÓRIO DE DESEMPENHO E ESTABILIDADE ESTATÍSTICA':^80}")
print(f"{'='*80}")

# Tabela comparativa de métricas
results_data = []
for name, model, s_roc, s_acc, probs, thresh in models_list:
    test_roc = roc_auc_score(y_test, probs)
    test_acc_std = accuracy_score(y_test, probs > 0.5)
    test_acc_opt = accuracy_score(y_test, probs > thresh)
    
    results_data.append({
        'Modelo': name,
        'CV ROC-AUC': f"{s_roc.mean():.4f} ± {s_roc.std():.2f}",
        'CV ACC': f"{s_acc.mean():.4f} ± {s_acc.std():.2f}",
        'Test ROC-AUC': f"{test_roc:.4f}",
        'Test ACC (0.5)': f"{test_acc_std:.4f}",
        'Best Thresh': f"{thresh:.3f}",
        'Test ACC (Opt)': f"{test_acc_opt:.4f}"
    })

df_results = pd.DataFrame(results_data)
print(df_results.to_string(index=False))

print(f"\n{'='*80}")
print(f"{'ANÁLISE DE SIGNIFICÂNCIA ESTATÍSTICA (T-TEST PAREADO)':^80}")
print(f"{'='*80}")

# Função para imprimir p-value formatado
def check_sig(p): return "SIM" if p < 0.05 else "NÃO"

t12, p12 = ttest_rel(s1, s2)
t23, p23 = ttest_rel(s2, s3)
t13, p13 = ttest_rel(s1, s3)

print(f"M1 vs M2: p-value = {p12:.4f} | Diferença Significativa? {check_sig(p12)}")
print(f"M2 vs M3: p-value = {p23:.4f} | Diferença Significativa? {check_sig(p23)}")
print(f"M1 vs M3: p-value = {p13:.4f} | Diferença Significativa? {check_sig(p13)}")

print(f"\n{'='*80}")
print(f"{'CONCLUSÃO TÉCNICA':^80}")
print(f"{'='*80}")

best_idx = df_results['Test ROC-AUC'].astype(float).idxmax()
vencedor = df_results.iloc[best_idx]['Modelo']

print(f"1. O modelo vencedor em generalização (Test ROC-AUC) é o: {vencedor}")
print("\n#Processo finalizado em:", time.strftime("%H:%M:%S"))


# ## 7. Salvando_hiperparametros

# Salvar Hiperparametros joblib
# apenas parametros
#joblib.dump(bayes_search.best_params_, 'parametros_RF_BAYER_v12.joblib') 
# modelo completo

mtd_scoring
joblib.dump(search_1.best_estimator_, 'modelo_RF_final_randsearch.'+mtd_scoring+'_v12.joblib')
joblib.dump(search_2.best_estimator_, 'modelo_RF_final_refine.'+mtd_scoring+'_v12.joblib')
joblib.dump(bayes_search.best_estimator_, 'modelo_RF_final.'+mtd_scoring+'_bayes_v12.joblib')

