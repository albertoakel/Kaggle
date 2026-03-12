#model_utils.py
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning,
                       message='Found unknown categories in columns')
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV,StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score,
                             average_precision_score,roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.base import clone



def run_model(Xo, yo, model, nome='sem nome', p=True):
    X_train, X_test, y_train, y_test = train_test_split(Xo, yo, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    resultados = {
        'Modelo': '🤖 ' + nome,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R²': round(r2, 4)}

    if p != False:
        for k, v in resultados.items():
            print(f"{k}: {v}")
    return r2


# função validacao_cruzada_parapipeline
def valida(Xo, yo, model, N=5, write=None):
    if write==None:
        print('Validação cruzada realizada!')
    kf = KFold(n_splits=N, shuffle=True, random_state=42)
    r2_scores = []

    for i, (train_idx, test_idx) in enumerate(kf.split(Xo, yo), 1):
        X_train, X_val = Xo.iloc[train_idx], Xo.iloc[test_idx]
        y_train, y_val = yo.iloc[train_idx], yo.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        r2_scores.append(r2)

    if write == 'on':
        print("=" * 44)
        print(f"validação cruzada (K-Fold Cross Validation)")
        print("=" * 44)

        for i, r2 in enumerate(r2_scores, 1):
            print(f"Fold {i}: R² = {r2:.4f}")

        print(f"\n📊 R² médio: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

    return r2_scores


def metricas_model(y_test, y_pred, nome_modelo='Modelo',write=None):
    """
    print_metricas
    """

    # Calcula as métricas
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Cria dicionário com resultados
    resultados = {
        'Modelo': nome_modelo,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R²': round(r2, 4)
    }

    # Imprime resultados com alinhamento
    if write=='on':
        print('=' * 44)
        print(f'🤖 {nome_modelo.upper()}')
        print('=' * 44)
        print(f"MAE:  {resultados['MAE']:>7}")
        print(f"RMSE: {resultados['RMSE']:>6}")
        print(f"R²:   {resultados['R²']:>6}")

    return resultados

def pipe_models(modelo, preprocessador):
    return Pipeline([('preprocess', clone(preprocessador)),
                     ('model', modelo)])



def best_threshold(model, X_test, y_test, start=0.3, stop=0.7, steps=41,print_results=False):
    """
    Encontra o threshold que maximiza a acurácia para um modelo de classificação.
    """
    # 1. Obtém as probabilidades da classe positiva
    y_probs = model.predict_proba(X_test)[:, 1]

    # 2. Define o range de busca
    thresholds = np.linspace(start, stop, steps)
    best_threshold = 0.5
    max_acc = 0

    # 3. Itera sobre os thresholds
    for t in thresholds:
        acc = accuracy_score(y_test, y_probs > t)
        if acc > max_acc:
            max_acc = acc
            best_threshold = t

    if print_results == True:
        print(f"{'=' * 40}")
        print(f"Melhor Threshold: {best_threshold:.3f}")
        print(f"Melhor Acurácia (Test): {max_acc:.4f}")
        print(f"{'=' * 40}")

    return best_threshold, max_acc

def best_threshold2(model, X_train, y_train, X_test, y_test, start=0.3, stop=0.7, steps=41, print_results=False):
    """
    Encontra o threshold que maximiza a acurácia para um modelo de classificação.(versão corrigida best_threshold)
    """
    # 1. Obtém as probabilidades da classe positiva
    # probabilidades do treino

    prob_train = model.predict_proba(X_train)[:, 1]

    # 2. Define o range de busca
    thresholds = np.linspace(start, stop, steps)
    best_threshold = 0.5
    max_acc0 = 0

    # 3. Itera sobre os thresholds( melhor threshold usando apenas o TREINO)
    for t in thresholds:
        acc = accuracy_score(y_train, prob_train > t)
        if acc > max_acc0:
            max_acc0 = acc
            best_threshold = t

    # probabilidades do TESTE
    prob_test = model.predict_proba(X_test)[:, 1]
    # Calculamos a acurácia aplicando o threshold do treino nas probabilidades do teste
    max_acc = accuracy_score(y_test, prob_test > best_threshold)

    if print_results == True:
        print(f"{'=' * 40}")
        print(f"Melhor Threshold: {best_threshold:.3f}")
        print(f"Melhor Acurácia (Test): {max_acc:.4f}")
        print(f"{'=' * 40}")

    return best_threshold, max_acc, prob_test

def evaluate_model(pipe,X_train,y_train,X_val,y_val,modelname='Modelo_sem_nome',pipe_fit=True):

    # 1)FIT
    if pipe_fit==True:
        pipe.fit(X_train, y_train)

    # 2)Pred
    y_pred=pipe.predict(X_val)

    # 3) Otimização do threshold de decisão
    best_th,max_acc,y_probs=best_threshold2(pipe, X_train, y_train,X_val,y_val)

    # 4) Validação cruzada
    mtd_scoring='roc_auc'
    cv_s = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipe, X_train, y_train,
                                 cv=cv_s,
                                 scoring=mtd_scoring,
                                 n_jobs=-1)

    print(f"\n{'='*70}")
    print(f" 📍 RESULTADOS {modelname.upper()}".center(70))
    print(f"{'='*70}")
    # =====================================================
    # 5) Avaliação por validação cruzada (Treino)
    # =====================================================

    print("📊 CROSS-VALIDATION")
    print(f"   Média {mtd_scoring}:       {cv_scores.mean():>15.4f} ± {cv_scores.std():.4f}")

    # =====================================================
    # 6) Avaliação no conjunto de teste
    # =====================================================
    print(f"\n✅ TEST SET")
    print(f"   Padrão (0.5):              {accuracy_score(y_val, y_pred):>10.4f}")
    print(f"   Otimizado:                 {max_acc:>10.4f} (threshold ={best_th:>6.3f})")
    print(f"   ROC-AUC:                   {roc_auc_score(y_val, y_probs):>10.4f}")
    print(f"   Avg precision:             {average_precision_score(y_val, y_probs):>10.4f}")
    resultados = {}
    resultados[modelname] = {
        'cv_scores_stat':[cv_scores.mean(),cv_scores.std()],
        'acc_score':accuracy_score(y_val, y_pred),
        'max_acc':max_acc,
        'best_t':best_th,
        'cv_scores':cv_scores,
        'roc_auc_score':roc_auc_score(y_val, y_probs),
        'avg_p_score': average_precision_score(y_val, y_probs),
        'y_probs':y_probs,
        'y_pred':y_pred}
    return pd.DataFrame(resultados).T



def print_hiper(search_best_params):
    print("🎯 Melhores Hiperparâmetros")
    print("-" * 50)
    for param, value in search_best_params.items():
        param_name = param.replace('model__', '').replace('_', ' ').title()
        print(f"• {param_name:<25} : {value}")
    print("-" * 50)