import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning,
                       message='Found unknown categories in columns')
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline



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
def valida(Xo, yo, model, N=5):
    print("=" * 44)
    print(f"validação cruzada (K-Fold Cross Validation)")
    print("=" * 44)

    kf = KFold(n_splits=N, shuffle=True, random_state=42)
    r2_scores = []
    for i, (train_idx, test_idx) in enumerate(kf.split(Xo, yo), 1):
        X_train, X_val = Xo.iloc[train_idx], Xo.iloc[test_idx]
        y_train, y_val = yo.iloc[train_idx], yo.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        r2 = r2_score(y_val, y_pred)
        r2_scores.append(r2)
        print(f"Fold {i}: R² = {r2:.4f}")

    print(f"\n📊 R² médio: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
    return r2_scores


def metricas_model(y_test, y_pred, nome_modelo='Modelo'):
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
    print('=' * 44)
    print(f'🤖 {nome_modelo.upper()}')
    print('=' * 44)
    print(f"MAE:  {resultados['MAE']:>7}")
    print(f"RMSE: {resultados['RMSE']:>6}")
    print(f"R²:   {resultados['R²']:>6}")

    return resultados

def pipe_models(modelo, preprocessador):
    return Pipeline([('preprocess', preprocessador),
                     ('model', modelo)])
