# 📓 Notebooks — House Prices (GUIA)

Esta pasta reúne os **notebooks principais do projeto House Prices**, organizados por etapa de análise e modelagem. Cada notebook é independente, mas todos compartilham o mesmo **pré-processamento padronizado** para garantir consistência e reprodutibilidade.

---

## 🗂 Estrutura dos Notebooks

### 🔍 Análise Exploratória de Dados (EDA)

* **EDA.ipynb**
  Análise estatística e visual das variáveis, identificação de outliers, distribuições, correlações e insights iniciais para engenharia de atributos.

---

### 📐 Modelos Lineares

* [**models_Linear.ipynb**](https://github.com/albertoakel/Kaggle/blob/master/HousePrices/notebook/02_Linear_model_compare.ipynb)
  Avaliação de modelos lineares como *baseline*:

  * Linear Regression
  * Ridge (default e ajustado)
  * Lasso (default e ajustado)


---

### 🌲 Modelos Ensemble — Random Forest

* [**models_Random_Forest.ipynb**](https://github.com/albertoakel/Kaggle/blob/master/HousePrices/notebook/02_RF_model_compare.ipynb)
  Modelos baseados em árvores para capturar não linearidades e interações:

  * Random Forest (configuração padrão)
  * Random Forest (hiperparâmetros ajustados)


---

### 🚀 Gradient Boosting — XGBoost

* [**models_XGBoost.ipynb**](https://github.com/albertoakel/Kaggle/blob/master/HousePrices/notebook/02_XGB_model_compare.ipynb)
  Modelos de *boosting* sequencial com regularização explícita:

  * XGBoost (baseline)
  * XGBoost (configuração intermediária)
  * XGBoost (configuração otimizada)


---

## ⚙️ Padrões adotados

* Pré-processamento carregado via `preprocessador_v0.joblib`
* Pipelines integrados (dados + modelo)
* Avaliação com MAE, RMSE e R²
* Validação cruzada (K-Fold) para análise de estabilidade

---

## ✅ Modelo final recomendado

➡️ **XGBoost (configuração otimizada)**
Melhor compromisso entre desempenho, estabilidade e generalização.

---
