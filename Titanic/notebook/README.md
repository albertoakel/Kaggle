# 📓 Notebooks — Titanic (GUIA)

Esta pasta reúne os **notebooks principais do projeto Titanic**, organizados por etapa de análise, modelagem e submissão. Cada notebook é relativamente independente, porém todos compartilham um **pré-processamento padronizado**, garantindo consistência, rastreabilidade e reprodutibilidade dos resultados ao longo do projeto.

---

## 🗂 Estrutura dos Notebooks

### 🔍 Análise Exploratória de Dados (EDA)

* [**EDA.ipynb**](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/EDA.ipynb): Análise exploratória completa do dataset Titanic, incluindo avaliação de qualidade dos dados, distribuições univariadas, análises bivariadas, dispersão, correlação e identificação de padrões associados à variável resposta **Survived**.
  Este notebook fundamenta diretamente as decisões de **pré-processamento e engenharia de atributos** utilizadas nos modelos preditivos.

---

### 🌲 Modelos Ensemble — Random Forest

* [**models_randomForest.ipynb**](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/models_randomForest.ipynb):
  Avaliação de modelos baseados em árvores do tipo Random Forest:

  * Random Forest (baseline)
  * Random Forest com hiperparâmetros ajustados

  Ênfase na captura de não linearidades, interações entre variáveis e análise de robustez.

* [**Hiperparameter_search_RF.ipynb**](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/Hiperparameter_search_RF.ipynb):
  Busca sistemática de hiperparâmetros para Random Forest, com validação cruzada e análise comparativa de desempenho.

* **RF_Submission.ipynb**
  Geração do arquivo de submissão Kaggle utilizando o melhor modelo Random Forest selecionado.

---

### 🚀 Gradient Boosting — XGBoost

* [**models_XGBoost.ipynb**](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/models_XGBoost.ipynb):
  Avaliação de modelos XGBoost em diferentes níveis de complexidade:

  * XGBoost (baseline)
  * XGBoost com ajustes intermediários
  * XGBoost otimizado

  Foco em desempenho preditivo e controle de overfitting.

* [**Hiperparameter_search_XGB.ipynb**](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/Hiperparameter_search_XGB.ipynb):
  Busca de hiperparâmetros do XGBoost, explorando regularização, profundidade e taxa de aprendizado.

* **XGB_Submission.ipynb**
  Notebook dedicado à geração da submissão Kaggle com o melhor modelo XGBoost.

---

### 🧠 Gradient Boosting — CatBoost

* [**models_CBTBoost.ipynb**](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/models_CBTBoost.ipynb):
  Modelagem utilizando CatBoost, explorando seu tratamento nativo de variáveis categóricas e estabilidade em datasets tabulares.

* [**Hiperparameter_search_CBT.ipynb**](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/Hiperparameter_search_CBT.ipynb):
  Otimização de hiperparâmetros do CatBoost com validação cruzada.

* **CBT_Submission.ipynb**
  Geração do arquivo de submissão baseado no melhor modelo CatBoost.

---

## ⚙️ Padrões adotados

* Pré-processamento centralizado e reutilizável (via objetos serializados)
* Pipelines integrados (pré-processamento + modelo)
* Avaliação com métricas adequadas à classificação binária (ex.: Accuracy, ROC-AUC, F1-score)
* Validação cruzada para análise de estabilidade e generalização

---

## ✅ Modelos finais e comparações

Os modelos **Random Forest**, **XGBoost** e **CatBoost** são comparados de forma consistente sob o mesmo pipeline de dados. A escolha do modelo final considera:

* Desempenho médio em validação cruzada
* Estabilidade entre folds
* Capacidade de generalização


## 📊 Resultados dos Modelos

| Modelo                            | CV ROC-AUC (±σ) | Test ROC-AUC | Test ACC (0.5) | Test ACC (Opt) | Status |
|-----------------------------------| --------------- | ------ |----------------| --------- | ------ |
| Random Forest (Base)              | 0.8600 ± 0.0501 | 0.8710 | 0.8060         | 0.8134    |        |
| **Random Forest (Random Search)** | **0.8531 ± 0.0643** | **0.8905** | **0.8246**       | **0.8358** | 🏆     |
| Random Forest (Refine)            | 0.8760 ± 0.0516 | 0.8688 | 0.8060         | 0.8284    | 🥈     |
| Random Forest (Bayes)             | 0.8750 ± 0.0554 | 0.8752 | 0.8022         | 0.8172    |        |
| XGBoost (Base)                    | 0.8235 ± 0.0485 | 0.8520 | 0.7836         | 0.8022    |        |
| XGBoost (Random Search)           | 0.8411 ± 0.0558 | 0.8738 | 0.8172         | 0.8209    |        |
| XGBoost (Refine)                  | 0.8491 ± 0.0494 | 0.8749 | 0.8321         | 0.8358    | 🥈     |
| XGBoost (Bayes)                   | 0.8475 ± 0.0408 | 0.8695 | 0.8134         | 0.8209    |        |
| CatBoost (Base)                   | 0.8523 ± 0.0529 | 0.8811 | 0.8172         | 0.8284    | 🥉     |
| CatBoost (Random Search)          | 0.8460 ± 0.0508 | 0.8730 | 0.8134         | 0.8134    |        |
| CatBoost (Refine)                 | 0.8460 ± 0.0466 | 0.8739 | 0.8246         | 0.8246    |        |
| CatBoost (Bayes)                  | 0.8539 ± 0.0535 | 0.8678 | 0.8246         | 0.8284    |        |

--