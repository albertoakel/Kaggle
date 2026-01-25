# 📓 Notebooks — Titanic (GUIA)

Esta pasta reúne os **notebooks principais do projeto Titanic**, organizados por etapa de análise, modelagem e submissão. Cada notebook é relativamente independente, porém todos compartilham um **pré-processamento padronizado**, garantindo consistência, rastreabilidade e reprodutibilidade dos resultados ao longo do projeto.

---

## 🗂 Estrutura dos Notebooks

### 🔍 Análise Exploratória de Dados (EDA)

* **EDA.ipynb**
  Análise exploratória completa do dataset Titanic, incluindo avaliação de qualidade dos dados, distribuições univariadas, análises bivariadas, dispersão, correlação e identificação de padrões associados à variável resposta **Survived**.
  Este notebook fundamenta diretamente as decisões de **pré-processamento e engenharia de atributos** utilizadas nos modelos preditivos.

---

### 🌲 Modelos Ensemble — Random Forest

* **models_randomForest.ipynb**
  Avaliação de modelos baseados em árvores do tipo Random Forest:

  * Random Forest (baseline)
  * Random Forest com hiperparâmetros ajustados

  Ênfase na captura de não linearidades, interações entre variáveis e análise de robustez.

* **Hiperparameter_search_RF.ipynb**
  Busca sistemática de hiperparâmetros para Random Forest, com validação cruzada e análise comparativa de desempenho.

* **RF_Submission.ipynb**
  Geração do arquivo de submissão Kaggle utilizando o melhor modelo Random Forest selecionado.

---

### 🚀 Gradient Boosting — XGBoost

* **models_XGBoost.ipynb**
  Avaliação de modelos XGBoost em diferentes níveis de complexidade:

  * XGBoost (baseline)
  * XGBoost com ajustes intermediários
  * XGBoost otimizado

  Foco em desempenho preditivo e controle de overfitting.

* **Hiperparameter_search_XGB.ipynb**
  Busca de hiperparâmetros do XGBoost, explorando regularização, profundidade e taxa de aprendizado.

* **XGB_Submission.ipynb**
  Notebook dedicado à geração da submissão Kaggle com o melhor modelo XGBoost.

---

### 🧠 Gradient Boosting — CatBoost

* **models_CBTBoost.ipynb**
  Modelagem utilizando CatBoost, explorando seu tratamento nativo de variáveis categóricas e estabilidade em datasets tabulares.

* **Hiperparameter_search_CBT.ipynb**
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

➡️ O modelo recomendado é aquele que apresenta o melhor compromisso entre desempenho e robustez, conforme os resultados obtidos nos notebooks de busca e validação.

--