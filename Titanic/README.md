
# 🚢 Titanic — Análise de Sobrevivência de Passageiros

Projeto baseado no desafio **[Titanic: Machine Learning from Disaster (Kaggle)]**.
O objetivo é prever **a sobrevivência dos passageiros do Titanic** a partir de características demográficas, socioeconômicas e de viagem, utilizando técnicas clássicas e modernas de **Machine Learning para classificação supervisionada**.


## 📌 Objetivo do Projeto

Desenvolver um pipeline completo de **análise exploratória, pré-processamento, modelagem e avaliação**, com foco em:

* interpretação dos dados
* engenharia de atributos
* comparação de modelos de classificação
* controle de overfitting

---

## 🔍 O que você vai encontrar neste projeto

* **EDA detalhada** com análise estatística e visual dos fatores de sobrevivência
* **Tratamento de dados ausentes** (`Age`, `Cabin`, `Embarked`)
* **Feature engineering** (tamanho da família, título do nome, variáveis binárias)
* **Pré-processamento completo**:

  * imputação
  * normalização
  * codificação categórica
* **Modelos avaliados**:

  * Random Forest Classifier
  * Gradient Boosting / XGBoost
* **Avaliação comparativa** com métricas de classificação
* **Modelo final pronto para submissão no Kaggle**

---

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

📌 Nota: A pontuação final submetida ao Kaggle foi **0.79425** (14/11/2025). 

---

## 🧠 Principais Aprendizados



---

## 📁 Estrutura do Projeto

```
Titanic/
│
├── app/                     # Aplicações futuras (deploy)
│
├── data/
│   ├── raw/                 # Dados originais do Kaggle
│   └── processed/           # Bases tratadas
│
├── image/                   # Gráficos e figuras
│
├── notebook/
│   ├── eda_titanic.ipynb
│   ├── models_baseline.ipynb
│   ├── models_ensemble.ipynb
│   └── submission.ipynb
│
├── sandbox/                 # Experimentos e testes
│
├── src/
│   ├── preprocess_utils.py
│   ├── feature_utils.py
│   ├── model_utils.py
│   └── best_model.joblib
│
├── requirements.txt
└── README.md
```

---

## 🚀 Como executar o projeto

### 1️⃣ Instalar dependências

```bash
pip install -r requirements.txt
```

### 2️⃣ Executar notebooks

```bash
jupyter notebook
```

### 3️⃣ Pré-processamento automatizado

```bash
python src/preprocess_utils_tic.py
```

---

## 📌 Observações Finais

Este projeto foi desenvolvido como um **estudo clássico de classificação supervisionada**, com foco em **interpretação, boas práticas e clareza metodológica**, servindo como:

* introdução sólida ao Machine Learning
* benchmark técnico
