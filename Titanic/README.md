
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



➡️ **Melhor desempenho geral:** 

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
