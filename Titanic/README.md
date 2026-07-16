
# 🚢 Titanic — Análise de Sobrevivência de Passageiros

Projeto baseado no desafio **[Titanic: Machine Learning from Disaster (Kaggle)]**.
O objetivo é prever **a sobrevivência dos passageiros do Titanic** a partir de características demográficas, socioeconômicas e de viagem, utilizando técnicas clássicas e modernas de **Machine Learning para classificação supervisionada**.


## 📌 Objetivo do Projeto

Desenvolver um pipeline completo de **análise exploratória, pré-processamento, modelagem e avaliação**, com foco em:

* Interpretação dos dados
* engenharia de atributos
* comparação de modelos de classificação

---

## 🔍 O que você vai encontrar neste projeto


### [🔍 Análise Exploratória de Dados (EDA)](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/00_EDA.ipynb)

Análise exploratória completa do dataset Titanic, incluindo avaliação de qualidade dos dados, distribuições univariadas, análises bivariadas, dispersão, correlação e identificação de padrões associados à variável resposta **Survived**.
Este notebook fundamenta diretamente as decisões de **pré-processamento e engenharia de atributos** utilizadas nos modelos preditivos.

### [Pré-processamento completo](https://github.com/albertoakel/Kaggle/blob/master/Titanic/src/Descritivo%20_das_transforma%C3%A7%C3%B5es.md)
O projeto adota pipelines integrados que conectam a modelagem a um pré-processamento
completo (imputação, normalização e codificação categórica). Esse fluxo é centralizado e reutilizável via objetos serializados,
garantindo consistência entre treino e inferência e evitando o vazamento de dados (data leakage).


### Modelagem Preditiva

* [**Random Forest**](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/02_RF_model_compare.ipynb):
Foco na captura de relações não lineares, interações entre variáveis e robustez do modelo.
* [**XGBoost (eXtreme Gradient Boosting)** ](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/02_XGB_model_compare.ipynb):
Foco em maximizar o desempenho preditivo através do controle rigoroso de overfitting.
* [**CatBoost (Categorical Boosting)**](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/02_CBT_model_compare.ipynb):
Exploração do tratamento nativo de variáveis categóricas e estabilidade do modelo em dados tabulares.
* [**SVM (Support Vector Machine)**](https://github.com/albertoakel/Kaggle/blob/master/Titanic/notebook/02_SVM_model_compare.ipynb):Análise de diferentes kernels (RBF e Linear), otimização de margens e o impacto do escalonamento dos dados.

---

## 📊 Resultados dos Modelos

| Modelo                            | CV ROC-AUC (±σ)     | Test ROC-AUC | Test ACC (0.5) | Test ACC (Opt) | Status |
|-----------------------------------|---------------------|--------------|----------------|----------------|--------|
| Random Forest (Base)              | 0.8600 ± 0.0501     | 0.8710       | 0.8060         | 0.8134         |        |
| **Random Forest (Random Search)** | **0.8531 ± 0.0643** | **0.8905**   | **0.8246**     | **0.8358**     | 🏆     |
| Random Forest (Refine)            | 0.8760 ± 0.0516     | 0.8688       | 0.8060         | 0.8284         | 🥈     |
| Random Forest (Bayes)             | 0.8750 ± 0.0554     | 0.8752       | 0.8022         | 0.8172         |        |
| XGBoost (Base)                    | 0.8235 ± 0.0485     | 0.8520       | 0.7836         | 0.8022         |        |
| XGBoost (Random Search)           | 0.8411 ± 0.0558     | 0.8738       | 0.8172         | 0.8209         |        |
| XGBoost (Refine)                  | 0.8491 ± 0.0494     | 0.8749       | 0.8321         | 0.8358         | 🥈     |
| XGBoost (Bayes)                   | 0.8475 ± 0.0408     | 0.8695       | 0.8134         | 0.8209         |        |
| CatBoost (Base)                   | 0.8523 ± 0.0529     | 0.8811       | 0.8172         | 0.8284         | 🥉     |
| CatBoost (Random Search)          | 0.8460 ± 0.0508     | 0.8730       | 0.8134         | 0.8134         |        |
| CatBoost (Refine)                 | 0.8460 ± 0.0466     | 0.8739       | 0.8246         | 0.8246         |        |
| CatBoost (Bayes)                  | 0.8539 ± 0.0535     | 0.8678       | 0.8246         | 0.8284         |        |
| SVM (Base)                        | 0.8513 ± 0.0596     | 0.8769       | 0.8209         | 0.8134         |        |
| SVM (Random Search)               | 0.8531 ± 0.0593     | 0.8649       | 0.8209         | 0.8134         |        |
| SVM (Grid Search)                 | 0.8488 ± 0.0634     | 0.8734       | 0.8172         | 0.8097         |        |    


📌 Nota: Em negrito o modelo como maior pontuação na plataforma  Kaggle (**0.79425**)  em 23/01/2026 

---

## 📁 Estrutura do Projeto

```
Titanic/
│
├── app/                                # Aplicações futuras (deploy)
│
├── data/
│   ├── raw/                            # Dados originais do Kaggle
│   └── processed/                      # Bases tratadas
├
├── models/                             # Gráficos e figuras
│
├── image/                              # Gráficos e figuras
│
├── notebook/
│   ├── eda_titanic.ipynb               # EDA
│   ├── models_.ipynb                   # modelo testado
│   ├── hiperparameter_tuning_.ipynb    # processos de tunning
│   └── Submission_.ipynb               # Template submissão 
│
├── sandbox/                            # Experimentos e testes
│
├── src/
│   ├── preprocess_utils.py            # Pré-processamento
│   ├── functions.py                   # funções graficas
│   ├── model_utils.py                 # funções estatiscas
│   ├── plot_metrica_class.py          # relatorio estatitico
│   └── pre_process.joblib             # preprocessadores
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
