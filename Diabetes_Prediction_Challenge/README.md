# 💉 Diabetes Prediction Challenge (Kaggle Playground S5E12)


Projeto desenvolvido para a competição [Kaggle Playground Series - Season 5, Episode 12]. O
objetivo é prever o diagnóstico de diabetes em pacientes utilizando dados clínicos e demográficos, 
aplicando modelos de ensemble e técnicas avançadas de Gradient Boosting.

## 📌 Objetivo do Projeto

Desenvolver um pipeline de Machine Learning de ponta a ponta para classificação binária,
focado em maximizar a métrica de avaliação da competição (ROC-AUC), com foco em:

* **Tratamento de Dados Sintéticos:** Lidar com as nuances de datasets gerados artificialmente.

* **Otimização de Hiperparâmetros:** Ajuste fino de modelos de alta performance.

* **Ensemble Learning:** Combinação de diferentes algoritmos para aumentar a estabilidade e acurácia.

---

## 🔍 O que você vai encontrar neste projeto

* **EDA (Análise Exploratória)**: Identificação de correlações entre glicose, IMC (BMI), idade e o diagnóstico positivo.
* **Feature Engineering**: Criação de indicadores clínicos e interação entre variáveis biométricas.
* **Pré-processamento Customizado**:
* Imputação estratégica de valores nulos (se houver).
* Normalização e Padronização via `StandardScaler`.
* Codificação de variáveis categóricas para modelos baseados em árvore.
* **Modelos Avaliados**:
* **LGBM (Light Gradient Boosting Machine)**: Alta velocidade e eficiência.
* **XGB (XGBoost)**: Robustez e regularização avançada.
* **Random Forest**: Modelo de baseline para capturar interações não lineares.
* **Ensemble (LGBM + XGB)**: Implementação de um `VotingClassifier` (Soft Voting) para combinar a força dos dois melhores modelos.

---

## 📊 Metodologia de Avaliação


* **Threshold Optimization**: Otimização do limiar de decisão via validação cruzada no conjunto de treinamento.
* **Estabilidade Estatística**: Comparação de performance utilizando `cross_val_score` (10 folds).
* **Métricas Principais**: ROC-AUC, Acurácia Otimizada e F1-Score.

---

