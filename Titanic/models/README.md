
## 📂 Guia de Nomenclatura de Modelos

Os artefatos de modelos salvos nesta pasta seguem uma estrutura padronizada para facilitar o rastreamento de experimentos e versões.

### 📝 Padrão do Nome

`modelo_[ML]_[estagio]_[metodo_busca]_[scoring]_[versao].joblib`

### 🔍 Dicionário de Termos

* **`[ML]` (Algoritmo):**
  * `RF`: Random Forest
  * `XGB`: XGBoost
  * `CATBST`: CatBoost


* **`[estagio]` (Nível de Maturação):**
  * `final`: Modelo treinado com o melhor set de hiperparâmetros.

* **`[metodo_busca]` (Otimização):**
  * `randsearch`: Randomized Search CV.
  * 'refine': ajuste apos Randomized Search CV
  * `bayes`: Otimização Bayesiana.

* **`[scoring]` (Métrica Alvo):**
  * `accuracy` ou `roc_auc`: A métrica principal usada para decidir o melhor modelo no buscador.

* **`[versao]`:**
Refere-se à versão do **preprocessador** utilizado (ex: `v12` para o pré-processamento 1.2).



---

### ✅ Exemplo Prático

`modelo_RF_final_bayes.accuracy_v12.joblib`

> Modelo **Random Forest** definitivo, cujos hiperparâmetros foram encontrados via **Busca Bayesiana** focada em **Acurácia**, utilizando a pipeline de dados **v1.2**.

---

### Como ler no Python

```python
import joblib

# Carregar o modelo e o preprocessador correspondente
model = joblib.load('models/modelo_RF_final_bayes.accuracy_v12.joblib')
preprocessor = joblib.load('models/preprocess_Titanic_v1.2.joblib')

```
