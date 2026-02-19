##  Descritivo de Transformações **Preprocessador Titanic v1.2**

Este documento detalha as etapas de engenharia de atributos e limpeza de dados aplicadas ao dataset Titanic para prepará-lo para modelos de Machine Learning.

### 1. Tratamento de Cabines e Decks

Para extrair valor da coluna `Cabin` (que possui muitos valores nulos), o preprocessador realiza duas operações:

* **HasCabin:** Cria uma variável binária (0 ou 1) indicando se o passageiro possuía uma cabine registrada.
* **Deck:** Extrai a primeira letra da cabine (ex: C, E, F), que representa o andar do navio.
* Valores ausentes ou a cabine incomum 'T' são classificados como **'U' (Unknown)**.
* A coluna original `Cabin` é removida.


### 2. Engenharia de Títulos (Extração de Nome)

A partir da coluna `Name`, o preprocessador isola o título social do passageiro:

* **Normalização:** Títulos equivalentes são agrupados (ex: *Mlle* e *Ms* tornam-se *Miss*; *Mme* torna-se *Mrs*).
* **Categorização:** Títulos frequentes (*Mr, Miss, Mrs, Master*) são mantidos, enquanto títulos nobres ou raros (como *Dr, Rev, Col, Lady*) são agrupados na categoria **'Rare'**.
* As colunas `Name` e `Ticket` são descartadas após essa extração.

### 3. Imputação Hierárquica de Idade (`Age2`)

Em vez de uma média simples, o modelo utiliza uma estratégia de preenchimento em níveis para a coluna `Age`:

1. **Nível 1:** Tenta preencher o nulo com a mediana de passageiros com o mesmo **Sexo, Classe (Pclass) e Posse de Cabine (HasCabin)**.
2. **Nível 2 (Fallback):** Se o grupo acima não existir, utiliza combinações simplificadas.
3. **Nível 3 (Global):** Caso ainda reste algum valor nulo, utiliza a mediana global de idade calculada no treino.

* O resultado é armazenado na nova coluna `Age2`.

### 4. Dinâmica Familiar

* **FamilySize:** Calcula o tamanho total da família a bordo somando `SibSp` (irmãos/cônjuges), `Parch` (pais/filhos) e o próprio passageiro (+1).

### 5. Tratamento de Dados Categóricos e Alinhamento

* **Embarked:** Preenche os valores ausentes com a **Moda** (valor mais frequente) identificada no conjunto de treino.
* **One-Hot Encoding:** Transforma variáveis categóricas (`Sex`, `Embarked`, `Deck`, `Title`) em colunas binárias (True/False).
* **Reindexação de Colunas:** Garante que o DataFrame final tenha exatamente as mesmas colunas do momento do `fit`, preenchendo com `0` caso alguma categoria não apareça nos dados de teste/validação. Isso previne erros de dimensão no modelo.
