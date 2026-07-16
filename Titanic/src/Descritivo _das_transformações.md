## Descritivo de Transformações **Preprocessador Titanic v1.2**

Este documento descreve as etapas de pré-processamento, engenharia de atributos e tratamento de dados realizadas pelo [**Preprocessador Titanic v1.2**](https://github.com/albertoakel/Kaggle/blob/master/Titanic/src/preprocess_utils_tic.py). Todas as estatísticas são aprendidas durante o método `fit()` e posteriormente aplicadas de forma consistente no método `transform()`, evitando vazamento de dados (*data leakage*).

---

### 1. Tratamento de Cabines e Decks

**Onde ocorre no código**

- **fit()**
  - **Seção 2:** criação da variável `HasCabin`;
  - **Seção 4.2:** criação da variável `Deck` e remoção de `Cabin`.

- **transform()**
  - **Seção 1:** criação de `HasCabin`, extração de `Deck` e remoção de `Cabin`.

**Descrição**

A coluna `Cabin` contém uma grande quantidade de valores ausentes. Para aproveitar essa informação, o preprocessador cria dois novos atributos:

- **HasCabin:** variável binária (`0` ou `1`) indicando se o passageiro possui cabine registrada.
- **Deck:** primeira letra da cabine (A, B, C, D...), representando o convés.
- Valores ausentes ou a cabine incomum `T` são convertidos para **`U` (Unknown)**.
- Após essa transformação, a coluna `Cabin` é removida.

---

### 2. Engenharia de Títulos (Title)

**Onde ocorre no código**

- **fit()**
  - **Seção 4.3**

- **transform()**
  - **Seção 6**

**Descrição**

A partir da coluna `Name`, o preprocessador extrai o título social do passageiro.

São realizadas duas etapas:

- Normalização:
  - `Mlle` e `Ms` → `Miss`
  - `Mme` → `Mrs`

- Agrupamento:

  Permanecem apenas

  - `Mr`
  - `Miss`
  - `Mrs`
  - `Master`

  Todos os demais são agrupados como **`Rare`**.

Após essa etapa, as colunas `Name` e `Ticket` são removidas.

---

### 3. Imputação Hierárquica da Idade (`Age2`)

**Onde ocorre no código**

- **fit()**
  - **Seção 3:** cálculo das medianas por grupos.

- **transform()**
  - **Seção 3:** aplicação da imputação.

**Descrição**

Durante o treinamento são calculadas medianas para:

1. Sexo + Classe + HasCabin
2. Sexo + Classe
3. Sexo

Na transformação, os valores ausentes são preenchidos seguindo exatamente essa ordem.

Caso ainda permaneçam valores nulos, utiliza-se a mediana global da idade.

O resultado é armazenado em `Age2`, enquanto `Age` é removida.

---

### 4. Imputação de Fare

**Onde ocorre no código**

- **fit()**
  - **Seção 3:** cálculo da mediana.

- **transform()**
  - **Seção 4:** preenchimento dos valores ausentes.

**Descrição**

A variável `Fare` possui sua mediana calculada durante o treinamento.

Na inferência, qualquer valor ausente é substituído por essa mediana.

---

### 5. FamilySize

**Onde ocorre no código**

- **fit()**
  - **Seção 4.1**

- **transform()**
  - **Seção 5**

**Descrição**

É criada a variável

```
FamilySize = SibSp + Parch + 1
```

representando o número total de familiares viajando com o passageiro.

---

### 6. Tratamento de Embarked

**Onde ocorre no código**

- **fit()**
  - **Seção 1:** cálculo da moda.

- **transform()**
  - **Seção 2:** preenchimento dos valores ausentes.

**Descrição**

Os valores ausentes de `Embarked` são preenchidos utilizando a moda aprendida durante o treinamento.

---

### 7. One-Hot Encoding

**Onde ocorre no código**

- **fit()**
  - **Seção 4.5:** geração das colunas de referência.

- **transform()**
  - **Seção 8:** aplicação do `pd.get_dummies()`.

**Descrição**

As variáveis categóricas são convertidas em colunas binárias utilizando One-Hot Encoding.

---

### 8. Alinhamento das Colunas

**Onde ocorre no código**

- **fit()**
  - **Seção 4.5:** armazenamento de `self.dummy_columns_`.

- **transform()**
  - **Seção 8:** reindexação das colunas.

**Descrição**

Após o One-Hot Encoding, o DataFrame é reindexado utilizando exatamente as colunas aprendidas durante o treinamento.

Categorias inexistentes recebem valor `0`, garantindo compatibilidade com o modelo treinado.

---

## Fluxo Geral do Preprocessador

| Etapa | `fit()` | `transform()` |
|--------|---------|---------------|
| Embarked | Seção 1 | Seção 2 |
| HasCabin | Seção 2 | Seção 1 |
| Estatísticas de Age/Fare | Seção 3 | — |
| FamilySize | Seção 4.1 | Seção 5 |
| Deck | Seção 4.2 | Seção 1 |
| Title | Seção 4.3 | Seção 6 |
| Remoção de colunas | Seção 4.4 | Seção 7 |
| One-Hot Encoding | Seção 4.5 | Seção 8 |
| Reindexação | — | Seção 8 |


---

## Nota sobre a documentação

Este documento foi elaborado com apoio de um **Modelo de Linguagem de Grande Escala (Large Language Model - LLM)**, utilizado como ferramenta de auxílio 
na organização da documentação.

A descrição das transformações foi produzida a partir da análise do código-fonte do 
preprocessador (`preprocess_utils_tic.py`) e revisada para refletir o comportamento implementado na versão **v1.2**. 
O código-fonte permanece como a referência definitiva para a implementação e funcionamento do preprocessador.