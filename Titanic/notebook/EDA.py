#!/usr/bin/env python
# coding: utf-8

# # Titanic (Machine Learning from Disaster) - Análise Exploratória dos Dados
# ## Contexto
# 
# ### 📌 Objetivos do EDA:
# * Entender a estrutura e qualidade dos dados
# * Identificar variáveis importantes
# * Detectar outliers, valores ausentes e distribuições
# * Formular hipóteses

# In[1]:


import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.patches as mpatches


from setup_notebook import setup_path
setup_path()
from src.functions import *

from matplotlib.colors import LinearSegmentedColormap
import warnings

warnings.filterwarnings("ignore")

color_palette21 = [
    "#004C4C", "#006666", "#008080", "#199191", "#29A3A3",
    "#40B5B5", "#55C7C7", "#66D9D9", "#80ECEC", "#99FFFF", 
    "#FFD580", "#FFC460", "#FFB240", "#FFA020", "#FF8E00",
    "#FF7C00", "#FF6400", "#FF4C00", "#FF3300", "#FF1A00", "#FF0000"]

# Definir cores
color_binary = {
    0: color_palette21[-7],  # Vermelho para não sobreviveu
    1: color_palette21[3]   # Azul para sobreviveu
}

sns.set_palette(sns.color_palette(color_palette21))
sns.color_palette(color_palette21)


# ---
# ## 2. Dataload & Pré-visualização dos Dados 🗂️

# In[2]:


# Carregando os dados
dfo = pd.read_csv("/home/akel/PycharmProjects/Kaggle/Titanic/data/raw/train.csv")
df=dfo.drop(columns='PassengerId')
NC=df.shape[1]
display(df.head(5))
inital_describe(df,True)
df.describe()


# ### 2.1 Pré-visualização dos Dados 🗂️
# 

# In[3]:


out=mult_plt2(df.drop(['Name','Ticket','Cabin'], axis=1),ncols=3,max_bins=10,figsize=(16, 16))
out


# In[16]:


out2=mult_plt2(df.drop(['Name','Ticket','Cabin'], axis=1),kind='box',ncols=5,max_bins=10,figsize=(16, 5))
out2


# #### B. Percentual de Sobreviventes/Sexo

# In[5]:


stats1,stats2= bar_bar_cat(df,'Sex','Survived',h=6)
display(stats1)
display(stats2)


# #### C. Percentual de Sobreviventes por classe

# In[6]:


stats1,stats2= bar_bar_cat(df,'Pclass','Survived',h=6)
display(stats1)
display(stats2)


# #### D. Percentual de Sobreviventes por Local de Embarque

# In[7]:


stats1,stats2= bar_bar_cat(df,'Embarked','Survived',h=6)
display(stats1)
display(stats2)


# #### E. Percentual de Sobreviventes por ter Cabine

# In[8]:


df['HasCabin'] = df['Cabin'].notnull().astype(int)
stats1,stats2= bar_bar_cat(df,'HasCabin','Survived',h=6)
display(stats1)
display(stats2)


# #### F. Percentual de Sobreviventes por faixa etária

# In[9]:


# Criar bins de idade

#df['Age'] = df['Age'].fillna(df['Age'].median())

df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 30, 50, 80], 
                         labels=['Criança (<12)', 'Adolescente (12-18)', 
                                 'Adulto Jovem (19-30)', 'Adulto (31-50)', 
                                 'Idoso (>50)'])

stats1,stats2= bar_bar_cat(df,'Age_Group','Survived',h=6)
display(stats1)
display(stats2)


# #### G. Percentual de Sobreviventes por Tamanho da familia

# In[10]:


df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
stats1,stats2= bar_bar_cat(df,'FamilySize','Survived',h=6)
display(stats1)
display(stats2)


# #### 3.Dispersão e distribuição

# In[11]:


out1=scatter_by_category(df,'Age','Fare',
    hue_var='Survived',category_var='HasCabin',
    ncols=2,category_name=['Sem cabine','Com cabine'])
display(out1)

out2=scatter_by_category(df,'FamilySize','Fare',
    hue_var='Survived', category_var='HasCabin',
    ncols=2,category_name=['Sem cabine','Com cabine'])
display(out2)

# out3=scatter_by_category(df,'FamilySize','Fare',
#     hue_var='Survived', category_var='Age_Group',
#     ncols=4)
# display(out3)


# In[12]:


#===================================================================
# out3=scatter_by_category(df,'FamilySize','Fare',
#     hue_var='Survived', category_var='Age_Group',
#     ncols=5)
# display(out3)


# In[13]:


#df['Age_Group'].isnull()


# In[14]:


# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np


dfa=df['Age']


# Configurar estilo
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 1, figsize=(14, 5))

# 1. Variável para número de barras
#nbins = 10

b = np.arange(0, 80+1, 5)

# --- Gráfico 1: Histograma básico ---
count, bins, patches = axes.hist(dfa, bins=b, color='skyblue', edgecolor='black', alpha=0.7)

# 2. Exibir contagens acima das barras no formato: val_percent % (val_count)
for i in range(len(count)):
    if count[i] > 0:  # Só mostrar se tiver contagem > 0
        # Calcular percentual
        total = count.sum()
        percent = (count[i] / total) * 100
        
        # Posicionar texto no meio da barra
        bin_center = (bins[i] + bins[i+1]) / 2
        bin_height = count[i]
        
        # Formatar texto: percentual % (contagem)
        text = f'{percent:.1f}% ({int(count[i])})'
        
        axes.text(bin_center, bin_height + max(count)*0.02, text,
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

axes.axvline(dfa.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {dfa.mean():.1f}')
axes.axvline(dfa.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {dfa.median():.1f}')
axes.set_title('Distribuição da Idade', fontsize=14, fontweight='bold')
axes.set_xlabel('Idade (anos)', fontsize=12)
axes.set_ylabel('Frequência', fontsize=12)
axes.legend()
axes.grid(True, alpha=0.3)

# Ajustar limite do eixo Y para dar espaço aos textos
axes.set_ylim(0, max(count) * 1.15)

plt.tight_layout()
plt.show()
display(dfa.describe())
display(dfa.isnull().sum())


# In[15]:


def impute_with_distribution(df, column):
   """
   Preenche NaN amostrando da distribuição real mantendo estatísticas
   """
   # Dados observados
   observed = df[column].dropna()
   
   if len(observed) == 0:
       return df  # Nada para fazer
   
   # Calcular estatísticas para verificação
   original_mean = observed.mean()
   original_std = observed.std()
   
   # Índices com NaN
   nan_idx = df[column].isna()
   num_nans = nan_idx.sum()
   
   if num_nans > 0:
       # Amostrar mantendo estatísticas (usando bootstrap)
       bootstrap_sample = np.random.choice(observed, size=num_nans, replace=True)
       
       # Garantir estatísticas similares (opcional)
       bootstrap_mean = bootstrap_sample.mean()
       bootstrap_std = bootstrap_sample.std()
       
       # Ajustar para manter média e desvio similares (opcional)
       if bootstrap_std > 0:
           adjusted_sample = (bootstrap_sample - bootstrap_mean) * (original_std / bootstrap_std) + original_mean
       else:
           adjusted_sample = bootstrap_sample
       
       # Aplicar
       df.loc[nan_idx, column] = adjusted_sample
   
   return df



# Aplicar
df2 = impute_with_distribution(df,'Age')

dfa=df2['Age']
# Configurar estilo
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 1, figsize=(14, 5))

# 1. Variável para número de barras
#nbins = 10
b = np.arange(0, 80+1, 5)

# --- Gráfico 1: Histograma básico ---
count, bins, patches = axes.hist(dfa, bins=b, color='skyblue', edgecolor='black', alpha=0.7)

# 2. Exibir contagens acima das barras no formato: val_percent % (val_count)
for i in range(len(count)):
   if count[i] > 0:  # Só mostrar se tiver contagem > 0
       # Calcular percentual
       total = count.sum()
       percent = (count[i] / total) * 100
       
       # Posicionar texto no meio da barra
       bin_center = (bins[i] + bins[i+1]) / 2
       bin_height = count[i]
       
       # Formatar texto: percentual % (contagem)
       text = f'{percent:.1f}% ({int(count[i])})'
       
       axes.text(bin_center, bin_height + max(count)*0.02, text,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

axes.axvline(dfa.mean(), color='red', linestyle='--', linewidth=2, label=f'Média: {dfa.mean():.1f}')
axes.axvline(dfa.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {dfa.median():.1f}')
axes.set_title('Distribuição da Idade', fontsize=14, fontweight='bold')
axes.set_xlabel('Idade (anos)', fontsize=12)
axes.set_ylabel('Frequência', fontsize=12)
axes.legend()
axes.grid(True, alpha=0.3)

# Ajustar limite do eixo Y para dar espaço aos textos
axes.set_ylim(0, max(count) * 1.325)

plt.tight_layout()
plt.show()
display(dfa.describe())
display(dfa.isnull().sum())


# In[ ]:




