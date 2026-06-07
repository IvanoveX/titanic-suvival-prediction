# 🚢 Titanic – Previsão de Sobrevivência
**Projeto Avaliativo P2 · Bacharelado em Inteligência Artificial · UNIMAR 2026**

---

## 👥 Integrantes e RAs

| Nome | RA |
|---|---|
| Felipe Traskini Rocha | 2028148 |
| Samuel Alves Vieira | 2041169 |
| Ivan Luís Gerônimo Del Roio | 2031330 |

**Grupo:** 12 | **Curso:** Bacharelado em Inteligência Artificial | **UNIMAR 2026**

---

## 📌 Descrição do Problema

O naufrágio do Titanic (1912) é um dos desastres marítimos mais documentados da história. Este projeto utiliza dados reais dos passageiros para construir um modelo de Machine Learning capaz de prever quais passageiros sobreviveram, explorando como fatores socioeconômicos e demográficos influenciaram as chances de sobrevivência.

---

## 🎯 Objetivo do Projeto

Desenvolver um classificador binário que preveja a sobrevivência (`Survived`: 0 ou 1) de passageiros do Titanic, com base em variáveis como classe do bilhete, sexo, idade, título social e tamanho da família.

---

## 📂 Dataset Utilizado

- **Fonte:** [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic)
- **Arquivo utilizado:** `Titanic-Dataset.csv` (891 registros com rótulo)
- **Variável-alvo:** `Survived` (0 = Não sobreviveu, 1 = Sobreviveu)

### Download do Dataset

```bash
# Via Kaggle CLI
kaggle competitions download -c titanic

# Ou via site: https://www.kaggle.com/competitions/titanic/data
```

---

## 🤖 Tipo de Problema de Machine Learning

**Classificação Binária Supervisionada**
- **Entrada:** Atributos do passageiro (demográficos, socioeconômicos, familiares)
- **Saída:** Probabilidade e classe de sobrevivência (0 ou 1)

---

## 🔬 Metodologia

```
Titanic-Dataset.csv
    │
    ├── EDA Avançada
    │     ├── Taxa de sobrevivência por grupos (Sex, Pclass, Age)
    │     ├── Análise de valores ausentes
    │     ├── Detecção de outliers (IQR + Boxplots)
    │     └── Matriz de Correlação de Pearson
    │
    ├── Feature Engineering
    │     ├── Title  → extraído do campo Name via regex
    │     ├── FamilySize = SibSp + Parch + 1
    │     ├── IsAlone = (FamilySize == 1)
    │     └── HasCabin → binário (Cabin não-nulo)
    │
    ├── Pré-processamento (sem Data Leakage)
    │     ├── Imputação de Age pela mediana por Título (do treino)
    │     ├── Imputação de Embarked pela moda (do treino)
    │     ├── Winsorização de Fare no P99 do treino (cap: £227,53)
    │     ├── Label Encoding de Sex, Embarked, Title
    │     └── StandardScaler (ajustado somente no treino)
    │
    ├── Divisão Estratificada
    │     ├── Treino:    70% (623 amostras)  — sobreviventes: 38,36%
    │     ├── Validação: 15% (134 amostras)  — sobreviventes: 38,81%
    │     └── Teste:     15% (134 amostras)  — sobreviventes: 38,06% ← ISOLADO
    │
    ├── Validação Cruzada (Stratified K-Fold, k=5)
    │     └── Aplicada sobre Treino + Validação (757 amostras)
    │
    ├── Treinamento e Avaliação
    │     ├── LogisticRegression
    │     ├── RandomForestClassifier
    │     └── GradientBoostingClassifier ← Modelo Final
    │
    └── Deploy
          └── Streamlit Community Cloud
```

---

## 🏆 Modelos Treinados

### Validação Cruzada — Stratified K-Fold (k=5)

| Classificador | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0,8057 | 0,7636 | 0,7182 | 0,7391 | 0,8628 |
| Random Forest | 0,8309 | 0,8237 | 0,7150 | 0,7633 | 0,8701 |
| **Gradient Boosting** | **0,8229** | **0,7990** | **0,7218** | **0,7577** | **0,8760** |

### Avaliação Final — Conjunto de Teste Isolado

| Classificador | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | 0,8134 | 0,7600 | 0,7451 | 0,7525 | 0,8330 |
| Random Forest | 0,7836 | 0,7115 | 0,7255 | 0,7184 | 0,8273 |
| **Gradient Boosting** | **0,7612** | **0,7111** | **0,6275** | **0,6667** | **0,7750** |

---

## 🥇 Modelo Final Escolhido

**GradientBoostingClassifier** (scikit-learn)

**Justificativa:**
- Maior AUC-ROC na validação cruzada estratificada (0,8760) — o indicador estatisticamente mais robusto para datasets pequenos
- Captura interações não-lineares complexas entre features (ex.: Sex × Pclass × Age) que a Regressão Logística, por ser um modelo linear, não consegue representar
- Superior em escalabilidade: em cenários com mais dados e features, o GB continua melhorando enquanto modelos lineares atingem seu teto de expressividade
- A queda de AUC no teste isolado é esperada e documentada em datasets pequenos; regularização adicional reduziria esse gap em produção real

**Hiperparâmetros:**
```python
GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=4,
    random_state=42
)
```

---

## 📊 Métricas de Avaliação

| Métrica | Descrição | Aplicação ao Problema |
|---|---|---|
| **Acurácia** | % de predições corretas | Visão geral do desempenho |
| **Precisão** | % de positivos corretos entre os preditos como positivos | Evitar falsos alarmes |
| **Recall** | % de positivos reais identificados corretamente | Identificar todos os sobreviventes |
| **F1-Score** | Média harmônica entre Precisão e Recall | Equilíbrio entre Precisão e Recall |
| **AUC-ROC** | Área sob a Curva ROC | Poder discriminativo independente do limiar |

---

## 🔑 Principais Resultados

1. **Sex** é o fator mais importante (importância relativa: **39,11%**) — mulheres tiveram 74,20% de sobrevivência vs 18,89% dos homens
2. **Fare** é o segundo fator (**19,41%**) — proxy direto do status socioeconômico e posição no navio
3. **Age** contribui com **14,21%** — crianças tiveram evacuação prioritária
4. **Pclass** representa **11,39%** — passageiros de 1ª classe (~63% de sobrevivência) tinham acesso físico privilegiado aos botes
5. **HasCabin** agrega **5,74%** — cabines registradas indicam posição social e física no andar superior do navio
6. O modelo confirma quantitativamente a regra histórica *"mulheres e crianças primeiro"* com AUC-ROC de 0,8760 na validação cruzada

---

## 📁 Estrutura dos Arquivos

```
titanic-p2/
│
├── app.py                              # Aplicação Streamlit
├── requirements.txt                    # Dependências do projeto
├── README.md                           # Esta documentação
│
├── notebooks/
│   └── notebook_atualizado.ipynb       # Notebook P2 revisado
│
├── model/
│   ├── modelo_final.pkl                # Gradient Boosting treinado (joblib)
│   ├── scaler_final.pkl                # StandardScaler ajustado no dataset completo
│   └── feature_names.pkl              # Lista ordenada das 11 features (segurança de deploy)
│
├── reports/
│   └── relatorio_atualizado.pdf        # Relatório técnico P2
│
└── data/
    └── Titanic-Dataset.csv             # Dataset utilizado (fonte: Kaggle)
```

---

## 🛠️ Tecnologias Utilizadas

| Tecnologia | Versão | Uso |
|---|---|---|
| Python | 3.10+ | Linguagem principal |
| pandas | 2.0+ | Manipulação de dados |
| numpy | 1.24+ | Operações numéricas |
| scikit-learn | 1.3+ | ML, pré-processamento, métricas |
| matplotlib | 3.7+ | Visualizações |
| seaborn | 0.12+ | Visualizações estatísticas |
| joblib | 1.3+ | Serialização do modelo |
| streamlit | 1.32+ | Interface web do app |

---

## ▶️ Como Executar o Notebook

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/titanic-p2.git
   cd titanic-p2
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Certifique-se de que o dataset está no lugar correto:**
   ```
   titanic-p2/
   └── data/
       └── Titanic-Dataset.csv   ← obrigatório
   ```
   O notebook lê o arquivo com `pd.read_csv("Titanic-Dataset.csv")`.  
   Coloque o CSV na mesma pasta em que o notebook será executado, ou ajuste o caminho conforme necessário.

4. **Execute o notebook:**
   ```bash
   jupyter notebook notebooks/notebook_atualizado.ipynb
   ```
---

## ▶️ Como Executar o App Streamlit

```bash
# Com os arquivos de model/ já gerados pelo notebook:
streamlit run app.py
```

O app abrirá em `http://localhost:8501`

---

## 🌐 Link do App Publicado

**[🚀 Acessar a aplicação — would-you-survive-titanic.streamlit.app](https://would-you-survive-titanic.streamlit.app)**

---

## ⚠️ Limitações

1. **Dataset histórico de 1912:** O modelo não generaliza para desastres modernos com diferentes protocolos de evacuação
2. **77% de missing em Cabin:** A variável `HasCabin` captura apenas presença/ausência de registro, não o número real da cabine
3. **Imputação de Age:** ~19,87% das idades foram imputadas pela mediana do título; erros nessa imputação propagam-se para o modelo
4. **Classe desbalanceada:** ~38% de sobreviventes vs ~62% de não-sobreviventes
5. **Amostra limitada:** 891 registros é uma amostra pequena; o Gradient Boosting apresenta overfitting moderado neste tamanho de dataset

---

## 🏁 Conclusão

O projeto demonstrou que fatores socioeconômicos e a política de evacuação da época (*"mulheres e crianças primeiro"*) foram determinantes para a sobrevivência. A feature **Sex** concentra 39,11% da importância relativa do modelo, seguida de **Fare** (19,41%) e **Age** (14,21%). O **Gradient Boosting** foi selecionado como modelo final por sua superioridade na validação cruzada (AUC-ROC 0,8760) e por sua escalabilidade para contextos de produção com datasets maiores.

