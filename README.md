# 🚢 Titanic – Previsão de Sobrevivência
**Projeto Avaliativo P2 · Bacharelado em Inteligência Artificial · UNIMAR 2026**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://seu-app.streamlit.app)

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
- **Arquivo principal:** `train.csv` (891 registros com rótulo)
- **Arquivo de submissão:** `test.csv` (418 registros sem rótulo)
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
train.csv
    │
    ├── EDA Avançada
    │     ├── Taxa de sobrevivência por grupos (Sex, Pclass, Age)
    │     ├── Análise de valores ausentes
    │     ├── Detecção de outliers (IQR + Boxplots)
    │     └── Matriz de Correlação de Pearson
    │
    ├── Feature Engineering
    │     ├── Title  → extraído do campo Name
    │     ├── FamilySize = SibSp + Parch + 1
    │     ├── IsAlone = (FamilySize == 1)
    │     └── HasCabin → binário (Cabin não-nulo)
    │
    ├── Pré-processamento (sem Data Leakage)
    │     ├── Imputação de Age pela mediana por Título (do treino)
    │     ├── Imputação de Embarked pela moda (do treino)
    │     ├── Winsorização de Fare (P99 do treino)
    │     ├── Label Encoding de Sex, Embarked, Title
    │     └── StandardScaler (ajustado só no treino)
    │
    ├── Divisão Estratificada
    │     ├── Treino:    70% (~623 amostras)
    │     ├── Validação: 15% (~134 amostras)
    │     └── Teste:     15% (~134 amostras) ← ISOLADO
    │
    ├── Validação Cruzada (Stratified K-Fold, k=5)
    │     └── Aplicada sobre Treino + Validação
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

| Classificador | Acurácia | Precisão | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.812 | ~0.765 | ~0.728 | ~0.746 | ~0.865 |
| Random Forest | ~0.825 | ~0.795 | ~0.731 | ~0.761 | ~0.878 |
| **Gradient Boosting** | **~0.832** | **~0.815** | **~0.725** | **~0.767** | **~0.885** |

> **Nota:** Os valores exatos são gerados na execução do notebook (Seção 4 e 6).

---

## 🥇 Modelo Final Escolhido

**GradientBoostingClassifier** (scikit-learn)

**Justificativa:**
- Melhor AUC-ROC e F1-Score entre os três classificadores
- AUC-ROC superior a 0.88 indica excelente capacidade discriminativa
- Robusto a outliers e não-linearidades nos dados
- Captura interações complexas entre features (ex.: Sex × Pclass)

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
| **F1-Score** | Média harmônica entre Precisão e Recall | Equilíbrio entre os dois |
| **AUC-ROC** | Área sob a Curva ROC | Poder discriminativo geral |

---

## 🔑 Principais Resultados

1. **Sex/Title** são os fatores mais importantes: mulheres tiveram ~74% de sobrevivência vs ~19% dos homens
2. **Pclass** é o segundo fator: 1ª classe ~63% de sobrevivência, 3ª classe ~24%
3. **HasCabin** captura indiretamente a posição socioeconômica e física no navio
4. O modelo confirma historicamente a aplicação da regra "mulheres e crianças primeiro"
5. AUC-ROC ≈ 0.89 no teste isolado indica excelente generalização

---

## 📁 Estrutura dos Arquivos

```
titanic-p2/
│
├── app.py                          # Aplicação Streamlit
├── requirements.txt                # Dependências do projeto
├── README.md                       # Esta documentação
│
├── notebooks/
│   └── notebook_atualizado.ipynb   # Notebook P2 revisado
│
├── model/
│   ├── modelo_final.pkl            # Gradient Boosting treinado (joblib)
│   └── scaler_final.pkl            # StandardScaler ajustado
│
├── reports/
│   └── relatorio_atualizado.pdf    # Relatório técnico P2
│
└── data/
    └── README_data.md              # Instruções para obter o dataset
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

3. **Obtenha o dataset:**
   ```bash
   kaggle competitions download -c titanic
   # Extraia train.csv e test.csv para a pasta data/
   ```

4. **Execute o notebook:**
   ```bash
   jupyter notebook notebooks/notebook_atualizado.ipynb
   ```
   > O notebook deve ser executado do início ao fim, em ordem. A Seção 8 gera os arquivos `model/modelo_final.pkl` e `model/scaler_final.pkl`, necessários para o app.

---

## ▶️ Como Executar o App Streamlit

```bash
# Com o modelo já gerado pelo notebook:
streamlit run app.py
```

O app abrirá em `http://localhost:8501`

---

<!--## 🌐 Link do App Publicado

**[🚀 Acessar a aplicação no Streamlit Community Cloud](https://seu-app.streamlit.app)**

> Substitua o link acima pela URL gerada após o deploy em [share.streamlit.io](https://share.streamlit.io)

--->

## ⚠️ Limitações

1. **Dataset histórico de 1912:** O modelo não generaliza para desastres modernos com diferentes protocolos de evacuação
2. **77% de missing em Cabin:** A variável `HasCabin` captura apenas presença/ausência de registro
3. **Imputação de Age:** ~20% das idades foram imputadas; erros nessa imputação propagam para o modelo
4. **Classe desbalanceada:** ~38% de sobreviventes vs 62% de não-sobreviventes
5. **Amostra limitada:** 891 registros de treino é uma amostra pequena para deep learning

---

## 🏁 Conclusão

O projeto demonstrou que fatores socioeconômicos e a política de evacuação da época ("mulheres e crianças primeiro") foram determinantes para a sobrevivência. O modelo **Gradient Boosting** capturou essas dinâmicas com AUC-ROC ≈ 0.89, confirmando que ML pode revelar e quantificar padrões históricos latentes em dados reais.

---

*Projeto Avaliativo P2 – Bacharelado em Inteligência Artificial – UNIMAR 2026*
