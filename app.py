"""
app.py – Aplicação Streamlit: Previsão de Sobrevivência no Titanic
Grupo 12 | Bacharelado em IA – UNIMAR | P2 2026
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ─────────────────────────────────────────────────────────────────────────────
# Configuração da Página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic – Previsão de Sobrevivência",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS Customizado
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a3c5e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1rem;
        color: #5a7a9a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-survived {
        background: linear-gradient(135deg, #1a7a4a, #27ae60);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(39,174,96,0.4);
    }
    .result-died {
        background: linear-gradient(135deg, #8b1a1a, #e74c3c);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        font-size: 1.6rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 15px rgba(231,76,60,0.4);
    }
    .prob-bar-label {
        font-size: 0.85rem;
        color: #555;
        margin-top: 0.3rem;
    }
    .info-box {
        background-color: #f0f6ff;
        border-left: 4px solid #2980b9;
        padding: 1rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
        font-size: 0.92rem;
        color: #333;
    }
    .metric-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .section-divider {
        border-top: 2px solid #e8ecf0;
        margin: 1.5rem 0;
    }
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento do Modelo
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Carregando modelo treinado...")
def load_model():
    model_path  = os.path.join(os.path.dirname(__file__), "model", "modelo_final.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler_final.pkl")

    if not os.path.exists(model_path):
        st.error(
            "⚠️ Arquivo do modelo não encontrado. "
            "Execute o notebook até a Seção 8 para gerar `model/modelo_final.pkl`."
        )
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(
            "⚠️ Arquivo do scaler não encontrado. "
            "Execute o notebook até a Seção 8 para gerar `model/scaler_final.pkl`."
        )
        st.stop()

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


model, scaler = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Funções Auxiliares
# ─────────────────────────────────────────────────────────────────────────────
TITLE_MAP = {
    "Mr (Homem adulto)":           "Mr",
    "Mrs (Mulher casada)":         "Mrs",
    "Miss (Mulher solteira)":      "Miss",
    "Master (Menino)":             "Master",
    "Rare (Título especial/nobre)":"Rare",
}

TITLE_AGE_MEDIAN = {
    "Master": 4.0,
    "Miss":   21.5,
    "Mr":     30.0,
    "Mrs":    35.0,
    "Rare":   45.0,
}

TITLE_LABEL_ENC = {"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rare": 4}
SEX_LABEL_ENC   = {"female": 0, "male": 1}
EMBARKED_ENC    = {"C": 0, "Q": 1, "S": 2}

FEATURE_ORDER = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "Title", "HasCabin", "FamilySize", "IsAlone"
]

FARE_CAP = 300.0  # percentil 99 aproximado do dataset Titanic


def encode_inputs(pclass, sex, age, sibsp, parch, fare,
                  embarked, title_raw, has_cabin):
    title   = TITLE_MAP[title_raw]
    age_val = age if age and age > 0 else TITLE_AGE_MEDIAN.get(title, 28.0)
    fare_val = min(fare, FARE_CAP)

    family_size = sibsp + parch + 1
    is_alone    = int(family_size == 1)

    row = {
        "Pclass":     pclass,
        "Sex":        SEX_LABEL_ENC[sex],
        "Age":        age_val,
        "SibSp":      sibsp,
        "Parch":      parch,
        "Fare":       fare_val,
        "Embarked":   EMBARKED_ENC[embarked],
        "Title":      TITLE_LABEL_ENC[title],
        "HasCabin":   int(has_cabin),
        "FamilySize": family_size,
        "IsAlone":    is_alone,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


def predict(df_input):
    X_scaled = scaler.transform(df_input)
    pred     = model.predict(X_scaled)[0]
    proba    = model.predict_proba(X_scaled)[0]
    return int(pred), float(proba[1])


# ─────────────────────────────────────────────────────────────────────────────
# Interface Principal
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🚢 Titanic – Previsão de Sobrevivência</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Bacharelado em Inteligência Artificial · UNIMAR · Grupo 12 · P2 2026</div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="info-box">
Insira os dados do passageiro nos campos abaixo e clique em <strong>Executar Predição</strong>.
O modelo de <em>Gradient Boosting</em> (AUC-ROC ≈ 0.89) irá estimar a probabilidade de sobrevivência
com base em padrões aprendidos do dataset histórico do Titanic (Kaggle).
</div>
""", unsafe_allow_html=True)

# ─── Sidebar: Informações do Modelo ──────────────────────────────────────────
with st.sidebar:
    st.header("📊 Sobre o Modelo")
    st.markdown("**Algoritmo:** Gradient Boosting Classifier")
    st.markdown("**Validação:** Stratified K-Fold (k=5)")

    st.markdown("---")
    st.subheader("Métricas no Teste")
    col1, col2 = st.columns(2)
    col1.metric("Acurácia",  "≈ 83%")
    col2.metric("AUC-ROC",   "≈ 0.89")
    col1.metric("F1-Score",  "≈ 0.77")
    col2.metric("Precisão",  "≈ 0.81")

    st.markdown("---")
    st.subheader("Top Features")
    st.markdown("""
    1. 🎩 **Title** (Título social)
    2. 👤 **Sex** (Gênero)
    3. 🎫 **Pclass** (Classe)
    4. 💰 **Fare** (Tarifa)
    5. 🛏️ **HasCabin** (Possui cabine)
    """)

    st.markdown("---")
    st.caption("Integrantes: Felipe T. Rocha | Samuel A. Vieira | Ivan L. G. Del Roio")

# ─── Formulário de Entrada ───────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.subheader("📋 Dados do Passageiro")

col_a, col_b = st.columns(2)

with col_a:
    pclass = st.selectbox(
        "🎫 Classe do Bilhete (Pclass)",
        options=[1, 2, 3],
        format_func=lambda x: {1: "1ª Classe (Luxo)", 2: "2ª Classe (Intermediária)",
                                 3: "3ª Classe (Econômica)"}[x],
        help="A classe social do passageiro. 1ª classe tinha maior prioridade nos botes."
    )

    sex = st.selectbox(
        "👤 Sexo",
        options=["female", "male"],
        format_func=lambda x: {"female": "Feminino", "male": "Masculino"}[x],
        help="Gênero do passageiro. Mulheres tiveram prioridade de evacuação."
    )

    age = st.number_input(
        "🎂 Idade (anos)",
        min_value=0.0, max_value=100.0, value=28.0, step=1.0,
        help="Se desconhecida, deixe 0 e o modelo usará a mediana do título."
    )

    title_display = st.selectbox(
        "🎩 Título Social",
        options=list(TITLE_MAP.keys()),
        index=2,
        help="Extraído do nome. Reflete gênero, estado civil e posição social."
    )

with col_b:
    sibsp = st.number_input(
        "👫 Cônjuges/Irmãos a bordo (SibSp)",
        min_value=0, max_value=10, value=0, step=1,
        help="Número de cônjuges ou irmãos embarcados junto."
    )

    parch = st.number_input(
        "👨‍👩‍👧 Pais/Filhos a bordo (Parch)",
        min_value=0, max_value=10, value=0, step=1,
        help="Número de pais ou filhos embarcados junto."
    )

    fare = st.number_input(
        "💰 Tarifa paga (£ libras)",
        min_value=0.0, max_value=600.0, value=32.0, step=0.5,
        help="Valor da passagem em libras esterlinas de 1912. Referência: 1ª classe ≈ £75."
    )

    embarked = st.selectbox(
        "⚓ Porto de Embarque",
        options=["S", "C", "Q"],
        format_func=lambda x: {"S": "S – Southampton (Inglaterra)",
                                "C": "C – Cherbourg (França)",
                                "Q": "Q – Queenstown (Irlanda)"}[x],
        help="Porto onde o passageiro embarcou. A maioria embarcou em Southampton."
    )

has_cabin = st.checkbox(
    "🛏️ Possui número de cabine registrado",
    value=False,
    help="Passageiros com cabine registrada geralmente eram de 1ª ou 2ª classe."
)

# ─── Resumo dos Dados Inseridos ───────────────────────────────────────────────
family_size_display = sibsp + parch + 1
is_alone_display = "Sim" if family_size_display == 1 else "Não"

with st.expander("📑 Resumo dos Dados Inseridos", expanded=False):
    df_display = pd.DataFrame({
        "Variável": ["Pclass", "Sexo", "Idade", "Título", "SibSp", "Parch",
                     "Tarifa (£)", "Embarque", "HasCabin", "FamilySize", "IsAlone"],
        "Valor": [
            f"{pclass}ª Classe",
            "Feminino" if sex == "female" else "Masculino",
            f"{age:.0f} anos" if age > 0 else "Não informada (imputada)",
            TITLE_MAP[title_display],
            sibsp, parch,
            f"£ {fare:.2f}",
            embarked,
            "Sim" if has_cabin else "Não",
            family_size_display,
            is_alone_display,
        ]
    })
    st.dataframe(df_display, use_container_width=True, hide_index=True)

# ─── Botão de Predição ───────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

btn_col1, btn_col2, btn_col3 = st.columns([1, 2, 1])
with btn_col2:
    predict_btn = st.button("🔍 Executar Predição", use_container_width=True, type="primary")

# ─── Resultado ───────────────────────────────────────────────────────────────
if predict_btn:
    df_input = encode_inputs(
        pclass=pclass, sex=sex, age=age, sibsp=sibsp,
        parch=parch, fare=fare, embarked=embarked,
        title_raw=title_display, has_cabin=has_cabin
    )

    prediction, prob_survived = predict(df_input)
    prob_died = 1.0 - prob_survived

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.subheader("🎯 Resultado da Predição")

    if prediction == 1:
        st.markdown(
            f'<div class="result-survived">✅ SOBREVIVEU<br>'
            f'<span style="font-size:1.1rem;font-weight:400;">'
            f'Probabilidade de sobrevivência: {prob_survived:.1%}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-died">❌ NÃO SOBREVIVEU<br>'
            f'<span style="font-size:1.1rem;font-weight:400;">'
            f'Probabilidade de sobrevivência: {prob_survived:.1%}</span></div>',
            unsafe_allow_html=True,
        )

    # Barra de probabilidade
    st.markdown("<br>", unsafe_allow_html=True)
    col_p1, col_p2 = st.columns(2)
    col_p1.metric("🟢 P(Sobreviveu)", f"{prob_survived:.1%}")
    col_p2.metric("🔴 P(Não Sobreviveu)", f"{prob_died:.1%}")

    st.progress(prob_survived, text=f"Probabilidade de Sobrevivência: {prob_survived:.1%}")

    # Interpretação contextual
    st.markdown("---")
    st.subheader("📖 Interpretação do Resultado")

    interpretacao = []

    if sex == "female":
        interpretacao.append(
            "👩 **Gênero Feminino:** A política 'mulheres e crianças primeiro' "
            "beneficiou fortemente as mulheres, que tiveram ~74% de taxa de sobrevivência."
        )
    else:
        interpretacao.append(
            "👨 **Gênero Masculino:** Homens tiveram apenas ~19% de taxa de sobrevivência, "
            "sendo os últimos a embarcar nos botes salva-vidas."
        )

    if pclass == 1:
        interpretacao.append(
            "🥇 **1ª Classe:** Passageiros de luxo ocupavam os andares superiores "
            "do navio, com acesso direto ao convés de botes (~63% de sobrevivência)."
        )
    elif pclass == 2:
        interpretacao.append(
            "🥈 **2ª Classe:** Taxa intermediária de sobrevivência (~47%). "
            "Localização no navio era mais favorável que a 3ª classe."
        )
    else:
        interpretacao.append(
            "🥉 **3ª Classe:** Passageiros econômicos enfrentaram barreiras físicas e "
            "organizacionais que reduziram drasticamente suas chances (~24% sobrevivência)."
        )

    title_key = TITLE_MAP[title_display]
    if title_key == "Master":
        interpretacao.append(
            "👦 **Título Master:** Indica menino. Crianças tiveram prioridade nas "
            "operações de evacuação."
        )
    elif title_key == "Miss":
        interpretacao.append(
            "👧 **Título Miss:** Mulher solteira ou jovem. Combinado com o gênero "
            "feminino, eleva consideravelmente as chances de sobrevivência."
        )

    if family_size_display == 1:
        interpretacao.append(
            "🧍 **Viajando sozinho:** Passageiros sem acompanhantes tinham mais "
            "mobilidade, mas também menos suporte mútuo durante a evacuação."
        )
    elif family_size_display > 4:
        interpretacao.append(
            f"👨‍👩‍👧‍👦 **Família grande ({family_size_display} pessoas):** Famílias numerosas "
            "tinham mais dificuldade de se deslocar rapidamente para os botes."
        )

    for item in interpretacao:
        st.markdown(f"- {item}")

    st.markdown("---")
    st.caption(
        "⚠️ **Nota:** Esta predição é baseada em padrões históricos de 1912. "
        "O modelo tem acurácia de ~83% no conjunto de teste. "
        "Resultados individuais podem divergir dos padrões estatísticos do grupo."
    )

# ─── Rodapé ──────────────────────────────────────────────────────────────────
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.8rem;">
    🎓 Projeto Avaliativo P2 · Bacharelado em Inteligência Artificial · UNIMAR 2026<br>
    Grupo 12: Felipe Traskini Rocha · Samuel Alves Vieira · Ivan Luís Gerônimo Del Roio<br>
    Modelo: Gradient Boosting Classifier | Dataset: Kaggle Titanic Competition
</div>
""", unsafe_allow_html=True)
