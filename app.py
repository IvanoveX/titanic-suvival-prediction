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
# CSS Customizado — Tema Dark com Azul Claro como destaque
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Tipografia e cores base ── */
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #60b4ff;
        text-align: center;
        margin-bottom: 0.25rem;
        letter-spacing: -0.5px;
        text-shadow: 0 0 40px rgba(96,180,255,0.3);
    }
    .subtitle {
        font-size: 0.92rem;
        color: #7eb8e8;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: 0.3px;
    }

    /* ── Hero / intro box ── */
    .hero-box {
        background: linear-gradient(135deg, rgba(30,60,100,0.55), rgba(15,35,65,0.75));
        border: 1px solid rgba(96,180,255,0.25);
        border-left: 4px solid #60b4ff;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin: 1.2rem 0 1.5rem 0;
        color: #d0e8ff;
        font-size: 0.93rem;
        line-height: 1.75;
    }
    .hero-box a { color: #60b4ff; text-decoration: underline; }
    .hero-box a:hover { color: #a0d0ff; }

    /* ── Seção de inputs ── */
    .section-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #60b4ff;
        margin-bottom: 0.6rem;
        margin-top: 1.4rem;
    }
    .section-divider {
        border: none;
        border-top: 1px solid rgba(96,180,255,0.15);
        margin: 1.4rem 0;
    }

    /* ── Radio buttons — visual limpo ── */
    div[data-testid="stRadio"] > label {
        color: #c8dcf0 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.2px;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 6px !important;
    }
    div[data-testid="stRadio"] label span {
        color: #b0ccec !important;
        font-size: 0.9rem !important;
    }

    /* ── Sliders — cor azul clara ── */
    div[data-testid="stSlider"] > label {
        color: #c8dcf0 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background-color: #60b4ff !important;
        border-color: #60b4ff !important;
    }
    div[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stThumbValue"] {
        color: #60b4ff !important;
        font-weight: 700 !important;
    }
    /* Trilha preenchida do slider */
    div[data-testid="stSlider"] [data-baseweb="slider"] div:first-child div:nth-child(3) {
        background: #60b4ff !important;
    }

    /* ── Selectbox ── */
    div[data-testid="stSelectbox"] > label {
        color: #c8dcf0 !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
    }

    /* ── Resultado sobreviveu ── */
    .result-survived {
        background: linear-gradient(135deg, #0d4d2a, #1a7a4a);
        border: 1px solid rgba(46,204,113,0.4);
        color: #a8ffcc;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        font-size: 1.55rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 4px 24px rgba(39,174,96,0.25);
        letter-spacing: 0.5px;
    }
    /* ── Resultado não sobreviveu ── */
    .result-died {
        background: linear-gradient(135deg, #4d0d0d, #8b1a1a);
        border: 1px solid rgba(231,76,60,0.4);
        color: #ffb3a8;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        font-size: 1.55rem;
        font-weight: 800;
        text-align: center;
        box-shadow: 0 4px 24px rgba(231,76,60,0.25);
        letter-spacing: 0.5px;
    }

    /* ── Resumo de dados ── */
    .summary-box {
        background: rgba(20,45,80,0.5);
        border: 1px solid rgba(96,180,255,0.18);
        border-radius: 8px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
        color: #b0ccec;
    }

    /* ── Nota de rodapé ── */
    .footer-note {
        text-align: center;
        color: #4a6a8a;
        font-size: 0.78rem;
        margin-top: 0.5rem;
        line-height: 1.7;
    }

    /* ── Progress bar azul ── */
    div[data-testid="stProgress"] > div > div > div {
        background-color: #60b4ff !important;
    }

    /* ── Botão primário ── */
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #1565c0, #1e88e5) !important;
        border: 1px solid #60b4ff !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px !important;
        transition: box-shadow 0.2s ease !important;
    }
    div[data-testid="stButton"] > button[kind="primary"]:hover {
        box-shadow: 0 0 18px rgba(96,180,255,0.45) !important;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento do Modelo
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Carregando modelo treinado...")
def load_model():
    model_path  = os.path.join(os.path.dirname(__file__), "model", "modelo_final.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler_final.pkl")
    if not os.path.exists(model_path):
        st.error("⚠️ modelo_final.pkl não encontrado em model/. Execute o notebook até a Seção 8.")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error("⚠️ scaler_final.pkl não encontrado em model/. Execute o notebook até a Seção 8.")
        st.stop()
    return joblib.load(model_path), joblib.load(scaler_path)


model, scaler = load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Constantes e mapeamentos
# ─────────────────────────────────────────────────────────────────────────────

# Faixas de tarifa reais observadas no dataset Titanic por classe
# (usadas para tornar o slider de Fare dinâmico)
FARE_RANGES = {
    1: (0,  300, 60),   # (min, max, default) — 1ª Classe
    2: (0,  100, 15),   # 2ª Classe
    3: (0,   60,  8),   # 3ª Classe
}

# Títulos disponíveis por gênero (lógica dinâmica)
TITLES_BY_GENDER = {
    "female": ["Miss", "Mrs", "Rare"],
    "male":   ["Mr", "Master", "Rare"],
}

# Descrições amigáveis dos títulos
TITLE_LABELS = {
    "Mr":     "Mr — Homem adulto",
    "Mrs":    "Mrs — Mulher casada",
    "Miss":   "Miss — Mulher solteira / jovem",
    "Master": "Master — Menino (< ~15 anos)",
    "Rare":   "Rare — Título especial / nobre",
}

# Encodings para o modelo (devem corresponder ao treino)
TITLE_ENC    = {"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Rare": 4}
SEX_ENC      = {"female": 0, "male": 1}
EMBARKED_ENC = {"C": 0, "Q": 1, "S": 2}

# Mediana de Age por título (fallback de imputação)
TITLE_AGE_MEDIAN = {"Master": 4.0, "Miss": 21.5, "Mr": 30.0, "Mrs": 35.0, "Rare": 45.0}

FEATURE_ORDER = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
    "Embarked", "Title", "HasCabin", "FamilySize", "IsAlone",
]


# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────────────────────
def encode_inputs(pclass, sex, age, sibsp, parch, fare, embarked, title, has_cabin):
    age_val     = age if age > 0 else TITLE_AGE_MEDIAN.get(title, 28.0)
    family_size = sibsp + parch + 1
    row = {
        "Pclass":     pclass,
        "Sex":        SEX_ENC[sex],
        "Age":        age_val,
        "SibSp":      sibsp,
        "Parch":      parch,
        "Fare":       float(fare),
        "Embarked":   EMBARKED_ENC[embarked],
        "Title":      TITLE_ENC[title],
        "HasCabin":   0,          # app não coleta — assume ausente (conservador)
        "FamilySize": family_size,
        "IsAlone":    int(family_size == 1),
    }
    return pd.DataFrame([row])[FEATURE_ORDER]


def run_prediction(df_input):
    X_scaled = scaler.transform(df_input)
    pred     = model.predict(X_scaled)[0]
    proba    = model.predict_proba(X_scaled)[0]
    return int(pred), float(proba[1])


# ─────────────────────────────────────────────────────────────────────────────
# ── CABEÇALHO / HERO ─────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="main-title">🚢 Titanic — Previsão de Sobrevivência</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtitle">Bacharelado em Inteligência Artificial · UNIMAR · Grupo 12 · P2 2026</div>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="hero-box">
    <strong>Bem-vindo a bordo, passageiro ⚓</strong><br><br>
    Este é um aplicativo de <em>Previsão de Sobrevivência do Titanic</em> desenvolvido com
    <strong>Streamlit</strong> e <strong>Python</strong> como projeto avaliativo do Bacharelado
    em Inteligência Artificial da UNIMAR. Ele utiliza um modelo de
    <em>Gradient Boosting</em> treinado para estimar a sua probabilidade de sobrevivência
    com base nas informações que você inserir abaixo.<br><br>
    Os dados usados no treinamento foram obtidos no
    <a href="https://www.kaggle.com/competitions/titanic" target="_blank">Kaggle — Titanic Competition</a>
    e contêm registros demográficos de <strong>891 dos 2.224 passageiros e tripulantes</strong>
    que estavam a bordo na noite de 14 para 15 de abril de 1912.<br><br>
    Preencha os campos abaixo, clique em <strong>Executar Predição</strong> e descubra
    quais seriam as suas chances. Uma ótima viagem! 🌊
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── SIDEBAR ──────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Sobre o Modelo")
    st.markdown("**Algoritmo:** Gradient Boosting Classifier")
    st.markdown("**Validação:** Stratified K-Fold (k = 5)")
    st.divider()

    st.markdown("**Métricas no Teste**")
    c1, c2 = st.columns(2)
    c1.metric("Acurácia",  "≈ 83%")
    c2.metric("AUC-ROC",  "≈ 0.89")
    c1.metric("F1-Score",  "≈ 0.77")
    c2.metric("Precisão", "≈ 0.81")
    st.divider()

    st.markdown("**Top Features**")
    st.markdown("""
1. 🎩 **Title** — Título social
2. 👤 **Sex** — Gênero
3. 🎫 **Pclass** — Classe
4. 💰 **Fare** — Tarifa
5. 🛏️ **HasCabin** — Cabine registrada
    """)
    st.divider()
    st.caption(
        "Grupo 12 · Felipe T. Rocha · Samuel A. Vieira · Ivan L. G. Del Roio"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── FORMULÁRIO DE ENTRADA ────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("### 🧭 Dados do Passageiro")
st.caption("Preencha todos os campos na ordem sugerida para melhores resultados.")

# ── 1. CLASSE DO BILHETE ──────────────────────────────────────────────────────
st.markdown('<p class="section-label">1 · Classe do Bilhete</p>', unsafe_allow_html=True)

pclass_label = st.radio(
    label="Classe do Bilhete",
    options=["1 — Luxo", "2 — Intermediária", "3 — Econômica"],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    help="A classe social do passageiro determina a localização no navio e o acesso aos botes.",
)
pclass = int(pclass_label[0])   # extrai o dígito: "1 — Luxo" → 1


# ── 2. GÊNERO ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">2 · Gênero</p>', unsafe_allow_html=True)

gender_label = st.radio(
    label="Gênero",
    options=["Mulher", "Homem"],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    help="Mulheres tiveram prioridade de evacuação (~74% de sobrevivência vs ~19% dos homens).",
)
sex = "female" if gender_label == "Mulher" else "male"


# ── 3. IDADE ──────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">3 · Idade</p>', unsafe_allow_html=True)

age = st.slider(
    label="Idade (anos)",
    min_value=1,
    max_value=100,
    value=28,
    step=1,
    label_visibility="collapsed",
    help="Crianças com menos de 12 anos tiveram prioridade nas operações de evacuação.",
)


# ── 4. IRMÃOS / CÔNJUGES ─────────────────────────────────────────────────────
st.markdown('<p class="section-label">4 · Irmãos ou Cônjuges a bordo (SibSp)</p>',
            unsafe_allow_html=True)

sibsp = st.slider(
    label="SibSp",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
    label_visibility="collapsed",
    help="Número de cônjuges ou irmãos embarcados junto com o passageiro.",
)


# ── 5. PAIS / FILHOS ──────────────────────────────────────────────────────────
st.markdown('<p class="section-label">5 · Pais ou Filhos a bordo (Parch)</p>',
            unsafe_allow_html=True)

parch = st.slider(
    label="Parch",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
    label_visibility="collapsed",
    help="Número de pais ou filhos embarcados junto com o passageiro.",
)


# ── 6. TARIFA — SLIDER DINÂMICO POR CLASSE ───────────────────────────────────
fare_min, fare_max, fare_default = FARE_RANGES[pclass]

st.markdown(
    f'<p class="section-label">'
    f'6 · Preço da Passagem — faixa da {pclass}ª Classe: £{fare_min}–£{fare_max}'
    f'</p>',
    unsafe_allow_html=True,
)

# Garante que o default seja múltiplo de 10 e dentro do range atual
fare_default_safe = min(
    max(round(fare_default / 10) * 10, fare_min),
    (fare_max // 10) * 10,
)

fare = st.slider(
    label="Tarifa (£)",
    min_value=fare_min,
    max_value=fare_max,
    value=fare_default_safe,
    step=10,
    label_visibility="collapsed",
    help=(
        "Valor da passagem em libras esterlinas de 1912. "
        "O intervalo se ajusta automaticamente à classe selecionada. "
        "Referência histórica: 1ª classe ≈ £60–300 | 2ª ≈ £10–60 | 3ª ≈ £5–30."
    ),
)


# ── 7. TÍTULO SOCIAL — DROPDOWN DINÂMICO POR GÊNERO ──────────────────────────
st.markdown('<p class="section-label">7 · Título Social</p>', unsafe_allow_html=True)

available_titles = TITLES_BY_GENDER[sex]
title_options    = [TITLE_LABELS[t] for t in available_titles]

title_label = st.selectbox(
    label="Título Social",
    options=title_options,
    index=0,
    label_visibility="collapsed",
    help=(
        "O título foi a feature mais preditiva do modelo (~24% de importância). "
        "As opções são filtradas automaticamente com base no gênero selecionado."
    ),
)
# Recupera a chave curta ("Mr", "Mrs", etc.) a partir do label selecionado
title = next(k for k, v in TITLE_LABELS.items() if v == title_label)


# ── 8. PORTO DE EMBARQUE ─────────────────────────────────────────────────────
st.markdown('<p class="section-label">8 · Porto de Embarque</p>', unsafe_allow_html=True)

embarked_label = st.radio(
    label="Porto de Embarque",
    options=[
        "Southampton 🇬🇧  (S)",
        "Cherbourg 🇫🇷  (C)",
        "Queenstown 🇮🇪  (Q)",
    ],
    index=0,
    horizontal=True,
    label_visibility="collapsed",
    help="Porto onde o passageiro embarcou. ~72% embarcou em Southampton.",
)
embarked = embarked_label[-2]   # extrai "S", "C" ou "Q" do final da string


# ─────────────────────────────────────────────────────────────────────────────
# ── RESUMO DOS DADOS INSERIDOS ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
family_size_calc = sibsp + parch + 1
is_alone_calc    = family_size_calc == 1

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

with st.expander("📋 Resumo dos dados inseridos", expanded=False):
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown(f"""
<div class="summary-box">
<b>Classe</b>: {pclass}ª — {pclass_label.split('—')[1].strip()}<br>
<b>Gênero</b>: {gender_label}<br>
<b>Idade</b>: {age} anos<br>
<b>Título</b>: {title}<br>
<b>Irmãos/Cônjuges</b>: {sibsp}
</div>""", unsafe_allow_html=True)
    with col_r2:
        st.markdown(f"""
<div class="summary-box">
<b>Pais/Filhos</b>: {parch}<br>
<b>Tarifa</b>: £{fare}<br>
<b>Porto</b>: {embarked} — {embarked_label.split('(')[0].strip()}<br>
<b>Tamanho da família</b>: {family_size_calc}<br>
<b>Viajando sozinho</b>: {"Sim" if is_alone_calc else "Não"}
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ── BOTÃO DE PREDIÇÃO ────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1, 2, 1])
with btn_col:
    predict_btn = st.button(
        "🔍 Executar Predição",
        use_container_width=True,
        type="primary",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── RESULTADO ────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    df_input = encode_inputs(
        pclass=pclass, sex=sex, age=age, sibsp=sibsp,
        parch=parch, fare=fare, embarked=embarked, title=title,
        has_cabin=False,
    )
    prediction, prob_survived = run_prediction(df_input)
    prob_died = 1.0 - prob_survived

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    st.markdown("### 🎯 Resultado da Predição")

    # ── Card de resultado ─────────────────────────────────────────────────────
    if prediction == 1:
        st.markdown(
            f'<div class="result-survived">'
            f'✅ SOBREVIVEU<br>'
            f'<span style="font-size:1rem;font-weight:500;">'
            f'Probabilidade estimada de sobrevivência: <strong>{prob_survived:.1%}</strong>'
            f'</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-died">'
            f'❌ NÃO SOBREVIVEU<br>'
            f'<span style="font-size:1rem;font-weight:500;">'
            f'Probabilidade estimada de sobrevivência: <strong>{prob_survived:.1%}</strong>'
            f'</span></div>',
            unsafe_allow_html=True,
        )

    # ── Métricas e barra de progresso ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    m1.metric("🔵 P(Sobreviveu)",      f"{prob_survived:.1%}")
    m2.metric("🔴 P(Não Sobreviveu)",  f"{prob_died:.1%}")
    st.progress(prob_survived,
                text=f"Probabilidade de Sobrevivência: {prob_survived:.1%}")

    # ── Interpretação contextual ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📖 Interpretação do Resultado")

    fatores = []

    # Gênero
    if sex == "female":
        fatores.append(
            "👩 **Gênero Feminino** — A regra *'mulheres e crianças primeiro'* garantiu "
            "prioridade de evacuação às mulheres, que tiveram **~74% de sobrevivência** "
            "versus ~19% dos homens."
        )
    else:
        fatores.append(
            "👨 **Gênero Masculino** — Homens foram os últimos a embarcar nos botes, "
            "resultando em apenas **~19% de sobrevivência**."
        )

    # Classe
    if pclass == 1:
        fatores.append(
            "🥇 **1ª Classe (Luxo)** — Passageiros de 1ª classe ocupavam os andares "
            "superiores do navio, com acesso direto e prioritário ao convés de botes "
            "(**~63% de sobrevivência**)."
        )
    elif pclass == 2:
        fatores.append(
            "🥈 **2ª Classe (Intermediária)** — Taxa de sobrevivência moderada (**~47%**). "
            "Localização mais favorável que a 3ª classe, mas sem a prioridade da 1ª."
        )
    else:
        fatores.append(
            "🥉 **3ª Classe (Econômica)** — Acomodados abaixo da linha d'água, com maior "
            "distância até o convés e barreiras físicas de evacuação (**~24% de sobrevivência**)."
        )

    # Título
    if title == "Master":
        fatores.append(
            "👦 **Título Master** — Indica menino (geralmente < 15 anos). "
            "Crianças foram incluídas na categoria de evacuação prioritária."
        )
    elif title == "Miss":
        fatores.append(
            "👧 **Título Miss** — Mulher solteira ou jovem. Combinado ao gênero feminino, "
            "eleva substancialmente a probabilidade de sobrevivência."
        )
    elif title == "Mrs":
        fatores.append(
            "💍 **Título Mrs** — Mulher casada. Também beneficiada pela prioridade feminina "
            "nas operações de evacuação."
        )

    # Família
    if is_alone_calc:
        fatores.append(
            "🧍 **Viajando sozinho** — Passageiros sem acompanhantes tinham maior "
            "mobilidade individual, mas nenhum suporte mútuo durante a evacuação."
        )
    elif family_size_calc > 4:
        fatores.append(
            f"👨‍👩‍👧‍👦 **Família grande ({family_size_calc} pessoas)** — Grupos numerosos "
            "tinham dificuldade de se deslocar rapidamente e de encontrar espaço suficiente "
            "nos botes salva-vidas."
        )

    for fator in fatores:
        st.markdown(f"- {fator}")

    st.markdown("---")
    st.caption(
        "⚠️ **Nota metodológica:** Esta predição é baseada em padrões estatísticos de 1912 "
        "aprendidos pelo modelo. O Gradient Boosting tem acurácia de ~83% e AUC-ROC de ~0.89 "
        "no conjunto de teste. Resultados individuais podem divergir dos padrões de grupo."
    )


# ─────────────────────────────────────────────────────────────────────────────
# ── RODAPÉ ───────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="footer-note">
    🎓 Projeto Avaliativo P2 · Bacharelado em Inteligência Artificial · UNIMAR 2026<br>
    Grupo 12 · Felipe Traskini Rocha · Samuel Alves Vieira · Ivan Luís Gerônimo Del Roio<br>
    Modelo: Gradient Boosting Classifier &nbsp;|&nbsp; Dataset: Kaggle Titanic Competition
</div>
""", unsafe_allow_html=True)
