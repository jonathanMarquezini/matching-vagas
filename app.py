import streamlit as st
import pandas as pd
import re
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================
# 🎨 CONFIGURAÇÃO DA PÁGINA
# =====================================================
st.set_page_config(
    page_title="Matching Inteligente de Vagas",
    layout="wide",
    page_icon="💼"
)

# =====================================================
# 🎨 ESTILO VISUAL
# =====================================================
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background-color: #0b1120;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 95%;
}

/* =========================
HEADER
========================= */

.top-title {
    font-size: 54px;
    font-weight: 700;
    color: white;
    margin-bottom: 5px;
}

.sub-title {
    font-size: 16px;
    color: #94a3b8;
    margin-top: -5px;
}

.company-title {
    text-align: right;
    font-size: 34px;
    font-weight: 700;
    color: white;
    margin-top: 15px;
}

/* =========================
CARDS
========================= */

.info-card {
    background: linear-gradient(145deg, #111827, #0f172a);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
}

/* =========================
BOTÕES
========================= */

.stButton > button {
    width: 100%;
    height: 48px;
    border-radius: 10px;
    border: none;
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    color: white;
    font-size: 16px;
    font-weight: 600;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #3b82f6, #2563eb);
    color: white;
}

/* =========================
INPUTS
========================= */

.stTextInput input {
    background-color: #111827;
    border: 1px solid #334155;
    color: white;
    border-radius: 10px;
}

div[data-baseweb="select"] > div {
    background-color: #111827;
    border: 1px solid #334155;
    border-radius: 10px;
}

/* =========================
DATAFRAME
========================= */

[data-testid="stDataFrame"] {
    border: 1px solid #1e293b;
    border-radius: 12px;
    overflow: hidden;
}

/* =========================
METRIC
========================= */

[data-testid="metric-container"] {
    background: linear-gradient(145deg, #111827, #0f172a);
    border: 1px solid #1e293b;
    padding: 15px;
    border-radius: 14px;
}

/* =========================
EXPANDER
========================= */

.streamlit-expanderHeader {
    font-size: 16px;
    font-weight: 600;
}

/* =========================
FOOTER
========================= */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    padding-top: 20px;
    padding-bottom: 10px;
}

.footer-title {
    font-size: 15px;
    font-weight: 600;
    color: white;
}

hr {
    border-color: #1e293b;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 🏢 HEADER
# =====================================================
col1, col2 = st.columns([4, 1])

with col1:
    st.markdown("""
    <div class="top-title">
        💼 Matching Inteligente de Vagas
    </div>
    <div class="sub-title">
        Plataforma corporativa para análise estratégica de aderência entre colaboradores e oportunidades internas com base em perfil profissional, senioridade, taxa e competências técnicas.
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="company-title">
        🏢 Indra Group | Minsait
    </div>
    """, unsafe_allow_html=True)

st.divider()

# =====================================================
# 🔧 LIMPEZA TEXTO
# =====================================================
def limpar_texto(texto):

    if pd.isna(texto):
        return ""

    texto = str(texto).lower()

    texto = re.sub(r"\n", " ", texto)
    texto = re.sub(r"\r", " ", texto)
    texto = re.sub(r"\t", " ", texto)

    texto = re.sub(r"[^\w\s]", " ", texto)

    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()

# =====================================================
# 🔧 TEXTO MODELO
# =====================================================
def limpar_texto_modelo(texto):

    if pd.isna(texto):
        return ""

    return str(texto)

# =====================================================
# 🔧 COLUNA SEGURA
# =====================================================
def get_coluna(df, nome):

    if nome in df.columns:
        return df[nome].fillna("").astype(str)

    return pd.Series([""] * len(df))

# =====================================================
# 🧠 PARSE ROL
# =====================================================
def parse_rol(rol):

    if pd.isna(rol):
        return {"tipo": "", "nivel": 0}

    rol = str(rol).strip().lower()

    partes = rol.split()

    tipo = partes[0]

    mapa_nivel = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5
    }

    nivel = 0

    if len(partes) > 1:
        nivel = mapa_nivel.get(partes[1], 0)

    return {
        "tipo": tipo,
        "nivel": nivel
    }

# =====================================================
# 🧠 REGRAS ROL
# =====================================================
def rol_compativel(rol_colab, rol_vaga):

    colab = parse_rol(rol_colab)
    vaga = parse_rol(rol_vaga)

    if colab["tipo"] != vaga["tipo"]:
        return False

    return colab["nivel"] >= vaga["nivel"]

# =====================================================
# 💰 TAXA
# =====================================================
def tratar_taxa(valor):

    if pd.isna(valor):
        return 0

    valor = str(valor)

    valor = valor.replace(",", ".")

    valor = re.sub(r"[^0-9.]", "", valor)

    try:
        return float(valor)

    except:
        return 0

# =====================================================
# 🧠 BOOST SKILLS
# =====================================================
def tem_skill_direta(perfil, vaga_texto):

    palavras = perfil.split()

    for skill in palavras:

        if len(skill) > 4 and skill in vaga_texto:
            return True

    return False

# =====================================================
# 📥 GERAR EXCEL
# =====================================================
def gerar_excel(df):

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Matching")

    processed_data = output.getvalue()

    return processed_data

# =====================================================
# 📂 UPLOAD
# =====================================================
st.subheader("📂 Upload das Bases")

col1, col2 = st.columns(2)

with col1:
    file_vagas = st.file_uploader(
        "Base de Vagas",
        type=["csv", "xlsx"]
    )

with col2:
    file_colab = st.file_uploader(
        "Base de Colaboradores",
        type=["csv", "xlsx"]
    )

# =====================================================
# 🚀 PROCESSAMENTO
# =====================================================
if file_vagas and file_colab:

    # =====================================================
    # 📂 LEITURA
    # =====================================================
    vagas = (
        pd.read_csv(file_vagas)
        if file_vagas.name.endswith(".csv")
        else pd.read_excel(file_vagas)
    )

    colab = (
        pd.read_csv(file_colab)
        if file_colab.name.endswith(".csv")
        else pd.read_excel(file_colab)
    )

    vagas.columns = vagas.columns.str.strip().str.lower()
    colab.columns = colab.columns.str.strip().str.lower()

    # =====================================================
    # 🔁 REMOVER DUPLICADAS
    # =====================================================
    if "necesidad" in vagas.columns:
        vagas = vagas.drop_duplicates(subset=["necesidad"])

    # =====================================================
    # 🧠 TEXTO DA VAGA
    # =====================================================
    vagas["texto"] = (

        get_coluna(vagas, "conocimientos tecnicos")
        + " " +

        get_coluna(vagas, "perfil solicitado resumido")
        + " " +

        get_coluna(vagas, "perfil solicitado detallado")
        + " " +

        get_coluna(vagas, "conocimientos funcionales")
        + " " +

        get_coluna(vagas, "perfil profesional")

    )

    vagas["texto"] = vagas["texto"].apply(limpar_texto)

    st.success("✅ Bases carregadas com sucesso")

    st.divider()

    # =====================================================
    # 🔍 IDENTIFICAR COLUNAS
    # =====================================================
    coluna_nome = next((
        c for c in [
            "nome_colaborador",
            "nome",
            "colaborador",
            "funcionario"
        ]
        if c in colab.columns
    ), None)

    coluna_matricula = next((
        c for c in [
            "matricula_colaborador",
            "matricula"
        ]
        if c in colab.columns
    ), None)

    if not coluna_nome:
        st.error("❌ Coluna de nome não encontrada")
        st.stop()

    # =====================================================
    # 🔎 SELEÇÃO
    # =====================================================
    st.subheader("🔎 Seleção de Colaborador")

    busca = st.text_input(
        "Digite nome ou matrícula"
    )

    if busca:

        filtro_nome = colab[coluna_nome].astype(str).str.contains(
            busca,
            case=False,
            na=False
        )

        if coluna_matricula:

            filtro_matricula = colab[coluna_matricula].astype(str).str.contains(
                busca,
                na=False
            )

            filtro = colab[
                filtro_nome | filtro_matricula
            ]

        else:

            filtro = colab[filtro_nome]

    else:
        filtro = colab

    selecionado = st.selectbox(
        "Selecione o colaborador",
        filtro[coluna_nome]
    )

    perfil_row = colab[
        colab[coluna_nome] == selecionado
    ].iloc[0]

    # =====================================================
    # 🧠 TEXTO COLABORADOR
    # =====================================================
    perfil_texto = limpar_texto(

        limpar_texto_modelo(
            perfil_row.get("descricao", "")
        )

    )

    st.divider()

    # =====================================================
    # 🚀 BUSCAR MATCH
    # =====================================================
    if st.button("🚀 Buscar Vagas Compatíveis"):

        taxa_colab = tratar_taxa(
            perfil_row.get("taxa")
        )

        vagas_filtradas = vagas[
            vagas.apply(
                lambda row:

                rol_compativel(
                    perfil_row.get("roll"),
                    row.get("rol reporting")
                )

                and

                taxa_colab <= tratar_taxa(
                    row.get("tasa máxima deseable")
                ),

                axis=1
            )
        ].copy()

        # =====================================================
        # ❌ SEM RESULTADO
        # =====================================================
        if len(vagas_filtradas) == 0:

            st.warning(
                "Nenhuma vaga compatível encontrada"
            )

            st.stop()

        # =====================================================
        # 🧠 IA MATCH
        # =====================================================
        vectorizer = TfidfVectorizer(
            stop_words=None,
            ngram_range=(1,2)
        )

        corpus = vagas_filtradas["texto"].tolist()

        corpus.append(perfil_texto)

        vectors = vectorizer.fit_transform(corpus)

        scores = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )[0]

        # =====================================================
        # 🔥 BOOST
        # =====================================================
        final_scores = []

        for i, row in enumerate(vagas_filtradas["texto"]):

            score = scores[i]

            if tem_skill_direta(perfil_texto, row):
                score += 0.15

            final_scores.append(round(score, 4))

        vagas_filtradas["match"] = final_scores

        # =====================================================
        # 📊 RESULTADO
        # =====================================================
        resultado = vagas_filtradas.sort_values(
            "match",
            ascending=False
        )

        resultado = resultado[
            resultado["match"] > 0.02
        ]

        # =====================================================
        # 📈 MÉTRICAS
        # =====================================================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Vagas Encontradas",
                len(resultado)
            )

        with col2:
            st.metric(
                "Maior Match",
                f"{resultado['match'].max():.2f}"
            )

        with col3:
            st.metric(
                "Colaborador",
                selecionado
            )

        st.divider()

        # =====================================================
        # 📋 COLUNAS
        # =====================================================
        colunas_exibir = [

            "proyecto",
            "solicitante",
            "necesidad",
            "rol reporting",
            "tasa máxima deseable",
            "match",
            "perfil profesional",
            "perfil solicitado resumido",
            "perfil solicitado detallado",
            "conocimientos funcionales",
            "conocimientos tecnicos"

        ]

        colunas_exibir = [
            c for c in colunas_exibir
            if c in resultado.columns
        ]

        # =====================================================
        # 📥 DOWNLOAD EXCEL
        # =====================================================
        excel = gerar_excel(
            resultado[colunas_exibir]
        )

        st.download_button(
            label="📥 Baixar Resultado em Excel",
            data=excel,
            file_name=f"matching_{selecionado}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.divider()

        # =====================================================
        # 📊 RESULTADOS EXPANSÍVEIS
        # =====================================================
        st.subheader("📌 Vagas Compatíveis")

        for index, row in resultado.head(20).iterrows():

            titulo = f"""
            {row.get('necesidad', 'Sem ID')} 
            • Match: {round(row.get('match', 0), 2)}
            • Rol: {row.get('rol reporting', '-')}
            """

            with st.expander(titulo):

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    **Projeto:**  
                    {row.get('proyecto', '-')}

                    **Solicitante:**  
                    {row.get('solicitante', '-')}
                    """)

                with col2:
                    st.markdown(f"""
                    **Rol:**  
                    {row.get('rol reporting', '-')}

                    **Taxa Máxima:**  
                    {row.get('tasa máxima deseable', '-')}
                    """)

                with col3:
                    st.markdown(f"""
                    **Score Match:**  
                    {round(row.get('match', 0), 4)}
                    """)

                st.divider()

                if "perfil profesional" in row:
                    st.markdown("### 👨‍💼 Perfil Profissional")
                    st.write(row["perfil profesional"])

                if "perfil solicitado resumido" in row:
                    st.markdown("### 📋 Perfil Solicitado")
                    st.write(row["perfil solicitado resumido"])

                if "perfil solicitado detallado" in row:
                    st.markdown("### 🧠 Descrição Completa")
                    st.write(row["perfil solicitado detallado"])

                if "conocimientos funcionales" in row:
                    st.markdown("### ⚙️ Conhecimentos Funcionais")
                    st.write(row["conocimientos funcionales"])

                if "conocimientos tecnicos" in row:
                    st.markdown("### 💻 Conhecimentos Técnicos")
                    st.write(row["conocimientos tecnicos"])

# =====================================================
# 🧾 FOOTER
# =====================================================
st.markdown("""
<hr>

<div class="footer">

    <div class="footer-title">
        Matching Inteligente de Vagas • v3.0
    </div>

    Plataforma interna de apoio estratégico para análise de aderência entre colaboradores e oportunidades corporativas.

    <br><br>

    Desenvolvido por <b>Jonathan Marquezini</b> • UGR

</div>
""", unsafe_allow_html=True)
