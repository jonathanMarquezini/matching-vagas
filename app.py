import streamlit as st
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from io import BytesIO

# =========================
# 🎨 CONFIG VISUAL
# =========================
st.set_page_config(
    page_title="Matching Inteligente de Vagas",
    layout="wide"
)

st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: "Segoe UI", sans-serif;
}

.main {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: #e6edf3;
}

.stButton > button {
    background-color: #1f6feb;
    color: white;
    border-radius: 10px;
    border: none;
    height: 45px;
    font-weight: 600;
    width: 100%;
}

.stButton > button:hover {
    background-color: #388bfd;
    color: white;
}

.stDownloadButton > button {
    background-color: #238636 !important;
    color: white !important;
    border-radius: 10px;
    border: none;
    height: 45px;
    font-weight: 600;
    width: 100%;
}

.stDownloadButton > button:hover {
    background-color: #2ea043 !important;
}

div[data-baseweb="select"] > div {
    background-color: #1c1f26;
}

.stTextInput input {
    background-color: #1c1f26;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

.footer-container {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 30px;
    margin-top: 50px;
    margin-bottom: 20px;
    text-align: center;
}

.footer-title {
    color: #f0f6fc;
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 15px;
}

.footer-text {
    color: #8b949e;
    font-size: 14px;
    line-height: 1.8;
}

.header-company {
    text-align: right;
    font-size: 28px;
    font-weight: 700;
    color: white;
    white-space: nowrap;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 🏢 HEADER
# =========================
st.title("💼 Matching Inteligente de Vagas")

col1, col2 = st.columns([5, 2])

with col1:
    st.caption(
        "Plataforma corporativa para análise estratégica de aderência entre colaboradores e oportunidades internas."
    )

with col2:
    st.markdown(
        """
        <div class="header-company">
            🏢 Indra Group | Minsait
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# =========================
# 🔧 LIMPEZA TEXTO
# =========================
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

# =========================
# 🔧 EVITAR NaN
# =========================
def limpar_texto_modelo(texto):

    if pd.isna(texto):
        return ""

    return str(texto)

# =========================
# 🔧 COLUNA SEGURA
# =========================
def get_coluna(df, nome):

    if nome in df.columns:
        return df[nome].fillna("").astype(str)

    return pd.Series([""] * len(df))

# =========================
# 🧠 PARSE DE ROL
# =========================
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

# =========================
# 🧠 REGRA DE ROL
# =========================
def rol_compativel(rol_colab, rol_vaga):

    colab = parse_rol(rol_colab)
    vaga = parse_rol(rol_vaga)

    if colab["tipo"] != vaga["tipo"]:
        return False

    return colab["nivel"] >= vaga["nivel"]

# =========================
# 💰 TAXA
# =========================
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

# =========================
# 🧠 BOOST SKILL
# =========================
def tem_skill_direta(perfil, vaga_texto):

    palavras = perfil.split()

    for skill in palavras:

        if len(skill) > 4 and skill in vaga_texto:
            return True

    return False

# =========================
# 📥 GERAR EXCEL
# =========================
def gerar_excel(df):

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Matching")

    output.seek(0)

    return output

# =========================
# 📂 UPLOAD
# =========================
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

# =========================
# 🚀 PROCESSAMENTO
# =========================
if file_vagas and file_colab:

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

    # =========================
    # 🔁 REMOVER DUPLICADAS
    # =========================
    if "necesidad" in vagas.columns:
        vagas = vagas.drop_duplicates(subset=["necesidad"])

    # =========================
    # 🧠 TEXTO DA VAGA
    # =========================
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

    # =========================
    # 🔍 IDENTIFICAR COLUNAS
    # =========================
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

    # =========================
    # 🔍 BUSCA
    # =========================
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

    # =========================
    # 🧠 TEXTO COLABORADOR
    # =========================
    perfil_texto = limpar_texto(

        limpar_texto_modelo(
            perfil_row.get("descricao", "")
        )

    )

    st.divider()

    # =========================
    # 🚀 MATCH
    # =========================
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

        # =========================
        # ❌ SEM RESULTADO
        # =========================
        if len(vagas_filtradas) == 0:

            st.warning(
                "Nenhuma vaga compatível encontrada"
            )

            st.stop()

        # =========================
        # 🧠 IA MATCH
        # =========================
        vectorizer = TfidfVectorizer(
            stop_words=None
        )

        corpus = vagas_filtradas["texto"].tolist()

        corpus.append(perfil_texto)

        vectors = vectorizer.fit_transform(corpus)

        scores = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )[0]

        # =========================
        # 🔥 BOOST
        # =========================
        final_scores = []

        for i, row in enumerate(vagas_filtradas["texto"]):

            score = scores[i]

            if tem_skill_direta(perfil_texto, row):
                score += 0.15

            final_scores.append(round(score, 4))

        vagas_filtradas["match"] = final_scores

        # =========================
        # 📊 RESULTADO
        # =========================
        resultado = vagas_filtradas.sort_values(
            "match",
            ascending=False
        )

        resultado = resultado[
            resultado["match"] > 0.02
        ]

        st.metric(
            "Vagas encontradas",
            len(resultado)
        )

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

        # =========================
        # 📊 TABELA PRINCIPAL
        # =========================
        st.dataframe(
            resultado[colunas_exibir],
            use_container_width=True,
            height=700
        )

        st.divider()

        # =========================
        # 📂 DETALHAMENTO EXPANSÍVEL
        # =========================
        st.subheader("📋 Detalhamento das Vagas")

        for idx, row in resultado.head(20).iterrows():

            titulo = f"{row.get('proyecto', 'Projeto')} | Match: {round(row['match'] * 100, 2)}%"

            with st.expander(titulo):

                st.markdown(f"""
                ### 📌 Informações da Vaga

                **Projeto:** {row.get('proyecto', '-')}

                **Solicitante:** {row.get('solicitante', '-')}

                **Necessidade:** {row.get('necesidad', '-')}

                **Rol:** {row.get('rol reporting', '-')}

                **Taxa Máxima:** {row.get('tasa máxima deseable', '-')}

                **Score Match:** {round(row['match'] * 100, 2)}%
                """)

                st.markdown("### 🧠 Perfil Profissional")
                st.write(row.get("perfil profesional", "-"))

                st.markdown("### 📄 Perfil Resumido")
                st.write(row.get("perfil solicitado resumido", "-"))

                st.markdown("### 📑 Perfil Detalhado")
                st.write(row.get("perfil solicitado detallado", "-"))

                st.markdown("### ⚙️ Conhecimentos Funcionais")
                st.write(row.get("conocimientos funcionales", "-"))

                st.markdown("### 💻 Conhecimentos Técnicos")
                st.write(row.get("conocimientos tecnicos", "-"))

        # =========================
        # 📥 DOWNLOAD EXCEL
        # =========================
        excel_file = gerar_excel(
            resultado[colunas_exibir]
        )

        st.download_button(
            label="📥 Baixar Resultado em Excel",
            data=excel_file,
            file_name=f"matching_{selecionado}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =========================
# 🧾 FOOTER
# =========================
footer_html = "<div class='footer-container'><div class='footer-title'>Matching Inteligente de Vagas • v3.0</div><div class='footer-text'>Plataforma corporativa de apoio estratégico para análise de aderência entre colaboradores e oportunidades internas.<br><br>Desenvolvido por <b>Jonathan Marquezini</b> • UGR</div></div>"

st.markdown(footer_html, unsafe_allow_html=True)
