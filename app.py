import streamlit as st
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# 🎨 CONFIG VISUAL
# =========================
st.set_page_config(
    page_title="Matching Inteligente de Vagas",
    layout="wide"
)

st.markdown("""
<style>

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

.metric-card {
    background-color: #161b22;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30363d;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 🏢 HEADER
# =========================
st.title("💼 Matching Inteligente de Vagas")

col1, col2 = st.columns([4, 1])

with col1:
    st.caption(
        "Análise baseada em Descrição Profissional • Rol • Taxa • Contexto da vaga"
    )

with col2:
    st.markdown("## 🏢 Minsait | Indra")

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

    # tipos diferentes não podem
    if colab["tipo"] != vaga["tipo"]:
        return False

    # nível colaborador precisa ser >= vaga
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

    # =========================
    # 📂 LEITURA
    # =========================
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

        st.dataframe(
            resultado[colunas_exibir],
            use_container_width=True,
            height=700
        )

    st.divider()

    # =========================
    # 📊 BASE COMPLETA
    # =========================
    if st.button("📊 Gerar Base Completa"):

        vagas_base = vagas.copy()

        vaga_para = {
            i: [] for i in range(len(vagas_base))
        }

        for _, colab_row in colab.iterrows():

            nome = colab_row[coluna_nome]

            perfil = limpar_texto(
                limpar_texto_modelo(
                    colab_row.get("descricao", "")
                )
            )

            taxa = tratar_taxa(
                colab_row.get("taxa")
            )

            vectorizer = TfidfVectorizer()

            corpus = vagas_base["texto"].tolist()

            corpus.append(perfil)

            vectors = vectorizer.fit_transform(corpus)

            scores = cosine_similarity(
                vectors[-1],
                vectors[:-1]
            )[0]

            for i, row in vagas_base.iterrows():

                score = scores[i]

                if tem_skill_direta(perfil, row["texto"]):
                    score += 0.15

                if (

                    rol_compativel(
                        colab_row.get("roll"),
                        row.get("rol reporting")
                    )

                    and

                    taxa <= tratar_taxa(
                        row.get("tasa máxima deseable")
                    )

                    and

                    score >= 0.05

                ):

                    vaga_para[i].append(nome)

        vagas_base["vaga_para"] = [

            ", ".join(vaga_para[i])
            if vaga_para[i]
            else "Sem match"

            for i in range(len(vagas_base))
        ]

        st.dataframe(
            vagas_base,
            use_container_width=True,
            height=700
        )

        csv = vagas_base.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Baixar CSV",
            data=csv,
            file_name="vagas_match.csv",
            mime="text/csv"
        )

# =========================
# 🧾 FOOTER
# =========================
st.markdown("""
<hr style="margin-top:50px; margin-bottom:10px;">

<div style="text-align:center; color:gray; font-size:14px;">

    Desenvolvido por <b>Jonathan Marquezini</b> • UGR

    <br>

    <span style="font-size:12px;">
        Matching Inteligente de Vagas v2.0
    </span>

</div>
""", unsafe_allow_html=True)
