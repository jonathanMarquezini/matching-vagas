import streamlit as st
import pandas as pd
import re
import pdfplumber
import unicodedata

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
    border: 1px solid #30363d;
}

div[data-testid="stExpander"] {
    border-radius: 12px !important;
    border: 1px solid #30363d !important;
    margin-bottom: 10px !important;
    overflow: hidden;
}

.header-company {
    text-align: right;
    font-size: 24px;
    font-weight: 700;
    color: white;
    white-space: nowrap;
    margin-top: 12px;
}

.cv-box {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 20px;
    margin-top: 10px;
    margin-bottom: 15px;
}

.footer-wrapper {
    margin-top: 60px;
    margin-bottom: 20px;
}

.footer-box {
    background: linear-gradient(135deg, #161b22 0%, #1c2330 100%);
    border: 1px solid #30363d;
    border-radius: 18px;
    padding: 35px 25px;
    text-align: center;
}

.footer-title {
    color: #f0f6fc;
    font-size: 24px;
    font-weight: 700;
    margin-bottom: 18px;
}

.footer-description {
    color: #8b949e;
    font-size: 15px;
    line-height: 1.8;
    margin-bottom: 18px;
}

.footer-author {
    color: #c9d1d9;
    font-size: 14px;
}

.footer-author b {
    color: white;
}

.col-hint-box {
    background-color: #1c1f26;
    border: 1px solid #f0883e55;
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 12px;
    color: #f0883e;
    font-size: 14px;
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
        "<div class='header-company'>🏢 Indra Group | Minsait</div>",
        unsafe_allow_html=True
    )

st.divider()

# =========================
# 🔧 NORMALIZAR COLUNAS
# =========================
def normalizar_col(nome):

    nome = str(nome).strip().lower()

    nome = unicodedata.normalize("NFD", nome)

    nome = "".join(
        c for c in nome
        if unicodedata.category(c) != "Mn"
    )

    nome = re.sub(r"[^\w\s]", "", nome)

    nome = re.sub(r"\s+", "_", nome)

    return nome

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

    texto = str(texto).strip()

    if texto.lower() == "nan":
        return ""

    return texto

# =========================
# 🔧 EXTRAIR TEXTO PDF
# =========================
def extrair_texto_pdf(arquivo_pdf):

    texto = ""

    try:

        with pdfplumber.open(arquivo_pdf) as pdf:

            for pagina in pdf.pages:

                conteudo = pagina.extract_text()

                if conteudo:
                    texto += " " + conteudo

    except Exception as e:

        st.warning(f"Erro ao ler PDF: {e}")

    return texto

# =========================
# 🔧 COLUNA SEGURA
# =========================
def get_coluna(df, nome):

    if nome in df.columns:
        return df[nome].fillna("").astype(str)

    return pd.Series([""] * len(df))

# =========================
# 🔍 ENCONTRAR COLUNA
# =========================
def encontrar_coluna(df, candidatos):

    cols_norm = {
        normalizar_col(c): c
        for c in df.columns
    }

    # Match exato
    for cand in candidatos:

        cand_norm = normalizar_col(cand)

        if cand_norm in cols_norm:
            return cols_norm[cand_norm]

    # Match parcial
    for cand in candidatos:

        cand_norm = normalizar_col(cand)

        for col_norm, col_real in cols_norm.items():

            if (
                cand_norm in col_norm
                or
                col_norm in cand_norm
            ):
                return col_real

    return None

# =========================
# 🧠 PARSE DE ROL
# =========================
def parse_rol(rol):

    if pd.isna(rol):
        return {"tipo": "", "nivel": 0}

    rol = str(rol).strip().lower()

    rol = re.sub(r"\s+", " ", rol)

    partes = rol.split()

    if len(partes) == 0:
        return {"tipo": "", "nivel": 0}

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
# Agora só permite MESMO nível
# =========================
def rol_compativel(rol_colab, rol_vaga):

    colab = parse_rol(rol_colab)

    vaga = parse_rol(rol_vaga)

    if colab["tipo"] != vaga["tipo"]:
        return False

    return colab["nivel"] == vaga["nivel"]

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
# 📂 UPLOAD BASES
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
    # 📖 LEITURA DOS ARQUIVOS
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

    # =========================
    # 🔧 NORMALIZA COLUNAS
    # =========================
    vagas.columns = [
        normalizar_col(c)
        for c in vagas.columns
    ]

    colab.columns = [
        normalizar_col(c)
        for c in colab.columns
    ]

    # =========================
    # 🔧 REMOVE NaN
    # =========================
    colab = colab.fillna("")
    vagas = vagas.fillna("")

    # =========================
    # 🔍 DEBUG COLUNAS
    # =========================
    with st.expander("🛠️ Debug — Colunas Detectadas"):

        st.write("### Base Colaboradores")
        st.write(colab.columns.tolist())

        st.write("### Base Vagas")
        st.write(vagas.columns.tolist())

    # =========================
    # 🔁 REMOVER DUPLICADAS
    # =========================
    if "necesidad" in vagas.columns:

        vagas = vagas.drop_duplicates(
            subset=["necesidad"]
        )

    # =========================
    # 🧠 TEXTO DA VAGA
    # =========================
    vagas["texto"] = (

        get_coluna(vagas, "conocimientos_tecnicos")
        + " " +

        get_coluna(vagas, "perfil_solicitado_resumido")
        + " " +

        get_coluna(vagas, "perfil_solicitado_detallado")
        + " " +

        get_coluna(vagas, "conocimientos_funcionales")
        + " " +

        get_coluna(vagas, "perfil_profesional")

    )

    vagas["texto"] = vagas["texto"].apply(
        limpar_texto
    )

    st.success("✅ Bases carregadas com sucesso")

    st.divider()

    # =========================
    # 🔍 IDENTIFICAR COLUNAS
    # =========================
    coluna_nome = encontrar_coluna(colab, [
        "nome_colaborador",
        "nome",
        "colaborador",
        "funcionario",
        "nombre",
        "empleado",
        "employee",
        "name"
    ])

    coluna_matricula = encontrar_coluna(colab, [
        "matricula",
        "matricula_colaborador",
        "employee_id",
        "codigo",
        "id"
    ])

    coluna_nome_perfil = encontrar_coluna(colab, [
        "nome_perfil",
        "perfil",
        "cargo",
        "funcao",
        "role"
    ])

    coluna_descricao = encontrar_coluna(colab, [
        "descricao",
        "description",
        "resumo",
        "summary"
    ])

    # =========================
    # ❌ SEM NOME
    # =========================
    if not coluna_nome:

        st.error(
            "❌ Não foi possível identificar automaticamente a coluna de nome."
        )

        st.write("Colunas encontradas:")

        st.write(colab.columns.tolist())

        coluna_nome = st.selectbox(
            "Selecione manualmente:",
            colab.columns.tolist()
        )

    # =========================
    # 🔧 LIMPA NOMES
    # =========================
    colab[coluna_nome] = (
        colab[coluna_nome]
        .astype(str)
        .str.strip()
    )

    colab = colab[
        colab[coluna_nome] != ""
    ]

    colab = colab[
        colab[coluna_nome].str.lower() != "nan"
    ]

    # =========================
    # 🔍 BUSCA
    # =========================
    st.subheader("🔎 Seleção de Colaborador")

    busca = st.text_input(
        "Digite nome ou matrícula"
    )

    if busca:

        filtro_nome = (
            colab[coluna_nome]
            .astype(str)
            .str.contains(
                busca,
                case=False,
                na=False
            )
        )

        if coluna_matricula:

            filtro_matricula = (
                colab[coluna_matricula]
                .astype(str)
                .str.contains(
                    busca,
                    case=False,
                    na=False
                )
            )

            filtro = colab[
                filtro_nome | filtro_matricula
            ]

        else:

            filtro = colab[
                filtro_nome
            ]

    else:

        filtro = colab

    # =========================
    # ❌ SEM RESULTADO
    # =========================
    if filtro.empty:

        st.warning(
            "Nenhum colaborador encontrado."
        )

        st.stop()

    nomes_validos = (
        filtro[coluna_nome]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    nomes_validos = sorted(nomes_validos)

    if len(nomes_validos) == 0:

        st.error(
            "Nenhum nome válido encontrado na coluna selecionada."
        )

        st.stop()

    selecionado = st.selectbox(
        "Selecione o colaborador",
        nomes_validos
    )

    # =========================
    # 🔧 PERFIL ROW
    # =========================
    perfil_filtrado = colab[
        colab[coluna_nome] == selecionado
    ]

    if perfil_filtrado.empty:

        st.error(
            "Erro ao localizar colaborador selecionado."
        )

        st.stop()

    perfil_row = perfil_filtrado.iloc[0]

    # =========================
    # 📄 UPLOAD CV
    # =========================
    st.markdown(f"""
    <div class="cv-box">

        <h4 style="margin-top:0;">
            📄 Currículo de {selecionado}
        </h4>

        <p style="color:#8b949e;">
            Você pode anexar um PDF para melhorar a precisão do matching.
        </p>

    </div>
    """, unsafe_allow_html=True)

    cv_pdf = st.file_uploader(
        "Anexar CV em PDF",
        type=["pdf"]
    )

    texto_cv = ""

    if cv_pdf:

        with st.spinner(
            "Extraindo informações do CV..."
        ):

            texto_cv = extrair_texto_pdf(cv_pdf)

        if texto_cv.strip():

            st.success(
                f"✅ CV carregado com sucesso ({len(texto_cv.split())} palavras extraídas)"
            )

        else:

            st.warning(
                "⚠️ Não foi possível extrair texto do PDF."
            )

    # =========================
    # 🧠 TEXTO COLABORADOR
    # =========================
    descricao_colab = ""

    if coluna_descricao:

        descricao_colab = limpar_texto_modelo(
            perfil_row.get(coluna_descricao, "")
        )

    nome_perfil = ""

    if coluna_nome_perfil:

        nome_perfil = limpar_texto_modelo(
            perfil_row.get(coluna_nome_perfil, "")
        )

    perfil_texto = limpar_texto(
        descricao_colab
        + " "
        + nome_perfil
        + " "
        + texto_cv
    )

    st.divider()

    # =========================
    # 🚀 MATCH
    # =========================
    if st.button("🚀 Buscar Vagas Compatíveis"):

        coluna_rol_colab = encontrar_coluna(colab, [
            "roll",
            "rol",
            "role",
            "nivel"
        ])

        coluna_rol_vaga = encontrar_coluna(vagas, [
            "rol_reporting",
            "rol",
            "role",
            "nivel"
        ])

        coluna_taxa_colab = encontrar_coluna(colab, [
            "taxa",
            "tasa",
            "rate"
        ])

        coluna_taxa_vaga = encontrar_coluna(vagas, [
            "tasa_maxima_deseable",
            "taxa_maxima",
            "tasa",
            "rate"
        ])

        taxa_colab = tratar_taxa(
            perfil_row.get(coluna_taxa_colab, 0)
        )

        # =========================
        # 🔍 FILTRO VAGAS
        # =========================
        def filtro_vaga(row):

            # 🔒 ROL
            if (
                coluna_rol_colab
                and
                coluna_rol_vaga
            ):

                if not rol_compativel(
                    perfil_row.get(coluna_rol_colab),
                    row.get(coluna_rol_vaga)
                ):
                    return False

            # 🔒 TAXA
            if coluna_taxa_vaga:

                taxa_max = tratar_taxa(
                    row.get(coluna_taxa_vaga)
                )

                if (
                    taxa_max > 0
                    and
                    taxa_colab > taxa_max
                ):
                    return False

            return True

        vagas_filtradas = vagas[
            vagas.apply(
                filtro_vaga,
                axis=1
            )
        ].copy()

        # =========================
        # ❌ SEM RESULTADO
        # =========================
        if len(vagas_filtradas) == 0:

            st.warning(
                "Nenhuma vaga compatível encontrada."
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

        vectors = vectorizer.fit_transform(
            corpus
        )

        scores = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )[0]

        # =========================
        # 🔥 BOOSTS
        # =========================
        texto_cv_limpo = limpar_texto(
            texto_cv
        )

        final_scores = []

        for i, row in enumerate(vagas_filtradas["texto"]):

            score = scores[i]

            # Skill direta
            if tem_skill_direta(
                perfil_texto,
                row
            ):
                score += 0.10

            # Nome perfil
            if (
                nome_perfil
                and
                nome_perfil.lower() in row
            ):
                score += 0.15

            # CV
            if (
                texto_cv_limpo
                and
                tem_skill_direta(
                    texto_cv_limpo,
                    row
                )
            ):
                score += 0.20

            final_scores.append(
                round(score, 4)
            )

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

        score_medio = 0

        if len(resultado) > 0:

            score_medio = round(
                resultado["match"].mean() * 100,
                1
            )

        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.metric(
                "Vagas encontradas",
                len(resultado)
            )

        with col_m2:
            st.metric(
                "Score médio",
                f"{score_medio}%"
            )

        with col_m3:

            cv_status = (
                "✅ Sim"
                if texto_cv.strip()
                else "❌ Não"
            )

            st.metric(
                "CV utilizado",
                cv_status
            )

        # =========================
        # 📊 TABELA
        # =========================
        colunas_exibir = [

            "proyecto",
            "solicitante",
            "necesidad",
            "rol_reporting",
            "tasa_maxima_deseable",
            "match",

            "perfil_profesional",
            "perfil_solicitado_resumido",
            "perfil_solicitado_detallado",
            "conocimientos_funcionales",
            "conocimientos_tecnicos"

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
        # 📋 DETALHAMENTO
        # =========================
        st.subheader(
            "📋 Detalhamento das Vagas"
        )

        for idx, row in resultado.head(20).iterrows():

            titulo = (
                f"{row.get('proyecto', 'Projeto')} "
                f"| Match: {round(row['match'] * 100, 2)}%"
            )

            with st.expander(
                titulo,
                expanded=False
            ):

                st.markdown(f"""
### 📌 Informações da Vaga

**Projeto:** {row.get('proyecto', '-')}

**Solicitante:** {row.get('solicitante', '-')}

**Necessidade:** {row.get('necesidad', '-')}

**Rol:** {row.get('rol_reporting', '-')}

**Taxa Máxima:** {row.get('tasa_maxima_deseable', '-')}

**Score Match:** {round(row['match'] * 100, 2)}%
""")

                st.markdown(
                    "### 🧠 Perfil Profissional"
                )

                st.write(
                    row.get(
                        "perfil_profesional",
                        "-"
                    )
                )

                st.markdown(
                    "### 📄 Perfil Resumido"
                )

                st.write(
                    row.get(
                        "perfil_solicitado_resumido",
                        "-"
                    )
                )

                st.markdown(
                    "### 📑 Perfil Detalhado"
                )

                st.write(
                    row.get(
                        "perfil_solicitado_detallado",
                        "-"
                    )
                )

                st.markdown(
                    "### ⚙️ Conhecimentos Funcionais"
                )

                st.write(
                    row.get(
                        "conocimientos_funcionales",
                        "-"
                    )
                )

                st.markdown(
                    "### 💻 Conhecimentos Técnicos"
                )

                st.write(
                    row.get(
                        "conocimientos_tecnicos",
                        "-"
                    )
                )

        # =========================
        # 📥 DOWNLOAD
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
st.markdown("""
<div class='footer-wrapper'>

    <div class='footer-box'>

        <div class='footer-title'>
            💼 Matching Inteligente de Vagas • v5.0
        </div>

        <div class='footer-description'>

            Plataforma corporativa de apoio estratégico para análise
            de aderência entre colaboradores e oportunidades internas,
            utilizando IA, Skills, Perfil Profissional e Currículo PDF.

        </div>

        <div class='footer-author'>

            Desenvolvido por <b>Jonathan Marquezini</b> • UGR

        </div>

    </div>

</div>
""", unsafe_allow_html=True)
