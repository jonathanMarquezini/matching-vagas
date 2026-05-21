import streamlit as st
import pandas as pd
import re
import pdfplumber
import unicodedata

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from io import BytesIO

# =========================================================
# 🎨 CONFIG VISUAL
# =========================================================
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

.info-box {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 15px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 🏢 HEADER
# =========================================================
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

# =========================================================
# 🔧 NORMALIZAR TEXTO
# =========================================================
def normalizar_texto(texto):

    if pd.isna(texto):
        return ""

    texto = str(texto)

    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(
        c for c in texto
        if unicodedata.category(c) != "Mn"
    )

    texto = texto.lower()

    texto = re.sub(r"\n", " ", texto)
    texto = re.sub(r"\r", " ", texto)
    texto = re.sub(r"\t", " ", texto)

    texto = re.sub(r"[^\w\s]", " ", texto)

    texto = re.sub(r"\s+", " ", texto)

    return texto.strip()

# =========================================================
# 🔧 EXTRAIR PDF
# =========================================================
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

# =========================================================
# 🔧 COLUNA SEGURA
# =========================================================
def get_coluna(df, coluna):

    if coluna and coluna in df.columns:
        return df[coluna].fillna("").astype(str)

    return pd.Series([""] * len(df))

# =========================================================
# 🔧 NORMALIZAR COLUNAS
# =========================================================
def normalizar_coluna(nome):

    nome = normalizar_texto(nome)
    nome = nome.replace(" ", "_")

    return nome

# =========================================================
# 🔍 DETECÇÃO FLEXÍVEL
# =========================================================
def encontrar_coluna(df, candidatos):

    mapa = {}

    for c in df.columns:
        mapa[normalizar_coluna(c)] = c

    # MATCH EXATO
    for candidato in candidatos:

        candidato_norm = normalizar_coluna(candidato)

        if candidato_norm in mapa:
            return mapa[candidato_norm]

    # MATCH PARCIAL
    for candidato in candidatos:

        candidato_norm = normalizar_coluna(candidato)

        for col_norm, col_real in mapa.items():

            if (
                candidato_norm in col_norm
                or
                col_norm in candidato_norm
            ):
                return col_real

    return None

# =========================================================
# 🔧 LIMPEZA DF
# =========================================================
def limpar_dataframe(df):

    df.columns = [
        normalizar_coluna(c)
        for c in df.columns
    ]

    return df

# =========================================================
# 🧠 PARSE ROL
# =========================================================
def parse_rol(rol):

    if pd.isna(rol):
        return {"tipo": "", "nivel": 0}

    rol = normalizar_texto(rol)

    partes = rol.split()

    if len(partes) == 0:
        return {"tipo": "", "nivel": 0}

    tipo = partes[0]

    mapa = {
        "i": 1,
        "ii": 2,
        "iii": 3,
        "iv": 4,
        "v": 5
    }

    nivel = 0

    if len(partes) > 1:
        nivel = mapa.get(partes[1], 0)

    return {
        "tipo": tipo,
        "nivel": nivel
    }

# =========================================================
# 🧠 VALIDAR ROL
# =========================================================
def rol_compativel(rol_colab, rol_vaga):

    if not rol_colab or not rol_vaga:
        return True

    c = parse_rol(rol_colab)
    v = parse_rol(rol_vaga)

    if c["tipo"] != v["tipo"]:
        return False

    return c["nivel"] == v["nivel"]

# =========================================================
# 💰 TAXA
# =========================================================
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

# =========================================================
# 🧠 BOOST SKILLS
# =========================================================
def tem_skill_direta(perfil, vaga):

    palavras = perfil.split()

    for palavra in palavras:

        if len(palavra) > 4 and palavra in vaga:
            return True

    return False

# =========================================================
# 📥 EXCEL
# =========================================================
def gerar_excel(df):

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Matching")

    output.seek(0)

    return output

# =========================================================
# 📂 UPLOAD
# =========================================================
st.subheader("📂 Upload das Bases")

col1, col2 = st.columns(2)

with col1:

    file_vagas = st.file_uploader(
        "Base de Vagas",
        type=["xlsx", "csv"]
    )

with col2:

    file_colab = st.file_uploader(
        "Base de Colaboradores",
        type=["xlsx", "csv"]
    )

# =========================================================
# 🚀 PROCESSAMENTO
# =========================================================
if file_vagas and file_colab:

    # =====================================================
    # 📥 LEITURA
    # =====================================================
    try:

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

    except Exception as e:

        st.error(f"Erro ao carregar arquivos: {e}")
        st.stop()

    # =====================================================
    # 🔧 LIMPAR DF
    # =====================================================
    vagas = limpar_dataframe(vagas)
    colab = limpar_dataframe(colab)

    # =====================================================
    # 🔍 DEBUG
    # =====================================================
    with st.expander("🛠️ Colunas identificadas"):

        st.write("### Base Colaboradores")
        st.write(colab.columns.tolist())

        st.write("### Base Vagas")
        st.write(vagas.columns.tolist())

    # =====================================================
    # 🔍 COLUNAS COLABORADORES
    # =====================================================
    coluna_nome = encontrar_coluna(colab, [
        "nome_colaborador",
        "nome",
        "colaborador",
        "funcionario",
        "employee",
        "empleado",
        "name"
    ])

    coluna_matricula = encontrar_coluna(colab, [
        "matricula",
        "matricula_colaborador",
        "employee_id",
        "codigo",
        "id"
    ])

    coluna_perfil = encontrar_coluna(colab, [
        "perfil",
        "nome_perfil",
        "cargo",
        "funcao",
        "role"
    ])

    coluna_descricao = encontrar_coluna(colab, [
        "descricao",
        "resumo",
        "summary",
        "perfil_resumo"
    ])

    coluna_rol_colab = encontrar_coluna(colab, [
        "roll",
        "rol",
        "role",
        "nivel"
    ])

    coluna_taxa_colab = encontrar_coluna(colab, [
        "taxa",
        "tasa",
        "rate"
    ])

    # =====================================================
    # 🔍 COLUNAS VAGAS
    # =====================================================
    coluna_rol_vaga = encontrar_coluna(vagas, [
        "rol_reporting",
        "rol",
        "role"
    ])

    coluna_taxa_vaga = encontrar_coluna(vagas, [
        "tasa_maxima_deseable",
        "taxa_maxima",
        "taxa",
        "rate"
    ])

    # =====================================================
    # ❌ NOME NÃO ENCONTRADO
    # =====================================================
    if not coluna_nome:

        st.error("❌ Não foi possível localizar a coluna de nome.")

        st.write("Colunas disponíveis:")
        st.write(colab.columns.tolist())

        st.stop()

    # =====================================================
    # 🧠 TEXTO VAGA
    # =====================================================
    vagas["texto"] = (

        get_coluna(vagas, encontrar_coluna(vagas, ["conocimientos_tecnicos"]))
        + " " +

        get_coluna(vagas, encontrar_coluna(vagas, ["perfil_solicitado_resumido"]))
        + " " +

        get_coluna(vagas, encontrar_coluna(vagas, ["perfil_solicitado_detallado"]))
        + " " +

        get_coluna(vagas, encontrar_coluna(vagas, ["conocimientos_funcionales"]))
        + " " +

        get_coluna(vagas, encontrar_coluna(vagas, ["perfil_profesional"]))
    )

    vagas["texto"] = vagas["texto"].apply(normalizar_texto)

    st.success("✅ Bases carregadas com sucesso")

    st.divider()

    # =====================================================
    # 🔎 BUSCA COLABORADOR
    # =====================================================
    st.subheader("🔎 Seleção de Colaborador")

    busca = st.text_input(
        "Digite nome ou matrícula"
    )

    filtro = colab.copy()

    # =====================================================
    # 🔍 BUSCA FLEXÍVEL
    # =====================================================
    if busca:

        busca_norm = normalizar_texto(busca)

        filtro_nome = (
            filtro[coluna_nome]
            .astype(str)
            .apply(normalizar_texto)
            .str.contains(busca_norm, na=False)
        )

        if coluna_matricula:

            filtro_matricula = (
                filtro[coluna_matricula]
                .astype(str)
                .apply(normalizar_texto)
                .str.contains(busca_norm, na=False)
            )

            filtro = filtro[
                filtro_nome | filtro_matricula
            ]

        else:

            filtro = filtro[
                filtro_nome
            ]

    # =====================================================
    # ❌ SEM RESULTADO
    # =====================================================
    if len(filtro) == 0:

        st.warning("Nenhum colaborador encontrado.")
        st.stop()

    # =====================================================
    # 🔧 REMOVER NAN
    # =====================================================
    filtro = filtro[
        filtro[coluna_nome]
        .notna()
    ]

    filtro = filtro[
        filtro[coluna_nome]
        .astype(str)
        .str.strip() != ""
    ]

    # =====================================================
    # 📋 LISTAGEM COLABORADORES
    # =====================================================
    opcoes = []

    for _, row in filtro.iterrows():

        nome = str(row[coluna_nome]).strip()

        matricula = ""

        if coluna_matricula:
            matricula = str(
                row[coluna_matricula]
            ).strip()

        texto = nome

        if matricula and matricula != "nan":
            texto += f" | {matricula}"

        opcoes.append(texto)

    selecionado = st.selectbox(
        "Selecione o colaborador",
        opcoes
    )

    nome_selecionado = selecionado.split("|")[0].strip()

    perfil_row = filtro[
        filtro[coluna_nome]
        .astype(str)
        .str.strip() == nome_selecionado
    ].iloc[0]

    # =====================================================
    # 📄 CV
    # =====================================================
    st.markdown(
        f"""
        <div class='cv-box'>
            <b>📄 Currículo de {nome_selecionado}</b><br>
            <span style='color:#8b949e;font-size:13px;'>
                Anexe o CV em PDF para enriquecer o matching.
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    cv_pdf = st.file_uploader(
        "Anexar CV PDF",
        type=["pdf"]
    )

    texto_cv = ""

    if cv_pdf:

        with st.spinner("📖 Lendo CV..."):
            texto_cv = extrair_texto_pdf(cv_pdf)

        if texto_cv.strip():

            st.success(
                f"✅ CV carregado com sucesso ({len(texto_cv.split())} palavras)"
            )

    # =====================================================
    # 🧠 TEXTO PERFIL
    # =====================================================
    descricao = ""

    if coluna_descricao:
        descricao = str(
            perfil_row.get(coluna_descricao, "")
        )

    perfil = ""

    if coluna_perfil:
        perfil = str(
            perfil_row.get(coluna_perfil, "")
        )

    perfil_texto = normalizar_texto(
        descricao
        + " "
        + perfil
        + " "
        + texto_cv
    )

    if not perfil_texto.strip():

        st.warning(
            "⚠️ Colaborador sem descrição e sem CV."
        )

    st.divider()

    # =====================================================
    # 🚀 MATCH
    # =====================================================
    if st.button("🚀 Buscar Vagas Compatíveis"):

        taxa_colab = tratar_taxa(
            perfil_row.get(coluna_taxa_colab, 0)
            if coluna_taxa_colab
            else 0
        )

        rol_colab = perfil_row.get(
            coluna_rol_colab,
            ""
        ) if coluna_rol_colab else ""

        vagas_filtradas = vagas.copy()

        # =================================================
        # 🔍 FILTRO ROL
        # =================================================
        if coluna_rol_vaga and rol_colab:

            vagas_filtradas = vagas_filtradas[
                vagas_filtradas[coluna_rol_vaga]
                .apply(
                    lambda x: rol_compativel(
                        rol_colab,
                        x
                    )
                )
            ]

        # =================================================
        # 🔍 FILTRO TAXA
        # =================================================
        if coluna_taxa_vaga and taxa_colab > 0:

            vagas_filtradas = vagas_filtradas[
                vagas_filtradas[coluna_taxa_vaga]
                .apply(tratar_taxa)
                >= taxa_colab
            ]

        # =================================================
        # ❌ SEM VAGAS
        # =================================================
        if len(vagas_filtradas) == 0:

            st.warning(
                "Nenhuma vaga encontrada pelos filtros."
            )

            st.stop()

        # =================================================
        # 🧠 TF-IDF
        # =================================================
        vectorizer = TfidfVectorizer()

        corpus = vagas_filtradas["texto"].tolist()

        corpus.append(perfil_texto)

        vectors = vectorizer.fit_transform(corpus)

        scores = cosine_similarity(
            vectors[-1],
            vectors[:-1]
        )[0]

        final_scores = []

        texto_cv_limpo = normalizar_texto(texto_cv)

        for i, vaga_texto in enumerate(vagas_filtradas["texto"]):

            score = scores[i]

            # BOOST PERFIL
            if perfil and normalizar_texto(perfil) in vaga_texto:
                score += 0.15

            # BOOST CV
            if texto_cv_limpo:

                if tem_skill_direta(
                    texto_cv_limpo,
                    vaga_texto
                ):
                    score += 0.20

            # BOOST SKILL
            if tem_skill_direta(
                perfil_texto,
                vaga_texto
            ):
                score += 0.10

            final_scores.append(
                round(score, 4)
            )

        vagas_filtradas["match"] = final_scores

        resultado = vagas_filtradas.sort_values(
            "match",
            ascending=False
        )

        resultado = resultado[
            resultado["match"] > 0.02
        ]

        # =================================================
        # 📊 MÉTRICAS
        # =================================================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Vagas encontradas",
                len(resultado)
            )

        with col2:

            media = 0

            if len(resultado) > 0:
                media = round(
                    resultado["match"].mean() * 100,
                    1
                )

            st.metric(
                "Score médio",
                f"{media}%"
            )

        with col3:

            st.metric(
                "CV utilizado",
                "✅ Sim" if texto_cv.strip() else "❌ Não"
            )

        st.divider()

        # =================================================
        # 📊 TABELA
        # =================================================
        colunas_exibir = [

            "proyecto",
            "solicitante",
            "necesidad",
            "rol_reporting",
            "tasa_maxima_deseable",
            "perfil_profesional",
            "perfil_solicitado_resumido",
            "match"
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

        # =================================================
        # 📋 DETALHAMENTO
        # =================================================
        st.subheader("📋 Detalhamento das Vagas")

        for _, row in resultado.head(20).iterrows():

            titulo = (
                f"{row.get('proyecto', 'Projeto')} "
                f"| Match: {round(row['match'] * 100, 2)}%"
            )

            with st.expander(titulo):

                st.markdown(f"""
### 📌 Informações da Vaga

**Projeto:** {row.get('proyecto', '-')}

**Solicitante:** {row.get('solicitante', '-')}

**Necessidade:** {row.get('necesidad', '-')}

**Rol:** {row.get('rol_reporting', '-')}

**Taxa Máxima:** {row.get('tasa_maxima_deseable', '-')}

**Score Match:** {round(row['match'] * 100, 2)}%
""")

                st.markdown("### 🧠 Perfil Profissional")
                st.write(row.get("perfil_profesional", "-"))

                st.markdown("### 📄 Perfil Resumido")
                st.write(row.get("perfil_solicitado_resumido", "-"))

                st.markdown("### 📑 Perfil Detalhado")
                st.write(row.get("perfil_solicitado_detallado", "-"))

                st.markdown("### ⚙️ Conhecimentos Funcionais")
                st.write(row.get("conocimientos_funcionales", "-"))

                st.markdown("### 💻 Conhecimentos Técnicos")
                st.write(row.get("conocimientos_tecnicos", "-"))

        # =================================================
        # 📥 DOWNLOAD
        # =================================================
        excel = gerar_excel(resultado)

        st.download_button(
            label="📥 Baixar Resultado Excel",
            data=excel,
            file_name=f"matching_{nome_selecionado}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =========================================================
# 🧾 FOOTER
# =========================================================
st.markdown(
    """
    <div class='footer-wrapper'>
        <div class='footer-box'>

            <div class='footer-title'>
                💼 Matching Inteligente de Vagas • v5.0
            </div>

            <div class='footer-description'>
                Plataforma corporativa de apoio estratégico para análise
                de aderência entre colaboradores e oportunidades internas,
                utilizando IA, Perfil Profissional e Currículo PDF.
            </div>

            <div class='footer-author'>
                Desenvolvido por <b>Jonathan Marquezini</b> • UGR
            </div>

        </div>
    </div>
    """,
    unsafe_allow_html=True
)
