import streamlit as st
import pandas as pd
import re
import pdfplumber
import unicodedata

from io import BytesIO

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================================================
# 🎨 CONFIG
# =========================================================
st.set_page_config(
    page_title="Matching Inteligente de Vagas",
    layout="wide"
)

# =========================================================
# 🎨 CSS
# =========================================================
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

.alert-box {
    background-color: #2d1b1b;
    border: 1px solid #ff6b6b;
    border-radius: 12px;
    padding: 15px;
    margin-top: 15px;
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
# 🔧 FUNÇÕES
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
# 🔍 DETECÇÃO FLEXÍVEL DE COLUNA
# =========================================================
def encontrar_coluna(df, candidatos):

    mapa = {}

    for c in df.columns:
        mapa[normalizar_texto(c)] = c

    # MATCH EXATO
    for candidato in candidatos:

        candidato_norm = normalizar_texto(candidato)

        if candidato_norm in mapa:
            return mapa[candidato_norm]

    # MATCH PARCIAL
    for candidato in candidatos:

        candidato_norm = normalizar_texto(candidato)

        for col_norm, col_real in mapa.items():

            if candidato_norm in col_norm:
                return col_real

    return None

# =========================================================
# 🔧 PEGAR COLUNA SEGURA
# =========================================================
def get_coluna(df, nome):

    if nome and nome in df.columns:
        return df[nome].fillna("").astype(str)

    return pd.Series([""] * len(df))

# =========================================================
# 📄 EXTRAIR PDF
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
# 💰 TRATAR TAXA
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
# 🧠 PARSE ROL
# =========================================================
def parse_rol(rol):

    if pd.isna(rol):
        return {
            "tipo": "",
            "nivel": 0
        }

    rol = normalizar_texto(rol)

    partes = rol.split()

    if not partes:
        return {
            "tipo": "",
            "nivel": 0
        }

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
# 🧠 REGRA ROL
# =========================================================
def rol_compativel(rol_colab, rol_vaga):

    colab = parse_rol(rol_colab)

    vaga = parse_rol(rol_vaga)

    # precisa ser mesmo tipo
    if colab["tipo"] != vaga["tipo"]:
        return False

    # colaborador pode pegar nível igual ou abaixo
    return colab["nivel"] >= vaga["nivel"]

# =========================================================
# 🧠 BOOST SKILL
# =========================================================
def tem_skill_direta(perfil, vaga):

    palavras = perfil.split()

    for palavra in palavras:

        if len(palavra) > 4 and palavra in vaga:
            return True

    return False

# =========================================================
# 📥 GERAR EXCEL
# =========================================================
def gerar_excel(df):

    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Matching"
        )

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
        type=["csv", "xlsx"]
    )

with col2:

    file_colab = st.file_uploader(
        "Base de Colaboradores",
        type=["csv", "xlsx"]
    )

# =========================================================
# 🚀 PROCESSAMENTO
# =========================================================
if file_vagas and file_colab:

    try:

        # =====================================================
        # 📥 LEITURA
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

        vagas.columns = vagas.columns.str.strip()
        colab.columns = colab.columns.str.strip()

        # =====================================================
        # 🔍 DETECTAR COLUNAS
        # =====================================================
        coluna_nome = encontrar_coluna(colab, [
            "nome colaborador",
            "nome_colaborador",
            "nome",
            "employee",
            "funcionario",
            "colaborador",
            "nombre"
        ])

        coluna_matricula = encontrar_coluna(colab, [
            "matricula",
            "matricula colaborador",
            "employee id",
            "codigo",
            "id"
        ])

        coluna_perfil = encontrar_coluna(colab, [
            "perfil",
            "cargo",
            "funcao",
            "role",
            "position",
            "nome perfil"
        ])

        coluna_descricao = encontrar_coluna(colab, [
            "descricao",
            "description",
            "resumo",
            "summary"
        ])

        coluna_rol_colab = encontrar_coluna(colab, [
            "roll",
            "rol",
            "role"
        ])

        coluna_rol_vaga = encontrar_coluna(vagas, [
            "rol reporting",
            "rol",
            "role"
        ])

        coluna_taxa_colab = encontrar_coluna(colab, [
            "taxa",
            "tasa",
            "rate"
        ])

        coluna_taxa_vaga = encontrar_coluna(vagas, [
            "tasa máxima deseable",
            "tasa maxima deseable",
            "taxa maxima",
            "rate"
        ])

        # =====================================================
        # ❌ ERRO NOME
        # =====================================================
        if not coluna_nome:

            st.error("❌ Não foi possível localizar a coluna de nome.")

            st.write("Colunas encontradas na base:")

            st.write(colab.columns.tolist())

            st.stop()

        # =====================================================
        # 🧹 LIMPEZA NOME
        # =====================================================
        colab[coluna_nome] = (
            colab[coluna_nome]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        colab = colab[
            colab[coluna_nome] != ""
        ]

        colab = colab[
            colab[coluna_nome].str.lower() != "nan"
        ]

        # =====================================================
        # 🧠 TEXTO VAGAS
        # =====================================================
        vagas["texto"] = (

            get_coluna(vagas, encontrar_coluna(vagas, ["conocimientos tecnicos"]))
            + " " +

            get_coluna(vagas, encontrar_coluna(vagas, ["perfil solicitado resumido"]))
            + " " +

            get_coluna(vagas, encontrar_coluna(vagas, ["perfil solicitado detallado"]))
            + " " +

            get_coluna(vagas, encontrar_coluna(vagas, ["conocimientos funcionales"]))
            + " " +

            get_coluna(vagas, encontrar_coluna(vagas, ["perfil profesional"]))

        )

        vagas["texto"] = vagas["texto"].apply(normalizar_texto)

        # =====================================================
        # 🔁 REMOVER DUPLICADAS
        # =====================================================
        necessidade_col = encontrar_coluna(vagas, ["necesidad"])

        if necessidade_col:
            vagas = vagas.drop_duplicates(subset=[necessidade_col])

        st.success("✅ Bases carregadas com sucesso")

        st.divider()

        # =====================================================
        # 🔎 BUSCA
        # =====================================================
        st.subheader("🔎 Seleção de Colaborador")

        busca = st.text_input(
            "Digite nome ou matrícula"
        )

        filtro = colab.copy()

        if busca:

            busca = busca.strip()

            filtro_nome = filtro[coluna_nome].astype(str).str.contains(
                busca,
                case=False,
                na=False
            )

            if coluna_matricula:

                filtro_matricula = filtro[coluna_matricula].astype(str).str.contains(
                    busca,
                    case=False,
                    na=False
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
        # 🧍 SELECTBOX
        # =====================================================
        lista_colaboradores = sorted(
            filtro[coluna_nome]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selecionado = st.selectbox(
            "Selecione o colaborador",
            lista_colaboradores
        )

        perfil_filtrado = filtro[
            filtro[coluna_nome].astype(str) == str(selecionado)
        ]

        if len(perfil_filtrado) == 0:

            st.error("Erro ao localizar colaborador.")

            st.stop()

        perfil_row = perfil_filtrado.iloc[0]

        # =====================================================
        # 📄 CV
        # =====================================================
        st.markdown(
            f"""
            <div class='cv-box'>
                <b>📄 Currículo de {selecionado}</b><br>
                <span style='color:#8b949e;font-size:13px;'>
                    Anexe o CV em PDF para enriquecer o matching com skills, experiências e formações.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        cv_pdf = st.file_uploader(
            "Anexar CV em PDF",
            type=["pdf"],
            key=f"cv_{selecionado}"
        )

        texto_cv = ""

        if cv_pdf:

            with st.spinner("📖 Extraindo informações do CV..."):

                texto_cv = extrair_texto_pdf(cv_pdf)

            if texto_cv.strip():

                st.success(
                    f"✅ CV carregado com sucesso • {len(texto_cv.split())} palavras extraídas"
                )

        # =====================================================
        # 🧠 PERFIL TEXTO
        # =====================================================
        descricao = ""

        if coluna_descricao:
            descricao = str(
                perfil_row.get(coluna_descricao, "")
            )

        nome_perfil = ""

        if coluna_perfil:
            nome_perfil = str(
                perfil_row.get(coluna_perfil, "")
            )

        perfil_texto = normalizar_texto(
            descricao + " " +
            nome_perfil + " " +
            texto_cv
        )

        if not perfil_texto.strip():

            st.warning(
                "⚠️ Colaborador sem descrição e sem CV. O match pode ficar fraco."
            )

        st.divider()

        # =====================================================
        # 🚀 MATCH
        # =====================================================
        if st.button("🚀 Buscar Vagas Compatíveis"):

            with st.spinner("🔍 Calculando matching..."):

                taxa_colab = 0

                if coluna_taxa_colab:
                    taxa_colab = tratar_taxa(
                        perfil_row.get(coluna_taxa_colab)
                    )

                # =================================================
                # 🎯 FILTRO
                # =================================================
                def validar_vaga(row):

                    # FILTRO ROL
                    if coluna_rol_colab and coluna_rol_vaga:

                        if not rol_compativel(
                            perfil_row.get(coluna_rol_colab),
                            row.get(coluna_rol_vaga)
                        ):
                            return False

                    # FILTRO TAXA
                    if coluna_taxa_vaga:

                        taxa_vaga = tratar_taxa(
                            row.get(coluna_taxa_vaga)
                        )

                        if taxa_vaga > 0:

                            if taxa_colab > taxa_vaga:
                                return False

                    return True

                vagas_filtradas = vagas[
                    vagas.apply(validar_vaga, axis=1)
                ].copy()

                # =================================================
                # ❌ SEM RESULTADO
                # =================================================
                if len(vagas_filtradas) == 0:

                    st.warning(
                        "Nenhuma vaga compatível encontrada."
                    )

                    st.stop()

                # =================================================
                # 🧠 IA
                # =================================================
                vectorizer = TfidfVectorizer()

                corpus = vagas_filtradas["texto"].tolist()

                corpus.append(perfil_texto)

                vectors = vectorizer.fit_transform(corpus)

                scores = cosine_similarity(
                    vectors[-1],
                    vectors[:-1]
                )[0]

                # =================================================
                # 🔥 BOOSTS
                # =================================================
                texto_cv_limpo = normalizar_texto(texto_cv)

                final_scores = []

                for i, vaga_texto in enumerate(vagas_filtradas["texto"]):

                    score = scores[i]

                    # BOOST PERFIL
                    if tem_skill_direta(
                        perfil_texto,
                        vaga_texto
                    ):
                        score += 0.10

                    # BOOST CARGO
                    if (
                        nome_perfil
                        and normalizar_texto(nome_perfil) in vaga_texto
                    ):
                        score += 0.15

                    # BOOST CV
                    if (
                        texto_cv_limpo
                        and tem_skill_direta(
                            texto_cv_limpo,
                            vaga_texto
                        )
                    ):
                        score += 0.25

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
                # 📊 KPIs
                # =================================================
                score_medio = 0

                if len(resultado) > 0:
                    score_medio = round(
                        resultado["match"].mean() * 100,
                        1
                    )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Vagas encontradas",
                        len(resultado)
                    )

                with c2:
                    st.metric(
                        "Score médio",
                        f"{score_medio}%"
                    )

                with c3:
                    st.metric(
                        "CV Utilizado",
                        "✅ Sim" if texto_cv else "❌ Não"
                    )

                # =================================================
                # 📋 COLUNAS
                # =================================================
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

                # =================================================
                # 📊 TABELA
                # =================================================
                st.dataframe(
                    resultado[colunas_exibir],
                    use_container_width=True,
                    height=700
                )

                st.divider()

                # =================================================
                # 📂 DETALHAMENTO
                # =================================================
                st.subheader("📋 Detalhamento das Vagas")

                for idx, row in resultado.head(20).iterrows():

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

                # =================================================
                # 📥 DOWNLOAD
                # =================================================
                excel_file = gerar_excel(
                    resultado[colunas_exibir]
                )

                st.download_button(
                    label="📥 Baixar Resultado em Excel",
                    data=excel_file,
                    file_name=f"matching_{selecionado}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    except Exception as e:

        st.markdown(
            f"""
            <div class='alert-box'>
                <b>❌ Erro identificado:</b><br><br>
                {str(e)}
            </div>
            """,
            unsafe_allow_html=True
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
                utilizando IA, Skills, Perfil Profissional e Currículo PDF.
            </div>

            <div class='footer-author'>
                Desenvolvido por <b>Jonathan Marquezini</b> • UGR
            </div>

        </div>
    </div>
    """,
    unsafe_allow_html=True
)
