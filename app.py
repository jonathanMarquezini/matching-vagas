import streamlit as st
import pandas as pd
import re
import pdfplumber
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

</style>
""", unsafe_allow_html=True)

# =========================
# 🛑 STOPWORDS PT + ES
# =========================
STOPWORDS = set([
    # Português
    "de", "da", "do", "das", "dos", "em", "na", "no", "nas", "nos",
    "para", "por", "com", "sem", "sob", "sobre", "entre", "até",
    "que", "se", "ou", "e", "a", "o", "as", "os", "um", "uma",
    "uns", "umas", "ao", "aos", "à", "às", "me", "te", "se", "nos",
    "vos", "lhe", "lhes", "meu", "minha", "seu", "sua", "seus", "suas",
    "este", "esta", "estes", "estas", "esse", "essa", "esses", "essas",
    "aquele", "aquela", "aqueles", "aquelas", "isso", "isto", "aquilo",
    "ele", "ela", "eles", "elas", "eu", "tu", "você", "nós", "vocês",
    "foi", "ser", "estar", "tem", "ter", "são", "há", "vai", "pode",
    "deve", "mais", "mas", "bem", "já", "não", "sim", "também",
    "quando", "onde", "como", "pelo", "pela", "pelos", "pelas",
    "num", "numa", "nuns", "numas", "desse", "dessa", "deste", "desta",
    "nesse", "nessa", "neste", "nesta", "ano", "anos", "mês", "meses",
    # Espanhol
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "en", "con", "por", "para", "sin", "sobre", "entre", "hasta",
    "que", "se", "si", "no", "ni", "y", "o", "pero", "sino",
    "del", "al", "lo", "le", "les", "me", "te", "nos", "vos",
    "su", "sus", "mi", "mis", "tu", "tus", "este", "esta", "estos",
    "estas", "ese", "esa", "esos", "esas", "aquel", "aquella",
    "él", "ella", "ellos", "ellas", "yo", "tú", "usted", "nosotros",
    "es", "son", "fue", "ser", "estar", "tiene", "tener", "hay",
    "más", "también", "cuando", "donde", "como", "muy", "bien",
    "año", "años", "mes", "meses", "nivel", "tipo", "conocimiento",
])

# =========================
# 🔧 SEÇÕES RELEVANTES DO CV
# =========================
SECOES_RELEVANTES = [
    "conhecimento",
    "conocimiento",
    "tecnológico",
    "tecnologico",
    "funcional",
    "experiência",
    "experiencia",
    "perfil",
    "especialização",
    "especializacion",
    "habilidade",
    "habilidad",
    "formação",
    "formacion",
    "angular",
    "javascript",
    "python",
    "java",
    "react",
    "typescript",
    "desenvolvimento",
    "desarrollo",
    "software",
    "sistema",
    "banco",
    "base",
    "dados",
    "datos",
    "cloud",
    "devops",
    "agile",
    "scrum",
]

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
    st.markdown("<div class='header-company'>🏢 Indra Group | Minsait</div>", unsafe_allow_html=True)

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
# 🔧 REMOVER STOPWORDS
# =========================
def remover_stopwords(texto):

    palavras = texto.split()

    filtradas = [
        p for p in palavras
        if p not in STOPWORDS and len(p) > 2
    ]

    return " ".join(filtradas)

# =========================
# 🔧 EVITAR NaN
# =========================
def limpar_texto_modelo(texto):

    if pd.isna(texto):
        return ""

    return str(texto)

# =========================
# 🔧 EXTRAIR TEXTO PDF
# com foco nas seções relevantes
# =========================
def extrair_texto_pdf(arquivo_pdf):

    texto_total = ""
    texto_relevante = ""

    try:

        with pdfplumber.open(arquivo_pdf) as pdf:

            for pagina in pdf.pages:

                conteudo = pagina.extract_text()

                if conteudo:
                    texto_total += " " + conteudo

    except Exception as e:
        st.warning(f"Erro ao ler PDF: {e}")
        return ""

    # Filtra linhas com termos relevantes
    linhas = texto_total.split("\n")

    for linha in linhas:

        linha_lower = linha.lower()

        if any(termo in linha_lower for termo in SECOES_RELEVANTES):
            texto_relevante += " " + linha

    # Se extraiu pouco conteúdo relevante, usa o texto completo
    if len(texto_relevante.split()) < 30:
        return texto_total

    return texto_relevante

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

    if not partes:
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
# 🧠 BOOST SKILL — conta
# quantas skills batem
# =========================
def contar_skills_diretas(perfil, vaga_texto):

    palavras = set(perfil.split())
    count = 0

    for skill in palavras:

        if len(skill) > 4 and skill in vaga_texto:
            count += 1

    return count

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
    # com remoção de stopwords
    # =========================
    vagas["texto_raw"] = (
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

    vagas["texto"] = vagas["texto_raw"].apply(
        lambda t: remover_stopwords(limpar_texto(t))
    )

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

    coluna_nome_perfil = next((
        c for c in [
            "nome_perfil",
            "perfil",
            "cargo",
            "funcao"
        ]
        if c in colab.columns
    ), None)

    # =========================
    # 🔍 BUSCA COLABORADOR
    # =========================
    st.subheader("🔎 Seleção de Colaborador")

    busca = st.text_input("Digite nome ou matrícula")

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

            filtro = colab[filtro_nome | filtro_matricula]

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
    # 📄 UPLOAD CV — vinculado
    # ao colaborador selecionado
    # =========================
    st.markdown(
        "<div class='cv-box'><b>📄 Currículo de " + str(selecionado) +
        " (Opcional)</b><br><span style='color:#8b949e;font-size:13px;'>" +
        "Anexe o CV em PDF para enriquecer o matching com skills, experiências e formações." +
        "</span></div>",
        unsafe_allow_html=True
    )

    cv_pdf = st.file_uploader(
        "Anexar CV em PDF",
        type=["pdf"],
        key=f"cv_{selecionado}"
    )

    texto_cv = ""

    if cv_pdf:

        with st.spinner("📖 Extraindo informações relevantes do CV..."):
            texto_cv = extrair_texto_pdf(cv_pdf)

        if texto_cv.strip():
            palavras_cv = len(limpar_texto(texto_cv).split())
            st.success(f"✅ CV de {selecionado} carregado — {palavras_cv} termos relevantes extraídos")
        else:
            st.warning("⚠️ Não foi possível extrair texto do PDF enviado.")

    # =========================
    # 🧠 TEXTO COLABORADOR
    # com remoção de stopwords
    # =========================
    descricao_colab = limpar_texto_modelo(perfil_row.get("descricao", ""))

    nome_perfil = ""

    if coluna_nome_perfil:
        nome_perfil = limpar_texto_modelo(
            perfil_row.get(coluna_nome_perfil, "")
        )

    texto_cv_limpo = remover_stopwords(limpar_texto(texto_cv))

    perfil_texto = remover_stopwords(
        limpar_texto(descricao_colab + " " + nome_perfil + " " + texto_cv)
    )

    # =========================
    # ⚠️ AVISO PERFIL VAZIO
    # =========================
    if not perfil_texto.strip():
        st.warning("⚠️ Este colaborador não possui descrição de perfil nem CV anexado. O match pode ter baixa precisão.")

    st.divider()

    # =========================
    # 🚀 MATCH
    # =========================
    if st.button("🚀 Buscar Vagas Compatíveis"):

        taxa_colab = tratar_taxa(perfil_row.get("taxa"))

        with st.spinner("🔍 Calculando compatibilidade das vagas..."):

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
                st.warning("Nenhuma vaga compatível encontrada")
                st.stop()

            # =========================
            # 🧠 IA MATCH — TF-IDF
            # com stopwords removidas
            # =========================
            vectorizer = TfidfVectorizer(
                stop_words=None,     # já removemos manualmente
                ngram_range=(1, 2),  # bigrams capturam "angular js", "spring boot"
                min_df=1,
                sublinear_tf=True    # suaviza termos muito frequentes
            )

            corpus = vagas_filtradas["texto"].tolist()
            corpus.append(perfil_texto)

            vectors = vectorizer.fit_transform(corpus)

            scores_tfidf = cosine_similarity(
                vectors[-1],
                vectors[:-1]
            )[0]

            # =========================
            # 🔥 BOOST MULTIPLICATIVO
            # não soma diretamente ao
            # score — multiplica por um
            # fator proporcional às skills
            # =========================
            final_scores = []

            for i, vaga_texto in enumerate(vagas_filtradas["texto"]):

                score_base = scores_tfidf[i]

                # Conta skills que batem entre perfil e vaga
                skills_desc = contar_skills_diretas(
                    remover_stopwords(limpar_texto(descricao_colab + " " + nome_perfil)),
                    vaga_texto
                )

                skills_cv = contar_skills_diretas(
                    texto_cv_limpo,
                    vaga_texto
                ) if texto_cv_limpo else 0

                # Fator multiplicativo baseado em skills encontradas
                # Cada skill adiciona 3% ao score, cap em 30%
                fator_skills_desc = min(1 + (skills_desc * 0.03), 1.30)
                fator_skills_cv   = min(1 + (skills_cv   * 0.04), 1.40)

                # Boost adicional se cargo/perfil bate diretamente na vaga
                fator_perfil = 1.10 if (
                    nome_perfil and
                    len(nome_perfil) > 3 and
                    nome_perfil.lower() in vaga_texto
                ) else 1.0

                score_final = score_base * fator_skills_desc * fator_skills_cv * fator_perfil

                # Normaliza para não ultrapassar 1.0
                score_final = min(round(score_final, 4), 1.0)

                final_scores.append(score_final)

            vagas_filtradas["match_raw"] = final_scores

            # Formata como percentual para exibição
            vagas_filtradas["match"] = vagas_filtradas["match_raw"].apply(
                lambda x: f"{round(x * 100, 1)}%"
            )

        # =========================
        # 📊 RESULTADO
        # =========================
        resultado = vagas_filtradas.sort_values("match_raw", ascending=False)

        resultado = resultado[
            resultado["match_raw"] >= 0.02
        ]

        if len(resultado) == 0:
            st.warning("Nenhuma vaga compatível encontrada.")
            st.stop()

        score_medio = round(resultado["match_raw"].mean() * 100, 1)
        score_top   = round(resultado["match_raw"].iloc[0] * 100, 1)

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)

        with col_m1:
            st.metric("Vagas encontradas", len(resultado))

        with col_m2:
            st.metric("Melhor match", f"{score_top}%")

        with col_m3:
            st.metric("Score médio", f"{score_medio}%")

        with col_m4:
            cv_status = "✅ Sim" if texto_cv.strip() else "❌ Não"
            st.metric("CV no match", cv_status)

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
        # 📂 DETALHAMENTO
        # =========================
        st.subheader("📋 Detalhamento das Vagas")

        for idx, row in resultado.head(20).iterrows():

            rol = row.get("rol reporting", "")
            rol_str = f"| {rol} " if rol and str(rol).strip() else ""

            titulo = (
                f"{row.get('proyecto', 'Projeto')} "
                f"{rol_str}"
                f"| Match: {row.get('match', '-')}"
            )

            with st.expander(titulo, expanded=False):

                st.markdown(f"""
### 📌 Informações da Vaga

**Projeto:** {row.get('proyecto', '-')}

**Solicitante:** {row.get('solicitante', '-')}

**Necessidade:** {row.get('necesidad', '-')}

**Rol:** {row.get('rol reporting', '-')}

**Taxa Máxima:** {row.get('tasa máxima deseable', '-')}

**Score Match:** {row.get('match', '-')}
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
        excel_file = gerar_excel(resultado[colunas_exibir])

        st.download_button(
            label="📥 Baixar Resultado em Excel",
            data=excel_file,
            file_name=f"matching_{selecionado}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# =========================
# 🧾 FOOTER
# =========================
st.markdown("<div class='footer-wrapper'><div class='footer-box'><div class='footer-title'>💼 Matching Inteligente de Vagas • v5.0</div><div class='footer-description'>Plataforma corporativa de apoio estratégico para análise de aderência entre colaboradores e oportunidades internas, utilizando IA, Skills, Perfil Profissional e Currículo PDF.</div><div class='footer-author'>Desenvolvido por <b>Jonathan Marquezini</b> • UGR</div></div></div>", unsafe_allow_html=True)
