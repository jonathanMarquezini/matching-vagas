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
# 🔧 EVITAR NaN
# =========================
def limpar_texto_modelo(texto):

    if pd.isna(texto):
        return ""

    return str(texto)

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
# 🔧 DETECÇÃO FLEXÍVEL DE COLUNAS
# =========================
def normalizar_col(nome):

    nome = str(nome).strip().lower()
    nome = unicodedata.normalize("NFD", nome)
    nome = "".join(c for c in nome if unicodedata.category(c) != "Mn")
    nome = re.sub(r"[\s_]+", "_", nome)

    return nome

def encontrar_coluna(df, candidatos):

    cols_norm = {normalizar_col(c): c for c in df.columns}

    # 1. Match exato normalizado
    for cand in candidatos:
        cand_norm = normalizar_col(cand)
        if cand_norm in cols_norm:
            return cols_norm[cand_norm]

    # 2. Match parcial
    for cand in candidatos:
        cand_norm = normalizar_col(cand)
        for col_norm, col_real in cols_norm.items():
            if cand_norm in col_norm or col_norm in cand_norm:
                return col_real

    return None

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
# 🕓 INICIALIZAR HISTÓRICO
# =========================
if "resultado_cache" not in st.session_state:
    st.session_state.resultado_cache = None

if "colunas_cache" not in st.session_state:
    st.session_state.colunas_cache = None

if "selecionado_cache" not in st.session_state:
    st.session_state.selecionado_cache = None

if "texto_cv_cache" not in st.session_state:
    st.session_state.texto_cv_cache = ""

if "col_obs_cache" not in st.session_state:
    st.session_state.col_obs_cache = None

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

    # =========================
    # 🔧 EXTRAIR CAMPO "Outros"
    # de Observaciones Necesidad
    # =========================
    col_obs = next((c for c in vagas.columns if "observaciones" in c and "necesidad" in c), None)
    if col_obs is None:
        col_obs = next((c for c in vagas.columns if "observaciones" in c), None)

    def extrair_outros(texto):
        if pd.isna(texto) or str(texto).strip() == "":
            return "-"
        texto = str(texto)
        match = re.search(r"(?i)(?:outros|otros)\s*:\s*(.+)", texto, re.DOTALL)
        if match:
            return match.group(1).strip()
        return "-"

    if col_obs:
        vagas["outros"] = vagas[col_obs].apply(extrair_outros)
    else:
        vagas["outros"] = "-"

    st.success("✅ Bases carregadas com sucesso")

    st.divider()

    # =========================
    # 🔍 IDENTIFICAR COLUNAS — DETECÇÃO FLEXÍVEL
    # =========================
    coluna_nome = encontrar_coluna(colab, [
        "nome_colaborador",
        "nome colaborador",
        "nome",
        "colaborador",
        "funcionario",
        "nombre_colaborador",
        "nombre colaborador",
        "nombre",
        "employee",
        "name",
        "empleado"
    ])

    coluna_matricula = encontrar_coluna(colab, [
        "matricula_colaborador",
        "matricula colaborador",
        "matricula",
        "employee_id",
        "cod_colaborador",
        "codigo",
        "id"
    ])

    # =========================
    # ❌ COLUNA NOME NÃO ENCONTRADA
    # =========================
    if not coluna_nome:

        st.error("❌ Coluna de nome não encontrada na Base de Colaboradores.")

        colunas_disponiveis = ", ".join([f"`{c}`" for c in colab.columns.tolist()])

        st.markdown(
            f"<div class='col-hint-box'>"
            f"⚠️ <b>Colunas detectadas na sua planilha:</b><br><br>{colunas_disponiveis}<br><br>"
            f"Selecione manualmente qual coluna representa o <b>nome do colaborador</b>:"
            f"</div>",
            unsafe_allow_html=True
        )

        coluna_nome = st.selectbox(
            "Qual coluna é o nome do colaborador?",
            options=colab.columns.tolist()
        )

        if not coluna_nome:
            st.stop()

    coluna_nome_perfil = encontrar_coluna(colab, [
        "nome_perfil",
        "perfil",
        "cargo",
        "funcao",
        "perfil_colaborador",
        "role",
        "position",
        "puesto"
    ])

    coluna_descricao = encontrar_coluna(colab, [
        "descricao",
        "descricao_colaborador",
        "description",
        "resumo",
        "summary",
        "perfil_resumo"
    ])

    coluna_rol_colab  = encontrar_coluna(colab, ["roll", "rol", "role", "nivel"])
    coluna_taxa_colab = encontrar_coluna(colab, ["taxa", "tasa", "rate", "valor"])
    coluna_rol_vaga   = encontrar_coluna(vagas, ["rol reporting", "rol", "role", "nivel"])
    coluna_taxa_vaga  = encontrar_coluna(vagas, [
        "tasa maxima deseable",
        "tasa máxima deseable",
        "taxa maxima",
        "taxa",
        "tasa",
        "rate_max"
    ])

    # =========================
    # 🔍 BUSCA COLABORADOR
    # =========================
    st.subheader("🔎 Seleção de Colaborador")

    busca = st.text_input("Digite nome ou matrícula")

    # Garante string limpa na coluna de nome
    colab[coluna_nome] = colab[coluna_nome].fillna("").astype(str).str.strip()

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

            filtro_df = colab[filtro_nome | filtro_matricula]

        else:
            filtro_df = colab[filtro_nome]

    else:
        filtro_df = colab

    if filtro_df.empty:
        st.warning("Nenhum colaborador encontrado com esse filtro.")
        st.stop()

    selecionado = st.selectbox(
        "Selecione o colaborador",
        filtro_df[coluna_nome].tolist()
    )

    # ✅ BUSCA SEGURA — evita IndexError
    linhas = filtro_df[filtro_df[coluna_nome] == selecionado]

    if linhas.empty:
        st.error("Colaborador não encontrado. Tente novamente.")
        st.stop()

    perfil_row = linhas.iloc[0]

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

        with st.spinner("📖 Extraindo informações do CV..."):
            texto_cv = extrair_texto_pdf(cv_pdf)

        if texto_cv.strip():
            st.success(f"✅ CV de {selecionado} carregado — {len(texto_cv.split())} palavras extraídas")
        else:
            st.warning("⚠️ Não foi possível extrair texto do PDF enviado.")

    # =========================
    # 🧠 TEXTO COLABORADOR
    # =========================
    descricao_colab = ""

    if coluna_descricao:
        descricao_colab = limpar_texto_modelo(perfil_row.get(coluna_descricao, ""))

    nome_perfil = ""

    if coluna_nome_perfil:
        nome_perfil = limpar_texto_modelo(
            perfil_row.get(coluna_nome_perfil, "")
        )

    perfil_texto = limpar_texto(
        descricao_colab + " " + nome_perfil + " " + texto_cv
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

        taxa_colab = tratar_taxa(
            perfil_row.get(coluna_taxa_colab)
        ) if coluna_taxa_colab else 0

        with st.spinner("🔍 Calculando compatibilidade das vagas..."):

            # Normaliza variações comuns de cargo
            def normalizar_cargo_filtro(texto):
                t = limpar_texto(texto)
                t = t.replace("full stack", "fullstack")
                t = t.replace("front end", "frontend")
                t = t.replace("back end",  "backend")
                return t

            # =========================
            # 🗺️ MAPA DE ÁREAS RELACIONADAS
            # Cada grupo define cargos que pertencem
            # à mesma área e podem ser intercambiáveis
            # =========================
            AREAS_RELACIONADAS = [
                # Design / UX / Produto
                {"ux", "ui", "design", "produto", "frontend", "front", "usabilidade",
                 "figma", "prototipo", "experiencia", "interface", "wireframe"},
                # Desenvolvimento Frontend / Mobile
                {"frontend", "front", "react", "angular", "vue", "mobile", "ios",
                 "android", "flutter", "web", "javascript", "typescript", "html", "css"},
                # Desenvolvimento Backend
                {"backend", "back", "java", "python", "node", "dotnet", "net",
                 "api", "microsservicos", "php", "ruby", "golang", "kotlin", "spring"},
                # Fullstack
                {"fullstack", "full", "frontend", "backend", "front", "back",
                 "web", "react", "angular", "node", "java", "python"},
                # Dados / Analytics / BI / ETL
                {"dados", "data", "analytics", "bi", "business", "intelligence",
                 "sql", "dba", "banco", "database", "engenheiro", "etl", "teradata",
                 "powercenter", "informatica", "microstrategy", "datawarehouse", "dw",
                 "bigdata", "spark", "hadoop", "databricks", "pipeline"},
                # Oracle / Database
                {"oracle", "sql", "plsql", "database", "dba", "banco", "teradata",
                 "mysql", "postgres", "sqlserver"},
                # DevOps / Infra / Cloud
                {"devops", "infra", "cloud", "aws", "azure", "gcp", "kubernetes",
                 "docker", "sre", "plataforma", "linux", "ansible", "terraform"},
                # RPA / Automacao
                {"rpa", "automacao", "automation", "uipath", "blueprism",
                 "powerautomate", "robotica"},
                # Suporte / Service Desk
                {"suporte", "support", "servicedesk", "helpdesk", "atendimento",
                 "infraestrutura", "sustentacao", "incidente"},
                # Gestao / PMO / Scrum
                {"pmo", "projeto", "gestao", "coordenacao", "coordenador", "gerente",
                 "manager", "scrum", "agil", "master", "product", "owner", "lideranca"},
                # Arquitetura de Solucoes / Sistemas
                {"arquiteto", "arquitetura", "solucao", "sistemas", "solucoes",
                 "enterprise", "microservicos", "integracao", "middleware"},
                # Analista Funcional / Negócios / BA
                {"funcional", "negocios", "negocio", "requisitos", "processos",
                 "produto", "business", "analista", "analyst", "funcional",
                 "levantamento", "mapeamento"},
                # SAP
                {"sap", "abap", "fiori", "hana", "erp", "s4hana"},
                # Seguranca
                {"seguranca", "security", "cyber", "pentest", "soc", "ciberseguranca"},
                # Administrativo / RH
                {"administrativo", "admin", "recursos", "humanos", "rh",
                 "financeiro", "contabil", "backoffice"},
                # Qualidade / Testes
                {"qualidade", "teste", "testes", "quality", "qa", "automacao",
                 "selenium", "cypress", "jira", "testador"},
                # .NET especifico
                {"net", "dotnet", "csharp", "aspnet", "azure", "microsoft"},
                # PHP / Web especifico
                {"php", "laravel", "symfony", "web", "wordpress", "drupal"},
            ]

            # Expande os termos do cargo com termos das áreas relacionadas
            termos_cargo_filtro = []
            termos_expandidos   = set()

            termos_genericos_global = {
                "analista", "desenvolvedor", "especialista", "consultor",
                "coordenador", "gerente", "manager", "senior", "pleno",
                "junior", "lead", "tecnico", "engenheiro", "arquiteto"
            }

            if nome_perfil:
                cargo_norm = normalizar_cargo_filtro(nome_perfil)
                termos_cargo_filtro = [t for t in cargo_norm.split() if len(t) >= 2]

                # Para cada termo do cargo, encontra o grupo de área
                for termo in termos_cargo_filtro:
                    for grupo in AREAS_RELACIONADAS:
                        if termo in grupo:
                            termos_expandidos.update(grupo)

                # Verifica se o cargo é genérico demais
                # (só tem termos genéricos ou siglas curtas ambíguas)
                termos_uteis_cargo = [
                    t for t in termos_cargo_filtro
                    if t not in termos_genericos_global and len(t) >= 2
                ]
                cargo_generico = len(termos_uteis_cargo) == 0

                # Se cargo genérico, extrai termos relevantes da descrição
                if cargo_generico and descricao_colab:
                    desc_norm = normalizar_cargo_filtro(descricao_colab)
                    desc_tokens = [t for t in desc_norm.split()
                                   if len(t) >= 3 and t not in termos_genericos_global]

                    # Filtra apenas tokens que aparecem em algum grupo de área
                    for token in desc_tokens:
                        for grupo in AREAS_RELACIONADAS:
                            if token in grupo:
                                termos_expandidos.update(grupo)
                                termos_cargo_filtro.append(token)

                # Se ainda não achou grupo, usa termos originais
                if not termos_expandidos:
                    termos_expandidos = set(termos_cargo_filtro)

            def filtro_vaga(row):

                # ── Filtro de Rol ────────────────────────────────────────
                if coluna_rol_colab and coluna_rol_vaga:
                    if not rol_compativel(
                        perfil_row.get(coluna_rol_colab),
                        row.get(coluna_rol_vaga)
                    ):
                        return False

                # ── Filtro de Taxa ───────────────────────────────────────
                if coluna_taxa_vaga:
                    taxa_max = tratar_taxa(row.get(coluna_taxa_vaga))
                    if taxa_max > 0 and taxa_colab > taxa_max:
                        return False

                # ── Filtro de Área / Cargo ───────────────────────────────
                # Usa os termos expandidos (área relacionada) para permitir
                # vagas de áreas próximas. Ex: UX → passa Frontend.
                # Mas bloqueia áreas completamente diferentes.
                # Ex: UX → não passa DBA, SAP, DevOps.
                if termos_expandidos and termos_cargo_filtro:
                    perfil_res = normalizar_cargo_filtro(
                        str(row.get("perfil solicitado resumido", ""))
                        + " " +
                        str(row.get("perfil profesional", ""))
                    )

                    # Verifica hits nos termos ORIGINAIS do cargo (não expandidos)
                    # Exige pelo menos 30% dos termos originais batendo
                    # Ex: "Desenvolvedor RPA" → ["desenvolvedor", "rpa"]
                    # → precisa de pelo menos 1 dos 2 termos específicos
                    # → só "desenvolvedor" sem "rpa" não passa se cargo tiver 2+ termos
                    hits_originais = sum(1 for t in termos_cargo_filtro if t in perfil_res)
                    min_hits = max(1, round(len(termos_cargo_filtro) * 0.30))

                    # Se cargo tem termos específicos de área (não genéricos),
                    # exige que pelo menos um deles bata
                    termos_genericos = {"analista", "desenvolvedor", "especialista",
                                        "consultor", "coordenador", "gerente", "senior",
                                        "pleno", "junior", "lead", "tecnico"}
                    termos_especificos = [t for t in termos_cargo_filtro
                                          if t not in termos_genericos and len(t) >= 2]

                    if termos_especificos:
                        # Tem termos específicos de área (ex: "rpa", "java", "ux")
                        # exige que pelo menos 1 deles bata
                        hits_especificos = sum(1 for t in termos_especificos if t in perfil_res)
                        # Também aceita via termos expandidos da área
                        hits_expandidos  = sum(1 for t in termos_expandidos if t in perfil_res)
                        if hits_especificos == 0 and hits_expandidos < 2:
                            return False
                    else:
                        # Cargo só tem termos genéricos — usa hits mínimos
                        if hits_originais < min_hits:
                            return False

                return True

            vagas_filtradas = vagas[
                vagas.apply(filtro_vaga, axis=1)
            ].copy()

            # =========================
            # ❌ SEM RESULTADO
            # Se não encontrou com todos os filtros,
            # relaxa apenas o filtro de cargo/área
            # mas mantém Rol e Taxa obrigatórios
            # =========================
            if len(vagas_filtradas) == 0:

                def filtro_vaga_relaxado(row):
                    if coluna_rol_colab and coluna_rol_vaga:
                        if not rol_compativel(
                            perfil_row.get(coluna_rol_colab),
                            row.get(coluna_rol_vaga)
                        ):
                            return False
                    if coluna_taxa_vaga:
                        taxa_max = tratar_taxa(row.get(coluna_taxa_vaga))
                        if taxa_max > 0 and taxa_colab > taxa_max:
                            return False
                    return True

                vagas_filtradas = vagas[
                    vagas.apply(filtro_vaga_relaxado, axis=1)
                ].copy()

                if len(vagas_filtradas) == 0:
                    st.warning("Nenhuma vaga compatível encontrada para este colaborador.")
                    st.stop()

            # =========================
            # 🧠 IA MATCH — TF-IDF
            # =========================
            vectorizer = TfidfVectorizer(stop_words=None)

            corpus = vagas_filtradas["texto"].tolist()
            corpus.append(perfil_texto if perfil_texto.strip() else "sem perfil")

            vectors = vectorizer.fit_transform(corpus)

            scores = cosine_similarity(
                vectors[-1],
                vectors[:-1]
            )[0]

            # =========================
            # ⚖️ PESOS BASE — média ponderada
            # =========================
            PESO_TFIDF = 0.40
            PESO_CARGO = 0.25
            PESO_ROL   = 0.15
            PESO_TAXA  = 0.10
            PESO_CV    = 0.10

            # Verifica se colaborador tem perfil suficiente para TF-IDF
            tem_perfil_suficiente = len(perfil_texto.split()) >= 10

            # Pré-processa termos do cargo — normaliza variações comuns
            def normalizar_cargo(texto):
                t = limpar_texto(texto)
                t = re.sub(r"full\s+stack", "fullstack", t)
                t = re.sub(r"front\s+end",  "frontend",  t)
                t = re.sub(r"back\s+end",   "backend",   t)
                return t

            termos_cargo = []
            cargo_normalizado = ""
            if nome_perfil:
                cargo_normalizado = normalizar_cargo(nome_perfil)
                termos_cargo = [t for t in cargo_normalizado.split() if len(t) > 2]

            texto_cv_limpo = limpar_texto(texto_cv)
            final_scores   = []
            breakdowns     = []

            for i, row_texto in enumerate(vagas_filtradas["texto"]):

                row_vaga_i = vagas_filtradas.iloc[i]

                # ── 1. TFIDF ────────────────────────────────────────────
                score_tfidf = scores[i]

                # ── 2. CARGO ─────────────────────────────────────────────
                perfil_resumido = normalizar_cargo(str(row_vaga_i.get("perfil solicitado resumido", "")))

                if termos_cargo and perfil_resumido:
                    hits_cargo  = sum(1 for t in termos_cargo if t in perfil_resumido)
                    score_cargo = hits_cargo / len(termos_cargo)
                elif cargo_normalizado and cargo_normalizado in perfil_resumido:
                    score_cargo = 1.0
                else:
                    score_cargo = 0.0

                # ── 3. ROL ───────────────────────────────────────────────
                rol_c = str(perfil_row.get(coluna_rol_colab, "")) if coluna_rol_colab else ""
                rol_v = str(row_vaga_i.get(coluna_rol_vaga,  "")) if coluna_rol_vaga  else ""
                score_rol = 1.0 if rol_compativel(rol_c, rol_v) else 0.0

                # ── 4. TAXA ──────────────────────────────────────────────
                taxa_c_loop = tratar_taxa(perfil_row.get(coluna_taxa_colab)) if coluna_taxa_colab else 0
                taxa_v_loop = tratar_taxa(row_vaga_i.get(coluna_taxa_vaga))  if coluna_taxa_vaga  else 0

                if taxa_v_loop > 0 and taxa_c_loop > 0:
                    score_taxa = 1.0 if taxa_c_loop <= taxa_v_loop else 0.0
                else:
                    score_taxa = 0.0

                # ── 5. CV ────────────────────────────────────────────────
                score_cv = 1.0 if (texto_cv_limpo and tem_skill_direta(texto_cv_limpo, row_texto)) else 0.0

                # ── PESOS DINÂMICOS ──────────────────────────────────────
                if tem_perfil_suficiente:
                    p_tfidf = PESO_TFIDF
                    p_cargo = PESO_CARGO
                    p_rol   = PESO_ROL
                    p_taxa  = PESO_TAXA
                    p_cv    = PESO_CV
                else:
                    extra   = PESO_TFIDF / 3
                    p_tfidf = 0.0
                    p_cargo = PESO_CARGO + extra
                    p_rol   = PESO_ROL   + extra
                    p_taxa  = PESO_TAXA  + extra
                    p_cv    = PESO_CV

                # ── SCORE FINAL ──────────────────────────────────────────
                score_final = (
                    score_tfidf * p_tfidf +
                    score_cargo * p_cargo +
                    score_rol   * p_rol   +
                    score_taxa  * p_taxa  +
                    score_cv    * p_cv
                )

                final_scores.append(round(score_final, 4))

                breakdowns.append({
                    "tfidf":             round(score_tfidf * p_tfidf, 4),
                    "cargo":             round(score_cargo * p_cargo, 4),
                    "rol":               round(score_rol   * p_rol,   4),
                    "taxa":              round(score_taxa  * p_taxa,  4),
                    "cv":                round(score_cv    * p_cv,    4),
                    "total":             round(score_final, 4),
                    "perfil_suficiente": tem_perfil_suficiente,
                })

            vagas_filtradas["match"]      = final_scores
            vagas_filtradas["_breakdown"] = breakdowns

        # =========================
        # 📊 RESULTADO
        # Filtro: apenas vagas com score >= 50%
        # =========================
        resultado_ordenado = vagas_filtradas.sort_values("match", ascending=False)
        resultado_50       = resultado_ordenado[resultado_ordenado["match"] >= 0.50]

        # Se nenhuma vaga atingir 50%, exibe todas com aviso
        sem_vaga_50 = len(resultado_50) == 0
        resultado   = resultado_50 if not sem_vaga_50 else resultado_ordenado

        score_medio = round(resultado["match"].mean() * 100, 1) if len(resultado) > 0 else 0

        # =========================
        # ⚠️ NENHUMA VAGA DISPONÍVEL
        # =========================
        if len(resultado) == 0:
            st.warning(
                "Nenhuma vaga compatível encontrada para este colaborador. "
                "Tente anexar o CV ou verifique se o perfil está preenchido na base."
            )
            st.stop()

        # =========================
        # 🏷️ BANNER — VAGAS COMPATÍVEIS
        # =========================
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, #1c2330 0%, #161b22 100%);
                border: 1px solid #1f6feb55;
                border-left: 4px solid #1f6feb;
                border-radius: 12px;
                padding: 18px 24px;
                margin-bottom: 20px;
            ">
                <div style="color:#8b949e; font-size:13px; margin-bottom:4px;">{"⚠️ Nenhuma vaga com score ≥ 50% encontrada — exibindo todas as vagas disponíveis" if sem_vaga_50 else "Resultado da análise — apenas vagas com score ≥ 50%"}</div>
                <div style="color:#e6edf3; font-size:22px; font-weight:700;">
                    🎯 Vagas compatíveis para <span style="color:#388bfd;">{selecionado}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        col_m1, col_m2, col_m3 = st.columns(3)

        with col_m1:
            st.metric("Vagas encontradas", len(resultado))

        with col_m2:
            st.metric("Score médio", f"{score_medio}%")

        with col_m3:
            cv_status = "✅ Sim" if texto_cv.strip() else "❌ Não"
            st.metric("CV utilizado no match", cv_status)

        colunas_exibir = [
            "proyecto",
            "solicitante",
            "necesidad",
            "estado necesidad",
            "rol reporting",
            "tasa máxima deseable",
            "match",
            "perfil profesional",
            "perfil solicitado resumido",
            "lugar de trabajo",
            "lugar de trabajo definitivo",
            "perfil solicitado detallado",
            "conocimientos funcionales",
            "conocimientos tecnicos",
            "observaciones necesidad",
            "outros"
        ]

        # garante que "observaciones necesidad" aponte para a coluna real
        if col_obs and col_obs not in colunas_exibir:
            colunas_exibir = [col_obs if c == "observaciones necesidad" else c for c in colunas_exibir]

        colunas_exibir = [
            c for c in colunas_exibir
            if c in resultado.columns
        ]

        # Persiste resultado e contexto no session_state
        # para sobreviver ao rerun da busca por necessidade
        st.session_state.resultado_cache   = resultado
        st.session_state.colunas_cache     = colunas_exibir
        st.session_state.selecionado_cache = selecionado
        st.session_state.texto_cv_cache    = texto_cv
        st.session_state.col_obs_cache     = col_obs

        # =========================

# =========================
# 📊 EXIBIÇÃO — lê do session_state
# Persiste mesmo após rerun do campo de busca
# =========================
if st.session_state.resultado_cache is not None:

    resultado      = st.session_state.resultado_cache
    colunas_exibir = st.session_state.colunas_cache
    selecionado    = st.session_state.selecionado_cache
    texto_cv       = st.session_state.texto_cv_cache
    col_obs        = st.session_state.col_obs_cache
    texto_cv_limpo = limpar_texto(texto_cv)

    # 📊 TABELA PRINCIPAL
    # =========================
    st.dataframe(
        resultado[colunas_exibir],
        use_container_width=True,
        height=700
    )

    st.divider()

    # =========================
    # 🏷️ BANNER — DETALHAMENTO
    # =========================
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1c2330 0%, #161b22 100%);
            border: 1px solid #238636aa;
            border-left: 4px solid #238636;
            border-radius: 12px;
            padding: 18px 24px;
            margin-bottom: 20px;
        ">
            <div style="color:#8b949e; font-size:13px; margin-bottom:4px;">Detalhamento completo</div>
            <div style="color:#e6edf3; font-size:22px; font-weight:700;">
                📋 Vagas detalhadas para <span style="color:#2ea043;">{selecionado}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =========================
    # 🔎 BUSCA POR NECESSIDADE
    # =========================

    if "busca_necesidad_val" not in st.session_state:
        st.session_state.busca_necesidad_val = ""

    if "busca_input_key" not in st.session_state:
        st.session_state.busca_input_key = 0

    col_busca, col_limpar = st.columns([5, 1])

    with col_busca:
        busca_necesidad = st.text_input(
            "🔍 Buscar vaga pelo número da necessidade",
            placeholder="Ex: 767648-01/26",
            key=f"busca_necesidad_{st.session_state.busca_input_key}"
        )
        st.session_state.busca_necesidad_val = busca_necesidad

    with col_limpar:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Limpar", key="limpar_busca"):
            # Incrementa a key — força Streamlit a recriar o widget do zero
            st.session_state.busca_input_key += 1
            st.session_state.busca_necesidad_val = ""
            st.rerun()

    # Define quais vagas exibir no detalhamento
    busca_atual = st.session_state.busca_necesidad_val.strip()
    if busca_atual:
        vagas_detalhe = resultado[
            resultado["necesidad"].astype(str).str.contains(
                busca_atual, case=False, na=False
            )
        ]
        if vagas_detalhe.empty:
            st.warning(f"Nenhuma vaga encontrada com a necessidade '{busca_atual}'.")
            vagas_detalhe = resultado.head(1)
    else:
        # Sem busca — mostra apenas o Top 1
        vagas_detalhe = resultado.head(1)

    # =========================
    # 📂 DETALHAMENTO
    # =========================
    for idx, row in vagas_detalhe.iterrows():

        rol     = row.get("rol reporting", "")
        rol_str = f"| {rol} " if rol and str(rol).strip() else ""

        titulo = (
            f"{row.get('proyecto', 'Projeto')} "
            f"{rol_str}"
            f"| Match: {round(row['match'] * 100, 2)}%"
        )

        with st.expander(titulo, expanded=True):

            st.markdown(f"""
    ### 📌 Informações da Vaga

    **Projeto:** {row.get('proyecto', '-')}

    **Solicitante:** {row.get('solicitante', '-')}

    **Necessidade:** {row.get('necesidad', '-')}

    **Rol:** {row.get('rol reporting', '-')}

    **Taxa Máxima:** {row.get('tasa máxima deseable', '-')}

    **Score Match:** {round(row['match'] * 100, 2)}%
    """)

            st.divider()

            # =========================
            # 🔍 PAINEL DE TRANSPARÊNCIA
            # =========================
            st.markdown("### 🔍 Por que esse resultado?")

            st.markdown(
                "<p style='color:#8b949e; font-size:13px; margin-top:-10px; margin-bottom:20px;'>"
                "Cada critério abaixo mostra quanto contribuiu para o score final. "
                "Critérios em vermelho ou laranja indicam o que faltou para chegar a 100%."
                "</p>",
                unsafe_allow_html=True
            )

            # ── helpers visuais ─────────────────────────────────────
            def _barra(pct, cor):
                pct_clip = min(pct, 100)
                return (
                    f"<div style='background:#21262d;border-radius:8px;height:14px;"
                    f"width:100%;overflow:hidden;'>"
                    f"<div style='width:{pct_clip}%;background:{cor};height:14px;"
                    f"border-radius:8px;transition:width .4s ease;'></div></div>"
                )

            def _linha(label, pct_exibir, pct_barra, cor, colab_val, vaga_val, ok, faltou=""):
                icone      = "✅" if ok else ("⚠️" if pct_exibir > 0 else "❌")
                status     = "Compatível" if ok else ("Parcialmente compatível" if pct_exibir > 0 else "Não compatível")
                cor_status = "#238636" if ok else ("#f0883e" if pct_exibir > 0 else "#f85149")
                detalhe_colab = f"<span style='color:#c9d1d9;'>Colaborador: <b>{colab_val}</b></span>"
                detalhe_vaga  = f"<span style='color:#8b949e;'>Vaga exige: <b>{vaga_val}</b></span>"
                contribuicao  = (
                    f"<span style='color:#8b949e;font-size:11px;'>"
                    f"Contribuição para o score: <b style='color:#e6edf3;'>{pct_exibir:.1f}%</b></span>"
                )
                alerta = (
                    f"<div style='margin-top:6px;background:#f0883e18;border-left:3px solid #f0883e;"
                    f"border-radius:4px;padding:6px 10px;font-size:12px;color:#f0883e;'>"
                    f"⚠️ {faltou}</div>"
                    if not ok and faltou else ""
                )
                return (
                    f"<div style='margin-bottom:20px;'>"
                    f"<div style='display:flex;justify-content:space-between;"
                    f"align-items:center;margin-bottom:6px;'>"
                    f"<span style='color:#e6edf3;font-weight:600;font-size:14px;'>{icone} {label}</span>"
                    f"<span style='background:{cor_status}22;color:{cor_status};font-size:12px;"
                    f"font-weight:600;padding:2px 10px;border-radius:20px;'>{status}</span>"
                    f"</div>"
                    f"{_barra(pct_barra, cor)}"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;"
                    f"margin-top:6px;font-size:12px;flex-wrap:wrap;gap:6px;'>"
                    f"<div style='display:flex;gap:16px;'>{detalhe_colab} &nbsp;|&nbsp; {detalhe_vaga}</div>"
                    f"{contribuicao}"
                    f"</div>"
                    f"{alerta}"
                    f"</div>"
                )

            # ── recupera breakdown real ──────────────────────────────
            bd = row.get("_breakdown", {
                "tfidf": 0, "cargo": 0, "rol": 0,
                "taxa": 0, "cv": 0, "total": row["match"]
            })

            score_pct = round(row["match"] * 100, 2)

            p_tfidf  = round(bd["tfidf"] * 100, 1)
            p_cargo  = round(bd["cargo"] * 100, 1)
            p_skills = round(bd["rol"]   * 100, 1)
            p_cv     = round(bd["cv"]    * 100, 1)
            p_taxa   = round(bd["taxa"]  * 100, 1)

            # ── informações de contexto ──────────────────────────────
            rol_colab_val   = str(perfil_row.get(coluna_rol_colab, "")) if coluna_rol_colab else ""
            rol_vaga_val    = str(row.get(coluna_rol_vaga, ""))         if coluna_rol_vaga  else ""
            rol_ok          = rol_compativel(rol_colab_val, rol_vaga_val)
            nome_perfil_val = nome_perfil if nome_perfil else "—"

            perfil_prof_vaga = str(row.get("perfil solicitado resumido", "")).strip() or "—"
            if len(perfil_prof_vaga) > 60:
                perfil_prof_vaga = perfil_prof_vaga[:60] + "..."

            taxa_c  = tratar_taxa(perfil_row.get(coluna_taxa_colab)) if coluna_taxa_colab else 0
            taxa_v  = tratar_taxa(row.get(coluna_taxa_vaga))          if coluna_taxa_vaga  else 0
            taxa_ok = (taxa_v == 0) or (taxa_c <= taxa_v)

            cv_presente = texto_cv_limpo.strip() != ""

            # ── CRITÉRIO 1 — SKILLS TÉCNICAS ────────────────────────
            tfidf_ok       = p_tfidf >= (score_pct * 0.4)
            breakdown_html = _linha(
                "As skills técnicas combinam com a vaga?",
                p_tfidf, p_tfidf,
                "#1f6feb" if tfidf_ok else ("#f0883e" if p_tfidf > 0 else "#f85149"),
                "Perfil, cargo e CV do colaborador",
                "Requisitos da vaga",
                tfidf_ok, ""
            )

            # ── CRITÉRIO 2 — CARGO ──────────────────────────────────
            cargo_ok   = p_cargo > 0
            faltou_cargo = "" if cargo_ok else (
                f"O cargo '{nome_perfil_val}' não foi encontrado nos requisitos da vaga."
            )
            breakdown_html += _linha(
                "O cargo do colaborador aparece na vaga?",
                p_cargo, p_cargo,
                "#238636" if cargo_ok else "#f85149",
                f"{nome_perfil_val}",
                f"{perfil_prof_vaga}",
                cargo_ok, faltou_cargo
            )

            # ── CRITÉRIO 3 — ROL ────────────────────────────────────
            faltou_rol_nivel = "" if rol_ok else (
                f"A vaga exige o nível '{rol_vaga_val or '—'}', mas o colaborador é '{rol_colab_val or '—'}'. Os níveis precisam ser iguais para pontuar."
            )
            breakdown_html += _linha(
                "O nível (Rol) é o mesmo que a vaga pede?",
                p_skills if rol_ok else 0,
                p_skills if rol_ok else 0,
                "#238636" if rol_ok else "#f85149",
                f"{rol_colab_val or '—'}",
                f"{rol_vaga_val or '—'}",
                rol_ok, faltou_rol_nivel
            )

            # ── CRITÉRIO 4 — TAXA ────────────────────────────────────
            if not taxa_c and not taxa_v:
                faltou_taxa = ""
            elif taxa_ok:
                faltou_taxa = ""
            else:
                faltou_taxa = f"A taxa do colaborador ({taxa_c}) é maior que o limite da vaga ({taxa_v}). Essa vaga seria descartada na análise."

            breakdown_html += _linha(
                "A remuneração está dentro do limite da vaga?",
                p_taxa if taxa_ok else 0,
                p_taxa if taxa_ok else 0,
                "#238636" if (taxa_ok and p_taxa > 0) else ("#8b949e" if (not taxa_c and not taxa_v) else "#f85149"),
                f"Taxa do colaborador: {taxa_c}" if taxa_c else "Não informada",
                f"Limite máximo da vaga: {taxa_v}" if taxa_v else "Não informado",
                taxa_ok, faltou_taxa
            )

            # ── CRITÉRIO 5 — CV ──────────────────────────────────────
            cv_ok     = cv_presente and p_cv > 0
            faltou_cv = (
                "" if cv_ok
                else ("O CV ainda não foi enviado. Anexe o PDF do colaborador para melhorar o resultado do matching."
                      if not cv_presente
                      else "O CV foi enviado, mas tem poucas palavras em comum com os requisitos dessa vaga.")
            )
            cv_label = "CV enviado e considerado na análise" if cv_presente else "CV não enviado"
            breakdown_html += _linha(
                "O CV foi considerado na análise?",
                p_cv, p_cv,
                "#238636" if cv_ok else ("#f0883e" if cv_presente else "#f85149"),
                cv_label,
                "Requisitos técnicos da vaga",
                cv_ok, faltou_cv
            )

            # ── Resultado geral ───────────────────────────────────────
            cor_total  = "#238636" if score_pct >= 70 else ("#f0883e" if score_pct >= 50 else "#f85149")
            nota_score = (
                "Ótima aderência — colaborador muito compatível com a vaga." if score_pct >= 70
                else "Aderência moderada — vale avaliar os critérios em laranja/vermelho." if score_pct >= 50
                else "Baixa aderência — o colaborador tem pouco em comum com essa vaga."
            )

            st.markdown(
                f"<div style='background:#161b22;border:1px solid #30363d;border-radius:12px;"
                f"padding:20px 24px;margin-bottom:20px;'>"
                f"{breakdown_html}"
                f"<div style='border-top:1px solid #30363d;padding-top:16px;margin-top:4px;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;'>"
                f"<div>"
                f"<div style='color:#e6edf3;font-size:15px;font-weight:700;'>🏁 Resultado geral</div>"
                f"<div style='color:#8b949e;font-size:12px;margin-top:2px;'>{nota_score}</div>"
                f"</div>"
                f"<span style='color:{cor_total};font-size:26px;font-weight:800;'>{score_pct}%</span>"
                f"</div>"
                f"{_barra(score_pct, cor_total)}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True
            )

            st.divider()

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
st.markdown("<div class='footer-wrapper'><div class='footer-box'><div class='footer-title'>💼 Matching Inteligente de Vagas • v4.5</div><div class='footer-description'>Plataforma corporativa de apoio estratégico para análise de aderência entre colaboradores e oportunidades internas, utilizando IA, Skills, Perfil Profissional e Currículo PDF.</div><div class='footer-author'>Desenvolvido por <b>Jonathan Marquezini</b> • UGR Brasil</div></div></div>", unsafe_allow_html=True)
