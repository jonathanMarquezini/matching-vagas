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

    coluna_rol_colab = encontrar_coluna(colab, ["roll", "rol", "role", "nivel"])
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

            def filtro_vaga(row):

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
                vagas.apply(filtro_vaga, axis=1)
            ].copy()

            # =========================
            # ❌ SEM RESULTADO
            # =========================
            if len(vagas_filtradas) == 0:
                st.warning("Nenhuma vaga compatível com os filtros de Rol/Taxa. Exibindo todas as vagas ranqueadas por match.")
                vagas_filtradas = vagas.copy()

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
            # 🔥 BOOST + BREAKDOWN
            # Guardamos cada parcela para
            # exibir no painel de transparência
            # =========================
            texto_cv_limpo = limpar_texto(texto_cv)
            final_scores   = []
            breakdowns     = []

            for i, row_texto in enumerate(vagas_filtradas["texto"]):

                score_tfidf  = scores[i]

                boost_perfil = 0.10 if tem_skill_direta(perfil_texto, row_texto) else 0.0
                boost_nome   = 0.15 if (nome_perfil and nome_perfil.lower() in row_texto) else 0.0
                boost_cv     = 0.20 if (texto_cv_limpo and tem_skill_direta(texto_cv_limpo, row_texto)) else 0.0

                score_final  = score_tfidf + boost_perfil + boost_nome + boost_cv

                final_scores.append(round(score_final, 4))

                breakdowns.append({
                    "tfidf":        round(score_tfidf,  4),
                    "boost_perfil": round(boost_perfil, 4),
                    "boost_nome":   round(boost_nome,   4),
                    "boost_cv":     round(boost_cv,     4),
                    "total":        round(score_final,  4),
                })

            vagas_filtradas["match"]      = final_scores
            vagas_filtradas["_breakdown"] = breakdowns

        # =========================
        # 📊 RESULTADO
        # =========================
        resultado = vagas_filtradas.sort_values("match", ascending=False)
        resultado = resultado[resultado["match"] > 0.02]

        score_medio = round(resultado["match"].mean() * 100, 1) if len(resultado) > 0 else 0

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
                <div style="color:#8b949e; font-size:13px; margin-bottom:4px;">Resultado da análise</div>
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
        # 📂 DETALHAMENTO
        # =========================
        for idx, row in resultado.head(10).iterrows():

            rol = row.get("rol reporting", "")
            rol_str = f"| {rol} " if rol and str(rol).strip() else ""

            titulo = (
                f"{row.get('proyecto', 'Projeto')} "
                f"{rol_str}"
                f"| Match: {round(row['match'] * 100, 2)}%"
            )

            with st.expander(titulo, expanded=False):

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
                # Usa os valores REAIS do score
                # calculado pelo TF-IDF + boost
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
                    icone    = "✅" if ok else ("⚠️" if pct_exibir > 0 else "❌")
                    status   = "Compatível" if ok else ("Parcialmente compatível" if pct_exibir > 0 else "Não compatível")
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

                # ── recupera o breakdown real desta vaga ─────────────────
                bd = row.get("_breakdown", {
                    "tfidf": 0, "boost_perfil": 0,
                    "boost_nome": 0, "boost_cv": 0, "total": row["match"]
                })

                score_real  = bd["total"]           # 0..∞ (pode passar 1.0 com boosts)
                score_pct   = round(row["match"] * 100, 2)   # % exibido na tabela

                # Normaliza cada parcela proporcionalmente ao score_pct
                # para que a soma do breakdown == score exibido
                denom = score_real if score_real > 0 else 1

                p_tfidf  = round((bd["tfidf"]        / denom) * score_pct, 1)
                p_perfil = round((bd["boost_perfil"]  / denom) * score_pct, 1)
                p_nome   = round((bd["boost_nome"]    / denom) * score_pct, 1)
                p_cv     = round((bd["boost_cv"]      / denom) * score_pct, 1)

                # ── informações de contexto ──────────────────────────────
                rol_colab_val = str(perfil_row.get(coluna_rol_colab, "")) if coluna_rol_colab else ""
                rol_vaga_val  = str(row.get(coluna_rol_vaga,  ""))        if coluna_rol_vaga  else ""
                rol_ok        = rol_compativel(rol_colab_val, rol_vaga_val)

                taxa_c  = tratar_taxa(perfil_row.get(coluna_taxa_colab)) if coluna_taxa_colab else 0
                taxa_v  = tratar_taxa(row.get(coluna_taxa_vaga))          if coluna_taxa_vaga  else 0
                taxa_ok = (taxa_v == 0) or (taxa_c <= taxa_v)

                cv_presente = texto_cv_limpo.strip() != ""

                # ── CRITÉRIO 1 — ADERÊNCIA GERAL DO PERFIL ───────────
                tfidf_ok     = p_tfidf >= (score_pct * 0.4)
                faltou_tfidf = "" if tfidf_ok else "O perfil do colaborador tem pouco em comum com o que essa vaga pede. Considere buscar vagas mais próximas da área de atuação dele."
                breakdown_html = _linha(
                    "O perfil combina com a vaga?",
                    p_tfidf, p_tfidf,
                    "#1f6feb" if tfidf_ok else ("#f0883e" if p_tfidf > 0 else "#f85149"),
                    "Perfil, cargo e CV do colaborador",
                    "Requisitos da vaga",
                    tfidf_ok, faltou_tfidf
                )

                # ── CRITÉRIO 2 — ROL ────────────────────────────────────
                faltou_rol = "" if rol_ok else (
                    f"A vaga exige o cargo '{rol_vaga_val or '—'}', mas o colaborador é '{rol_colab_val or '—'}'. Os cargos precisam ser iguais para pontuar."
                )
                breakdown_html += _linha(
                    "O cargo é o mesmo que a vaga pede?",
                    p_perfil if rol_ok else 0,
                    p_perfil if rol_ok else 0,
                    "#238636" if rol_ok else "#f85149",
                    f"Colaborador: {rol_colab_val or '—'}",
                    f"Vaga: {rol_vaga_val or '—'}",
                    rol_ok, faltou_rol
                )

                # ── CRITÉRIO 3 — TAXA ────────────────────────────────────
                if not taxa_c and not taxa_v:
                    taxa_nota = "Taxa não informada — critério não aplicado"
                    faltou_taxa = ""
                elif taxa_ok:
                    taxa_nota = f"A taxa do colaborador ({taxa_c}) está dentro do limite da vaga ({taxa_v})"
                    faltou_taxa = ""
                else:
                    taxa_nota = ""
                    faltou_taxa = f"A taxa do colaborador ({taxa_c}) está acima do limite aceito pela vaga ({taxa_v}). Isso pode inviabilizar a alocação."

                breakdown_html += _linha(
                    "A remuneração está dentro do limite da vaga?",
                    p_nome if taxa_ok else 0,
                    p_nome if taxa_ok else 0,
                    "#238636" if taxa_ok else "#f0883e",
                    f"Taxa atual: {taxa_c}" if taxa_c else "Não informada",
                    f"Limite máximo: {taxa_v}" if taxa_v else "Não informado",
                    taxa_ok, faltou_taxa
                )

                # ── CRITÉRIO 4 — CV ──────────────────────────────────────
                cv_ok    = cv_presente and p_cv > 0
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

                # ── Score total — igual ao da tabela ─────────────────────
                cor_total  = "#238636" if score_pct >= 70 else ("#f0883e" if score_pct >= 40 else "#f85149")
                nota_score = (
                    "Ótima aderência — colaborador muito compatível com a vaga." if score_pct >= 70
                    else "Aderência moderada — vale avaliar os critérios em laranja/vermelho." if score_pct >= 40
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
st.markdown("<div class='footer-wrapper'><div class='footer-box'><div class='footer-title'>💼 Matching Inteligente de Vagas • v4.4</div><div class='footer-description'>Plataforma corporativa de apoio estratégico para análise de aderência entre colaboradores e oportunidades internas, utilizando IA, Skills, Perfil Profissional e Currículo PDF.</div><div class='footer-author'>Desenvolvido por <b>Jonathan Marquezini</b> • UGR</div></div></div>", unsafe_allow_html=True)
