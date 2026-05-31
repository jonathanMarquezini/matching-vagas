import streamlit as st
import pandas as pd
import re
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from io import BytesIO
import unicodedata

# =========================
# 🎨 CONFIG VISUAL
# =========================
st.set_page_config(
    page_title="Matching Inteligente de Vagas",
    layout="wide"
)

st.markdown("""
<style>
html, body, [class*="css"] { font-family: "Segoe UI", sans-serif; }
.main { background-color: #0e1117; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
h1, h2, h3 { color: #e6edf3; }
.stButton > button {
    background-color: #1f6feb; color: white; border-radius: 10px;
    border: none; height: 45px; font-weight: 600; width: 100%;
}
.stButton > button:hover { background-color: #388bfd; color: white; }
.stDownloadButton > button {
    background-color: #238636 !important; color: white !important;
    border-radius: 10px; border: none; height: 45px; font-weight: 600; width: 100%;
}
.stDownloadButton > button:hover { background-color: #2ea043 !important; }
div[data-baseweb="select"] > div { background-color: #1c1f26; }
.stTextInput input { background-color: #1c1f26; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid #30363d; }
div[data-testid="stExpander"] {
    border-radius: 12px !important; border: 1px solid #30363d !important;
    margin-bottom: 10px !important; overflow: hidden;
}
.header-company {
    text-align: right; font-size: 24px; font-weight: 700;
    color: white; white-space: nowrap; margin-top: 12px;
}
.cv-box {
    background-color: #161b22; border: 1px solid #30363d;
    border-radius: 14px; padding: 20px; margin-top: 10px; margin-bottom: 15px;
}
.footer-wrapper { margin-top: 60px; margin-bottom: 20px; }
.footer-box {
    background: linear-gradient(135deg, #161b22 0%, #1c2330 100%);
    border: 1px solid #30363d; border-radius: 18px; padding: 35px 25px; text-align: center;
}
.footer-title { color: #f0f6fc; font-size: 24px; font-weight: 700; margin-bottom: 18px; }
.footer-description { color: #8b949e; font-size: 15px; line-height: 1.8; margin-bottom: 18px; }
.footer-author { color: #c9d1d9; font-size: 14px; }
.footer-author b { color: white; }
.col-hint-box {
    background-color: #1c1f26; border: 1px solid #f0883e55;
    border-radius: 10px; padding: 14px 18px; margin-bottom: 12px;
    color: #f0883e; font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🏢 HEADER
# =========================
st.title("💼 Matching Inteligente de Vagas")

col1, col2 = st.columns([5, 2])
with col1:
    st.caption("Plataforma corporativa para análise estratégica de aderência entre colaboradores e oportunidades internas.")
with col2:
    st.markdown("<div class='header-company'>🏢 Indra Group | Minsait</div>", unsafe_allow_html=True)

st.divider()

# =========================
# 🔧 UTILITÁRIOS
# =========================
def limpar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    texto = re.sub(r"[\n\r\t]", " ", texto)
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()

def limpar_texto_modelo(texto):
    if pd.isna(texto):
        return ""
    return str(texto)

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

def get_coluna(df, nome):
    if nome in df.columns:
        return df[nome].fillna("").astype(str)
    return pd.Series([""] * len(df))

def normalizar_col(nome):
    nome = str(nome).strip().lower()
    nome = unicodedata.normalize("NFD", nome)
    nome = "".join(c for c in nome if unicodedata.category(c) != "Mn")
    nome = re.sub(r"[\s_]+", "_", nome)
    return nome

def encontrar_coluna(df, candidatos):
    """Match exato normalizado, depois parcial."""
    cols_norm = {normalizar_col(c): c for c in df.columns}
    for cand in candidatos:
        cand_norm = normalizar_col(cand)
        if cand_norm in cols_norm:
            return cols_norm[cand_norm]
    for cand in candidatos:
        cand_norm = normalizar_col(cand)
        for col_norm, col_real in cols_norm.items():
            if cand_norm in col_norm or col_norm in cand_norm:
                return col_real
    return None

def parse_rol(rol):
    if pd.isna(rol):
        return {"tipo": "", "nivel": 0}
    rol = str(rol).strip().lower()
    partes = rol.split()
    if not partes:
        return {"tipo": "", "nivel": 0}
    tipo = partes[0]
    mapa_nivel = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5}
    nivel = mapa_nivel.get(partes[1], 0) if len(partes) > 1 else 0
    return {"tipo": tipo, "nivel": nivel}

def rol_compativel(rol_colab, rol_vaga):
    colab = parse_rol(rol_colab)
    vaga = parse_rol(rol_vaga)
    if colab["tipo"] != vaga["tipo"]:
        return False
    return colab["nivel"] == vaga["nivel"]

def tratar_taxa(valor):
    if pd.isna(valor):
        return 0
    valor = re.sub(r"[^0-9.,]", "", str(valor)).replace(",", ".")
    try:
        return float(valor)
    except:
        return 0

def tem_skill_direta(perfil, vaga_texto):
    for skill in perfil.split():
        if len(skill) > 4 and skill in vaga_texto:
            return True
    return False

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
    file_vagas = st.file_uploader("Base de Vagas", type=["csv", "xlsx"])
with col2:
    file_colab = st.file_uploader("Base de Colaboradores", type=["csv", "xlsx"])

# =========================
# 🚀 PROCESSAMENTO
# =========================
if file_vagas and file_colab:

    vagas = (
        pd.read_csv(file_vagas) if file_vagas.name.endswith(".csv")
        else pd.read_excel(file_vagas)
    )
    colab = (
        pd.read_csv(file_colab) if file_colab.name.endswith(".csv")
        else pd.read_excel(file_colab)
    )

    # Normaliza nomes de colunas: strip + lowercase
    vagas.columns = vagas.columns.str.strip().str.lower()
    colab.columns = colab.columns.str.strip().str.lower()

    if "necesidad" in vagas.columns:
        vagas = vagas.drop_duplicates(subset=["necesidad"])

    # Texto da vaga
    vagas["texto"] = (
        get_coluna(vagas, "conocimientos tecnicos") + " " +
        get_coluna(vagas, "perfil solicitado resumido") + " " +
        get_coluna(vagas, "perfil solicitado detallado") + " " +
        get_coluna(vagas, "conocimientos funcionales") + " " +
        get_coluna(vagas, "perfil profesional")
    ).apply(limpar_texto)

    st.success("✅ Bases carregadas com sucesso")
    st.divider()

    # =========================
    # 🔍 DETECTAR COLUNA DE NOME
    # Inclui "nome_colaborador" como prioridade máxima
    # =========================
    coluna_nome = encontrar_coluna(colab, [
        "nome_colaborador", "nome colaborador",
        "nome", "colaborador", "funcionario",
        "nombre_colaborador", "nombre colaborador", "nombre",
        "employee", "name", "empleado"
    ])

    if not coluna_nome:
        st.error("❌ Coluna de nome não encontrada na Base de Colaboradores.")
        colunas_disponiveis = ", ".join([f"`{c}`" for c in colab.columns.tolist()])
        st.markdown(
            f"<div class='col-hint-box'>⚠️ <b>Colunas detectadas:</b><br><br>{colunas_disponiveis}<br><br>"
            f"Selecione manualmente a coluna do <b>nome do colaborador</b>:</div>",
            unsafe_allow_html=True
        )
        coluna_nome = st.selectbox("Qual coluna é o nome do colaborador?", options=colab.columns.tolist())
        if not coluna_nome:
            st.stop()

    coluna_matricula = encontrar_coluna(colab, [
        "matricula_colaborador", "matricula colaborador", "matricula",
        "employee_id", "cod_colaborador", "codigo", "id"
    ])

    coluna_nome_perfil = encontrar_coluna(colab, [
        "nome_perfil", "perfil", "cargo", "funcao",
        "perfil_colaborador", "role", "position", "puesto"
    ])

    coluna_descricao = encontrar_coluna(colab, [
        "descricao", "descricao_colaborador", "description",
        "resumo", "summary", "perfil_resumo"
    ])

    coluna_rol_colab = encontrar_coluna(colab, ["roll", "rol", "role", "nivel"])
    coluna_taxa_colab = encontrar_coluna(colab, ["taxa", "tasa", "rate", "valor"])
    coluna_rol_vaga  = encontrar_coluna(vagas, ["rol reporting", "rol", "role", "nivel"])
    coluna_taxa_vaga = encontrar_coluna(vagas, [
        "tasa maxima deseable", "tasa máxima deseable",
        "taxa maxima", "taxa", "tasa", "rate_max"
    ])

    # =========================
    # 🔍 BUSCA COLABORADOR
    # =========================
    st.subheader("🔎 Seleção de Colaborador")

    busca = st.text_input("Digite nome ou matrícula")

    # Garante que a coluna de nome seja string limpa
    colab[coluna_nome] = colab[coluna_nome].fillna("").astype(str).str.strip()

    if busca:
        filtro_nome = colab[coluna_nome].str.contains(busca, case=False, na=False)
        if coluna_matricula:
            filtro_mat = colab[coluna_matricula].astype(str).str.contains(busca, na=False)
            mask = filtro_nome | filtro_mat
        else:
            mask = filtro_nome
        filtro_df = colab[mask]
    else:
        filtro_df = colab

    if filtro_df.empty:
        st.warning("Nenhum colaborador encontrado com esse filtro.")
        st.stop()

    # Lista de nomes para o selectbox — usa índice original para lookup seguro
    opcoes = filtro_df[coluna_nome].tolist()
    selecionado = st.selectbox("Selecione o colaborador", opcoes)

    # ✅ BUSCA SEGURA: usa o índice da linha do filtro_df
    linhas = filtro_df[filtro_df[coluna_nome] == selecionado]
    if linhas.empty:
        st.error("Colaborador não encontrado. Tente novamente.")
        st.stop()

    perfil_row = linhas.iloc[0]

    # =========================
    # 📄 UPLOAD CV
    # =========================
    st.markdown(
        f"<div class='cv-box'><b>📄 Currículo de {selecionado} (Opcional)</b><br>"
        f"<span style='color:#8b949e;font-size:13px;'>"
        f"Anexe o CV em PDF para enriquecer o matching com skills, experiências e formações."
        f"</span></div>",
        unsafe_allow_html=True
    )

    cv_pdf = st.file_uploader("Anexar CV em PDF", type=["pdf"], key=f"cv_{selecionado}")

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
    descricao_colab = limpar_texto_modelo(perfil_row.get(coluna_descricao, "")) if coluna_descricao else ""
    nome_perfil = limpar_texto_modelo(perfil_row.get(coluna_nome_perfil, "")) if coluna_nome_perfil else ""
    perfil_texto = limpar_texto(descricao_colab + " " + nome_perfil + " " + texto_cv)

    if not perfil_texto.strip():
        st.warning("⚠️ Este colaborador não possui descrição de perfil nem CV anexado. O match pode ter baixa precisão.")

    st.divider()

    # =========================
    # 🚀 MATCH
    # =========================
    if st.button("🚀 Buscar Vagas Compatíveis"):

        taxa_colab = tratar_taxa(perfil_row.get(coluna_taxa_colab)) if coluna_taxa_colab else 0

        with st.spinner("🔍 Calculando compatibilidade das vagas..."):

            def filtro_vaga(row):
                if coluna_rol_colab and coluna_rol_vaga:
                    if not rol_compativel(perfil_row.get(coluna_rol_colab), row.get(coluna_rol_vaga)):
                        return False
                if coluna_taxa_vaga:
                    taxa_max = tratar_taxa(row.get(coluna_taxa_vaga))
                    if taxa_max > 0 and taxa_colab > taxa_max:
                        return False
                return True

            vagas_filtradas = vagas[vagas.apply(filtro_vaga, axis=1)].copy()

            if len(vagas_filtradas) == 0:
                st.warning("Nenhuma vaga compatível com os filtros de Rol/Taxa. Exibindo todas as vagas ranqueadas por match.")
                vagas_filtradas = vagas.copy()

            # TF-IDF
            vectorizer = TfidfVectorizer(stop_words=None)
            corpus = vagas_filtradas["texto"].tolist()
            corpus.append(perfil_texto if perfil_texto.strip() else "sem perfil")
            vectors = vectorizer.fit_transform(corpus)
            scores = cosine_similarity(vectors[-1], vectors[:-1])[0]

            texto_cv_limpo = limpar_texto(texto_cv)
            final_scores = []

            for i, row_texto in enumerate(vagas_filtradas["texto"]):
                score = scores[i]
                if tem_skill_direta(perfil_texto, row_texto):
                    score += 0.10
                if nome_perfil and nome_perfil.lower() in row_texto:
                    score += 0.15
                if texto_cv_limpo and tem_skill_direta(texto_cv_limpo, row_texto):
                    score += 0.20
                final_scores.append(round(score, 4))

            vagas_filtradas["match"] = final_scores

        resultado = vagas_filtradas.sort_values("match", ascending=False)
        resultado = resultado[resultado["match"] > 0.02]

        score_medio = round(resultado["match"].mean() * 100, 1) if len(resultado) > 0 else 0

        # =========================
        # 🏷️ TÍTULO COM NOME DO COLABORADOR
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
            st.metric("CV utilizado no match", "✅ Sim" if texto_cv.strip() else "❌ Não")

        colunas_exibir = [
            "proyecto", "solicitante", "necesidad", "rol reporting",
            "tasa máxima deseable", "match", "perfil profesional",
            "perfil solicitado resumido", "perfil solicitado detallado",
            "conocimientos funcionales", "conocimientos tecnicos"
        ]
        colunas_exibir = [c for c in colunas_exibir if c in resultado.columns]

        st.dataframe(resultado[colunas_exibir], use_container_width=True, height=700)
        st.divider()

        # =========================
        # 📋 DETALHAMENTO COM NOME
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

        for idx, row in resultado.head(20).iterrows():
            rol = row.get("rol reporting", "")
            rol_str = f"| {rol} " if rol and str(rol).strip() else ""
            titulo = f"{row.get('proyecto', 'Projeto')} {rol_str}| Match: {round(row['match'] * 100, 2)}%"

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
st.markdown(
    "<div class='footer-wrapper'><div class='footer-box'>"
    "<div class='footer-title'>💼 Matching Inteligente de Vagas • v4.3</div>"
    "<div class='footer-description'>Plataforma corporativa de apoio estratégico para análise de aderência entre "
    "colaboradores e oportunidades internas, utilizando IA, Skills, Perfil Profissional e Currículo PDF.</div>"
    "<div class='footer-author'>Desenvolvido por <b>Jonathan Marquezini</b> • UGR</div>"
    "</div></div>",
    unsafe_allow_html=True
)
