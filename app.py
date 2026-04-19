import streamlit as st
import pandas as pd
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# =========================
# 🎨 CONFIG VISUAL
# =========================
st.set_page_config(page_title="Matching de Vagas", layout="wide")

st.markdown("""
<style>
body {
    background-color: #0e1117;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #e6edf3;
}
.stButton>button {
    background-color: #1f6feb;
    color: white;
    border-radius: 8px;
    height: 45px;
    font-weight: bold;
}
.stTextInput>div>div>input {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 🏢 HEADER
# =========================
st.title("💼 Matching Inteligente de Vagas")
st.caption("Análise baseada em Skills • Rol • Taxa • Contexto da vaga")

st.divider()

# =========================
# 🔧 LIMPEZA DE SKILLS
# =========================
def limpar_texto(texto):
    if pd.isna(texto):
        return ""
    
    texto = str(texto).lower()
    partes = texto.split("//")
    skills_limpas = []

    for p in partes:
        p = p.strip()
        p = re.sub(r"\(.*?\)", "", p)
        p = p.replace("tecnologías digitales /", "")
        p = re.sub(r"princ\..*", "", p)
        skills_limpas.append(p.strip())

    return " ".join(skills_limpas)

# =========================
# 🔧 COLUNA SEGURA
# =========================
def get_coluna(df, nome):
    return df[nome].astype(str) if nome in df.columns else pd.Series([""] * len(df))

# =========================
# 🧠 PARSE DE ROL
# =========================
def parse_rol(rol):
    if pd.isna(rol):
        return {"tipo": "", "nivel": 0}

    rol = str(rol).strip().lower()
    partes = rol.split()

    if len(partes) == 1:
        return {"tipo": partes[0], "nivel": 0}

    mapa_nivel = {"i":1,"ii":2,"iii":3,"iv":4,"v":5}
    return {"tipo": partes[0], "nivel": mapa_nivel.get(partes[1], 0)}

# =========================
# 🧠 REGRA DE ROL
# =========================
def rol_compativel(rol_colab, rol_vaga):
    colab = parse_rol(rol_colab)
    vaga = parse_rol(rol_vaga)

    if colab["tipo"] == "sp":
        return (vaga["tipo"] == "sp" or (vaga["tipo"] == "t" and vaga["nivel"] <= 1))

    if colab["tipo"] == "d":
        return vaga["tipo"] == "d"

    if colab["tipo"] == "g":
        return vaga["tipo"] in ["g", "d"]

    if colab["tipo"] == "t":
        return vaga["tipo"] == "t" and colab["nivel"] >= vaga["nivel"]

    if colab["tipo"] == "s":
        return vaga["tipo"] == "s" and colab["nivel"] >= vaga["nivel"]

    if colab["tipo"] == "cd":
        return True

    return False

# =========================
# 💰 TAXA
# =========================
def tratar_taxa(valor):
    if pd.isna(valor):
        return 0
    try:
        return float(str(valor).replace(",", "."))
    except:
        return 0

# =========================
# 🧠 BOOST
# =========================
def tem_skill_direta(perfil, vaga_texto):
    return any(skill in vaga_texto for skill in perfil.split())

# =========================
# 📂 UPLOAD
# =========================
st.subheader("📂 Upload das Bases")

col1, col2 = st.columns(2)

with col1:
    file_vagas = st.file_uploader("Base de Vagas", type=["csv", "xlsx"])

with col2:
    file_colab = st.file_uploader("Base de Colaboradores", type=["csv", "xlsx"])

if file_vagas and file_colab:

    vagas = pd.read_csv(file_vagas) if file_vagas.name.endswith(".csv") else pd.read_excel(file_vagas)
    colab = pd.read_csv(file_colab) if file_colab.name.endswith(".csv") else pd.read_excel(file_colab)

    vagas.columns = vagas.columns.str.strip().str.lower()
    colab.columns = colab.columns.str.strip().str.lower()

    if "necesidad" in vagas.columns:
        vagas = vagas.drop_duplicates(subset=["necesidad"])

    vagas["skills_tratadas"] = vagas["conocimientos tecnicos"].apply(limpar_texto)

    st.success("Bases carregadas com sucesso")

    st.divider()

    # =========================
    # 🔍 BUSCA
    # =========================
    st.subheader("🔎 Seleção de Colaborador")

    coluna_nome = next((c for c in ["nome","colaborador","funcionario"] if c in colab.columns), None)

    busca = st.text_input("Digite nome ou matrícula")

    if busca:
        filtro = colab[
            colab[coluna_nome].str.contains(busca, case=False, na=False) |
            colab["matricula"].astype(str).str.contains(busca, na=False)
        ]
    else:
        filtro = colab

    selecionado = st.selectbox("Selecione", filtro[coluna_nome])

    perfil_row = colab[colab[coluna_nome] == selecionado].iloc[0]
    perfil_texto = str(perfil_row.get("skills","")).lower()

    st.divider()

    # =========================
    # 🔎 MATCH
    # =========================
    if st.button("🚀 Buscar Vagas Compatíveis"):

        vagas["texto"] = vagas["skills_tratadas"] + " " + get_coluna(vagas,"area")
        taxa_colab = tratar_taxa(perfil_row.get("taxa"))

        vagas_filtradas = vagas[vagas.apply(
            lambda row: rol_compativel(perfil_row.get("rol"), row.get("rol reporting")) and
                        taxa_colab <= tratar_taxa(row.get("tasa máxima deseable")),
            axis=1
        )].copy()

        if len(vagas_filtradas) == 0:
            st.warning("Nenhuma vaga encontrada")
            st.stop()

        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(vagas_filtradas["texto"].tolist() + [perfil_texto])
        scores = cosine_similarity(vectors[-1], vectors[:-1])[0]

        final_scores = []
        for i, row in enumerate(vagas_filtradas["texto"]):
            score = scores[i]
            if tem_skill_direta(perfil_texto, row):
                score += 0.5
            final_scores.append(score)

        vagas_filtradas["match"] = final_scores
        resultado = vagas_filtradas.sort_values("match", ascending=False)

        st.metric("Vagas encontradas", len(resultado))

        st.dataframe(resultado[[
            "proyecto",
            "solicitante",
            "necesidad",
            "rol reporting",
            "tasa máxima deseable",
            "match",
            "conocimientos tecnicos"
        ]], use_container_width=True)

    st.divider()

    # =========================
    # 📊 BASE FINAL
    # =========================
    if st.button("📊 Gerar Base Completa"):

        vagas_base = vagas.copy()
        vagas_base["texto"] = vagas_base["skills_tratadas"] + " " + get_coluna(vagas_base,"area")

        vaga_para = {i: [] for i in range(len(vagas_base))}

        for _, colab_row in colab.iterrows():
            nome = colab_row[coluna_nome]
            perfil = str(colab_row.get("skills","")).lower()
            taxa = tratar_taxa(colab_row.get("taxa"))

            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform(vagas_base["texto"].tolist() + [perfil])
            scores = cosine_similarity(vectors[-1], vectors[:-1])[0]

            for i, row in vagas_base.iterrows():
                score = scores[i]
                if tem_skill_direta(perfil, row["texto"]):
                    score += 0.5

                if (
                    rol_compativel(colab_row.get("rol"), row.get("rol reporting")) and
                    taxa <= tratar_taxa(row.get("tasa máxima deseable")) and
                    score >= 0.10
                ):
                    vaga_para[i].append(nome)

        vagas_base["vaga_para"] = [
            ", ".join(vaga_para[i]) if vaga_para[i] else "Sem match"
            for i in range(len(vagas_base))
        ]

        st.dataframe(vagas_base, use_container_width=True)

        st.download_button(
            "📥 Baixar CSV",
            vagas_base.to_csv(index=False).encode("utf-8"),
            "vagas_match.csv"
        )
