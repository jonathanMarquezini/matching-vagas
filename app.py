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
</style>
""", unsafe_allow_html=True)

# =========================
# 🏢 HEADER
# =========================
st.title("💼 Matching Inteligente de Vagas")
st.caption("Skills • Rol • Taxa • Contexto enriquecido")

st.divider()

# =========================
# 🔧 LIMPEZA
# =========================
def limpar_texto(texto):
    if pd.isna(texto):
        return ""
    texto = str(texto).lower()
    partes = texto.split("//")

    resultado = []
    for p in partes:
        p = p.strip()
        p = re.sub(r"\(.*?\)", "", p)
        p = p.replace("tecnologías digitales /", "")
        p = re.sub(r"princ\..*", "", p)
        resultado.append(p.strip())

    return " ".join(resultado)

def limpar_texto_modelo(texto):
    if pd.isna(texto):
        return ""
    return str(texto)

def get_coluna(df, nome):
    return df[nome].fillna("").astype(str) if nome in df.columns else pd.Series([""] * len(df))

# =========================
# 🧠 ROL
# =========================
def parse_rol(rol):
    if pd.isna(rol):
        return {"tipo": "", "nivel": 0}

    rol = str(rol).lower().strip()
    partes = rol.split()

    mapa = {"i":1,"ii":2,"iii":3,"iv":4,"v":5}

    if len(partes) == 1:
        return {"tipo": partes[0], "nivel": 0}

    return {"tipo": partes[0], "nivel": mapa.get(partes[1], 0)}

def rol_compativel(rol_colab, rol_vaga):
    c = parse_rol(rol_colab)
    v = parse_rol(rol_vaga)

    if c["tipo"] == "sp":
        return (v["tipo"] == "sp" or (v["tipo"] == "t" and v["nivel"] <= 1))

    if c["tipo"] == "d":
        return v["tipo"] == "d"

    if c["tipo"] == "g":
        return v["tipo"] in ["g","d"]

    if c["tipo"] == "t":
        return v["tipo"] == "t" and c["nivel"] >= v["nivel"]

    if c["tipo"] == "s":
        return v["tipo"] == "s" and c["nivel"] >= v["nivel"]

    if c["tipo"] == "cd":
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
def tem_skill_direta(perfil, vaga):
    return any(p in vaga for p in perfil.split())

# =========================
# 📂 UPLOAD
# =========================
st.subheader("📂 Upload")

col1, col2 = st.columns(2)
file_vagas = col1.file_uploader("Base de Vagas", type=["csv","xlsx"])
file_colab = col2.file_uploader("Base de Colaboradores", type=["csv","xlsx"])

# =========================
# 🚀 PROCESSAMENTO
# =========================
if file_vagas and file_colab:

    vagas = pd.read_csv(file_vagas) if file_vagas.name.endswith(".csv") else pd.read_excel(file_vagas)
    colab = pd.read_csv(file_colab) if file_colab.name.endswith(".csv") else pd.read_excel(file_colab)

    vagas.columns = vagas.columns.str.lower().str.strip()
    colab.columns = colab.columns.str.lower().str.strip()

    if "necesidad" in vagas.columns:
        vagas = vagas.drop_duplicates(subset=["necesidad"])

    vagas["skills_tratadas"] = get_coluna(vagas,"conocimientos tecnicos").apply(limpar_texto)

    st.success("Bases carregadas 🚀")

    # =========================
    # 🔎 SELEÇÃO
    # =========================
    coluna_nome = next((c for c in ["nome","colaborador","funcionario"] if c in colab.columns), None)

    busca = st.text_input("Buscar colaborador")

    if busca:
        filtro = colab[
            colab[coluna_nome].str.contains(busca, case=False, na=False) |
            colab["matricula"].astype(str).str.contains(busca, na=False)
        ]
    else:
        filtro = colab

    selecionado = st.selectbox("Selecione", filtro[coluna_nome])
    perfil_row = colab[colab[coluna_nome] == selecionado].iloc[0]

    perfil_texto = limpar_texto_modelo(perfil_row.get("skills","")).lower()

    st.divider()

    # =========================
    # 🔎 MATCH
    # =========================
    if st.button("🚀 Buscar Vagas"):

        vagas["texto"] = (
            get_coluna(vagas,"skills_tratadas") + " " +
            get_coluna(vagas,"area") + " " +
            get_coluna(vagas,"perfil profesional") + " " +
            get_coluna(vagas,"perfil solicitado resumido") + " " +
            get_coluna(vagas,"perfil solicitado detallado") + " " +
            get_coluna(vagas,"conocimientos funcionales")
        )

        taxa_colab = tratar_taxa(perfil_row.get("taxa"))

        vagas_filtradas = vagas[vagas.apply(
            lambda r: rol_compativel(perfil_row.get("rol"), r.get("rol reporting")) and
                      taxa_colab <= tratar_taxa(r.get("tasa máxima deseable")),
            axis=1
        )].copy()

        if vagas_filtradas.empty:
            st.warning("Nenhuma vaga encontrada")
            st.stop()

        vectorizer = TfidfVectorizer()

        corpus = vagas_filtradas["texto"].apply(limpar_texto_modelo).tolist()
        corpus.append(perfil_texto)

        vectors = vectorizer.fit_transform(corpus)
        scores = cosine_similarity(vectors[-1], vectors[:-1])[0]

        final_scores = []
        for i, txt in enumerate(vagas_filtradas["texto"]):
            score = scores[i]
            if tem_skill_direta(perfil_texto, txt):
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
            "perfil profesional",
            "perfil solicitado resumido",
            "perfil solicitado detallado",
            "conocimientos funcionales",
            "conocimientos tecnicos"
        ]], use_container_width=True)

    # =========================
    # 📊 BASE FINAL
    # =========================
    if st.button("📊 Gerar Base"):

        vagas_base = vagas.copy()

        vagas_base["texto"] = (
            get_coluna(vagas_base,"skills_tratadas") + " " +
            get_coluna(vagas_base,"area") + " " +
            get_coluna(vagas_base,"perfil profesional") + " " +
            get_coluna(vagas_base,"perfil solicitado resumido") + " " +
            get_coluna(vagas_base,"perfil solicitado detallado") + " " +
            get_coluna(vagas_base,"conocimientos funcionales")
        )

        vaga_para = {i: [] for i in range(len(vagas_base))}

        for _, c in colab.iterrows():

            nome = c[coluna_nome]
            perfil = limpar_texto_modelo(c.get("skills","")).lower()
            taxa = tratar_taxa(c.get("taxa"))

            vectorizer = TfidfVectorizer()

            corpus = vagas_base["texto"].apply(limpar_texto_modelo).tolist()
            corpus.append(perfil)

            vectors = vectorizer.fit_transform(corpus)
            scores = cosine_similarity(vectors[-1], vectors[:-1])[0]

            for i, row in vagas_base.iterrows():
                score = scores[i]

                if tem_skill_direta(perfil, row["texto"]):
                    score += 0.5

                if (
                    rol_compativel(c.get("rol"), row.get("rol reporting")) and
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
