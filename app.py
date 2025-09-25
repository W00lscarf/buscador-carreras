# app.py
# -------------------------------------------------------------
# Buscador de carreras (Chile) con datos SIES 2024–2025
# -------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

# Opcional: correlaciones
try:
    from scipy.stats import pearsonr, spearmanr
    SCIPY = True
except Exception:
    SCIPY = False

st.set_page_config(page_title="Buscador de Carreras – Chile", layout="wide")
st.title("Buscador de Carreras – Chile (SIES 2024–2025)")
st.caption(
    "Fuente: Subsecretaría de Educación Superior – Portal MiFuturo.cl (SIES 2024–2025).\n"
    "Este prototipo integra estadísticas por **carrera genérica** con el detalle por **institución**."
)

# -------------------------------------------------------------
# Funciones auxiliares
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_excels_from_files(file1_path: str, file2_path: str):
    df1 = pd.read_excel(file1_path, sheet_name="Estadísticas x CG", header=1)
    df2 = pd.read_excel(file2_path, sheet_name="Buscador Carreras  2024-2025")

    # Normalizar nombres de columnas
    df1.columns = df1.columns.str.strip().str.replace("\n", " ", regex=False)
    df2.columns = df2.columns.str.strip().str.replace("\n", " ", regex=False)

    return df1, df2


def tidy_df1(df1: pd.DataFrame) -> pd.DataFrame:
    cols_map = {
        "Área": "area",
        "Tipo de institución": "tipo_institucion",
        "Carrera genérica": "carrera_generica",
        "1er año": "ingreso_1a",
        "2° año": "ingreso_2a",
        "3er año": "ingreso_3a",
        "4° año": "ingreso_4a",
        "Empleabilidad 1er año": "empleab_1a",
        "Empleabilidad 2° año": "empleab_2a",
    }
    keep = [c for c in cols_map.keys() if c in df1.columns]
    out = df1[keep].rename(columns=cols_map).copy()
    return out


def tidy_df2(df2: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Nombre institución",
        "Nombre carrera",
        "Región",
        "Jornada",
        "Sede",
        "Arancel Anual 2025",
        "Promedio PAES 2024 de Matrícula 1er año 2024",
        "Promedio NEM 2024 de Matrícula 2024",
        "Vacantes 1er semestre ",
        "Área Carrera Genérica",
    ]
    have = [c for c in cols if c in df2.columns]
    df = df2[have].copy()

    rename_map = {
        "Nombre institución": "institucion",
        "Nombre carrera": "carrera",
        "Región": "region",
        "Jornada": "jornada",
        "Sede": "sede",
        "Arancel Anual 2025": "arancel_2025",
        "Promedio PAES 2024 de Matrícula 1er año 2024": "prom_paes_2024",
        "Promedio NEM 2024 de Matrícula 2024": "prom_nem_2024",
        "Vacantes 1er semestre ": "vacantes_1s",
        "Área Carrera Genérica": "carrera_generica",
    }
    df = df.rename(columns=rename_map)

    # Convertir a numérico
    for c in ["prom_paes_2024", "prom_nem_2024", "arancel_2025"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_master(df1_tidy: pd.DataFrame, df2_tidy: pd.DataFrame) -> pd.DataFrame:
    # PAES promedio por carrera genérica
    paes_group = (
        df2_tidy[["carrera_generica", "prom_paes_2024"]]
        .groupby("carrera_generica", as_index=False)
        .mean(numeric_only=True)
    )

    master = pd.merge(df1_tidy, paes_group, on="carrera_generica", how="inner")

    # Filtrar solo universitarias con PAES válido
    master = master[master["prom_paes_2024"].notna() & (master["prom_paes_2024"] > 0)]
    master = master[~master["carrera_generica"].str.contains("Técnico", case=False, na=False)]

    # Normalización min-max
    def minmax(s: pd.Series):
        if s.min() == s.max():
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    master["ingreso_2a_norm"] = minmax(master["ingreso_2a"])
    master["empleab_1a_norm"] = minmax(master["empleab_1a"])
    master["paes_norm"] = minmax(master["prom_paes_2024"])

    # Índice de eficiencia
    master["eficiencia"] = (
        0.5 * master["ingreso_2a_norm"]
        + 0.3 * master["empleab_1a_norm"]
        + 0.2 * (1 - master["paes_norm"])
    )

    master = master.sort_values("eficiencia", ascending=False).reset_index(drop=True)
    master["ranking"] = np.arange(1, len(master) + 1)
    master["eficiencia"] = master["eficiencia"].round(4)

    return master


# -------------------------------------------------------------
# Carga de datos (de fábrica, ya en el repo)
# -------------------------------------------------------------
DATA_FILE1 = Path("Buscador_Estadisticas_por_carrera_2024_2025_SIES (2).xlsx")
DATA_FILE2 = Path("Buscador_de_Carreras_2024_2025_SIES_-2.xlsx")

if not (DATA_FILE1.exists() and DATA_FILE2.exists()):
    st.error(
        "No se encontraron los archivos de datos.\n"
        "Debes colocar junto a `app.py`:\n\n"
        "- Buscador_Estadisticas_por_carrera_2024_2025_SIES (2).xlsx\n"
        "- Buscador_de_Carreras_2024_2025_SIES_-2.xlsx"
    )
    st.stop()

df1_raw, df2_raw = read_excels_from_files(str(DATA_FILE1), str(DATA_FILE2))
df1_tidy = tidy_df1(df1_raw)
df2_tidy = tidy_df2(df2_raw)
master = build_master(df1_tidy, df2_tidy)

st.success(f"Ranking construido con {len(master)} carreras genéricas.")

# -------------------------------------------------------------
# Interfaz
# -------------------------------------------------------------
tabs = st.tabs(["🔎 Buscar carrera", "🏆 Ranking", "📈 PAES vs Ingreso"])

# --- Tab 1: Buscar carrera ---
with tabs[0]:
    carreras = sorted(master["carrera_generica"].unique())
    q = st.selectbox("Selecciona carrera genérica", options=carreras)

    if q:
        row = master.loc[master["carrera_generica"] == q]
        st.metric("Ingreso 2º año", f"$ {row['ingreso_2a'].iloc[0]:,.0f}")
        st.metric("Empleabilidad 1er año", f"{row['empleab_1a'].iloc[0]*100:.1f}%")
        st.metric("Promedio PAES 2024", f"{row['prom_paes_2024'].iloc[0]:.1f}")
        st.metric("Índice de eficiencia", f"{row['eficiencia'].iloc[0]:.2f}")

        st.markdown("#### Detalle institucional")
        inst = df2_tidy[df2_tidy["carrera_generica"] == q][
            ["institucion", "carrera", "sede", "region", "jornada", "arancel_2025", "prom_paes_2024", "prom_nem_2024", "vacantes_1s"]
        ].copy()
        st.dataframe(inst)

# --- Tab 2: Ranking ---
with tabs[1]:
    st.dataframe(master[["ranking", "carrera_generica", "eficiencia", "ingreso_2a", "empleab_1a", "prom_paes_2024"]])

# --- Tab 3: PAES vs Ingreso ---
with tabs[2]:
    fig = px.scatter(
        master,
        x="prom_paes_2024",
        y="ingreso_2a",
        hover_data=["carrera_generica"],
        trendline="ols",
        labels={"prom_paes_2024": "Promedio PAES", "ingreso_2a": "Ingreso 2º año"},
    )
    if SCIPY:
        r_p, _ = pearsonr(master["prom_paes_2024"], master["ingreso_2a"])
        r_s, _ = spearmanr(master["prom_paes_2024"], master["ingreso_2a"])
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
            text=f"Pearson r = {r_p:.2f} | Spearman ρ = {r_s:.2f}", bgcolor="white"
        )
    st.plotly_chart(fig, use_container_width=True)

st.caption("© 2025 – Prototipo académico. Fuente: Subsecretaría de Educación Superior – MiFuturo.cl (SIES 2024–2025).")
