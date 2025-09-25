# app.py
# -------------------------------------------------------------
# Buscador de carreras (Chile) con datos SIES 2024–2025
# - Integra 2 archivos Excel:
#   1) "Buscador_Estadisticas_por_carrera_2024_2025_SIES (2).xlsx" (hoja: "Estadísticas x CG")
#   2) "Buscador_de_Carreras_2024_2025_SIES_-2.xlsx" (hoja: "Buscador Carreras  2024-2025")
# - Calcula ranking de eficiencia por carrera genérica
# - Explora oferta por institución (PAES/NEM, aranceles, vacantes)
# - Gráficos de correlación PAES vs ingreso
# -------------------------------------------------------------

import io
import numpy as np
import pandas as pd
import streamlit as st

# Gráficos
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Utilidad: correlaciones
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

# -----------------------------
# Utilidades
# -----------------------------
@st.cache_data(show_spinner=False)
def read_excels_from_files(file1_path: str, file2_path: str):
    df1 = pd.read_excel(file1_path, sheet_name="Estadísticas x CG", header=1)
    df2 = pd.read_excel(file2_path, sheet_name="Buscador Carreras  2024-2025")

    # Normalizar columnas (remover saltos de línea y espacios)
df1.columns = df1.columns.str.strip().str.replace("\n", " ", regex=False)
df2.columns = df2.columns.str.strip().str.replace("\n", " ", regex=False)


    return df1, df2


def tidy_df1(df1: pd.DataFrame) -> pd.DataFrame:
    """Selecciona y renombra columnas clave de la base genérica (estadísticas)."""
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
    """Selecciona columnas clave de la base por institución y limpia PAES/NEM numéricos."""
    cols = [
        "Código único de carrera ", "Código institución ", "Área del conocimiento", "Tipo de institución",
        "Nombre institución", "Nombre carrera", "Región", "Jornada", "Sede", "Arancel Anual 2025",
        "Costo de titulación", "Duración Formal (semestres)", "Nivel carrera ",
        "Matrícula Total Femenina 2024", "Matrícula Total Masculina 2024", "Matrícula Total 2024",
        "Promedio PAES 2024 de Matrícula 1er año 2024", "Promedio NEM 2024 de Matrícula 2024",
        "Vacantes 1er semestre ", "NEM", "Ranking ", "PAES Lenguaje", "PAES Matemáticas ",
        "PAES Matemáticas 2", "PAES Historia ", "PAES Ciencias ", "Otros ", "Área Carrera Genérica"
    ]
    have = [c for c in cols if c in df2.columns]
    df = df2[have].copy()

    # Renombrar
    rename_map = {
        "Código único de carrera ": "codigo_carrera",
        "Código institución ": "codigo_institucion",
        "Área del conocimiento": "area_conocimiento",
        "Tipo de institución": "tipo_institucion",
        "Nombre institución": "institucion",
        "Nombre carrera": "carrera",
        "Región": "region",
        "Jornada": "jornada",
        "Sede": "sede",
        "Arancel Anual 2025": "arancel_2025",
        "Costo de titulación": "costo_titulacion",
        "Duración Formal (semestres)": "duracion_sem",
        "Nivel carrera ": "nivel",
        "Matrícula Total Femenina 2024": "matricula_f_2024",
        "Matrícula Total Masculina 2024": "matricula_m_2024",
        "Matrícula Total 2024": "matricula_total_2024",
        "Promedio PAES 2024 de Matrícula 1er año 2024": "prom_paes_2024",
        "Promedio NEM 2024 de Matrícula 2024": "prom_nem_2024",
        "Vacantes 1er semestre ": "vacantes_1s",
        "NEM": "pondera_nem",
        "Ranking ": "pondera_ranking",
        "PAES Lenguaje": "pondera_paes_len",
        "PAES Matemáticas ": "pondera_paes_mat",
        "PAES Matemáticas 2": "pondera_paes_m2",
        "PAES Historia ": "pondera_paes_hist",
        "PAES Ciencias ": "pondera_paes_cien",
        "Otros ": "pondera_otros",
        "Área Carrera Genérica": "carrera_generica",
    }
    df = df.rename(columns=rename_map)

    # Convertir PAES/NEM a numérico
    for c in ["prom_paes_2024", "prom_nem_2024"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_master(df1_tidy: pd.DataFrame, df2_tidy: pd.DataFrame, excluir_tecnicas=True, exigir_paes=True) -> pd.DataFrame:
    """Une bases por carrera_genérica y construye índice de eficiencia.
    Eficiencia = 0.5 * ingreso_2a_norm + 0.3 * empleab_1a_norm + 0.2 * (1 - paes_norm)
    """
    # Agregar PAES promedio por carrera genérica
    paes_group = (df2_tidy[["carrera_generica", "prom_paes_2024"]]
                  .groupby("carrera_generica", as_index=False)
                  .mean(numeric_only=True))

    master = pd.merge(
        df1_tidy,
        paes_group,
        on="carrera_generica",
        how="inner",
    )

    # Filtros solicitados
    if exigir_paes:
        master = master[master["prom_paes_2024"].notna() & (master["prom_paes_2024"] > 0)]
    if excluir_tecnicas:
        master = master[~master["carrera_generica"].str.contains("Técnico", case=False, na=False)]

    # Normalizaciones (min-max)
    def minmax(s: pd.Series):
        if s.min() == s.max():
            return pd.Series(0.5, index=s.index)
        return (s - s.min()) / (s.max() - s.min())

    master["ingreso_2a_norm"] = minmax(master["ingreso_2a"])  # mayor es mejor
    master["empleab_1a_norm"] = minmax(master["empleab_1a"])  # mayor es mejor
    master["paes_norm"] = minmax(master["prom_paes_2024"])     # menor es mejor

    master["eficiencia"] = (
        0.5 * master["ingreso_2a_norm"] +
        0.3 * master["empleab_1a_norm"] +
        0.2 * (1 - master["paes_norm"])  # penaliza PAES alto
    )

    master = master.sort_values("eficiencia", ascending=False).reset_index(drop=True)
    master["ranking"] = np.arange(1, len(master) + 1)

    # Redondeos para presentación
    master["eficiencia"] = master["eficiencia"].round(4)
    return master


# -----------------------------
# Carga "de fábrica" (sin subir archivos)
# -----------------------------
from pathlib import Path
DATA_FILE1 = Path("Buscador_Estadisticas_por_carrera_2024_2025_SIES (2).xlsx")
DATA_FILE2 = Path("Buscador_de_Carreras_2024_2025_SIES_-2.xlsx")

if not (DATA_FILE1.exists() and DATA_FILE2.exists()):
    st.error(
        "No se encontraron los archivos de datos junto a la app.
"
        "Asegúrate de colocar estos dos archivos en la misma carpeta que `app.py`:

"
        "- Buscador_Estadisticas_por_carrera_2024_2025_SIES (2).xlsx
"
        "- Buscador_de_Carreras_2024_2025_SIES_-2.xlsx"
    )
    st.stop()

# Leer y preparar
df1_raw, df2_raw = read_excels_from_files(str(DATA_FILE1), str(DATA_FILE2))

# Leer y preparar
# (ya cargado arriba con read_excels_from_files), f2.getvalue())
df1_tidy = tidy_df1(df1_raw)
df2_tidy = tidy_df2(df2_raw)
master = build_master(df1_tidy, df2_tidy, excluir_tecnicas=True, exigir_paes=True)

st.success(f"Se cargaron {len(df1_tidy):,} filas (estadísticas) y {len(df2_tidy):,} filas (instituciones). Ranking con {len(master):,} carreras genéricas.")

# -----------------------------
# Sidebar filtros
# -----------------------------
st.sidebar.header("Filtros")
areas = sorted(master["area"].dropna().unique()) if "area" in master.columns else []
area_sel = st.sidebar.multiselect("Área (carrera genérica)", areas, default=[])

if area_sel:
    view = master[master["area"].isin(area_sel)].copy()
else:
    view = master.copy()

# -----------------------------
# Layout principal con Tabs
# -----------------------------
T1, T2, T3, T4 = st.tabs([
    "🔎 Buscar carrera",
    "🏆 Ranking de eficiencia",
    "🏫 Instituciones",
    "📈 PAES vs Ingresos",
])

# --- Tab 1: Buscar carrera ---
with T1:
    st.subheader("Buscar carrera genérica")
    carreras = sorted(view["carrera_generica"].unique())
    q = st.selectbox("Selecciona la carrera genérica", options=carreras)

    if q:
        row = view.loc[view["carrera_generica"] == q]
        st.markdown("#### Indicadores clave (promedios)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ingreso promedio 2º año", f"$ {row['ingreso_2a'].iloc[0]:,.0f}")
        c2.metric("Empleabilidad 1er año", f"{row['empleab_1a'].iloc[0]*100:.1f}%")
        c3.metric("Promedio PAES 2024", f"{row['prom_paes_2024'].iloc[0]:.1f}")
        c4.metric("Índice de eficiencia", f"{row['eficiencia'].iloc[0]:.2f}")

        st.markdown("#### Perfil de ingreso por institución (PAES/NEM, aranceles, vacantes)")
        inst = df2_tidy[df2_tidy["carrera_generica"] == q][[
            "institucion", "carrera", "sede", "region", "jornada",
            "arancel_2025", "prom_paes_2024", "prom_nem_2024", "vacantes_1s"
        ]].copy()
        inst = inst.sort_values(["prom_paes_2024", "arancel_2025"], ascending=[False, True])
        st.dataframe(inst, use_container_width=True)

        csv = inst.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar detalle institucional (CSV)", data=csv, file_name=f"detalle_instituciones_{q}.csv")

# --- Tab 2: Ranking ---
with T2:
    st.subheader("Ranking de eficiencia (solo universitarias con PAES)")
    show_cols = ["ranking", "carrera_generica", "eficiencia", "ingreso_2a", "empleab_1a", "prom_paes_2024", "area"]
    show_cols = [c for c in show_cols if c in view.columns]
    st.dataframe(view[show_cols], use_container_width=True)

    # Descargas
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        view[show_cols].to_excel(writer, index=False, sheet_name="ranking")
    st.download_button("Descargar ranking (Excel)", data=xlsx_buf.getvalue(), file_name="ranking_eficiencia.xlsx")

# --- Tab 3: Instituciones ---
with T3:
    st.subheader("Explorar instituciones por carrera genérica")
    csel = st.selectbox("Carrera genérica", options=carreras, key="cinst")
    if csel:
        inst2 = df2_tidy[df2_tidy["carrera_generica"] == csel][[
            "institucion", "carrera", "sede", "region", "jornada", "arancel_2025",
            "prom_paes_2024", "prom_nem_2024", "vacantes_1s"
        ]].copy()
        st.dataframe(inst2.sort_values("prom_paes_2024", ascending=False), use_container_width=True)

# --- Tab 4: PAES vs Ingresos ---
with T4:
    st.subheader("Relación entre puntaje PAES promedio e ingresos (2º año)")
    dv = view.dropna(subset=["prom_paes_2024", "ingreso_2a"]).copy()

    fig = px.scatter(
        dv,
        x="prom_paes_2024",
        y="ingreso_2a",
        hover_data=["carrera_generica"],
        trendline="ols",  # agrega línea de tendencia
        labels={"prom_paes_2024": "Promedio PAES 2024", "ingreso_2a": "Ingreso 2º año ($)"},
        title="Scatter: PAES vs Ingreso 2º año",
    )

    # Calcular correlaciones
    x = dv["prom_paes_2024"].to_numpy()
    y = dv["ingreso_2a"].to_numpy()

    if SCIPY:
        r_p, _ = pearsonr(x, y)
        r_s, _ = spearmanr(x, y)
    else:
        # Pearson con numpy; Spearman vía ranking
        r_p = np.corrcoef(x, y)[0, 1]
        rx = pd.Series(x).rank().to_numpy()
        ry = pd.Series(y).rank().to_numpy()
        r_s = np.corrcoef(rx, ry)[0, 1]

    # Agregar anotación
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
        text=f"Pearson r = {r_p:.2f} | Spearman ρ = {r_s:.2f}",
        bgcolor="white"
    )

    st.plotly_chart(fig, use_container_width=True)

# Footer
st.caption("© 2025 – Prototipo académico. Fuente: Subsecretaría de Educación Superior – MiFuturo.cl (SIES 2024–2025).")
