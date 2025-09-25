
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
import os, uuid
from posthog import Posthog

POSTHOG_API_KEY = os.getenv("POSTHOG_API_KEY")
POSTHOG_HOST = os.getenv("POSTHOG_HOST", "https://us.i.posthog.com")
posthog = Posthog(POSTHOG_API_KEY, host=POSTHOG_HOST) if POSTHOG_API_KEY else None

# ID único por visitante (por sesión de Streamlit)
if "distinct_id" not in st.session_state:
    st.session_state["distinct_id"] = str(uuid.uuid4())

def track(event_name, **props):
    if posthog:
        posthog.capture(
            distinct_id=st.session_state["distinct_id"],
            event=event_name,
            properties=props
        )

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
# Utilidades
# -------------------------------------------------------------
def fix_percent_series(s: pd.Series) -> pd.Series:
    """Asegura que una serie de porcentajes esté en [0,1] (si viene en 0-100 la divide por 100)."""
    s2 = pd.to_numeric(s, errors="coerce")
    if s2.dropna().max() is not None and s2.dropna().max() > 1.5:
        return s2 / 100.0
    return s2

@st.cache_data(show_spinner=False)
def read_excels_from_files(file1_path: str, file2_path: str):
    df1 = pd.read_excel(file1_path, sheet_name="Estadísticas x CG", header=1)
    df2 = pd.read_excel(file2_path, sheet_name="Buscador Carreras  2024-2025")

    # Normalizar nombres de columnas (sin cortes de comillas)
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

    # Empleabilidades en [0,1]
    if "empleab_1a" in out.columns:
        out["empleab_1a"] = fix_percent_series(out["empleab_1a"])
    if "empleab_2a" in out.columns:
        out["empleab_2a"] = fix_percent_series(out["empleab_2a"])

    # Ingresos a numérico
    for c in ["ingreso_1a", "ingreso_2a", "ingreso_3a", "ingreso_4a"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def tidy_df2(df2: pd.DataFrame) -> pd.DataFrame:
    # Columnas clave + Tipo de institución para filtrar Universidades
    cols = [
        "Nombre institución",
        "Nombre carrera",
        "Región",
        "Jornada",
        "Sede",
        "Arancel Anual 2025",
        "Promedio PAES 2024 de Matrícula 1er año 2024",
        "Promedio NEM 2024 de Matrícula 2024",
        "Vacantes 1er semestre",
        "Área Carrera Genérica",
        "Tipo de institución",
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
        "Vacantes 1er semestre": "vacantes_1s",
        "Área Carrera Genérica": "carrera_generica",
        "Tipo de institución": "tipo_institucion",
    }
    df = df.rename(columns=rename_map)

    # Numéricos
    for c in ["prom_paes_2024", "prom_nem_2024", "arancel_2025", "vacantes_1s"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def build_master(df1_tidy: pd.DataFrame, df2_tidy: pd.DataFrame) -> pd.DataFrame:
    # Filtrar solo Universidades en ambos (si existe la columna)
    df1u = df1_tidy.copy()
    if "tipo_institucion" in df1u.columns:
        df1u = df1u[df1u["tipo_institucion"].str.contains("Universidad", case=False, na=False)]

    df2u = df2_tidy.copy()
    if "tipo_institucion" in df2u.columns:
        df2u = df2u[df2u["tipo_institucion"].str.contains("Universidad", case=False, na=False)]

    # PAES promedio por carrera genérica (solo universidades)
    paes_group = (
        df2u[["carrera_generica", "prom_paes_2024"]]
        .groupby("carrera_generica", as_index=False)
        .mean(numeric_only=True)
    )

    master = pd.merge(df1u, paes_group, on="carrera_generica", how="inner")

    # Filtrar: PAES válido y excluir Técnicos
    master = master[master["prom_paes_2024"].notna() & (master["prom_paes_2024"] > 0)]
    master = master[~master["carrera_generica"].str.contains("Técnico", case=False, na=False)]

    # Normalización min-max
    def minmax(s: pd.Series):
        s = pd.to_numeric(s, errors="coerce")
        if s.dropna().empty or s.min() == s.max():
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
# Carga "de fábrica" (sin subir archivos)
# -------------------------------------------------------------
DATA_FILE1 = Path("Buscador_Estadisticas_por_carrera_2024_2025_SIES (2).xlsx")
DATA_FILE2 = Path("Buscador_de_Carreras_2024_2025_SIES_-2.xlsx")

if not (DATA_FILE1.exists() and DATA_FILE2.exists()):
    st.error(
        "No se encontraron los archivos de datos junto a la app.\n\n"
        "Asegúrate de colocar estos dos archivos en la misma carpeta que `app.py`:\n"
        "- Buscador_Estadisticas_por_carrera_2024_2025_SIES (2).xlsx\n"
        "- Buscador_de_Carreras_2024_2025_SIES_-2.xlsx"
    )
    st.stop()

df1_raw, df2_raw = read_excels_from_files(str(DATA_FILE1), str(DATA_FILE2))
df1_tidy = tidy_df1(df1_raw)
df2_tidy = tidy_df2(df2_raw)
master = build_master(df1_tidy, df2_tidy)

st.success(
    f"Datos cargados. Estadísticas: {len(df1_tidy):,} filas | Instituciones: {len(df2_tidy):,} filas | "
    f"Ranking (solo Universidades): {len(master):,} carreras genéricas."
)
track("app_loaded")
# -------------------------------------------------------------
# Sidebar filtros
# -------------------------------------------------------------
st.sidebar.header("Filtros")
areas = sorted(master["area"].dropna().unique()) if "area" in master.columns else []
area_sel = st.sidebar.multiselect("Área (carrera genérica)", areas, default=[])

if area_sel:
    view = master[master["area"].isin(area_sel)].copy()
else:
    view = master.copy()

# -------------------------------------------------------------
# Layout principal
# -------------------------------------------------------------
T1, T2, T3 = st.tabs(["🔎 Buscar carrera", "🏆 Ranking de eficiencia", "📈 PAES vs Ingreso"])

# --- Tab 1: Buscar carrera ---
with T1:
    st.subheader("Buscar carrera genérica")
    carreras = sorted(view["carrera_generica"].unique())
    q = st.selectbox("Selecciona la carrera genérica", options=carreras)

    if q:
        row = view.loc[view["carrera_generica"] == q]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ingreso promedio 2º año", f"$ {row['ingreso_2a'].iloc[0]:,.0f}")
        emp1 = float(row["empleab_1a"].iloc[0]) if not pd.isna(row["empleab_1a"].iloc[0]) else np.nan
        c2.metric("Empleabilidad 1er año", f"{emp1*100:.1f}%" if pd.notna(emp1) else "—")
        c3.metric("Promedio PAES 2024", f"{row['prom_paes_2024'].iloc[0]:.1f}")
        c4.metric("Índice de eficiencia", f"{row['eficiencia'].iloc[0]:.2f}")

        st.markdown("#### Detalle institucional (solo Universidades)")
        # Filtro instituciones de la carrera seleccionada y SOLO Universidades
        mask = df2_tidy["carrera_generica"].eq(q)
        if "tipo_institucion" in df2_tidy.columns:
            mask &= df2_tidy["tipo_institucion"].str.contains("Universidad", case=False, na=False)
        inst = df2_tidy.loc[mask].copy()

        desired_cols = [
            "institucion", "carrera", "sede", "region", "jornada",
            "arancel_2025", "prom_paes_2024", "prom_nem_2024", "vacantes_1s"
        ]
        available = [c for c in desired_cols if c in inst.columns]
        inst = inst[available].sort_values(["prom_paes_2024", "arancel_2025"], ascending=[False, True])

        st.dataframe(inst, use_container_width=True)

        # Descarga CSV
        if not inst.empty:
            csv = inst.to_csv(index=False).encode("utf-8")
            st.download_button("Descargar detalle institucional (CSV)", data=csv, file_name=f"detalle_instituciones_{q}.csv")

# --- Tab 2: Ranking ---
with T2:
    st.subheader("Ranking de eficiencia (solo Universidades con PAES)")
    show_cols = ["ranking", "carrera_generica", "eficiencia", "ingreso_2a", "empleab_1a", "prom_paes_2024", "area"]
    show_cols = [c for c in show_cols if c in view.columns]
    st.dataframe(view[show_cols], use_container_width=True)

    # Descarga Excel
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="xlsxwriter") as writer:
        view[show_cols].to_excel(writer, index=False, sheet_name="ranking")
    st.download_button("Descargar ranking (Excel)", data=xlsx_buf.getvalue(), file_name="ranking_eficiencia.xlsx")

# --- Tab 3: PAES vs Ingreso ---
with T3:
    st.subheader("Relación entre puntaje PAES promedio e ingresos (2º año)")
    dv = view.dropna(subset=["prom_paes_2024", "ingreso_2a"]).copy()

    fig = px.scatter(
        dv,
        x="prom_paes_2024",
        y="ingreso_2a",
        hover_data=["carrera_generica"],
        labels={"prom_paes_2024": "Promedio PAES", "ingreso_2a": "Ingreso 2º año ($)"},
        title="Scatter: PAES vs Ingreso 2º año (Universidades)",
    )

    # Línea de regresión simple + correlaciones
    if len(dv) >= 2:
        x = dv["prom_paes_2024"].to_numpy()
        y = dv["ingreso_2a"].to_numpy()
        m, b = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = m * x_line + b
        line_fig = px.line(x=x_line, y=y_line)
        for tr in line_fig.data:
            fig.add_trace(tr)

        if SCIPY:
            r_p, _ = pearsonr(x, y)
            r_s, _ = spearmanr(x, y)
        else:
            r_p = np.corrcoef(x, y)[0, 1]
            rx = pd.Series(x).rank().to_numpy()
            ry = pd.Series(y).rank().to_numpy()
            r_s = np.corrcoef(rx, ry)[0, 1]

        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
            text=f"Pearson r = {r_p:.2f} | Spearman ρ = {r_s:.2f}",
            bgcolor="white"
        )

    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# Explicación de la fórmula de eficiencia
# -------------------------------------------------------------
st.markdown("""
---
### 📌 Fórmula de cálculo de la eficiencia

El **índice de eficiencia** combina tres factores normalizados (min–max):

- **Ingresos 2º año** (pondera 50%) – a mayor ingreso, mayor eficiencia.
- **Empleabilidad 1er año** (pondera 30%) – a mayor empleabilidad, mayor eficiencia.
- **Promedio PAES 2024** (pondera 20%, invertido) – mientras más bajo el puntaje de ingreso, mayor eficiencia relativa.

La fórmula es:

$$
Eficiencia = 0.5 \\times Ingreso_{norm} + 0.3 \\times Empleabilidad_{norm} + 0.2 \\times (1 - PAES_{norm})
$$

De esta forma, carreras que logran **buenos ingresos y empleabilidad con menores puntajes de entrada** aparecen más arriba en el ranking.
""")

# Footer
st.caption("© 2025 – Prototipo académico. Fuente: Subsecretaría de Educación Superior – MiFuturo.cl (SIES 2024–2025).")
