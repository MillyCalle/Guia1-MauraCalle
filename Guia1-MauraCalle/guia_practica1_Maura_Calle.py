# app.py — EDA Compras Públicas Ecuador (API search_ocds)
import streamlit as st
import pandas as pd
import numpy as np
import requests, time
import plotly.express as px
from datetime import datetime
from scipy.stats import linregress, pearsonr, spearmanr

API_URL = "https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/search_ocds"

st.set_page_config(page_title="EDA Compras Públicas Ecuador", layout="wide")
st.title("📊 EDA: Compras Públicas Ecuador (API search_ocds)")

# ==================== FUNCIONES ====================

def safe_get_json(url, params=None, timeout=30):
    """Obtiene datos de la API con control de errores"""
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def extract_fields(rec):
    """Extrae campos relevantes de un registro JSON de la API search_ocds"""
    flat = {}
    flat["ocid"] = rec.get("ocid") or rec.get("id")

    # Título de la contratación
    flat["tender_title"] = None
    if isinstance(rec.get("tender"), dict):
        flat["tender_title"] = rec["tender"].get("title")

    # Buyer (puede ser str o dict)
    buyer_data = rec.get("buyer")
    if isinstance(buyer_data, dict):
        flat["buyer"] = buyer_data.get("name")
    elif isinstance(buyer_data, str):
        flat["buyer"] = buyer_data
    else:
        flat["buyer"] = None

    # Supplier (anidado dentro de awards)
    flat["supplier"] = None
    if "awards" in rec and isinstance(rec["awards"], list) and len(rec["awards"]) > 0:
        award = rec["awards"][0]
        suppliers = award.get("suppliers", [])
        if suppliers and isinstance(suppliers, list) and isinstance(suppliers[0], dict):
            flat["supplier"] = suppliers[0].get("name")

        # Valor del contrato adjudicado
        val = award.get("value", {})
        if isinstance(val, dict):
            flat["total"] = val.get("amount")
            flat["currency"] = val.get("currency")
        else:
            flat["total"] = None
            flat["currency"] = None

        flat["award_date"] = award.get("date")
    else:
        tender_val = rec.get("tender", {}).get("value", {})
        if isinstance(tender_val, dict):
            flat["total"] = tender_val.get("amount")
            flat["currency"] = tender_val.get("currency")
        else:
            flat["total"] = None
            flat["currency"] = None

        flat["award_date"] = rec.get("tender", {}).get("tenderPeriod", {}).get("startDate")

    # Tipo de contratación
    tender_data = rec.get("tender")
    if isinstance(tender_data, dict):
        flat["internal_type"] = tender_data.get("mainProcurementCategory")
    else:
        flat["internal_type"] = None

    return flat

@st.cache_data(ttl=3600)
def load_api_data(years, keyword="obra", max_pages=5, sleep=0.5):
    """Carga datos desde la API por rango de años"""
    all_rows = []
    for year in years:
        page = 1
        while page <= max_pages:
            params = {"year": year, "search": keyword, "page": page}
            try:
                js = safe_get_json(API_URL, params=params)
            except Exception as e:
                st.warning(f"Error año={year}, página={page}: {e}")
                break
            data = js.get("data") or []
            if not data:
                break
            for rec in data:
                flat = extract_fields(rec)
                flat["year_api"] = year
                all_rows.append(flat)
            page += 1
            time.sleep(sleep)
    df = pd.DataFrame(all_rows)
    return df

def normalize_df(df):
    """Limpieza y estandarización"""
    df = df.copy()
    if "award_date" in df.columns:
        df["date"] = pd.to_datetime(df["award_date"], errors="coerce")
    if "total" in df.columns:
        df["total"] = pd.to_numeric(df["total"], errors="coerce")
    if "buyer" in df.columns:
        df["buyer"] = df["buyer"].astype(str).str.title()
    if "internal_type" in df.columns:
        df["internal_type"] = df["internal_type"].astype(str).str.title()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["contracts"] = 1
    df = df.drop_duplicates().dropna(subset=["total"])
    return df

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Configuración")
years = list(range(2015, datetime.now().year + 1))
years_sel = st.sidebar.multiselect("Años a analizar", options=years, default=[2024, 2025])

# Lista de palabras clave
keywords_list = ["obra", "servicio", "suministro", "construcción"]
keyword = st.sidebar.selectbox("Palabra clave (search)", options=keywords_list, index=0)

max_pages = st.sidebar.slider("Páginas por año (limit API)", 1, 20, 5)
load_btn = st.sidebar.button("Cargar datos desde API")

if not load_btn:
    st.info("Selecciona parámetros y presiona **Cargar datos desde API**")
    st.stop()

# ==================== CARGA DE DATOS ====================
with st.spinner("Descargando datos desde la API..."):
    df_raw = load_api_data(years_sel, keyword=keyword, max_pages=max_pages)
    st.success(f"Datos cargados: {len(df_raw)} registros.")

st.dataframe(df_raw.head())

# ==================== NORMALIZACIÓN ====================
df = normalize_df(df_raw)
st.write("Shape después de limpieza:", df.shape)

# ==================== FILTROS DINÁMICOS ====================
st.sidebar.header("Filtros EDA")
year_f = st.sidebar.selectbox("Filtrar por año", options=["Todos"] + sorted(df["year"].dropna().unique().tolist()))
type_f = st.sidebar.selectbox("Tipo contratación", options=["Todos"] + sorted(df["internal_type"].dropna().unique().tolist()))
buyer_f = st.sidebar.selectbox("Entidad compradora", options=["Todas"] + sorted(df["buyer"].dropna().unique().tolist()))

df_f = df.copy()
if year_f != "Todos":
    df_f = df_f[df_f["year"] == int(year_f)]
if type_f != "Todos":
    df_f = df_f[df_f["internal_type"] == type_f]
if buyer_f != "Todas":
    df_f = df_f[df_f["buyer"] == buyer_f]

# ==================== KPIs ====================
st.markdown("### 🧮 KPIs generales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total registros", f"{len(df_f):,}")
col2.metric("Monto total (USD)", f"{df_f['total'].sum():,.2f}")
col3.metric("Promedio por registro", f"{df_f['total'].mean():,.2f}")
col4.metric("Entidades únicas", f"{df_f['buyer'].nunique()}")

# ==================== VISUALIZACIONES ====================
st.markdown("## 📊 Visualizaciones")

# a) Total por tipo
if "internal_type" in df_f.columns:
    agg_type = df_f.groupby("internal_type")["total"].sum().reset_index().sort_values("total", ascending=False)
    fig1 = px.bar(agg_type, x="internal_type", y="total", title="Total de Montos por Tipo de Contratación")
    st.plotly_chart(fig1, use_container_width=True)

# b) Serie temporal mensual
if "date" in df_f.columns:
    monthly = df_f.groupby(df_f["date"].dt.to_period("M"))["total"].sum().reset_index()
    monthly["date"] = monthly["date"].astype(str)
    fig2 = px.line(monthly, x="date", y="total", title="Evolución mensual de montos", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

# c) Buyer x monto
if "buyer" in df_f.columns:
    top_buyers = df_f.groupby("buyer")["total"].sum().reset_index().sort_values("total", ascending=False).head(10)
    fig3 = px.bar(top_buyers, x="buyer", y="total", title="Top 10 Entidades por Monto", text_auto=".2s")
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

# d) Pie tipo
if "internal_type" in df_f.columns:
    counts = df_f["internal_type"].value_counts().reset_index()
    counts.columns = ["Tipo", "Cantidad"]
    fig4 = px.pie(counts, names="Tipo", values="Cantidad", title="Distribución de tipos de contratación")
    st.plotly_chart(fig4, use_container_width=True)

# e) Heatmap año x mes
if {"year", "month", "total"}.issubset(df_f.columns):
    heat = df_f.pivot_table(index="year", columns="month", values="total", aggfunc="sum", fill_value=0)
    fig5 = px.imshow(heat, aspect="auto", labels={"x": "Mes", "y": "Año", "color": "Monto total"},
                     title="Heatmap Año × Mes (Monto total)")
    st.plotly_chart(fig5, use_container_width=True)

# ==================== CORRELACIONES ====================
st.markdown("## 📈 Correlaciones con `total`")
num_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != "total"]
corr_out = []
for c in num_cols:
    mask = df_f[["total", c]].dropna()
    if len(mask) < 10:
        continue
    r_p, p_p = pearsonr(mask["total"], mask[c])
    r_s, p_s = spearmanr(mask["total"], mask[c])
    corr_out.append((c, r_p, p_p, r_s, p_s))
corr_df = pd.DataFrame(corr_out, columns=["variable", "pearson_r", "pearson_p", "spearman_r", "spearman_p"])
st.dataframe(corr_df.style.format({'pearson_r':'{:.3f}','pearson_p':'{:.4f}','spearman_r':'{:.3f}','spearman_p':'{:.4f}'}))

# ==================== DESCARGA ====================
csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button("📥 Descargar datos filtrados (CSV)", data=csv, file_name="compras_ecuador_filtrado.csv", mime="text/csv")

st.markdown("### 📝 Conclusiones")
st.markdown("""
- Los datos provienen directamente del API público `search_ocds`.  
- Los KPIs y gráficos permiten identificar tendencias por año, tipo y entidad compradora.  
- La correlación entre variables numéricas es exploratoria y no implica causalidad.  
""")
