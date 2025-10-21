# app.py — EDA Compras Públicas Ecuador (API search_ocds) - VERSIÓN MEJORADA
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
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error en la petición: {str(e)}")
        return None

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

@st.cache_data(ttl=3600, show_spinner=False)
def load_api_data(years, keyword="obra", max_pages=5, sleep=0.5):
    """Carga datos desde la API por rango de años"""
    all_rows = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_iterations = len(years) * max_pages
    current = 0
    
    for year in years:
        page = 1
        records_found = 0
        while page <= max_pages:
            params = {"year": year, "search": keyword, "page": page}
            status_text.text(f"Cargando año {year}, página {page}/{max_pages}...")
            
            try:
                js = safe_get_json(API_URL, params=params)
                if js is None:
                    break
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
                records_found += 1
                
            page += 1
            current += 1
            progress_bar.progress(min(current / total_iterations, 1.0))
            time.sleep(sleep)
        
        st.info(f"Año {year}: {records_found} registros encontrados")
    
    progress_bar.empty()
    status_text.empty()
    
    if not all_rows:
        st.error("⚠️ No se encontraron datos. Intenta con otros parámetros.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    return df

def normalize_df(df):
    """Limpieza y estandarización"""
    if df.empty:
        return df
        
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
    
    # Remover duplicados y nulos
    initial_count = len(df)
    df = df.drop_duplicates().dropna(subset=["total"])
    final_count = len(df)
    
    if initial_count > final_count:
        st.info(f"Se eliminaron {initial_count - final_count} registros duplicados o sin monto")
    
    return df

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Configuración")
years = list(range(2015, datetime.now().year + 1))
years_sel = st.sidebar.multiselect("Años a analizar", options=years, default=[2024, 2025])

# Lista de palabras clave
keywords_list = ["obra", "servicio", "suministro", "construcción", "consultoria"]
keyword = st.sidebar.selectbox("Palabra clave (search)", options=keywords_list, index=0)

max_pages = st.sidebar.slider("Páginas por año (limit API)", 1, 20, 5)

# Botón para limpiar caché
if st.sidebar.button("🔄 Limpiar caché"):
    st.cache_data.clear()
    st.sidebar.success("Caché limpiado")

load_btn = st.sidebar.button("🚀 Cargar datos desde API", type="primary")

# ==================== CARGA DE DATOS ====================
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None

if load_btn:
    if not years_sel:
        st.error("⚠️ Selecciona al menos un año")
        st.stop()
    
    with st.spinner("Descargando datos desde la API..."):
        df_raw = load_api_data(years_sel, keyword=keyword, max_pages=max_pages)
        
        if df_raw.empty:
            st.error("No se encontraron datos con los parámetros seleccionados")
            st.stop()
        
        st.success(f"✅ Datos cargados: {len(df_raw)} registros.")
        st.session_state.df_loaded = df_raw

# Usar datos de session_state si existen
if st.session_state.df_loaded is not None:
    df_raw = st.session_state.df_loaded
else:
    st.info("👈 Selecciona parámetros en el panel lateral y presiona **🚀 Cargar datos desde API**")
    st.markdown("""
    ### Instrucciones:
    1. Selecciona uno o más **años** a analizar
    2. Elige una **palabra clave** para buscar
    3. Ajusta el número de **páginas por año** (más páginas = más datos)
    4. Presiona **🚀 Cargar datos desde API**
    
    ⏱️ La carga puede tomar varios minutos dependiendo de los parámetros.
    """)
    st.stop()

# Mostrar muestra de datos crudos
with st.expander("🔍 Ver muestra de datos crudos"):
    st.dataframe(df_raw.head(20))
    st.write(f"Columnas: {', '.join(df_raw.columns.tolist())}")

# ==================== NORMALIZACIÓN ====================
df = normalize_df(df_raw)

if df.empty:
    st.error("⚠️ No hay datos válidos después de la limpieza")
    st.stop()

st.success(f"✅ Datos procesados: {df.shape[0]} registros, {df.shape[1]} columnas")

# ==================== FILTROS DINÁMICOS ====================
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filtros EDA")

# Filtro de año
years_available = sorted(df["year"].dropna().unique().tolist())
year_f = st.sidebar.selectbox("Filtrar por año", options=["Todos"] + years_available)

# Filtro de tipo
types_available = sorted(df["internal_type"].dropna().unique().tolist())
type_f = st.sidebar.selectbox("Tipo contratación", options=["Todos"] + types_available)

# Filtro de comprador
buyers_available = sorted(df["buyer"].dropna().unique().tolist())
buyer_f = st.sidebar.selectbox("Entidad compradora", options=["Todas"] + buyers_available[:100])  # Limitar a 100

# Aplicar filtros
df_f = df.copy()
if year_f != "Todos":
    df_f = df_f[df_f["year"] == year_f]
if type_f != "Todos":
    df_f = df_f[df_f["internal_type"] == type_f]
if buyer_f != "Todas":
    df_f = df_f[df_f["buyer"] == buyer_f]

if df_f.empty:
    st.warning("⚠️ Los filtros seleccionados no devuelven resultados. Ajusta los filtros.")
    st.stop()

# ==================== KPIs ====================
st.markdown("### 🧮 KPIs generales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total registros", f"{len(df_f):,}")
col2.metric("Monto total (USD)", f"${df_f['total'].sum():,.0f}")
col3.metric("Promedio por registro", f"${df_f['total'].mean():,.0f}")
col4.metric("Entidades únicas", f"{df_f['buyer'].nunique()}")

# ==================== VISUALIZACIONES ====================
st.markdown("---")
st.markdown("## 📊 Visualizaciones")

# a) Total por tipo
if "internal_type" in df_f.columns and not df_f["internal_type"].isna().all():
    agg_type = df_f.groupby("internal_type")["total"].sum().reset_index().sort_values("total", ascending=False)
    fig1 = px.bar(agg_type, x="internal_type", y="total", 
                  title="Total de Montos por Tipo de Contratación",
                  labels={"total": "Monto (USD)", "internal_type": "Tipo"})
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)

# b) Serie temporal mensual
if "date" in df_f.columns and not df_f["date"].isna().all():
    df_f_dated = df_f.dropna(subset=["date"])
    if not df_f_dated.empty:
        monthly = df_f_dated.groupby(df_f_dated["date"].dt.to_period("M"))["total"].sum().reset_index()
        monthly["date"] = monthly["date"].astype(str)
        fig2 = px.line(monthly, x="date", y="total", 
                      title="Evolución mensual de montos", 
                      markers=True,
                      labels={"total": "Monto (USD)", "date": "Mes"})
        st.plotly_chart(fig2, use_container_width=True)

# c) Buyer x monto
if "buyer" in df_f.columns and not df_f["buyer"].isna().all():
    top_buyers = df_f.groupby("buyer")["total"].sum().reset_index().sort_values("total", ascending=False).head(10)
    fig3 = px.bar(top_buyers, x="buyer", y="total", 
                  title="Top 10 Entidades por Monto", 
                  text_auto=".2s",
                  labels={"total": "Monto (USD)", "buyer": "Entidad"})
    fig3.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig3, use_container_width=True)

# d) Pie tipo
if "internal_type" in df_f.columns and not df_f["internal_type"].isna().all():
    counts = df_f["internal_type"].value_counts().reset_index()
    counts.columns = ["Tipo", "Cantidad"]
    fig4 = px.pie(counts, names="Tipo", values="Cantidad", 
                  title="Distribución de tipos de contratación")
    st.plotly_chart(fig4, use_container_width=True)

# e) Heatmap año x mes
if {"year", "month", "total"}.issubset(df_f.columns):
    df_heat = df_f.dropna(subset=["year", "month"])
    if not df_heat.empty:
        heat = df_heat.pivot_table(index="year", columns="month", values="total", aggfunc="sum", fill_value=0)
        fig5 = px.imshow(heat, aspect="auto", 
                        labels={"x": "Mes", "y": "Año", "color": "Monto total"},
                        title="Heatmap Año × Mes (Monto total)")
        st.plotly_chart(fig5, use_container_width=True)

# ==================== CORRELACIONES ====================
st.markdown("---")
st.markdown("## 📈 Correlaciones con `total`")
num_cols = df_f.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != "total" and df_f[c].notna().sum() > 10]

if num_cols:
    corr_out = []
    for c in num_cols:
        mask = df_f[["total", c]].dropna()
        if len(mask) < 10:
            continue
        try:
            r_p, p_p = pearsonr(mask["total"], mask[c])
            r_s, p_s = spearmanr(mask["total"], mask[c])
            corr_out.append((c, r_p, p_p, r_s, p_s))
        except:
            continue
    
    if corr_out:
        corr_df = pd.DataFrame(corr_out, columns=["variable", "pearson_r", "pearson_p", "spearman_r", "spearman_p"])
        st.dataframe(corr_df.style.format({
            'pearson_r': '{:.3f}',
            'pearson_p': '{:.4f}',
            'spearman_r': '{:.3f}',
            'spearman_p': '{:.4f}'
        }))
    else:
        st.info("No hay suficientes datos para calcular correlaciones")
else:
    st.info("No hay columnas numéricas para calcular correlaciones")

# ==================== DESCARGA ====================
st.markdown("---")
csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Descargar datos filtrados (CSV)", 
    data=csv, 
    file_name=f"compras_ecuador_{keyword}_{datetime.now().strftime('%Y%m%d')}.csv", 
    mime="text/csv"
)

# ==================== ESTADÍSTICAS ADICIONALES ====================
with st.expander("📊 Estadísticas adicionales"):
    st.write("**Estadísticas descriptivas de montos:**")
    st.write(df_f["total"].describe())
    
    if "buyer" in df_f.columns:
        st.write(f"\n**Total de entidades compradoras:** {df_f['buyer'].nunique()}")
    
    if "supplier" in df_f.columns:
        st.write(f"**Total de proveedores:** {df_f['supplier'].nunique()}")

st.markdown("---")
st.markdown("### 📝 Conclusiones")
st.markdown("""
- Los datos provienen directamente del API público `search_ocds` de Compras Públicas Ecuador.  
- Los KPIs y gráficos permiten identificar tendencias por año, tipo y entidad compradora.  
- La correlación entre variables numéricas es exploratoria y no implica causalidad.  
- Los datos se cachean por 1 hora para mejorar el rendimiento.
""")

# Footer
st.markdown("---")
st.markdown("*Desarrollado con Streamlit | Datos: Portal de Compras Públicas Ecuador*")
