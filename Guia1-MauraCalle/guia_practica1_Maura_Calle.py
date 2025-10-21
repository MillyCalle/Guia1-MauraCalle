# app.py — EDA Compras Públicas Ecuador (VERSIÓN DEBUG)
import streamlit as st
import pandas as pd
import numpy as np
import requests, time
import plotly.express as px
from datetime import datetime
from scipy.stats import pearsonr, spearmanr

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
    
    for year in years:
        page = 1
        records_found = 0
        while page <= max_pages:
            params = {"year": year, "search": keyword, "page": page}
            
            try:
                js = safe_get_json(API_URL, params=params)
                if js is None:
                    st.warning(f"❌ No se pudo obtener datos para año {year}, página {page}")
                    break
            except Exception as e:
                st.warning(f"❌ Error año={year}, página={page}: {e}")
                break
                
            data = js.get("data") or []
            if not data:
                st.info(f"ℹ️ No hay más datos en año {year}, página {page}")
                break
                
            for rec in data:
                flat = extract_fields(rec)
                flat["year_api"] = year
                all_rows.append(flat)
                records_found += 1
                
            page += 1
            time.sleep(sleep)
        
        if records_found > 0:
            st.success(f"✅ Año {year}: {records_found} registros")
    
    if not all_rows:
        st.error("⚠️ No se encontraron datos. Verifica la API o cambia los parámetros.")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_rows)
    return df

def normalize_df(df):
    """Limpieza y estandarización"""
    if df.empty:
        return df
        
    df = df.copy()
    
    # Convertir fecha
    if "award_date" in df.columns:
        df["date"] = pd.to_datetime(df["award_date"], errors="coerce")
        st.write(f"🔍 Fechas válidas: {df['date'].notna().sum()} de {len(df)}")
    
    # Convertir total a numérico
    if "total" in df.columns:
        df["total"] = pd.to_numeric(df["total"], errors="coerce")
        st.write(f"🔍 Montos válidos: {df['total'].notna().sum()} de {len(df)}")
        st.write(f"💰 Rango de montos: ${df['total'].min():.2f} - ${df['total'].max():.2f}")
    
    # Normalizar texto
    if "buyer" in df.columns:
        df["buyer"] = df["buyer"].astype(str).str.title()
        st.write(f"🔍 Compradores únicos: {df['buyer'].nunique()}")
    
    if "internal_type" in df.columns:
        df["internal_type"] = df["internal_type"].astype(str).str.title()
        st.write(f"🔍 Tipos de contratación únicos: {df['internal_type'].nunique()}")
        st.write(f"📋 Tipos encontrados: {df['internal_type'].unique().tolist()}")
    
    # Crear columnas derivadas
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["contracts"] = 1
    
    # Remover duplicados y nulos
    initial_count = len(df)
    df = df.drop_duplicates()
    after_dupes = len(df)
    df = df.dropna(subset=["total"])
    final_count = len(df)
    
    st.write(f"🧹 Duplicados eliminados: {initial_count - after_dupes}")
    st.write(f"🧹 Registros sin monto eliminados: {after_dupes - final_count}")
    st.write(f"✅ **Registros finales: {final_count}**")
    
    return df

# ==================== SIDEBAR ====================
st.sidebar.header("⚙️ Configuración")
years = list(range(2020, datetime.now().year + 1))
years_sel = st.sidebar.multiselect("Años a analizar", options=years, default=[2024])

keywords_list = ["obra", "servicio", "suministro", "construcción", "consultoria"]
keyword = st.sidebar.selectbox("Palabra clave (search)", options=keywords_list, index=0)

max_pages = st.sidebar.slider("Páginas por año (limit API)", 1, 10, 3)

if st.sidebar.button("🔄 Limpiar caché"):
    st.cache_data.clear()
    st.sidebar.success("Caché limpiado")
    st.rerun()

load_btn = st.sidebar.button("🚀 Cargar datos desde API", type="primary")

# ==================== CARGA DE DATOS ====================
if 'df_loaded' not in st.session_state:
    st.session_state.df_loaded = None
    st.session_state.df_clean = None

if load_btn:
    if not years_sel:
        st.error("⚠️ Selecciona al menos un año")
        st.stop()
    
    st.markdown("### 📡 Cargando datos desde API...")
    df_raw = load_api_data(years_sel, keyword=keyword, max_pages=max_pages)
    
    if df_raw.empty:
        st.error("❌ No se encontraron datos con los parámetros seleccionados")
        st.stop()
    
    st.success(f"✅ Datos cargados: {len(df_raw)} registros brutos")
    st.session_state.df_loaded = df_raw
    
    # Normalizar inmediatamente
    st.markdown("### 🧹 Limpiando y procesando datos...")
    df_clean = normalize_df(df_raw)
    
    if df_clean.empty:
        st.error("❌ No hay datos válidos después de la limpieza")
        st.stop()
    
    st.session_state.df_clean = df_clean
    st.success("✅ Datos procesados y listos para análisis")

# Verificar si hay datos cargados
if st.session_state.df_clean is None:
    st.info("👈 Selecciona parámetros en el panel lateral y presiona **🚀 Cargar datos desde API**")
    st.markdown("""
    ### 📋 Instrucciones:
    1. Selecciona uno o más **años** a analizar
    2. Elige una **palabra clave** para buscar
    3. Ajusta el número de **páginas por año**
    4. Presiona **🚀 Cargar datos desde API**
    
    ⏱️ La carga puede tomar algunos minutos.
    """)
    
    # MODO DE PRUEBA: Generar datos sintéticos
    if st.checkbox("🧪 Modo prueba: Generar datos sintéticos"):
        st.warning("Generando datos de prueba...")
        np.random.seed(42)
        n = 500
        df = pd.DataFrame({
            'ocid': [f'OCID-{i}' for i in range(n)],
            'buyer': np.random.choice(['Municipio de Quito', 'Ministerio de Salud', 'GAD Guayaquil'], n),
            'supplier': np.random.choice(['Empresa A', 'Empresa B', 'Empresa C'], n),
            'total': np.random.uniform(1000, 1000000, n),
            'internal_type': np.random.choice(['Obras', 'Servicios', 'Bienes'], n),
            'date': pd.date_range('2024-01-01', periods=n, freq='D'),
            'year': 2024,
            'month': np.random.randint(1, 13, n)
        })
        st.session_state.df_clean = df
        st.success(f"✅ Datos de prueba generados: {len(df)} registros")
    else:
        st.stop()

# Usar datos limpios
df = st.session_state.df_clean

# Mostrar info de datos cargados
st.markdown("---")
st.markdown("## 📊 Datos Cargados")
col1, col2, col3 = st.columns(3)
col1.metric("Registros totales", len(df))
col2.metric("Columnas", len(df.columns))
col3.metric("Rango de fechas", f"{df['year'].min():.0f} - {df['year'].max():.0f}" if 'year' in df.columns else "N/A")

with st.expander("🔍 Ver muestra de datos"):
    st.dataframe(df.head(20))
    st.write("**Columnas disponibles:**", df.columns.tolist())
    st.write("**Info del DataFrame:**")
    st.write(df.dtypes)

# ==================== FILTROS DINÁMICOS ====================
st.sidebar.markdown("---")
st.sidebar.header("🔍 Filtros de Análisis")

df_f = df.copy()

# Filtro de año
if 'year' in df.columns and df['year'].notna().any():
    years_available = sorted([int(y) for y in df["year"].dropna().unique()])
    year_f = st.sidebar.selectbox("Filtrar por año", options=["Todos"] + years_available)
    if year_f != "Todos":
        df_f = df_f[df_f["year"] == int(year_f)]

# Filtro de tipo
if 'internal_type' in df.columns and df['internal_type'].notna().any():
    types_available = sorted(df["internal_type"].dropna().unique().tolist())
    type_f = st.sidebar.selectbox("Tipo contratación", options=["Todos"] + types_available)
    if type_f != "Todos":
        df_f = df_f[df_f["internal_type"] == type_f]

# Filtro de comprador
if 'buyer' in df.columns and df['buyer'].notna().any():
    buyers_available = sorted(df["buyer"].dropna().unique().tolist())[:50]
    buyer_f = st.sidebar.selectbox("Entidad compradora", options=["Todas"] + buyers_available)
    if buyer_f != "Todas":
        df_f = df_f[df_f["buyer"] == buyer_f]

st.sidebar.info(f"Registros después de filtros: {len(df_f)}")

if df_f.empty:
    st.warning("⚠️ Los filtros eliminaron todos los datos. Ajusta los filtros.")
    st.stop()

# ==================== KPIs ====================
st.markdown("---")
st.markdown("## 🧮 KPIs Generales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total registros", f"{len(df_f):,}")
col2.metric("Monto total (USD)", f"${df_f['total'].sum():,.0f}")
col3.metric("Promedio por registro", f"${df_f['total'].mean():,.0f}")
col4.metric("Entidades únicas", f"{df_f['buyer'].nunique()}" if 'buyer' in df_f.columns else "N/A")

# ==================== VISUALIZACIONES ====================
st.markdown("---")
st.markdown("## 📈 Visualizaciones")

# Gráfico 1: Total por tipo
st.markdown("### 1️⃣ Monto Total por Tipo de Contratación")
if "internal_type" in df_f.columns and df_f["internal_type"].notna().sum() > 0:
    agg_type = df_f.groupby("internal_type")["total"].sum().reset_index().sort_values("total", ascending=False)
    st.write(f"Registros para este gráfico: {len(agg_type)}")
    if not agg_type.empty:
        fig1 = px.bar(agg_type, x="internal_type", y="total", 
                      title="Monto Total por Tipo de Contratación",
                      labels={"total": "Monto Total (USD)", "internal_type": "Tipo"},
                      text_auto='.2s')
        fig1.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("No hay datos de tipo de contratación")
else:
    st.warning("⚠️ No hay datos válidos para el gráfico de tipos")

# Gráfico 2: Serie temporal
st.markdown("### 2️⃣ Evolución Temporal de Montos")
if "date" in df_f.columns and df_f["date"].notna().sum() > 0:
    df_dated = df_f.dropna(subset=["date"]).copy()
    st.write(f"Registros con fecha válida: {len(df_dated)}")
    if len(df_dated) > 0:
        monthly = df_dated.groupby(df_dated["date"].dt.to_period("M"))["total"].sum().reset_index()
        monthly["date"] = monthly["date"].astype(str)
        st.write(f"Meses únicos: {len(monthly)}")
        fig2 = px.line(monthly, x="date", y="total", 
                      title="Evolución Mensual de Montos", 
                      markers=True,
                      labels={"total": "Monto (USD)", "date": "Mes"})
        fig2.update_layout(height=500)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No hay suficientes datos con fechas válidas")
else:
    st.warning("⚠️ No hay datos de fecha para el gráfico temporal")

# Gráfico 3: Top compradores
st.markdown("### 3️⃣ Top 10 Entidades por Monto")
if "buyer" in df_f.columns and df_f["buyer"].notna().sum() > 0:
    top_buyers = df_f.groupby("buyer")["total"].sum().reset_index().sort_values("total", ascending=False).head(10)
    st.write(f"Entidades únicas: {len(top_buyers)}")
    if not top_buyers.empty:
        fig3 = px.bar(top_buyers, x="buyer", y="total", 
                      title="Top 10 Entidades Compradoras por Monto", 
                      text_auto=".2s",
                      labels={"total": "Monto Total (USD)", "buyer": "Entidad"})
        fig3.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("No hay datos de compradores")
else:
    st.warning("⚠️ No hay datos de compradores")

# Gráfico 4: Distribución tipo (pie)
st.markdown("### 4️⃣ Distribución de Tipos de Contratación")
if "internal_type" in df_f.columns and df_f["internal_type"].notna().sum() > 0:
    counts = df_f["internal_type"].value_counts().reset_index()
    counts.columns = ["Tipo", "Cantidad"]
    st.write(f"Tipos únicos: {len(counts)}")
    if not counts.empty:
        fig4 = px.pie(counts, names="Tipo", values="Cantidad", 
                      title="Distribución por Tipo de Contratación",
                      hole=0.3)
        fig4.update_layout(height=500)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("No hay datos para el gráfico circular")
else:
    st.warning("⚠️ No hay datos de tipo para el gráfico circular")

# Gráfico 5: Heatmap
st.markdown("### 5️⃣ Heatmap Año × Mes")
if all(col in df_f.columns for col in ["year", "month", "total"]):
    df_heat = df_f.dropna(subset=["year", "month"])
    st.write(f"Registros para heatmap: {len(df_heat)}")
    if len(df_heat) > 0:
        heat = df_heat.pivot_table(index="year", columns="month", values="total", aggfunc="sum", fill_value=0)
        st.write(f"Dimensiones del heatmap: {heat.shape}")
        if not heat.empty:
            fig5 = px.imshow(heat, aspect="auto", 
                            labels={"x": "Mes", "y": "Año", "color": "Monto (USD)"},
                            title="Heatmap de Montos por Año y Mes",
                            color_continuous_scale="Viridis")
            fig5.update_layout(height=500)
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.warning("No hay suficientes datos para crear el heatmap")
    else:
        st.warning("No hay datos con año y mes válidos")
else:
    st.warning("⚠️ Faltan columnas necesarias para el heatmap")

# ==================== DESCARGA ====================
st.markdown("---")
csv = df_f.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Descargar datos filtrados (CSV)", 
    data=csv, 
    file_name=f"compras_ecuador_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", 
    mime="text/csv"
)

st.markdown("---")
st.markdown("*Desarrollado con Streamlit | Datos: Compras Públicas Ecuador*")
