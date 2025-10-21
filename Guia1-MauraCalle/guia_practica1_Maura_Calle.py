# guia_practica1_Maura_Calle.py
import streamlit as st
import pandas as pd
import numpy as np
import requests, time
import plotly.express as px
from datetime import datetime
from scipy.stats import linregress, pearsonr, spearmanr

BASE_ANALYSIS = "https://datosabiertos.compraspublicas.gob.ec/PLATAFORMA/api/search_ocds"

st.set_page_config(page_title="EDA Compras Públicas - Maura Calle", layout="wide")

st.title("EDA: Compras Públicas (API) — Maura Calle")

# ---------- Helpers ----------
def safe_get_json(url, params=None, timeout=30):
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def extract_record_to_flat_dict(rec):
    out = {}
    def deep_search(o, keyset):
        if o is None:
            return None
        if isinstance(o, dict):
            for k,v in o.items():
                if k.lower() in keyset:
                    return v
            for v in o.values():
                if isinstance(v, (dict,list)):
                    res = deep_search(v, keyset)
                    if res is not None:
                        return res
        elif isinstance(o, list):
            for e in o:
                res = deep_search(e, keyset)
                if res is not None:
                    return res
        return None
    candidates = {
        'date': ['date','fecha','award_date','tender_date'],
        'total': ['total','monto','amount','value'],
        'contracts': ['contracts','contract_count','num_contracts'],
        'internal_type': ['internal_type','type','procurement_type','tipo'],
        'province': ['province','provincia','region','department'],
        'supplier': ['supplier','supplier_name','vendor','proveedor'],
        'buyer': ['buyer','buyer_name','entity','entidad']
    }
    for k, keys in candidates.items():
        val = deep_search(rec, set([x.lower() for x in keys]))
        out[k] = val
    out['ocid'] = rec.get('ocid') or rec.get('id') or None
    return out

def fetch_analysis_page(year=None, region=None, itype=None, page=1, per_page=100, sleep=0.2):
    params = {}
    if year is not None: params['year'] = year
    if region is not None: params['region'] = region
    if itype is not None: params['type'] = itype
    params['page'] = page
    params['per_page'] = per_page
    js = safe_get_json(BASE_ANALYSIS, params=params)
    time.sleep(sleep)
    return js

@st.cache_data(ttl=3600)
def load_data_from_api(years, regions=None, types=None, max_pages=20, sleep=0.1):
    rows = []
    for yr in years:
        combos = [(None,None)]
        if regions and types:
            combos = [(r,t) for r in regions for t in types]
        elif regions:
            combos = [(r,None) for r in regions]
        elif types:
            combos = [(None,t) for t in types]
        for region, itype in combos:
            page = 1
            while True:
                try:
                    js = fetch_analysis_page(year=yr, region=region, itype=itype, page=page, per_page=100, sleep=sleep)
                except Exception as e:
                    st.warning(f"Error fetch year={yr} page={page}: {e}")
                    break
                data = js.get('data') or js.get('results') or js.get('records') or js.get('items') or []
                if not data:
                    break
                for rec in data:
                    flat = extract_record_to_flat_dict(rec)
                    flat['year_requested'] = yr
                    rows.append(flat)
                page += 1
                if page > max_pages:
                    break
    df = pd.DataFrame(rows)
    return df

# ---------- Sidebar filtros ----------
st.sidebar.header("Opciones de carga")
mode = st.sidebar.radio("Fuente de datos", ("API (get_analysis)", "Subir CSV"))
years_range = range(2015, datetime.now().year+1)

if mode == "API (get_analysis)":
    years_sel = st.sidebar.multiselect("Años (selección)", options=list(years_range), default=[datetime.now().year])
    load_btn = st.sidebar.button("Cargar desde API")
    if load_btn:
        with st.spinner("Descargando datos desde la API..."):
            df_raw = load_data_from_api(years=years_sel, regions=None, types=None, max_pages=30, sleep=0.1)
            st.success(f"Datos descargados: {len(df_raw)} registros (raw)")
    else:
        st.info("Presiona 'Cargar desde API' para iniciar la descarga.")
        st.stop()
else:
    uploaded = st.sidebar.file_uploader("Sube CSV", type=["csv"])
    if uploaded is None:
        st.stop()
    df_raw = pd.read_csv(uploaded)

# ---------- Normalización ----------
def normalize_df(df):
    df = df.copy()
    
    if 'date' not in df.columns:
        if 'year_requested' in df.columns:
            df['date'] = pd.to_datetime(df['year_requested'].astype(str) + '-01-01', errors='coerce')
        else:
            df['date'] = pd.NaT
    
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['year_month'] = df['date'].dt.to_period('M').astype(str) if 'date' in df.columns else None

    if 'total' in df.columns:
        df['total'] = pd.to_numeric(df['total'].astype(str).str.replace(r'[^\d\.\-]', '', regex=True), errors='coerce')
    if 'contracts' in df.columns:
        df['contracts'] = pd.to_numeric(df['contracts'], errors='coerce')
    if 'province' in df.columns:
        df['province'] = df['province'].astype(str).str.strip().str.title()
    if 'internal_type' in df.columns:
        df['internal_type'] = df['internal_type'].astype(str).str.strip().str.lower()
    
    df = df.drop_duplicates()
    df = df[~df['date'].isna()] if 'date' in df.columns else df
    df = df[~df['total'].isna()] if 'total' in df.columns else df
    return df

df = normalize_df(df_raw)
st.write("Shape post-normalización:", df.shape)
st.dataframe(df.head(5))

# ---------- Filtros dinámicos ----------
st.sidebar.header("Filtros EDA")
years = sorted(df['year'].dropna().unique().tolist())
sel_year = st.sidebar.selectbox("Año", options=["Todos"] + years)
provinces = sorted(df['province'].dropna().unique().tolist()) if 'province' in df.columns else []
sel_province = st.sidebar.selectbox("Provincia / Región", options=["Todas"] + provinces) if provinces else "Todas"
types = sorted(df['internal_type'].dropna().unique().tolist()) if 'internal_type' in df.columns else []
sel_type = st.sidebar.selectbox("Tipo Contratación", options=["Todos"] + types) if types else "Todos"

dff = df.copy()
if sel_year != "Todos":
    dff = dff[dff['year'] == int(sel_year)]
if sel_province != "Todas":
    dff = dff[dff['province'] == sel_province]
if sel_type != "Todos":
    dff = dff[dff['internal_type'] == sel_type]

st.markdown("### KPIs generales")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total registros", f"{len(dff):,}")
col2.metric("Monto total (USD)", f"{dff['total'].sum():,.2f}" if 'total' in dff.columns else "N/A")
col3.metric("Promedio por registro", f"{dff['total'].mean():,.2f}" if 'total' in dff.columns else "N/A")
col4.metric("Proveedores distintos", f"{dff['supplier'].nunique() if 'supplier' in dff.columns else 'N/A'}")

# ---------- Visualizaciones ----------
st.markdown("## Visualizaciones")

if 'internal_type' in dff.columns:
    agg_type = dff.groupby('internal_type')['total'].sum().reset_index().sort_values('total', ascending=False)
    fig1 = px.bar(agg_type, x='internal_type', y='total', title='Total de Montos por Tipo de Contratación')
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Interpretación:** Observa qué tipo concentra mayor gasto. (Explicar picos si conoces cambios regulatorios).")

if 'date' in dff.columns:
    monthly = dff.groupby('year_month')['total'].sum().reset_index().sort_values('year_month')
    fig2 = px.line(monthly, x='year_month', y='total', title='Evolución Mensual de Montos Totales', markers=True)
    fig2.update_xaxes(tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)
    try:
        kpy = dff.groupby('year')['total'].sum().reset_index().sort_values('year')
        if len(kpy) >= 3:
            slope, intercept, r_val, p_val, _ = linregress(kpy['year'], kpy['total'])
            st.write(f"Tendencia anual: slope={slope:.2f} USD/año, R={r_val:.3f}, p={p_val:.4f}")
            if slope > 0:
                st.success("Tendencia: CRECIENTE")
            elif slope < 0:
                st.warning("Tendencia: DECRECIENTE")
            else:
                st.info("Tendencia: ESTABLE")
    except Exception as e:
        st.write("No fue posible calcular tendencia:", e)

if {'internal_type','month','total'}.issubset(dff.columns):
    pivot = dff.groupby(['month','internal_type'])['total'].sum().reset_index()
    fig3 = px.bar(pivot, x='month', y='total', color='internal_type', title='Total por Tipo de Contratación por Mes', barmode='stack')
    st.plotly_chart(fig3, use_container_width=True)

if 'internal_type' in dff.columns:
    counts = dff['internal_type'].value_counts().reset_index()
    counts.columns = ['internal_type','count']
    fig4 = px.pie(counts, names='internal_type', values='count', title='Proporción de contratos por tipo')
    st.plotly_chart(fig4, use_container_width=True)

if {'contracts','total'}.issubset(dff.columns):
    fig5 = px.scatter(dff, x='contracts', y='total', color='internal_type' if 'internal_type' in dff.columns else None,
                      title='Dispersión: Monto Total vs Cantidad de Contratos', trendline='ols')
    st.plotly_chart(fig5, use_container_width=True)

if {'year','month','total'}.issubset(dff.columns):
    heat = dff.pivot_table(index='year', columns='month', values='total', aggfunc='sum', fill_value=0)
    fig6 = px.imshow(heat, labels=dict(x="Mes", y="Año", color="Total"), title='Heatmap Año × Mes (Total)')
    st.plotly_chart(fig6, use_container_width=True)

st.markdown("## Correlaciones con `total`")
num_cols = dff.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'total']
corr_out = []
for c in num_cols:
    mask = dff[['total', c]].dropna()
    if len(mask) < 10:
        continue
    r_p, p_p = pearsonr(mask['total'], mask[c])
    r_s, p_s = spearmanr(mask['total'], mask[c])
    corr_out.append((c, r_p, p_p, r_s, p_s))
corr_df = pd.DataFrame(corr_out, columns=['variable','pearson_r','pearson_p','spearman_r','spearman_p']).sort_values('pearson_r', key=abs, ascending=False)
st.dataframe(corr_df.style.format({'pearson_r':'{:.3f}','pearson_p':'{:.4f}','spearman_r':'{:.3f}','spearman_p':'{:.4f}'}))

csv = dff.to_csv(index=False).encode('utf-8')
st.download_button("Descargar CSV filtrado / procesado", data=csv, file_name="compras_procesadas.csv", mime='text/csv')

st.markdown("### Notas y conclusiones")
st.markdown("""
- Revisa la consistencia de `total` (monedas, registros atípicos).  
- Si observas picos anómalos en la serie temporal, investiga los `ocid` asociados.  
- Las correlaciones reportadas son exploratorias; para afirmaciones causales requerirías más análisis y control de variables.
""")
