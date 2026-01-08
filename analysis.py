import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import numpy as np
from itertools import combinations

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="Quantum Benchmark Dashboard",
    page_icon="⚛️",
    layout="wide"
)

# --- 1. FUNCIONES DE CARGA Y PROCESAMIENTO ---

@st.cache_data
def load_data(uploaded_files):
    """
    Carga, normaliza y consolida los archivos Excel subidos.
    Detecta automáticamente el nombre de la máquina.
    """
    dfs = {"test": [], "train": [], "total": []}
    
    for file in uploaded_files:
        # Detección del nombre de la máquina
        filename = file.name
        if "Slave1" in filename:
            machine_name = "Slave 1"
        elif "Slave2" in filename:
            machine_name = "Slave 2"
        elif "Slave6" in filename:
            machine_name = "Slave 6"
        else:
            # Fallback: intenta extraer del nombre del archivo o usa el nombre completo
            parts = filename.split('_')
            machine_name = parts[1] if len(parts) > 1 else filename.split('.')[0]
        
        try:
            xls = pd.ExcelFile(file)
            # Normalizar nombres de pestañas a minúsculas
            sheet_map = {name.lower(): name for name in xls.sheet_names}
            
            for key in dfs.keys():
                if key in sheet_map:
                    df = pd.read_excel(file, sheet_name=sheet_map[key])
                    df['Machine'] = machine_name
                    df['Source File'] = filename
                    dfs[key].append(df)
        except Exception as e:
            st.error(f"Error procesando {filename}: {e}")

    # Concatenar DataFrames
    final_dfs = {}
    for key, df_list in dfs.items():
        if df_list:
            final_dfs[key] = pd.concat(df_list, ignore_index=True)
        else:
            final_dfs[key] = pd.DataFrame()
            
    return final_dfs

def get_time_columns(phase, df):
    """Devuelve las columnas de tiempo relevantes según la fase, incluyendo Affinity si existe."""
    if phase == "total":
        base_cols = ['Total Time', 'Penny Time Total', 'Resto Time Total', 'SVM iteraciones']
    else:
        base_cols = ['Penny Time', 'Resto Time']
    
    # Agregar Affinity si existe en el DataFrame
    if 'Affinity' in df.columns:
        base_cols.append('Affinity')
    
    return base_cols

# --- 2. INTERFAZ: BARRA LATERAL ---

st.sidebar.title("⚛️ Panel de Control")
st.sidebar.markdown("---")

# Carga de archivos
st.sidebar.subheader("1. Carga de Datos")
uploaded_files = st.sidebar.file_uploader(
    "Subir excels (Slave1, Slave2, Slave6...)", 
    type=["xlsx"], 
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("👋 Sube los archivos Excel para comenzar.")
    st.markdown("""
    **Estructura esperada:**
    - Archivos Excel con pestañas: `test`, `train`, `total`.
    - Columnas clave: `Backend`, `Qubits`, `Mode`, `Penny Time`, etc.
    """)
    st.stop()

# Procesamiento inicial
data_dict = load_data(uploaded_files)

# Selector de Fase
st.sidebar.subheader("2. Selección de Fase")
phase_input = st.sidebar.radio("Fase a analizar:", ["Test", "Train", "Total"], index=2, horizontal=True)
phase_key = phase_input.lower()
df_main = data_dict[phase_key]

if df_main.empty:
    st.error(f"No hay datos para la fase {phase_input}.")
    st.stop()

# Filtros Globales
st.sidebar.subheader("3. Filtros Globales")
all_machines = sorted(df_main['Machine'].unique())
all_backends = sorted(df_main['Backend'].unique())
all_qubits = sorted(df_main['Qubits'].unique())

sel_machines = st.sidebar.multiselect("Máquinas", all_machines, default=all_machines)
sel_backends = st.sidebar.multiselect("Backend", all_backends, default=all_backends)
sel_qubits = st.sidebar.multiselect("Qubits", all_qubits, default=all_qubits)

# Filtro de Affinity (si existe)
if 'Affinity' in df_main.columns:
    all_affinities = sorted(df_main['Affinity'].unique())
    sel_affinities = st.sidebar.multiselect("Affinity", all_affinities, default=all_affinities)
else:
    sel_affinities = None

# Aplicar filtros
if sel_affinities is not None:
    df_filtered = df_main[
        (df_main['Machine'].isin(sel_machines)) &
        (df_main['Backend'].isin(sel_backends)) &
        (df_main['Qubits'].isin(sel_qubits)) &
        (df_main['Affinity'].isin(sel_affinities))
    ]
else:
    df_filtered = df_main[
        (df_main['Machine'].isin(sel_machines)) &
        (df_main['Backend'].isin(sel_backends)) &
        (df_main['Qubits'].isin(sel_qubits))
    ]

# --- 3. DASHBOARD PRINCIPAL ---

st.title(f"Dashboard de Rendimiento: Fase {phase_input}")

# Métricas Generales (Top Row)
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
time_cols = get_time_columns(phase_key, df_filtered)
main_metric = time_cols[0] # Por defecto el primer tiempo (Total o Penny)

with col_m1:
    st.metric("Registros Totales", len(df_filtered))
with col_m2:
    st.metric("Máquinas Analizadas", df_filtered['Machine'].nunique())
with col_m3:
    st.metric("Backends Únicos", df_filtered['Backend'].nunique())
with col_m4:
    avg_time = df_filtered[main_metric].mean()
    st.metric(f"Promedio {main_metric}", f"{avg_time:.2f} s")

st.markdown("---")

# PESTAÑAS
tabs = st.tabs(["📊 Exploración Visual", "🧮 Análisis Estadístico Profundo", "⏱️ Desglose de Tiempos", "💾 Datos Crudos"])

# === TAB 1: EXPLORACIÓN VISUAL ===
with tabs[0]:
    st.subheader("Comparativa Visual Dinámica")
    
    # Opciones de agrupación y color (añadir Affinity si existe)
    grouping_options = ['Qubits', 'Backend', 'Machine', 'Mode', 'Block Type']
    color_options = ['Machine', 'Backend', 'Mode', 'Block Type']
    
    if 'Affinity' in df_filtered.columns:
        grouping_options.append('Affinity')
        color_options.append('Affinity')
    
    c1, c2, c3 = st.columns(3)
    with c1:
        y_axis = st.selectbox("Métrica (Eje Y)", time_cols, index=0)
    with c2:
        x_axis = st.selectbox("Agrupación (Eje X)", grouping_options)
    with c3:
        color_dims = st.multiselect("Color / Leyenda (una o más)", color_options, default=[color_options[0]])
    
    # Crear columna combinada si se seleccionan múltiples dimensiones
    if len(color_dims) > 1:
        df_filtered['_combined_color'] = df_filtered[color_dims].astype(str).agg(' | '.join, axis=1)
        color_dim = '_combined_color'
    elif len(color_dims) == 1:
        color_dim = color_dims[0]
    else:
        color_dim = None

    # Contenedor de gráficos
    chart_type = st.radio("Tipo de Gráfico", ["Boxplot (Distribución)", "Violin (Densidad)", "Barras (Promedio + Error)", "Líneas (Escalabilidad)", "Heatmap (Comparativa)"], horizontal=True)
    
    if chart_type == "Boxplot (Distribución)":
        fig = px.box(df_filtered, x=x_axis, y=y_axis, color=color_dim, points="all", 
                     title=f"Distribución de {y_axis} por {x_axis}")
        
    elif chart_type == "Violin (Densidad)":
        fig = px.violin(df_filtered, x=x_axis, y=y_axis, color=color_dim, box=True, points="all",
                        title=f"Densidad de {y_axis} por {x_axis}")
        
    elif chart_type == "Barras (Promedio + Error)":
        # Agrupar para calcular media y desviación
        if color_dim and color_dim != x_axis:
            grp = [x_axis, color_dim]
        else:
            grp = [x_axis]
            color_dim = None
            
        df_stats = df_filtered.groupby(grp)[y_axis].agg(['mean', 'std']).reset_index()
        fig = px.bar(df_stats, x=x_axis, y='mean', error_y='std', color=color_dim, barmode='group',
                     title=f"Promedio de {y_axis} con Desviación Estándar")
        
    elif chart_type == "Líneas (Escalabilidad)":
        # Ideal para Qubits en el eje X
        if color_dim:
            df_stats = df_filtered.groupby([x_axis, color_dim])[y_axis].mean().reset_index()
        else:
            df_stats = df_filtered.groupby([x_axis])[y_axis].mean().reset_index()
        fig = px.line(df_stats, x=x_axis, y=y_axis, color=color_dim, markers=True,
                      title=f"Tendencia de {y_axis} al aumentar {x_axis}")
                      
    elif chart_type == "Heatmap (Comparativa)":
    # Pivot table para heatmap
    if color_dim:
        pivot_data = df_filtered.pivot_table(
            values=y_axis,
            index=x_axis,
            columns=color_dim,
            aggfunc='mean'
        )
        fig = px.imshow(
            pivot_data,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="Viridis",
            title=f"Mapa de Calor: Promedio de {y_axis}"
        )
    else:
        st.warning("⚠️ Selecciona al menos una dimensión de color para generar el Heatmap")

    st.plotly_chart(fig, use_container_width=True) if color_dim or chart_type != "Heatmap (Comparativa)" else None

# === TAB 2: ESTADÍSTICA ===
with tabs[1]:
    st.subheader("Motor de Análisis Estadístico")
    st.markdown("Este módulo verifica premisas y selecciona automáticamente entre pruebas paramétricas y no paramétricas, ejecutando pruebas Post-Hoc si es necesario.")

    # Opciones de factor (añadir Affinity si existe)
    factor_options = ['Machine', 'Backend', 'Mode', 'Block Type', 'Qubits']
    if 'Affinity' in df_filtered.columns:
        factor_options.append('Affinity')

    c_stat1, c_stat2 = st.columns(2)
    with c_stat1:
        stat_target = st.selectbox("Variable Dependiente (Numérica)", time_cols, key="st_tgt")
    with c_stat2:
        stat_factor = st.selectbox("Factor de Grupo (Categórica)", factor_options, key="st_fct")

    if st.button("🚀 Ejecutar Análisis Estadístico Completo"):
        st.divider()
        
        # Preparación de datos
        data_clean = df_filtered[[stat_target, stat_factor]].dropna()
        groups = sorted(data_clean[stat_factor].unique())
        group_data = [data_clean[data_clean[stat_factor] == g][stat_target] for g in groups]
        
        if len(groups) < 2:
            st.error("⚠️ Se necesitan al menos 2 grupos para comparar.")
        else:
            # 1. ESTADÍSTICA DESCRIPTIVA
            st.markdown("#### 1. Resumen Descriptivo")
            desc_stats = data_clean.groupby(stat_factor)[stat_target].agg(['count', 'mean', 'median', 'std', 'min', 'max']).reset_index()
            st.dataframe(desc_stats, use_container_width=True)
            
            # 2. VERIFICACIÓN DE PREMISAS
            st.markdown("#### 2. Verificación de Premisas")
            col_prem1, col_prem2 = st.columns(2)
            
            # Shapiro-Wilk (Normalidad)
            all_normal = True
            with col_prem1:
                st.write("**Test de Normalidad (Shapiro-Wilk)**")
                for i, g in enumerate(groups):
                    if len(group_data[i]) >= 3:
                        s_stat, s_p = stats.shapiro(group_data[i])
                        is_norm = s_p > 0.05
                        icon = "✅" if is_norm else "❌"
                        st.caption(f"{icon} {g}: p={s_p:.4f}")
                        if not is_norm: all_normal = False
                    else:
                        st.caption(f"⚠️ {g}: Datos insuficientes (N<3)")
                        all_normal = False # Conservador
            
            # Levene (Homocedasticidad)
            homoscedastic = False
            with col_prem2:
                st.write("**Test de Homogeneidad (Levene)**")
                try:
                    l_stat, l_p = stats.levene(*group_data)
                    homoscedastic = l_p > 0.05
                    icon_l = "✅" if homoscedastic else "❌"
                    st.write(f"{icon_l} Varianzas Iguales: p={l_p:.4f}")
                except:
                    st.write("⚠️ No se pudo calcular Levene.")
            
            # 3. TEST DE HIPÓTESIS
            st.markdown("#### 3. Test de Diferencia (Omnibus)")
            
            is_parametric = all_normal and homoscedastic
            p_global = 1.0
            
            if len(groups) == 2:
                if is_parametric:
                    test_name = "T-Test (Student)"
                    stat_val, p_global = stats.ttest_ind(group_data[0], group_data[1])
                elif all_normal and not homoscedastic:
                    test_name = "T-Test (Welch)"
                    stat_val, p_global = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
                else:
                    test_name = "Mann-Whitney U (No Paramétrico)"
                    stat_val, p_global = stats.mannwhitneyu(group_data[0], group_data[1])
            else:
                if is_parametric:
                    test_name = "ANOVA One-Way"
                    stat_val, p_global = stats.f_oneway(*group_data)
                else:
                    test_name = "Kruskal-Wallis (No Paramétrico)"
                    stat_val, p_global = stats.kruskal(*group_data)
            
            res_col1, res_col2 = st.columns([3, 1])
            with res_col1:
                st.info(f"ℹ️ Dado el cumplimiento de premisas, se ha seleccionado: **{test_name}**")
            with res_col2:
                st.metric("P-Value Global", f"{p_global:.4e}")

            # Interpretación Global
            if p_global < 0.05:
                st.success("🔴 **Resultado:** Existen diferencias estadísticamente significativas entre los grupos.")
                
                # 4. POST-HOC
                if len(groups) > 2:
                    st.markdown("#### 4. Análisis Post-Hoc (Comparaciones Par a Par)")
                    st.caption("Corrección de Bonferroni aplicada para ajustar la significancia.")
                    
                    # Generar combinaciones
                    combs = list(combinations(groups, 2))
                    posthoc_res = []
                    alpha_corrected = 0.05 / len(combs)
                    
                    for g1, g2 in combs:
                        d1 = data_clean[data_clean[stat_factor] == g1][stat_target]
                        d2 = data_clean[data_clean[stat_factor] == g2][stat_target]
                        
                        # Usar T-test o Mann-Whitney según lo decidido arriba (simplificación coherente)
                        if is_parametric:
                            _, p_pair = stats.ttest_ind(d1, d2, equal_var=homoscedastic)
                            method = "T-Test"
                        else:
                            _, p_pair = stats.mannwhitneyu(d1, d2)
                            method = "Mann-Whitney"
                            
                        sig = "SI" if p_pair < alpha_corrected else "NO"
                        
                        posthoc_res.append({
                            "Grupo A": g1,
                            "Grupo B": g2,
                            "Diferencia Medias": d1.mean() - d2.mean(),
                            "P-Value": p_pair,
                            "Significativo (Bonferroni)": sig
                        })
                    
                    ph_df = pd.DataFrame(posthoc_res)
                    # Formatear visualmente
                    st.dataframe(ph_df.style.format({"P-Value": "{:.4e}", "Diferencia Medias": "{:.4f}"})
                                 .applymap(lambda v: 'color: red; font-weight: bold' if v == 'SI' else None, subset=['Significativo (Bonferroni)']), 
                                 use_container_width=True)
                    
                    # Botón descarga Post-hoc
                    st.download_button("📥 Descargar Resultados Post-Hoc", ph_df.to_csv(index=False), "posthoc_results.csv")
                    
            else:
                st.success("🟢 **Resultado:** No hay evidencia suficiente para afirmar que los grupos son diferentes.")

# === TAB 3: TIEMPOS (PROFILING) ===
with tabs[2]:
    st.subheader("Desglose de Tiempo de Ejecución")
    st.markdown("Análisis de cuánto tiempo consume la librería cuántica (Penny) vs el resto del cómputo.")
    
    # Identificar columnas para apilar
    stack_cols = []
    if phase_key == "total":
        stack_cols = ['Penny Time Total', 'Resto Time Total']
    else:
        stack_cols = ['Penny Time', 'Resto Time']
    
    # Opciones de agrupación (añadir Affinity si existe)
    stack_group_options = ['Backend', 'Machine', 'Qubits', 'Mode', 'Block Type']
    if 'Affinity' in df_filtered.columns:
        stack_group_options.append('Affinity')
        
    stack_group = st.selectbox("Agrupar Tiempos por:", stack_group_options, key="stack_k")
    
    # Preparar datos
    df_stack = df_filtered.groupby(stack_group)[stack_cols].mean().reset_index()
    df_melt = df_stack.melt(id_vars=stack_group, value_vars=stack_cols, var_name='Componente', value_name='Segundos')
    
    # Gráfico Stacked Bar
    fig_stack = px.bar(df_melt, x=stack_group, y='Segundos', color='Componente', 
                       title=f"Composición de Tiempos Promedio por {stack_group}", 
                       barmode='stack', text_auto='.2f')
    fig_stack.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_stack, use_container_width=True)

# === TAB 4: DATOS CRUDOS ===
with tabs[3]:
    st.subheader("Datos Filtrados")
    st.dataframe(df_filtered, use_container_width=True)
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Descargar Datos Filtrados (CSV)",
        data=csv,
        file_name=f'data_{phase_key}_filtered.csv',
        mime='text/csv',
    )