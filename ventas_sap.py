# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gdown
import tempfile

# Configuración de página
st.set_page_config(
    layout="wide", 
    page_title="Análisis de Ventas SAP", 
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# CONSTANTES
MESES_ORDEN = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
               'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']

COLORES = {
    'positivo': '#2ecc71',
    'negativo': '#e74c3c',
    'neutro': '#3498db',
    'fondo': '#f8f9fa',
    'texto': '#2c3e50',
    'advertencia': '#f39c12'
}

# Estilos CSS personalizados
st.markdown(f"""
    <style>
        .css-18e3th9 {{padding: 1rem 1rem 10rem;}}
        .css-1d391kg {{padding-top: 3.5rem;}}
        .stProgress > div > div > div > div {{background-color: {COLORES['positivo']}}}
        .st-bb {{background-color: {COLORES['fondo']}}}
        .st-at {{background-color: {COLORES['fondo']}}}
        .css-1vq4p4l {{padding: 1rem;}}
        .css-1q8dd3e {{color: {COLORES['texto']};}}
        .css-1q8dd3e:hover {{color: {COLORES['positivo']};}}
    </style>
""", unsafe_allow_html=True)

# Carga de datos con gdown
@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    file_id = "1dnyGUW_pOdhjNgcX39MgdDpNpDwCTGVX"
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        cols = ['Fecha', 'Year', 'Mes', 'Vendedor', 'Ce.', 'SKU', 
               'Texto breve de material', 'Nombre Cliente', 'Cantidad vendida',
               'Monto vendido', 'PRECIO PROM', 'Factura', 'Oferta']

        dtypes = {
            'Fecha': 'str',
            'Year': 'int16',
            'Mes': 'str',
            'Vendedor': 'category',
            'Ce.': 'category',
            'SKU': 'category',
            'Texto breve de material': 'category',
            'Nombre Cliente': 'category',
            'Cantidad vendida': 'float32',
            'Monto vendido': 'float32',
            'PRECIO PROM': 'float32',
            'Factura': 'category',
            'Oferta': 'int8'
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            gdown.download(url, tmp.name, quiet=True)
            df = pd.read_excel(tmp.name, usecols=cols, dtype=dtypes)

        df['Fecha'] = pd.to_datetime(df['Fecha'], dayfirst=True, errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('int16')
        df['Mes'] = df['Mes'].astype('category')

        meses_validos = [m for m in MESES_ORDEN if m in df['Mes'].unique()]
        df = df[df['Mes'].isin(meses_validos)]

        mes_orden = {mes: i+1 for i, mes in enumerate(MESES_ORDEN)}
        df['Mes_Orden'] = df['Mes'].map(mes_orden).astype('int8')

        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def load_quota_data():
    file_id = "1M6-7LbM7wSSIbBcntZ5-Jc8wFS0s-TnB"
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            gdown.download(url, tmp.name, quiet=True)
            df_quota = pd.read_excel(tmp.name)

        required_cols = ['Ce.', 'Vendedor', 'Material', 'Texto breve de material', 
                         'Monto meta', 'Cantidad meta', 'Mes', 'Year']

        missing_cols = [col for col in required_cols if col not in df_quota.columns]
        if missing_cols:
            st.error(f"Columnas faltantes en el archivo de cuotas: {missing_cols}")
            return pd.DataFrame()

        df_quota = df_quota[required_cols].copy()

        if df_quota['Monto meta'].dtype == 'object':
            df_quota['Monto meta'] = pd.to_numeric(
                df_quota['Monto meta'].astype(str).str.replace(',', ''), 
                errors='coerce'
            ).fillna(0).astype('float32')

        df_quota['Cantidad meta'] = pd.to_numeric(
            df_quota['Cantidad meta'], errors='coerce'
        ).fillna(0).astype('int32')

        cat_cols = ['Ce.', 'Vendedor', 'Material', 'Texto breve de material', 'Mes']
        for col in cat_cols:
            df_quota[col] = df_quota[col].astype('category')

        df_quota['Year'] = df_quota['Year'].astype('int16')

        return df_quota
    except Exception as e:
        st.error(f"Error al cargar el archivo de cuotas: {str(e)}")
        return pd.DataFrame()

# Cargar datos
with st.spinner("Cargando datos..."):
    df = load_data()
    quota_df = load_quota_data()

if df.empty:
    st.warning("No se pudieron cargar los datos. Verifica la ruta del archivo.")
    st.stop()

# Sidebar para filtros
with st.sidebar:
    st.title("Filtros Globales")
    
    if st.button("🔄 Actualizar Datos", key="btn_actualizar"):
        st.cache_data.clear()
        st.experimental_rerun()
    
    year_filter = st.selectbox(
        "Seleccionar Año", 
        options=sorted(df['Year'].unique(), reverse=True),
        key="filtro_anio"
    )
    
    centros = ['Todos'] + sorted(df['Ce.'].dropna().unique().tolist())
    centro_filter = st.selectbox(
        "Seleccionar Centro", 
        options=centros,
        key="filtro_centro"
    )
    
    meses_disponibles = sorted(
        df['Mes'].unique(), 
        key=lambda x: MESES_ORDEN.index(x) if x in MESES_ORDEN else 999
    )
    meses_seleccionados = st.multiselect(
        "Seleccionar Meses",
        options=meses_disponibles,
        default=meses_disponibles[0] if meses_disponibles else None,
        key="filtro_meses"
    )

# Aplicar filtros optimizados
@st.cache_data(show_spinner=False)
def apply_filters(_df, year, centro, meses):
    filtered = _df[_df['Year'] == year].copy()
    if centro != 'Todos':
        filtered = filtered[filtered['Ce.'] == centro]
    if meses:
        filtered = filtered[filtered['Mes'].isin(meses)]
    return filtered

filtered_df = apply_filters(df, year_filter, centro_filter, meses_seleccionados)

# Función para mostrar KPIs rápidos
def display_kpis(df):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_ventas = df['Monto vendido'].sum()
        st.metric("Ventas Totales", f"${total_ventas:,.2f}")
    with col2:
        total_cajas = df['Cantidad vendida'].sum()
        st.metric("Cajas Vendidas", f"{total_cajas:,.0f}")
    with col3:
        clientes_unicos = df['Nombre Cliente'].nunique()
        st.metric("Clientes Únicos", clientes_unicos)
    with col4:
        facturas_unicas = df['Factura'].nunique()
        ticket_promedio = total_ventas / facturas_unicas if facturas_unicas > 0 else 0
        st.metric("Ticket Promedio", f"${ticket_promedio:,.2f}")

# Tabs principales
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Resumen Ventas", 
    "🎯 Cumplimiento Metas", 
    "📊 Análisis 20/80", 
    "🔄 Evolución Productos", 
    "🔗 Asociación Productos", 
    "💰 Simulación Ofertas",
    "🔍 Análisis General",
    "📚 Manual de Usuario"
])

# ------------------------- TAB 1: Resumen Ventas -------------------------
with tab1:
    st.header("📊 Resumen de Ventas por Vendedor")
    
    # Mostrar filtros aplicados
    filtros = [f"**Año:** {year_filter}"]
    if centro_filter != 'Todos':
        filtros.append(f"**Centro:** {centro_filter}")
    if meses_seleccionados:
        filtros.append(f"**Meses:** {', '.join(meses_seleccionados)}")
    
    st.markdown(" | ".join(filtros), unsafe_allow_html=True)
    
    # KPIs rápidos
    display_kpis(filtered_df)
    
    # Gráfico de ventas por mes
    if not filtered_df.empty:
        ventas_mensuales = filtered_df.groupby(['Mes', 'Mes_Orden'], observed=True)['Monto vendido'].sum().reset_index().sort_values('Mes_Orden')
        fig_ventas_mensuales = px.bar(
            ventas_mensuales,
            x='Mes',
            y='Monto vendido',
            title='Ventas Mensuales',
            labels={'Monto vendido': 'Ventas ($)', 'Mes': 'Mes'},
            color_discrete_sequence=[COLORES['positivo']]
        )
        fig_ventas_mensuales.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_ventas_mensuales, use_container_width=True)
    
    # Tabla pivot optimizada
    st.subheader("Ventas por Vendedor y Mes")
    if not filtered_df.empty:
        pivot_table = pd.pivot_table(
            filtered_df,
            values='Monto vendido',
            index='Vendedor',
            columns='Mes',
            aggfunc='sum',
            fill_value=0,
            margins=True,
            margins_name='Total'
        )
        
        cols_ordenadas = [m for m in MESES_ORDEN if m in pivot_table.columns]
        if 'Total' in pivot_table.columns:
            cols_ordenadas.append('Total')
        pivot_table = pivot_table[cols_ordenadas]
        
        st.dataframe(
            pivot_table.style.format("{:,.2f}")
            .background_gradient(cmap='Blues')
            .highlight_null(color='lightgray'),
            use_container_width=True,
            height=400
        )
    
    # Comparativo de desempeño mejorado
    st.subheader("Comparativo de Desempeño")
    selected_month = st.selectbox(
        "Seleccionar Mes para Comparar", 
        options=meses_disponibles,
        key="mes_comparacion"
    )
    
    if not filtered_df.empty:
        ventas_mensuales = filtered_df.groupby(['Vendedor', 'Mes'], observed=True)['Monto vendido'].sum().unstack()
        resumen = ventas_mensuales.agg(['sum', 'mean'], axis=1)
        resumen.columns = ['Total Año', 'Promedio Mensual']
        resumen[f'Ventas {selected_month}'] = ventas_mensuales[selected_month]
        resumen['Variación'] = resumen[f'Ventas {selected_month}'] - resumen['Promedio Mensual']
        resumen['% Variación'] = (resumen['Variación'] / resumen['Promedio Mensual']) * 100
        
        st.dataframe(
            resumen.style.format({
                'Total Año': "{:,.2f}",
                'Promedio Mensual': "{:,.2f}",
                f'Ventas {selected_month}': "{:,.2f}",
                'Variación': "{:,.2f}",
                '% Variación': "{:.1f}%"
            }).applymap(
                lambda x: 'color: green' if x > 0 else 'color: red', 
                subset=['Variación', '% Variación']
            ),
            use_container_width=True,
            height=400
        )

# ------------------------- TAB 2: Cumplimiento Metas -------------------------
with tab2:
    st.header("🎯 Cumplimiento de Metas")
    
    if not meses_seleccionados:
        st.warning("Por favor seleccione al menos un mes en los filtros")
        st.stop()
    
    mes_analisis = meses_seleccionados[0]
    df_mes = filtered_df[filtered_df['Mes'] == mes_analisis]
    
    if df_mes.empty:
        st.warning(f"No hay datos disponibles para {mes_analisis}")
        st.stop()
    
    with st.spinner("Calculando cumplimiento..."):
        resumen = df_mes.groupby(['Ce.', 'Vendedor'], observed=True).agg({
            'Monto vendido': 'sum',
            'Cantidad vendida': 'sum',
            'Nombre Cliente': 'nunique'
        }).reset_index()
        
        metas = quota_df[
            (quota_df['Mes'] == mes_analisis) & 
            (quota_df['Year'] == year_filter)
        ]
        
        if not metas.empty:
            metas_grouped = metas.groupby(['Ce.', 'Vendedor'], observed=True).agg({
                'Monto meta': 'sum',
                'Cantidad meta': 'sum'
            }).reset_index()
            
            # Merge sin fillna inmediato
            resumen = pd.merge(
                resumen,
                metas_grouped,
                on=['Ce.', 'Vendedor'],
                how='left'
            )
            
            # Llenar NA solo para las columnas numéricas
            numeric_cols = ['Monto meta', 'Cantidad meta']
            resumen[numeric_cols] = resumen[numeric_cols].fillna(0)
            
            resumen['% Cumplimiento Ventas'] = np.where(
                resumen['Monto meta'] > 0,
                (resumen['Monto vendido'] / resumen['Monto meta']) * 100,
                0
            )
            
            resumen['% Cumplimiento Cajas'] = np.where(
                resumen['Cantidad meta'] > 0,
                (resumen['Cantidad vendida'] / resumen['Cantidad meta']) * 100,
                0
            )
            
            st.subheader(f"Resultados para {mes_analisis} {year_filter}")
            st.dataframe(
                resumen.rename(columns={
                    'Monto vendido': 'Ventas Actuales',
                    'Monto meta': 'Meta Ventas',
                    'Cantidad vendida': 'Cajas Actuales',
                    'Cantidad meta': 'Meta Cajas',
                    'Nombre Cliente': 'Clientes Atendidos'
                }).style.format({
                    'Ventas Actuales': "${:,.2f}",
                    'Meta Ventas': "${:,.2f}",
                    '% Cumplimiento Ventas': "{:.1f}%",
                    'Cajas Actuales': "{:,.0f}",
                    'Meta Cajas': "{:,.0f}",
                    '% Cumplimiento Cajas': "{:.1f}%",
                    'Clientes Atendidos': "{:,}"
                }).apply(
                    lambda x: ['background-color: #2ecc71' if x.name in ['% Cumplimiento Ventas', '% Cumplimiento Cajas'] and v > 90 
                              else 'background-color: #f39c12' if x.name in ['% Cumplimiento Ventas', '% Cumplimiento Cajas'] and v > 60 
                              else 'background-color: #e74c3c' if x.name in ['% Cumplimiento Ventas', '% Cumplimiento Cajas'] and v <= 60 
                              else '' for v in x], 
                    axis=0
                ),
                use_container_width=True,
                height=400
            )
            
            fig = make_subplots(rows=1, cols=2, subplot_titles=('Ventas vs Meta', 'Cajas vs Meta'))
            
            fig.add_trace(
                go.Bar(
                    x=resumen['Vendedor'],
                    y=resumen['Monto vendido'],
                    name='Ventas Actuales',
                    marker_color=COLORES['positivo']
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=resumen['Vendedor'],
                    y=resumen['Monto meta'],
                    name='Meta Ventas',
                    marker_color=COLORES['neutro']
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(
                    x=resumen['Vendedor'],
                    y=resumen['Cantidad vendida'],
                    name='Cajas Actuales',
                    marker_color=COLORES['positivo']
                ),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=resumen['Vendedor'],
                    y=resumen['Cantidad meta'],
                    name='Meta Cajas',
                    marker_color=COLORES['neutro']
                ),
                row=1, col=2
            )
            
            fig.update_layout(
                barmode='group',
                showlegend=True,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_text=f'Cumplimiento de Metas - {mes_analisis}',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"No se encontraron metas para {mes_analisis} {year_filter}")

# ------------------------- TAB 3: Análisis 20/80 -------------------------
with tab3:
    st.header("Análisis 20/80 (Principio de Pareto)")
    
    if filtered_df.empty:
        st.warning("No hay datos para mostrar")
        st.stop()
    
    productos = filtered_df.groupby(['SKU', 'Texto breve de material'], observed=True).agg({
        'Monto vendido': 'sum',
        'Cantidad vendida': 'sum'
    }).reset_index()
    productos.columns = ['SKU', 'Producto', 'Ventas Totales', 'Unidades Vendidas']
    productos = productos.sort_values('Ventas Totales', ascending=False)
    
    productos['% Acumulado Ventas'] = (productos['Ventas Totales'].cumsum() / productos['Ventas Totales'].sum()) * 100
    productos['% Acumulado Unidades'] = (productos['Unidades Vendidas'].cumsum() / productos['Unidades Vendidas'].sum()) * 100
    productos['Posición'] = range(1, len(productos)+1)
    
    pareto = productos[productos['% Acumulado Ventas'] <= 80].copy()
    
    st.subheader(f"Top {len(pareto)} Productos (Generan 80% de Ventas)")
    st.dataframe(
        pareto.style.format({
            'Ventas Totales': "{:,.2f}",
            'Unidades Vendidas': "{:,.0f}",
            '% Acumulado Ventas': "{:.1f}%",
            '% Acumulado Unidades': "{:.1f}%"
        }),
        use_container_width=True,
        height=400
    )
    
    fig = px.line(
        productos,
        x='Posición',
        y='% Acumulado Ventas',
        title='Diagrama de Pareto - Ventas por Producto'
    )
    fig.add_scatter(
        x=productos['Posición'],
        y=productos['% Acumulado Unidades'],
        mode='lines',
        name='% Acumulado Unidades'
    )
    fig.add_hline(
        y=80,
        line_dash="dash",
        line_color="red",
        annotation_text="80% Ventas"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Análisis por Vendedor")
    vendedor = st.selectbox(
        "Seleccionar Vendedor", 
        options=filtered_df['Vendedor'].unique(),
        key="vendedor_pareto"
    )
    
    productos_vendedor = filtered_df[filtered_df['Vendedor'] == vendedor].groupby(
        ['SKU', 'Texto breve de material'], observed=True
    ).agg({
        'Monto vendido': 'sum',
        'Cantidad vendida': 'sum'
    }).reset_index()
    
    productos_vendedor.columns = ['SKU', 'Producto', 'Ventas Totales', 'Unidades Vendidas']
    productos_vendedor = productos_vendedor.sort_values('Ventas Totales', ascending=False)
    
    productos_vendedor['% Acumulado Ventas'] = (
        productos_vendedor['Ventas Totales'].cumsum() / productos_vendedor['Ventas Totales'].sum()
    ) * 100
    
    pareto_vendedor = productos_vendedor[productos_vendedor['% Acumulado Ventas'] <= 80].copy()
    
    st.write(f"**Top {len(pareto_vendedor)} Productos para {vendedor}**")
    st.dataframe(
        pareto_vendedor.style.format({
            'Ventas Totales': "{:,.2f}",
            'Unidades Vendidas': "{:,.0f}",
            '% Acumulado Ventas': "{:.1f}%"
        }),
        use_container_width=True,
        height=300
    )

# ------------------------- TAB 4: Evolución Productos -------------------------
with tab4:
    st.header("Evolución de Productos")
    
    if filtered_df.empty:
        st.warning("No hay datos para mostrar")
        st.stop()
    
    productos = filtered_df['Texto breve de material'].unique()
    seleccionados = st.multiselect(
        "Seleccionar productos para analizar",
        options=productos,
        default=productos[:2] if len(productos) >= 2 else productos,
        key="productos_evolucion"
    )
    
    if seleccionados:
        evolucion = filtered_df[filtered_df['Texto breve de material'].isin(seleccionados)].groupby(
            ['Mes', 'Mes_Orden', 'Texto breve de material'], observed=True
        ).agg({
            'Monto vendido': 'sum',
            'Cantidad vendida': 'sum'
        }).reset_index().sort_values('Mes_Orden')
        
        fig1 = px.line(
            evolucion,
            x='Mes',
            y='Monto vendido',
            color='Texto breve de material',
            title='Evolución de Ventas por Mes',
            markers=True
        )
        
        fig2 = px.line(
            evolucion,
            x='Mes',
            y='Cantidad vendida',
            color='Texto breve de material',
            title='Evolución de Unidades Vendidas por Mes',
            markers=True
        )
        
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Clientes por Producto")
        for producto in seleccionados:
            clientes = filtered_df[filtered_df['Texto breve de material'] == producto]['Nombre Cliente'].nunique()
            st.write(f"**{producto}**: {clientes} clientes distintos")
            
            top_clientes = filtered_df[filtered_df['Texto breve de material'] == producto].groupby(
                'Nombre Cliente', observed=True
            ).agg({
                'Monto vendido': 'sum',
                'Cantidad vendida': 'sum'
            }).nlargest(5, 'Monto vendido').reset_index()
            
            st.dataframe(
                top_clientes.style.format({
                    'Monto vendido': "{:,.2f}",
                    'Cantidad vendida': "{:,.0f}"
                }),
                use_container_width=True,
                height=200
            )

# ------------------------- TAB 5: Asociación Productos -------------------------
with tab5:
    st.header("Asociación de Productos")
    
    if filtered_df.empty:
        st.warning("No hay datos para mostrar")
        st.stop()
    
    producto = st.selectbox(
        "Seleccionar producto para análisis de asociación",
        options=filtered_df['Texto breve de material'].unique(),
        key="producto_asociacion"
    )
    
    if producto:
        facturas = filtered_df[
            (filtered_df['Texto breve de material'] == producto) & 
            (filtered_df['Oferta'] != 1)
        ]['Factura'].unique()
        
        asociados = filtered_df[
            (filtered_df['Factura'].isin(facturas)) &
            (filtered_df['Texto breve de material'] != producto)
        ]
        
        if not asociados.empty:
            frecuencia = asociados.groupby('Texto breve de material', observed=True).agg({
                'Factura': 'nunique',
                'Monto vendido': 'sum',
                'Cantidad vendida': 'sum'
            }).reset_index()
            
            frecuencia.columns = ['Producto', 'N° Facturas', 'Ventas Totales', 'Unidades']
            frecuencia = frecuencia.sort_values('N° Facturas', ascending=False)
            
            st.subheader(f"Productos frecuentemente comprados con {producto}")
            st.dataframe(
                frecuencia.style.format({
                    'N° Facturas': "{:,}",
                    'Ventas Totales': "{:,.2f}",
                    'Unidades': "{:,.0f}"
                }),
                use_container_width=True,
                height=400
            )
            
            fig1 = px.bar(
                frecuencia.head(10),
                x='Producto',
                y='N° Facturas',
                title='Top 10 Productos por Frecuencia de Asociación'
            )
            
            fig2 = px.bar(
                frecuencia.head(10),
                x='Producto',
                y='Ventas Totales',
                title='Top 10 Productos por Ventas en Asociación'
            )
            
            st.plotly_chart(fig1, use_container_width=True)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning(f"No se encontraron productos asociados a {producto}")

# ------------------------- TAB 6: Simulación Ofertas -------------------------
with tab6:
    st.header("💰 Simulación de Ofertas")
    st.info("Esta funcionalidad estará disponible en la próxima actualización")

# ------------------------- TAB 7: Análisis General Comparativo -------------------------
with tab7:
    st.header("🔍 Análisis General Comparativo")
    st.markdown("""
    Comparativa del mes seleccionado vs otros meses, con recomendaciones estratégicas
    """)
    
    if filtered_df.empty:
        st.warning("No hay datos para mostrar")
        st.stop()
    
    mes_analisis = st.selectbox(
        "Seleccionar Mes para Análisis", 
        options=meses_disponibles,
        key="mes_analisis_comparativo"
    )
    
    df_mes = filtered_df[filtered_df['Mes'] == mes_analisis]
    
    if not df_mes.empty:
        st.subheader("🏭 Desempeño por Centro")
        
        centro_analisis = df_mes.groupby('Ce.', observed=True).agg({
            'Monto vendido': ['sum', 'mean', 'count'],
            'Cantidad vendida': 'sum'
        }).reset_index()
        centro_analisis.columns = ['Centro', 'Ventas Totales', 'Ticket Promedio', 'Transacciones', 'Unidades']
        
        otros_meses = filtered_df[filtered_df['Mes'] != mes_analisis].groupby('Ce.', observed=True).agg({
            'Monto vendido': 'mean'
        }).reset_index()
        otros_meses.columns = ['Centro', 'Ventas Promedio Otros Meses']
        
        centro_analisis = pd.merge(centro_analisis, otros_meses, on='Centro', how='left')
        centro_analisis['Variación vs Promedio'] = ((centro_analisis['Ventas Totales'] / centro_analisis['Ventas Promedio Otros Meses']) - 1) * 100
        
        def color_variacion(val):
            color = 'red' if val < 0 else 'green'
            return f'color: {color}'
        
        st.dataframe(
            centro_analisis.style.format({
                'Ventas Totales': "${:,.2f}",
                'Ticket Promedio': "${:,.2f}",
                'Ventas Promedio Otros Meses': "${:,.2f}",
                'Variación vs Promedio': "{:.1f}%",
                'Unidades': "{:,.0f}"
            }).applymap(color_variacion, subset=['Variación vs Promedio']),
            use_container_width=True,
            height=300
        )
        
        fig_centros = px.bar(
            centro_analisis,
            x='Centro',
            y=['Ventas Totales', 'Ventas Promedio Otros Meses'],
            barmode='group',
            title=f'Comparativo de Ventas: {mes_analisis} vs Promedio Otros Meses',
            labels={'value': 'Ventas ($)', 'variable': 'Periodo'}
        )
        st.plotly_chart(fig_centros, use_container_width=True)
        
        st.subheader("👤 Desempeño por Vendedor")
        
        vendedor_analisis = df_mes.groupby('Vendedor', observed=True).agg({
            'Monto vendido': ['sum', 'mean', 'count'],
            'Cantidad vendida': 'sum',
            'PRECIO PROM': 'mean'
        }).reset_index()
        vendedor_analisis.columns = ['Vendedor', 'Ventas Totales', 'Ticket Promedio', 'Transacciones', 'Unidades', 'Precio Promedio']
        
        otros_meses_vend = filtered_df[filtered_df['Mes'] != mes_analisis].groupby('Vendedor', observed=True).agg({
            'Monto vendido': 'mean'
        }).reset_index()
        otros_meses_vend.columns = ['Vendedor', 'Ventas Promedio Otros Meses']
        
        vendedor_analisis = pd.merge(vendedor_analisis, otros_meses_vend, on='Vendedor', how='left')
        vendedor_analisis['Variación vs Promedio'] = ((vendedor_analisis['Ventas Totales'] / vendedor_analisis['Ventas Promedio Otros Meses']) - 1) * 100
        
        vendedor_analisis_clean = vendedor_analisis.dropna(subset=['Ticket Promedio'])
        vendedor_analisis_clean['Ticket Promedio'] = vendedor_analisis_clean['Ticket Promedio'].fillna(0)
        
        fig_vendedores = px.scatter(
            vendedor_analisis_clean,
            x='Unidades',
            y='Ventas Totales',
            color='Variación vs Promedio',
            size='Ticket Promedio',
            hover_name='Vendedor',
            title='Desempeño Relativo de Vendedores',
            labels={
                'Unidades': 'Unidades Vendidas',
                'Ventas Totales': 'Ventas Totales ($)',
                'Variación vs Promedio': 'Variación %',
                'Ticket Promedio': 'Ticket Promedio ($)'
            },
            color_continuous_scale=px.colors.diverging.RdYlGn,
            size_max=20
        )
        fig_vendedores.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_vendedores, use_container_width=True)
        
        st.subheader("📈 Tendencias Temporales")
        
        evolucion = filtered_df.groupby(['Mes', 'Mes_Orden'], observed=True).agg({
            'Monto vendido': 'sum',
            'Cantidad vendida': 'sum'
        }).reset_index().sort_values('Mes_Orden')
        
        evolucion['Variación Ventas'] = evolucion['Monto vendido'].pct_change() * 100
        evolucion['Variación Unidades'] = evolucion['Cantidad vendida'].pct_change() * 100
        
        fig_tendencia = px.line(
            evolucion,
            x='Mes',
            y='Monto vendido',
            title='Tendencia de Ventas Mensuales',
            markers=True,
            labels={'Monto vendido': 'Ventas ($)'}
        )
        fig_tendencia.add_annotation(
            x=mes_analisis,
            y=evolucion[evolucion['Mes'] == mes_analisis]['Monto vendido'].values[0],
            text="Mes Analizado",
            showarrow=True,
            arrowhead=1
        )
        fig_tendencia.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_tendencia, use_container_width=True)
        
        st.subheader("🏆 Ranking de Vendedores")
        
        # Crear ranking completo
        ranking = vendedor_analisis.copy()
        ranking['Ranking Ventas'] = ranking['Ventas Totales'].rank(ascending=False)
        ranking['Ranking Variación'] = ranking['Variación vs Promedio'].rank(ascending=False)
        ranking['Ranking Final'] = (ranking['Ranking Ventas'] * 0.6 + ranking['Ranking Variación'] * 0.4).rank()
        
        # Top performers
        top_performers = ranking.nsmallest(5, 'Ranking Final')[['Vendedor', 'Ventas Totales', 'Variación vs Promedio']]
        st.markdown("### Top 5 Vendedores")
        st.dataframe(
            top_performers.style.format({
                'Ventas Totales': "${:,.2f}",
                'Variación vs Promedio': "{:.1f}%"
            }).apply(
                lambda x: ['background-color: #e6f7e6' if x.name == 'Variación vs Promedio' and v > 0 
                          else 'background-color: #ffcccc' if x.name == 'Variación vs Promedio' and v <= 0 
                          else '' for v in x], 
                axis=0
            ),
            use_container_width=True
        )
        
        # Bottom performers
        bottom_performers = ranking.nlargest(5, 'Ranking Final')[['Vendedor', 'Ventas Totales', 'Variación vs Promedio']]
        st.markdown("### Vendedores a Mejorar")
        st.dataframe(
            bottom_performers.style.format({
                'Ventas Totales': "${:,.2f}",
                'Variación vs Promedio': "{:.1f}%"
            }).apply(
                lambda x: ['background-color: #e6f7e6' if x.name == 'Variación vs Promedio' and v > 0 
                          else 'background-color: #ffcccc' if x.name == 'Variación vs Promedio' and v <= 0 
                          else '' for v in x], 
                axis=0
            ),
            use_container_width=True
        )
        
        st.subheader("💡 Recomendaciones Estratégicas")
        
        # Análisis automático para generar recomendaciones
        ventas_totales = df_mes['Monto vendido'].sum()
        ventas_promedio = filtered_df[filtered_df['Mes'] != mes_analisis]['Monto vendido'].mean()
        variacion_total = ((ventas_totales - ventas_promedio) / ventas_promedio) * 100
        
        if variacion_total > 10:
            st.success(f"**Excelente desempeño general**: Las ventas en {mes_analisis} están un {variacion_total:.1f}% por encima del promedio. Recomendamos:")
            st.markdown("""
            - Capitalizar el momentum con promociones adicionales
            - Identificar y replicar las mejores prácticas de los vendedores top
            - Considerar aumentar las metas para los próximos meses
            """)
        elif variacion_total < -10:
            st.warning(f"**Desempeño inferior al promedio**: Las ventas en {mes_analisis} están un {abs(variacion_total):.1f}% por debajo del promedio. Recomendamos:")
            st.markdown("""
            - Analizar causas específicas (feriados, clima, etc.)
            - Implementar capacitación para los vendedores con bajo desempeño
            - Considerar promociones correctivas para el próximo mes
            """)
        else:
            st.info(f"**Desempeño estable**: Las ventas en {mes_analisis} están cercanas al promedio ({variacion_total:.1f}% de variación). Recomendamos:")
            st.markdown("""
            - Enfocarse en mejorar los puntos débiles identificados
            - Mantener las estrategias que están funcionando
            - Monitorear de cerca a los vendedores con mayor variación negativa
            """)
        
        # Recomendaciones específicas por producto
        productos_top = df_mes.groupby('Texto breve de material')['Monto vendido'].sum().nlargest(3).index.tolist()
        productos_debiles = df_mes.groupby('Texto breve de material')['Monto vendido'].sum().nsmallest(3).index.tolist()
        
        st.markdown("### Recomendaciones por Producto")
        st.markdown(f"""
        **Productos destacados** (considerar potenciar):
        - {productos_top[0]}
        - {productos_top[1]}
        - {productos_top[2]}
        
        **Productos con bajo desempeño** (considerar promocionar o analizar):
        - {productos_debiles[0]}
        - {productos_debiles[1]}
        - {productos_debiles[2]}
        """)
    
    else:
        st.warning(f"No hay datos disponibles para el mes de {mes_analisis}")

# ------------------------- TAB 8: Manual de Usuario -------------------------
with tab8:
    st.header("📚 Manual de Usuario")
    
    st.markdown("""
    ## Guía para el uso del Dashboard de Análisis de Ventas SAP
    
    ### Filtros Globales
    - **Año**: Selecciona el año a analizar
    - **Centro**: Filtra por centro de distribución (selecciona "Todos" para ver todos)
    - **Meses**: Permite seleccionar uno o varios meses para el análisis
    
    ### Pestañas Disponibles
    
    #### 1. 📈 Resumen Ventas
    - Muestra KPIs principales: Ventas totales, cajas vendidas, clientes únicos y ticket promedio
    - Gráfico de ventas mensuales
    - Tabla pivot de ventas por vendedor y mes
    - Comparativo de desempeño por vendedor
    
    #### 2. 🎯 Cumplimiento Metas
    - Compara ventas reales vs metas establecidas
    - Muestra porcentaje de cumplimiento para ventas y cantidades
    - Gráficos comparativos de desempeño
    - **Escala de colores**:
      - 🔴 Rojo: 0-60% de cumplimiento
      - 🟡 Amarillo: 61-90% de cumplimiento
      - 🟢 Verde: >90% de cumplimiento
    
    #### 3. 📊 Análisis 20/80
    - Identifica los productos que generan el 80% de las ventas (Principio de Pareto)
    - Permite analizar por vendedor específico
    
    #### 4. 🔄 Evolución Productos
    - Muestra tendencia de ventas para productos seleccionados
    - Identifica los principales clientes para cada producto
    
    #### 5. 🔗 Asociación Productos
    - Analiza qué productos se compran juntos frecuentemente
    - Identifica oportunidades de cross-selling
    
    #### 6. 💰 Simulación Ofertas
    - (Próximamente) Permitirá simular el impacto de ofertas
    
    #### 7. 🔍 Análisis General
    - Comparativo de desempeño por centro y vendedor
    - Tendencias temporales y variaciones
    - Ranking de vendedores (top/bottom performers)
    - Recomendaciones estratégicas basadas en datos
    
    ### Interpretación de Colores
    - **Verde**: Indicadores positivos (crecimiento, cumplimiento de metas >90%)
    - **Amarillo**: Indicadores intermedios (cumplimiento 61-90%)
    - **Rojo**: Indicadores negativos (decrecimiento, bajo desempeño, cumplimiento <60%)
    - **Azul**: Indicadores neutrales o de referencia
    
    ### Actualización de Datos
    - Los datos se actualizan, Lunes, Miercoles y Viernes
    - Puedes forzar una actualización con el botón "🔄 Actualizar Datos"
    
    ### Consideraciones
    - Los análisis se basan exclusivamente en los datos cargados desde SAP
    - Los filtros aplicados afectan a todas las pestañas
    - Para análisis específicos, usa los filtros por vendedor o producto
    - Los valores de cajas se muestran sin decimales (unidades completas)
    """)

# Pie de página
st.sidebar.markdown("---")
st.sidebar.markdown(f"🔄 La fecha de actualizacion es al dia {datetime.now().strftime('%d/%m/%Y %H:%M')}")
st.sidebar.markdown("📊 Los datos de este reporte son de SAP")