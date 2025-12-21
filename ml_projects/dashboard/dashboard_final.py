"""
Dashboard Final - Sistema de Inteligência Policial PMMA
Usando dados reais das ocorrências (2014-2024)
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import folium
from streamlit_folium import st_folium
from collections import Counter
import re
import os

# Configuração da página
st.set_page_config(
    page_title="Sistema de Inteligência Policial - PMMA",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🚔 Sistema de Inteligência Policial - PMMA")
st.markdown("*Análise de dados reais das ocorrências (2014-2024)*")
st.markdown("---")

# Carregar dados reais
@st.cache_data
def load_data():
    """Carrega os dados reais da PMMA"""

    # Tentar diferentes caminhos possíveis
    paths = [
        '../output/pmma_unificado_oficial.parquet',
        '../../output/pmma_unificado_oficial.parquet',
        '/Users/tgt/Documents/dados_pmma_copy/output/pmma_unificado_oficial.parquet',
        './output/pmma_unificado_oficial.parquet'
    ]

    for path in paths:
        if os.path.exists(path):
            df = pd.read_parquet(path)

            # Limpeza e preparação
            df = df.dropna(subset=['data'])
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
            df = df.dropna(subset=['data'])

            # Garantir que hora_num é numérico
            df['hora_num'] = pd.to_numeric(df['hora_num'], errors='coerce').fillna(0)
            df['hora_num'] = df['hora_num'].astype(int)

            # Limpar áreas
            df['area'] = df['area'].fillna('Não Informada')
            df['area'] = df['area'].str.lower().str.strip()

            # Padronizar áreas principais
            area_mapping = {
                'norte': 'Norte',
                'sul': 'Sul',
                'leste': 'Leste',
                'oeste': 'Oeste',
                'centro': 'Centro'
            }

            # Aplicar mapeamento para áreas padronizadas
            df['area_padrao'] = df['area'].apply(
                lambda x: next((v for k, v in area_mapping.items() if k in str(x).lower()), x)
            )

            # Extrair hora válida
            def extract_hour(hora_str):
                if pd.isna(hora_str):
                    return 12
                try:
                    if ':' in str(hora_str):
                        return int(str(hora_str).split(':')[0])
                    else:
                        hora_int = int(float(str(hora_str)))
                        return hora_int if 0 <= hora_int <= 23 else 12
                except:
                    return 12

            df['hora_valida'] = df['hora'].apply(extract_hour)

            # Adicionar dia da semana
            df['dia_semana'] = df['data'].dt.day_name()
            df['mes'] = df['data'].dt.month
            df['ano'] = df['data'].dt.year

            return df

    return None

# Sidebar para navegação
st.sidebar.title("Navegação")
page = st.sidebar.selectbox(
    "Selecione uma página:",
    ["📊 Visão Geral", "📈 Análise Temporal", "🎯 Análise Geográfica", "📋 Tipos de Ocorrência"]
)

# Tentar carregar dados
try:
    df = load_data()
    if df is not None:
        st.sidebar.success(f"✅ {len(df):,} registros carregados")
        data_loaded = True
    else:
        st.sidebar.error("❌ Dados não encontrados")
        data_loaded = False
except Exception as e:
    st.sidebar.error(f"❌ Erro: {str(e)}")
    data_loaded = False

# Página 1: Visão Geral
if page == "📊 Visão Geral" and data_loaded:
    st.header("📊 Visão Geral das Ocorrências")

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_ocorrencias = len(df)
        st.metric("Total de Ocorrências", f"{total_ocorrencias:,}")

    with col2:
        anos = df['ano'].nunique()
        st.metric("Anos Analisados", anos)

    with col3:
        media_diaria = total_ocorrencias / (df['data'].dt.date.nunique())
        st.metric("Média Diária", f"{media_diaria:.0f}")

    with col4:
        areas = df['area_padrao'].nunique()
        st.metric("Áreas Principais", areas)

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ocorrências por Ano")
        ano_counts = df.groupby('ano').size().reset_index(name='count')
        fig = px.line(ano_counts, x='ano', y='count', markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribuição por Área")
        area_counts = df['area_padrao'].value_counts()
        fig = px.pie(values=area_counts.values, names=area_counts.index)
        st.plotly_chart(fig, use_container_width=True)

    # Mapa de calor
    st.subheader("Mapa de Calor - Ocorrências por Hora e Dia da Semana")

    # Criar pivot table
    heatmap_data = df.groupby(['dia_semana', 'hora_valida']).size().unstack(fill_value=0)
    dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = heatmap_data.reindex(dias_ordem, fill_value=0)

    fig = px.imshow(
        heatmap_data.values,
        x=[f"{h:02d}:00" for h in heatmap_data.columns],
        y=['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'],
        title="Intensidade de Ocorrências",
        color_continuous_scale="Reds"
    )
    st.plotly_chart(fig, use_container_width=True)

# Página 2: Análise Temporal
elif page == "📈 Análise Temporal" and data_loaded:
    st.header("📈 Análise Temporal")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Padrão por Hora")
        hora_counts = df.groupby('hora_valida').size()
        fig = px.bar(x=hora_counts.index, y=hora_counts.values,
                     labels={'x': 'Hora', 'y': 'Ocorrências'})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Padrão por Mês")
        mes_counts = df.groupby('mes').size()
        mes_nomes = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                    'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        mes_counts.index = mes_counts.index.map(lambda x: mes_nomes[x-1])
        fig = px.bar(x=mes_counts.index, y=mes_counts.values)
        st.plotly_chart(fig, use_container_width=True)

    # Análise por dia da semana
    st.subheader("Ocorrências por Dia da Semana")
    dia_counts = df['dia_semana'].value_counts()
    fig = px.bar(x=dia_counts.values, y=dia_counts.index, orientation='h')
    st.plotly_chart(fig, use_container_width=True)

# Página 3: Análise Geográfica
elif page == "🎯 Análise Geográfica" and data_loaded:
    st.header("🎯 Análise Geográfica")

    # Mapa de São Luís
    st.subheader("Distribuição das Ocorrências")

    m = folium.Map(location=[-2.53, -44.30], zoom_start=11)

    # Adicionar marcadores para áreas principais
    coords = {
        'norte': (-2.48, -44.30),
        'sul': (-2.55, -44.28),
        'leste': (-2.52, -44.25),
        'oeste': (-2.53, -44.33),
        'centro': (-2.53, -44.28)
    }

    area_counts = df['area_padrao'].value_counts()

    for area, count in area_counts.items():
        if area in coords:
            lat, lon = coords[area]
            folium.Circle(
                location=[lat, lon],
                radius=1000 + count/100,
                popup=f"{area.title()}: {count:,} ocorrências",
                color='red',
                fill=True,
                fillOpacity=0.3
            ).add_to(m)

    st_folium(m, width=700, height=500)

# Página 4: Tipos de Ocorrência
elif page == "📋 Tipos de Ocorrência" and data_loaded:
    st.header("📋 Análise dos Tipos de Ocorrência")

    if 'descricao_tipo' in df.columns:
        # Top tipos
        tipo_counts = df['descricao_tipo'].value_counts().head(20)

        fig = px.bar(
            x=tipo_counts.values,
            y=tipo_counts.index,
            orientation='h',
            title="Top 20 Tipos de Ocorrência"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Estatísticas
        st.subheader("Estatísticas")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Tipos Diferentes", len(df['descricao_tipo'].unique()))
            st.metric("Tipo Mais Comum", tipo_counts.index[0])

        with col2:
            st.metric("Ocorrências do Tipo Principal", f"{tipo_counts.iloc[0]:,}")

            # Percentual
            percentual = (tipo_counts.iloc[0] / len(df)) * 100
            st.metric("Percentual do Total", f"{percentual:.1f}%")

elif not data_loaded:
    st.error("Não foi possível carregar os dados. Verifique se o arquivo 'pmma_unificado_oficial.parquet' existe no diretório de output.")

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🚔 <b>Sistema de Inteligência Policial - PMMA</b></p>
        <p>Análise de dados reais das ocorrências</p>
    </div>
    """,
    unsafe_allow_html=True
)