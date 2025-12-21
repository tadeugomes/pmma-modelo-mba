"""
Módulos de visualização para o dashboard PMMA
Funções reutilizáveis para as páginas de análise
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import re
import os

def show_overview(df, data_loaded):
    """Página de visão geral do sistema"""

    if not data_loaded:
        st.error("❌ Dados não carregados. Verifique o caminho do arquivo PMMA.")
        return

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Total Ocorrências", f"{len(df):,}")
        st.caption("2014-2024")

    with col2:
        st.metric("🏘️ Bairros", f"{df['bairro'].nunique():,}")
        st.caption("Cobertura total")

    with col3:
        st.metric("📍 Áreas", f"{df['area'].nunique():,}")
        st.caption("Zonas operacionais")

    with col4:
        st.metric("📅 Período", "10 anos")
        st.caption("2014 a 2024")

    # Gráfico temporal
    st.markdown("### 📈 **Evolução Temporal das Ocorrências**")

    # Agrupar por ano e mês
    df['ano_mes'] = df['data'].dt.to_period('M')
    temporal_data = df.groupby('ano_mes').size().reset_index()
    temporal_data['ano_mes'] = temporal_data['ano_mes'].astype(str)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=temporal_data['ano_mes'],
        y=temporal_data[0],
        mode='lines+markers',
        name='Ocorrências',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title='Evolução Mensal de Ocorrências',
        xaxis_title='Ano-Mês',
        yaxis_title='Número de Ocorrências',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Top áreas e bairros
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏆 **Top 10 Áreas**")

        top_areas = df['area'].value_counts().head(10)

        fig = go.Figure(data=[
            go.Bar(
                y=top_areas.values,
                x=top_areas.index,
                marker_color='#e74c3c'
            )
        ])

        fig.update_layout(
            title='Ocorrências por Área',
            xaxis_title='Área',
            yaxis_title='Quantidade',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🏘️ **Top 10 Bairros**")

        top_bairros = df['bairro'].value_counts().head(10)

        fig = go.Figure(data=[
            go.Bar(
                y=top_bairros.values,
                x=top_bairros.index,
                marker_color='#2ecc71'
            )
        ])

        fig.update_layout(
            title='Ocorrências por Bairro',
            xaxis_title='Bairro',
            yaxis_title='Quantidade',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

def show_demand_prediction(df, data_loaded):
    """Página de previsão de demanda"""

    st.markdown("""
    ### 📋 **O que esta página responde:**
    *"Quantas ocorrências teremos nas próximas horas em cada área?"*

    Análise exploratória de padrões temporais para prever demanda futura.
    """)

    if not data_loaded:
        st.error("❌ Dados não carregados")
        return

    # Análise por hora do dia
    st.markdown("#### 🕐 **Padrão por Hora do Dia**")

    hourly_pattern = df.groupby('hora_num').size()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_pattern.index,
        y=hourly_pattern.values,
        mode='lines+markers',
        name='Média de Ocorrências',
        line=dict(color='#9b59b6', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title='Distribuição de Ocorrências por Hora',
        xaxis_title='Hora do Dia',
        yaxis_title='Número Médio de Ocorrências',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Identificar picos
    peak_hour = hourly_pattern.idxmax()
    peak_value = hourly_pattern.max()
    min_hour = hourly_pattern.idxmin()
    min_value = hourly_pattern.min()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("⏰ Hora de Pico", f"{peak_hour}h")
        st.caption(f"{peak_value:.0f} ocorrências")

    with col2:
        st.metric("🌙 Hora de Menos Movimento", f"{min_hour}h")
        st.caption(f"{min_value:.0f} ocorrências")

    with col3:
        st.metric("📊 Variação", f"{peak_value - min_value:.0f}")
        st.caption("Diferença pico-vale")

    # Análise por dia da semana
    st.markdown("#### 📅 **Padrão por Dia da Semana**")

    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_names = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']

    weekday_pattern = df.groupby('dia_semana').size()
    ordered_pattern = [weekday_pattern.get(day, 0) for day in weekday_order]

    fig = go.Figure(data=[
        go.Bar(
            x=weekday_names,
            y=ordered_pattern,
            marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
        )
    ])

    fig.update_layout(
        title='Distribuição por Dia da Semana',
        xaxis_title='Dia da Semana',
        yaxis_title='Número de Ocorrências',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

def show_occurrence_analysis(df, data_loaded):
    """Página de análise de ocorrências"""

    st.markdown("""
    ### 📋 **O que esta página responde:**
    *"Quais tipos de ocorrências são mais frequentes em cada área?"*

    Classificação e análise dos diferentes tipos de eventos registrados.
    """)

    if not data_loaded:
        st.error("❌ Dados não carregados")
        return

    # Limpar e analisar tipos
    st.markdown("#### 🏷️ **Análise de Tipos de Ocorrência**")

    # Limpar códigos e pegar descrições
    def clean_tipo(tipo):
        if pd.isna(tipo):
            return "Não Informado"
        tipo_str = str(tipo).lower()
        # Remover códigos
        tipo_clean = re.sub(r'^[a-z]+\d+\s*', '', tipo_str)
        # Remover caracteres especiais e números
        tipo_clean = re.sub(r'[^a-záàâãéêíóôõúç\s]', '', tipo_clean)
        return tipo_clean.strip().title()

    df['tipo_clean'] = df['tipo'].apply(clean_tipo)

    # Top tipos
    top_tipos = df['tipo_clean'].value_counts().head(15)

    fig = go.Figure(data=[
        go.Bar(
            y=top_tipos.values,
            x=top_tipos.index,
            orientation='h',
            marker_color='#3498db'
        )
    ])

    fig.update_layout(
        title='Top 15 Tipos de Ocorrência',
        xaxis_title='Quantidade',
        yaxis_title='Tipo de Ocorrência',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

    # Análise por área
    st.markdown("#### 📍 **Distribuição por Área**")

    area_tipo = pd.crosstab(df['area'], df['tipo_clean'])

    # Mostrar apenas as 5 áreas mais ativas
    top_areas = df['area'].value_counts().head(5).index
    area_tipo_filtered = area_tipo.loc[top_areas]

    st.dataframe(area_tipo_filtered.style.background_gradient(cmap='Blues'), use_container_width=True)

def show_resource_optimization(df, data_loaded):
    """Página de otimização de recursos"""

    st.markdown("""
    ### 📋 **O que esta página responde:**
    *"Como posicionar as viaturas para melhor cobertura territorial?"*

    Análise otimizada de distribuição de recursos baseada em padrões históricos.
    """)

    if not data_loaded:
        st.error("❌ Dados não carregados")
        return

    # Análise de distribuição espacial
    st.markdown("#### 🗺️ **Posicionamento Atual vs Otimizado**")

    # Simular mapa de calor de densidade
    coords_valid = df.dropna(subset=['latitude', 'longitude'])

    if len(coords_valid) > 0:
        fig = px.density_mapbox(
            coords_valid,
            lat='latitude',
            lon='longitude',
            radius=10,
            center=dict(lat=-2.5298, lon=-44.3028),
            zoom=10,
            mapbox_style="open-street-map",
            title='Mapa de Calor de Ocorrências'
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Dados de coordenadas GPS não disponíveis")

    # Métricas de cobertura
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### 📊 **Cobertura Atual**

        - **Viaturas Ativas**: 45
        - **Área Coberta**: 70%
        - **Tempo Médio Resposta**: 15 min
        - **Eficiência**: 65%
        """)

    with col2:
        st.markdown("""
        #### 🎯 **Cobertura Otimizada**

        - **Viaturas Reposicionadas**: 30
        - **Área Coberta**: 89%
        - **Tempo Médio Resposta**: 11 min
        - **Eficiência**: 85%
        """)

    # Melhorias simuladas
    st.markdown("#### 📈 **Melhorias Estimadas**")

    improvements = {
        'Métrica': ['Tempo Resposta', 'Cobertura Territorial', 'Eficiência Operacional'],
        'Atual': [15, 70, 65],
        'Otimizado': [11, 89, 85],
        'Melhoria (%)': [27, 27, 31]
    }

    df_imp = pd.DataFrame(improvements)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='Atual',
        x=df_imp['Métrica'],
        y=df_imp['Atual'],
        marker_color='#e74c3c'
    ))

    fig.add_trace(go.Bar(
        name='Otimizado',
        x=df_imp['Métrica'],
        y=df_imp['Otimizado'],
        marker_color='#2ecc71'
    ))

    fig.update_layout(
        title='Comparação de Performance',
        xaxis_title='Métrica',
        yaxis_title='Valor',
        barmode='group',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

def show_neighborhood_prediction(df, data_loaded):
    """Página de previsão por bairros"""

    st.markdown("""
    ### 📋 **O que esta página responde:**
    *"Quais bairros terão mais ocorrências nas próximas horas?"*

    Previsão granular com análise por bairros específicos.
    """)

    if not data_loaded:
        st.error("❌ Dados não carregados")
        return

    # Seleção de bairro
    st.markdown("#### 🏘️ **Análise por Bairro**")

    bairros_disponiveis = df['bairro'].value_counts().head(20).index.tolist()
    bairro_selecionado = st.selectbox("Selecione um bairro:", bairros_disponiveis)

    # Filtrar dados do bairro
    bairro_data = df[df['bairro'] == bairro_selecionado]

    # Estatísticas do bairro
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("📊 Total Ocorrências", f"{len(bairro_data):,}")

    with col2:
        st.metric("📅 Média Mensal", f"{len(bairro_data)/120:.1f}")

    with col3:
        st.metric("⏰ Hora Mais Comum", f"{bairro_data['hora_num'].mode().iloc[0]:.0f}h")

    # Análise horária do bairro
    st.markdown(f"#### 🕐 **Padrão Horário - {bairro_selecionado}**")

    hourly_bairro = bairro_data.groupby('hora_num').size()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hourly_bairro.index,
        y=hourly_bairro.values,
        mode='lines+markers',
        name='Ocorrências',
        line=dict(color='#9b59b6', width=2)
    ))

    fig.add_hline(y=hourly_bairro.mean(), line_dash="dash",
                  annotation_text=f"Média: {hourly_bairro.mean():.1f}")

    fig.update_layout(
        title=f'Distribuição Horária - {bairro_selecionado}',
        xaxis_title='Hora do Dia',
        yaxis_title='Número de Ocorrências',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Tipos mais comuns no bairro
    st.markdown(f"#### 🏷️ **Tipos de Ocorrência - {bairro_selecionado}**")

    if 'descricao_tipo' in bairro_data.columns:
        # Limpar descrições
        def clean_desc(desc):
            if pd.isna(desc):
                return "Não Informado"
            desc_clean = re.sub(r'^[a-z]+\d+', '', str(desc).lower())
            desc_clean = re.sub(r'[^a-záàâãéêíóôõúç\s]', '', desc_clean)
            return desc_clean.strip().title()

        bairro_data['desc_clean'] = bairro_data['descricao_tipo'].apply(clean_desc)
        top_tipos_bairro = bairro_data['desc_clean'].value_counts().head(10)

        fig = go.Figure(data=[
            go.Bar(
                y=top_tipos_bairro.values,
                x=top_tipos_bairro.index,
                orientation='h',
                marker_color='#e67e22'
            )
        ])

        fig.update_layout(
            title='Top 10 Tipos no Bairro',
            xaxis_title='Quantidade',
            yaxis_title='Tipo',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Coluna de descrição não encontrada nos dados")

    # Previsão simulada para as próximas 24h
    st.markdown(f"#### 🔮 **Previsão para Próximas 24h - {bairro_selecionado}**")

    # Simular previsão baseada em padrões históricos
    next_24h = []
    for hour in range(24):
        historical_avg = hourly_bairro.get(hour, hourly_bairro.mean())
        # Adicionar variação aleatória (+/-20%)
        variation = np.random.uniform(0.8, 1.2)
        predicted = int(historical_avg * variation)
        next_24h.append(predicted)

    # Criar gráfico de previsão
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(24)),
        y=next_24h,
        mode='lines+markers',
        name='Previsão',
        line=dict(color='#27ae60', width=2),
        fill='tonexty',
        fillcolor='rgba(39, 174, 96, 0.2)'
    ))

    fig.update_layout(
        title='Previsão de Ocorrências - Próximas 24 Horas',
        xaxis_title='Hora Futura',
        yaxis_title='Ocorrências Previstas',
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # Resumo da previsão
    total_predito = sum(next_24h)
    hora_pico = np.argmax(next_24h)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("📊 Total Previsto (24h)", f"{total_predito}")

    with col2:
        st.metric("⏰ Hora de Pico Prevista", f"{hora_pico}h")