"""
Componente de Dashboard para Previsão por Bairros
Integração com o Streamlit para visualização das predições nivel bairro
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from bairro_prediction_model import BairroPredictionModel, get_bairro_hotspots
import torch


def show_bairro_prediction_page(df=None):
    """Página completa de previsão por bairros"""

    st.header("🏘️ Previsão de Ocorrências por Bairros")

    # Caixa informativa
    st.markdown("""
    <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #856404; margin-bottom: 10px;">🤔 Pergunta do Modelo:</h2>
        <h3 style="color: black; margin-bottom: 5px;">"QUAIS bairros terão mais ocorrências nas próximas horas?"</h3>
        <p style="color: black;">O modelo por bairros prevê a demanda em nível granular, permitindo alocação direcionada de recursos para as áreas mais críticas.</p>
    </div>
    """, unsafe_allow_html=True)

    # Verificar se os dados foram passados
    if df is None:
        st.error("Dados não carregados. Não é possível exibir a página de previsão por bairros.")
        return

    # Garantir que as colunas necessárias existam
    if 'bairro' not in df.columns:
        st.error("Coluna 'bairro' não encontrada nos dados.")
        return

    # Sidebar com configurações
    st.sidebar.subheader("⚙️ Configurações")

    # Seleção de bairros
    bairro_counts = df['bairro'].value_counts()
    top_bairros = bairro_counts.head(30)

    selected_bairros = st.sidebar.multiselect(
        "📍 Bairros para Análise",
        options=top_bairros.index,
        default=list(top_bairros.head(5).index),
        help="Selecione os bairros que deseja analisar"
    )

    # Período de previsão
    prediction_hours = st.sidebar.slider(
        "⏱️ Horas de Previsão",
        min_value=1,
        max_value=48,
        value=24,
        help="Quantidade de horas para frente na previsão"
    )

    # Modelo (simulação por enquanto)
    model_loaded = False

    # Seção 1: Hotspots Atuais
    st.subheader("🔥 Bairros Críticos (Hotspots)")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Gráfico de barras dos bairros com mais ocorrências
        hotspots = get_bairro_hotspots(df, top_n=15)

        fig = go.Figure(data=[
            go.Bar(
                x=hotspots.values,
                y=hotspots.index,
                orientation='h',
                marker_color='red',
                opacity=0.7
            )
        ])

        fig.update_layout(
            title="Top 15 Bairros - Total de Ocorrências",
            xaxis_title="Número de Ocorrências",
            yaxis_title="Bairro",
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Métricas dos hotspots
        st.markdown("### 📊 Estatísticas dos Hotspots")

        if len(hotspots) > 0:
            total_hotspots = hotspots.sum()
            total_geral = bairro_counts.sum()
            percent_hotspots = (total_hotspots / total_geral) * 100

            st.metric("Total Hotspots", f"{total_hotspots:,.0f}")
            st.metric("% do Total", f"{percent_hotspots:.1f}%")
            st.metric("Bairros Analisados", f"{len(hotspots)}/{len(bairro_counts)}")

            st.markdown("### 🎯 Top 5")
            for i, (bairro, count) in enumerate(hotspots.head().items(), 1):
                st.markdown(f"{i}. **{bairro}**: {count:,}")

    # Seção 2: Análise Temporal por Bairro
    if selected_bairros:
        st.subheader("📈 Análise Temporal dos Bairros Selecionados")

        # Preparar dados temporais
        df['hora'] = df['data'].dt.hour
        df['dia_semana'] = df['data'].dt.day_name()

        # Gráfico de padrão diário
        hourly_pattern = []
        for bairro in selected_bairros:
            bairro_data = df[df['bairro'] == bairro]
            hourly_count = bairro_data.groupby('hora').size()
            hourly_pattern.append(hourly_count)

        fig = go.Figure()

        for i, pattern in enumerate(hourly_pattern):
            fig.add_trace(go.Scatter(
                x=pattern.index,
                y=pattern.values,
                mode='lines+markers',
                name=selected_bairros[i],
                line=dict(width=2)
            ))

        fig.update_layout(
            title="Padrão Diário de Ocorrências por Hora",
            xaxis_title="Hora do Dia",
            yaxis_title="Número de Ocorrências",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Gráfico de padrão semanal
        weekly_pattern = []
        dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        for bairro in selected_bairros:
            bairro_data = df[df['bairro'] == bairro]
            weekly_count = bairro_data.groupby('dia_semana').size()
            # Garantir ordem dos dias
            weekly_count = weekly_count.reindex(dias_ordem, fill_value=0)
            weekly_pattern.append(weekly_count)

        fig = go.Figure()

        for i, pattern in enumerate(weekly_pattern):
            fig.add_trace(go.Scatter(
                x=['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'],
                y=pattern.values,
                mode='lines+markers',
                name=selected_bairros[i],
                line=dict(width=2)
            ))

        fig.update_layout(
            title="Padrão Semanal de Ocorrências",
            xaxis_title="Dia da Semana",
            yaxis_title="Número de Ocorrências",
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

    # Seção 3: Previsões (Simulação)
    st.subheader("🔮 Previsão para as Próximas Horas")

    if selected_bairros:
        col1, col2 = st.columns([2, 1])

        with col1:
            # Simular previsões
            hours = list(range(1, prediction_hours + 1))

            fig = go.Figure()

            for bairro in selected_bairros[:5]:  # Limitar a 5 para não poluir
                # Simular previsão baseada no histórico
                bairro_data = df[df['bairro'] == bairro]
                media_historica = len(bairro_data) / (df['data'].max() - df['data'].min()).days * 24

                # Adicionar variação sazonal
                base_predictions = []
                for h in hours:
                    hour_of_day = (datetime.now().hour + h) % 24
                    seasonal_factor = 1.0

                    # Fatores sazonais baseados no padrão geral
                    if 18 <= hour_of_day <= 23:  # Pico noturno
                        seasonal_factor = 1.5
                    elif 6 <= hour_of_day <= 11:  # Manhã
                        seasonal_factor = 0.8
                    elif 12 <= hour_of_day <= 17:  # Tarde
                        seasonal_factor = 1.0

                    # Adicionar ruído aleatório
                    prediction = media_historica * seasonal_factor * np.random.uniform(0.8, 1.2)
                    base_predictions.append(max(0, prediction))

                fig.add_trace(go.Scatter(
                    x=hours,
                    y=base_predictions,
                    mode='lines+markers',
                    name=bairro,
                    line=dict(width=2)
                ))

            fig.update_layout(
                title=f"Previsão de Ocorrências - Próximas {prediction_hours} Horas",
                xaxis_title="Horas à Frente",
                yaxis_title="Ocorrências Previstas",
                hovermode='x unified'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📋 Resumo da Previsão")

            # Simular alertas
            alertas = []
            for bairro in selected_bairros[:5]:
                # Simular número esperado
                bairro_data = df[df['bairro'] == bairro]
                media = len(bairro_data) / (df['data'].max() - df['data'].min()).days * 24

                if media > 50:
                    nivel = "🔴 Alto"
                elif media > 20:
                    nivel = "🟡 Médio"
                else:
                    nivel = "🟢 Baixo"

                alertas.append((bairro, media, nivel))

            # Exibir alertas
            for bairro, media, nivel in alertas:
                st.markdown(f"**{bairro}**")
                st.markdown(f"Média/hora: {media:.1f} {nivel}")
                st.markdown("---")

    # Seção 4: Mapa de Calor dos Bairros
    st.subheader("🗺️ Mapa de Calor - Ocorrências por Bairro")

    # Agrupar dados por bairro
    bairro_stats = df.groupby('bairro').agg({
        'id_ocorrencia': 'count',
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()

    bairro_stats.columns = ['bairro', 'ocorrencias', 'lat_media', 'lon_media']

    # Remover bairros sem coordenadas
    bairro_stats = bairro_stats.dropna(subset=['lat_media', 'lon_media'])

    if len(bairro_stats) > 0:
        # Ajustar tamanho dos marcadores
        max_size = bairro_stats['ocorrencias'].max()
        min_size = bairro_stats['ocorrencias'].min()

        # Criar mapa de calor usando scatter_geo
        fig = go.Figure(data=go.Scattergeo(
            lat=bairro_stats['lat_media'],
            lon=bairro_stats['lon_media'],
            text=bairro_stats['bairro'],
            mode='markers',
            marker=dict(
                size=10 + (bairro_stats['ocorrencias'] - min_size) / (max_size - min_size) * 30,
                color=bairro_stats['ocorrencias'],
                colorscale='Reds',
                showscale=True,
                colorbar_title="Ocorrências",
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Ocorrências: %{marker.color:,.0f}<br>"
                "<extra></extra>"
            )
        ))

        # Configurar o mapa
        fig.update_geos(
            scope='south america',
            resolution=50,
            showframe=True,
            framecolor='white',
            framewidth=2,
            showcountries=True,
            countrycolor='#EAEAEA',
            showland=True,
            landcolor='#F0F0F0',
            showocean=True,
            oceancolor='#E6F3FF',
            projection_type='natural earth'
        )

        # Ajustar layout
        fig.update_layout(
            height=600,
            title=dict(
                text="Distribuição Geográfica das Ocorrências por Bairro",
                x=0.5,
                font=dict(size=16)
            ),
            margin=dict(r=20, t=50, b=20, l=20),
            geo=dict(
                center=dict(lat=-2.5297, lon=-44.3028),
                projection_scale=5000
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Adicionar filtro para mostrar apenas bairros selecionados no mapa
        if selected_bairros:
            st.markdown("**Bairros selecionados no mapa:**")
            for bairro in selected_bairros:
                if bairro in bairro_stats['bairro'].values:
                    stats = bairro_stats[bairro_stats['bairro'] == bairro].iloc[0]
                    st.write(f"• {bairro}: {stats['ocorrencias']:,} ocorrências")

    # Tipos de Ocorrências por Bairro
    if selected_bairros:
        st.subheader("🏷️ Tipos de Ocorrências por Bairro")

        # Preparar dados de tipos de ocorrência - Priorizar descrições
        desc_cols = ['descricao_tipo', 'descricao_subtipo', 'motivo_finalizacao_descricao']
        tipo_cols = ['natureza', 'subtipo', 'tipo']

        # Primeiro tentar encontrar colunas de descrição
        tipo_coluna = None
        tipo_label = None

        # Verificar cada coluna de descrição
        for col in desc_cols:
            if col in df.columns:
                # Verificar se tem dados válidos (não apenas códigos)
                sample_data = df[col].dropna().head(20)
                valid_count = 0
                for val in sample_data:
                    val_str = str(val).strip()
                    if not val_str.startswith(('#', 'cp', 'le', 'lpa', 'a')) and len(val_str) > 2:
                        valid_count += 1

                if valid_count > 5:  # Se tiver mais de 5 descrições válidas
                    tipo_coluna = col
                    tipo_label = "Tipo"
                    st.info(f"Usando descrições de ocorrências da coluna: {col}")
                    break

        # Se não encontrar descrições válidas, verificar outras colunas
        if tipo_coluna is None:
            st.warning("Colunas de descrição não encontradas ou inválidas. Verificando outras colunas...")
            for col in tipo_cols:
                if col in df.columns and df[col].notna().sum() > 1000:  # Se tiver dados suficientes
                    tipo_coluna = col
                    tipo_label = "Categoria"
                    st.info(f"Usando categoria: {col}")
                    break

        if tipo_coluna is None:
            st.error("Não foi possível encontrar uma coluna adequada para análise de tipos.")
            tipo_coluna = None
            tipo_label = None

        if tipo_coluna:
            # Criar abas para cada bairro selecionado
            tabs = st.tabs(selected_bairros[:5])  # Limitar a 5 abas

            for i, bairro in enumerate(tabs[:5]):
                with tabs[i]:
                    # Filtrar dados do bairro
                    bairro_data = df[df['bairro'] == selected_bairros[i]]

                    # Contar tipos de ocorrência
                    tipo_counts = bairro_data[tipo_coluna].value_counts().head(10)

                    # Função para limpar e padronizar os tipos
                    def limpar_tipo(tipo):
                        if pd.isna(tipo) or tipo == '' or tipo == ' ':
                            return None
                        tipo_str = str(tipo).strip().lower()

                        # Remover códigos no início (padrões como #a21, cp129, etc.)
                        if '#' in tipo_str or tipo_str.startswith(('cp', 'le', 'lpa', 'a')):
                            # Se for código puro, não mostrar
                            if tipo_str.startswith('#') or all(c.isdigit() or c in 'cpale' for c in tipo_str):
                                return None

                        # Converter para título capitalizado
                        return tipo_str.title()

                    # Aplicar limpeza
                    tipo_counts.index = tipo_counts.index.map(limpar_tipo)
                    tipo_counts = tipo_counts.dropna()

                    # Remover duplicatas após limpeza
                    tipo_counts = tipo_counts.groupby(tipo_counts.index).sum().sort_values(ascending=False).head(10)

                    if len(tipo_counts) > 0:
                        # Gráfico de barras
                        fig = go.Figure(data=[
                            go.Bar(
                                x=tipo_counts.values,
                                y=tipo_counts.index,
                                orientation='h',
                                marker_color='steelblue'
                            )
                        ])

                        fig.update_layout(
                            title=f"Top 10 {tipo_label}s de Ocorrência - {selected_bairros[i]}",
                            xaxis_title="Número de Ocorrências",
                            yaxis_title=tipo_label,
                            height=400,
                            yaxis={'categoryorder': 'total ascending'}
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        # Tabela com estatísticas
                        st.markdown("**Estatísticas Detalhadas:**")

                        # Calcular percentuais
                        total = tipo_counts.sum()
                        tipo_stats = pd.DataFrame({
                            tipo_label: tipo_counts.index,
                            'Quantidade': tipo_counts.values,
                            'Percentual': (tipo_counts.values / total * 100).round(1)
                        })

                        # Formatar coluna do tipo para truncar textos longos
                        tipo_stats[tipo_label] = tipo_stats[tipo_label].astype(str).str.wrap(50)

                        st.dataframe(tipo_stats, use_container_width=True)

                        # Horários mais comuns para este bairro
                        st.markdown(f"**Padrão Horário - {selected_bairros[i]}:**")
                        hora_counts = bairro_data['hora_num'].value_counts().sort_index()

                        if len(hora_counts) > 0:
                            fig_hora = go.Figure(data=[
                                go.Scatter(
                                    x=hora_counts.index,
                                    y=hora_counts.values,
                                    mode='lines+markers',
                                    line=dict(width=3, color='firebrick')
                                )
                            ])

                            fig_hora.update_layout(
                                title="Distribuição por Hora do Dia",
                                xaxis_title="Hora",
                                yaxis_title="Ocorrências",
                                height=300
                            )

                            st.plotly_chart(fig_hora, use_container_width=True)
                    else:
                        st.warning(f"Sem dados de tipos de ocorrência para {selected_bairros[i]}")

    # Recomendações
    st.subheader("💡 Recomendações Operacionais")

    if selected_bairros:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🚓 Patrulhamento")
            for bairro in selected_bairros[:3]:
                st.write(f"• Intensificar em {bairro}")

        with col2:
            st.markdown("#### ⏰ Horários Críticos")
            st.write("• 18:00 - 23:00 (Pico)")
            st.write("• Fins de semana")
            st.write("• Áreas comerciais")

        with col3:
            st.markdown("#### 📊 Monitoramento")
            st.write("• Centro histórico")
            st.write("• Via expressa")
            st.write("• Bairros periféricos")


if __name__ == "__main__":
    show_bairro_prediction_page()