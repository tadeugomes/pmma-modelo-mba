"""
Dashboard Streamlit com dados REAIS da PMMA
Versão que utiliza os dados reais fornecidos
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import folium
from streamlit_folium import st_folium
from collections import Counter
import re

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
    import os

    # Tentar diferentes caminhos possíveis
    paths = [
        '../output/pmma_unificado_oficial.parquet',
        '../../output/pmma_unificado_oficial.parquet',
        '/Users/tgt/Documents/dados_pmma_copy/output/pmma_unificado_oficial.parquet'
    ]

    for path in paths:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            st.success(f"✅ Dados carregados de: {path}")

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
                    # Tentar extrair hora de vários formatos
                    if ':' in str(hora_str):
                        return int(str(hora_str).split(':')[0])
                    elif '.' in str(hora_str):
                        return int(float(str(hora_str)))
                    else:
                        hora_int = int(str(hora_str))
                        return hora_int if 0 <= hora_int <= 23 else 12
                except:
                    return 12

            df['hora_valida'] = df['hora'].apply(extract_hour)

            # Adicionar dia da semana
            df['dia_semana'] = df['data'].dt.day_name()
            df['mes'] = df['data'].dt.month
            df['ano'] = df['data'].dt.year

            return df

    st.error("❌ Arquivo de dados não encontrado!")
    return None
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
            # Tentar extrair hora de vários formatos
            if ':' in str(hora_str):
                return int(str(hora_str).split(':')[0])
            elif '.' in str(hora_str):
                return int(float(str(hora_str)))
            else:
                hora_int = int(str(hora_str))
                return hora_int if 0 <= hora_int <= 23 else 12
        except:
            return 12

    df['hora_valida'] = df['hora'].apply(extract_hour)

    # Adicionar dia da semana
    df['dia_semana'] = df['data'].dt.day_name()
    df['mes'] = df['data'].dt.month
    df['ano'] = df['data'].dt.year

    return df

# Sidebar para navegação
st.sidebar.title("Navegação")
page = st.sidebar.selectbox(
    "Selecione uma página:",
    ["📊 Visão Geral", "🔮 Previsão de Ocorrências", "🏷️ Análise de Classificação", "🎯 Análise de Recursos"]
)

# Filtros globais na sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Filtros Globais")

try:
    df = load_data()
    data_loaded = True

    # Filtros
    anos_disponiveis = sorted(df['ano'].unique())
    ano_selecionado = st.sidebar.multiselect(
        "Selecione o(s) ano(s)",
        options=anos_disponiveis,
        default=[anos_disponiveis[-1]]  # Último ano por padrão
    )

    areas_principais = ['Norte', 'Sul', 'Leste', 'Oeste', 'Centro']
    areas_disponiveis = ['Todas'] + areas_principais
    area_selecionada = st.sidebar.selectbox(
        "Selecione a área",
        options=areas_disponiveis,
        index=0
    )

    # Aplicar filtros ao dataframe
    df_filtrado = df.copy()
    if ano_selecionado:
        df_filtrado = df_filtrado[df_filtrado['ano'].isin(ano_selecionado)]

    if area_selecionada != 'Todas':
        df_filtrado = df_filtrado[df_filtrado['area_padrao'] == area_selecionada.lower()]

    st.sidebar.success(f"✅ {len(df_filtrado):,} registros carregados")

except Exception as e:
    st.error(f"❌ Erro ao carregar dados: {str(e)}")
    data_loaded = False

# Página 1: Visão Geral
if page == "📊 Visão Geral":
    st.header("📊 Visão Geral das Ocorrências")

    if data_loaded:
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_ocorrencias = len(df_filtrado)
            st.metric("Total de Ocorrências", f"{total_ocorrencias:,}")

        with col2:
            media_diaria = total_ocorrencias / max(1, len(df_filtrado['data'].dt.date.unique()))
            st.metric("Média Diária", f"{media_diaria:.0f}")

        with col3:
            if len(df_filtrado) > 0:
                hora_pico = df_filtrado.groupby('hora_valida').size().idxmax()
                st.metric("Horário de Pico", f"{hora_pico:02d}:00")
            else:
                st.metric("Horário de Pico", "N/A")

        with col4:
            areas_atendidas = df_filtrado['area_padrao'].nunique()
            st.metric("Áreas Atendidas", areas_atendidas)

        # Gráficos
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ocorrências por Mês")
            mes_counts = df_filtrado.groupby('mes').size().reset_index(name='count')
            meses_nome = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
            mes_counts['mes_nome'] = mes_counts['mes'].apply(lambda x: meses_nome[x-1])

            fig = px.bar(
                mes_counts,
                x='mes_nome',
                y='count',
                title="Distribuição Mensal"
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Tipos de Ocorrência (Top 10)")
            if 'descricao_tipo' in df_filtrado.columns:
                top_tipos = df_filtrado['descricao_tipo'].value_counts().head(10).reset_index()
                top_tipos.columns = ['tipo', 'count']

                fig = px.bar(
                    top_tipos,
                    x='count',
                    y='tipo',
                    orientation='h',
                    title="Top 10 Tipos de Ocorrência"
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

        # Mapa de calor temporal real
        st.subheader("Mapa de Calor - Ocorrências por Hora e Dia da Semana")

        # Criar pivot table com dados reais
        heatmap_data = df_filtrado.groupby(['dia_semana', 'hora_valida']).size().unstack(fill_value=0)

        # Ordenar dias da semana
        dias_ordem = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dias_portugues = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
        heatmap_data = heatmap_data.reindex(dias_ordem, fill_value=0)

        fig = px.imshow(
            heatmap_data.values,
            x=[f"{h:02d}:00" for h in heatmap_data.columns],
            y=dias_portugues,
            title="Intensidade de Ocorrências (Dados Reais)",
            labels={'x': 'Hora', 'y': 'Dia da Semana', 'color': 'Ocorrências'},
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Estatísticas detalhadas
        st.subheader("📈 Estatísticas Detalhadas")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Por Área:**")
            area_stats = df_filtrado['area_padrao'].value_counts()
            for area, count in area_stats.head().items():
                percentual = (count / len(df_filtrado)) * 100
                st.write(f"- {area}: {count:,} ({percentual:.1f}%)")

        with col2:
            st.write("**Por Período do Dia:**")
            df_filtrado['periodo'] = pd.cut(
                df_filtrado['hora_valida'],
                bins=[0, 6, 12, 18, 24],
                labels=['Madrugada', 'Manhã', 'Tarde', 'Noite']
            )
            periodo_stats = df_filtrado['periodo'].value_counts()
            for periodo, count in periodo_stats.items():
                percentual = (count / len(df_filtrado)) * 100
                st.write(f"- {periodo}: {count:,} ({percentual:.1f}%)")

# Página 2: Previsão de Ocorrências
elif page == "🔮 Previsão de Ocorrências":
    st.header("🔮 Análise Preditiva de Ocorrências")
    st.info("📌 Esta seção mostra padrões históricos que podem ajudar na previsão de demanda")

    if data_loaded:
        # Análise de padrões sazonais
        st.subheader("📅 Padrões Sazonais")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Padrão por Hora do Dia:**")
            hora_counts = df_filtrado.groupby('hora_valida').size()
            fig = px.line(
                x=hora_counts.index,
                y=hora_counts.values,
                title="Ocorrências por Hora",
                labels={'x': 'Hora', 'y': 'Número de Ocorrências'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Padrão por Dia da Semana:**")
            dia_counts = df_filtrado.groupby('dia_semana').size()
            fig = px.bar(
                x=dia_counts.index,
                y=dia_counts.values,
                title="Ocorrências por Dia da Semana"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Análise de áreas críticas
        st.subheader("🎯 Análise de Áreas Críticas")

        # Top 5 áreas com mais ocorrências
        area_counts = df_filtrado['area_padrao'].value_counts().head(5)

        fig = go.Figure(data=[
            go.Bar(name='Ocorrências', x=area_counts.index, y=area_counts.values)
        ])
        fig.update_layout(
            title="Top 5 Áreas com Mais Ocorrências",
            xaxis_title="Área",
            yaxis_title="Número de Ocorrências"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Análise de tendência temporal
        if len(df_filtrado['ano'].unique()) > 1:
            st.subheader("📈 Tendência Temporal")

            tendencia = df_filtrado.groupby('ano').size().reset_index(name='count')

            fig = px.line(
                tendencia,
                x='ano',
                y='count',
                title="Evolução das Ocorrências ao Longo dos Anos",
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

        # Insights baseados nos dados
        st.subheader("💡 Insights dos Dados")

        insights = []

        # Hora de pico
        hora_pico = df_filtrado.groupby('hora_valida').size().idxmax()
        hora_pico_count = df_filtrado.groupby('hora_valida').size().max()
        insights.append(f"O horário de pico é às {hora_pico:02d}:00 com {hora_pico_count:,} ocorrências")

        # Dia mais movimentado
        dia_pico = df_filtrado.groupby('dia_semana').size().idxmax()
        insights.append(f"O dia mais movimentado é {dia_pico}")

        # Área crítica
        area_critica = df_filtrado['area_padrao'].value_counts().index[0]
        insights.append(f"A área mais crítica é {area_critica.title()}")

        for insight in insights:
            st.write(f"• {insight}")

# Página 3: Análise de Classificação
elif page == "🏷️ Análise de Classificação":
    st.header("🏷️ Análise dos Tipos de Ocorrência")

    if data_loaded:
        if 'descricao_tipo' in df_filtrado.columns:
            # Top tipos
            st.subheader("📊 Principais Tipos de Ocorrência")

            tipo_counts = df_filtrado['descricao_tipo'].value_counts()

            col1, col2 = st.columns([3, 1])

            with col1:
                fig = px.bar(
                    x=tipo_counts.values[:20],
                    y=tipo_counts.index[:20],
                    orientation='h',
                    title="Top 20 Tipos de Ocorrência"
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Estatísticas")
                st.metric("Tipos Diferentes", len(tipo_counts))
                st.metric("Tipo Mais Comum", tipo_counts.index[0])
                st.metric("Ocorrências do Tipo Principal", f"{tipo_counts.iloc[0]:,}")

            # Análise de subtipos
            if 'descricao_subtipo' in df_filtrado.columns:
                st.subheader("🔍 Análise de Subtipos")

                # Selecionar tipo para analisar
                tipo_selecionado = st.selectbox(
                    "Selecione um tipo para analisar os subtipos:",
                    options=tipo_counts.index[:10]
                )

                df_tipo = df_filtrado[df_filtrado['descricao_tipo'] == tipo_selecionado]
                subtipo_counts = df_tipo['descricao_subtipo'].value_counts().head(10)

                if len(subtipo_counts) > 0:
                    fig = px.bar(
                        x=subtipo_counts.values,
                        y=subtipo_counts.index,
                        orientation='h',
                        title=f"Subtipos de {tipo_selecionado}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Palavras mais comuns nas descrições
            if 'titulo' in df_filtrado.columns:
                st.subheader("📝 Palavras Mais Comuns nos Títulos")

                # Extrair palavras
                all_words = []
                for titulo in df_filtrado['titulo'].dropna():
                    words = re.findall(r'\b\w+\b', str(titulo).lower())
                    all_words.extend([w for w in words if len(w) > 3])

                word_counts = Counter(all_words).most_common(20)
                words_df = pd.DataFrame(word_counts, columns=['palavra', 'frequencia'])

                fig = px.bar(
                    words_df,
                    x='frequencia',
                    y='palavra',
                    orientation='h',
                    title="Top 20 Palavras nos Títulos"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Coluna 'descricao_tipo' não encontrada nos dados")

# Página 4: Análise de Recursos
elif page == "🎯 Análise de Recursos":
    st.header("🎯 Análise de Distribuição de Recursos")
    st.info("📌 Análise baseada na distribuição geográfica das ocorrências")

    if data_loaded:
        # Mapa de ocorrências por área
        st.subheader("📍 Distribuição Geográfica das Ocorrências")

        # Coordenadas aproximadas para São Luís e áreas
        coords_areas = {
            'norte': (-2.48, -44.30),
            'sul': (-2.55, -44.28),
            'leste': (-2.52, -44.25),
            'oeste': (-2.53, -44.33),
            'centro': (-2.53, -44.28),
            'não informada': (-2.53, -44.30)
        }

        # Criar mapa
        m = folium.Map(
            location=[-2.53, -44.30],
            zoom_start=11,
            tiles="OpenStreetMap"
        )

        # Adicionar círculos para cada área
        area_counts = df_filtrado['area_padrao'].value_counts()

        for area, count in area_counts.items():
            if area in coords_areas:
                lat, lon = coords_areas[area]

                # Tamanho do círculo baseado no número de ocorrências
                radius = min(2000, 500 + count / 100)

                folium.Circle(
                    location=[lat, lon],
                    radius=radius,
                    popup=f"<b>Área: {area.title()}</b><br>Ocorrências: {count:,}",
                    tooltip=f"{area.title()}: {count:,} ocorrências",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.3
                ).add_to(m)

        # Exibir mapa
        st_data = st_folium(m, width=700, height=500)

        # Análise de distribuição de CPAMs
        if 'cpam' in df_filtrado.columns:
            st.subheader("🏢 Análise por CPAM")

            cpam_counts = df_filtrado['cpam'].value_counts().head(10)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    x=cpam_counts.values,
                    y=cpam_counts.index,
                    orientation='h',
                    title="Top 10 CPAMs por Número de Ocorrências"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.write("**Distribuição:**")
                for cpam, count in cpam_counts.head().items():
                    percentual = (count / len(df_filtrado)) * 100
                    st.write(f"{cpam}: {percentual:.1f}%")

        # Recomendações baseadas nos dados
        st.subheader("💡 Recomendações Operacionais")

        # Calcular densidade por hora
        densidade_horaria = df_filtrado.groupby(['area_padrao', 'hora_valida']).size().unstack(fill_value=0)

        # Identificar áreas e horários críticos
        criticos = []
        for area in densidade_horaria.index:
            if area in ['norte', 'sul', 'leste', 'oeste']:
                hora_critica = densidade_horaria.loc[area].idxmax()
                max_ocorr = densidade_horaria.loc[area].max()
                criticos.append((area.title(), hora_critica, max_ocorr))

        # Ordenar por número de ocorrências
        criticos.sort(key=lambda x: x[2], reverse=True)

        st.write("**Pontos Críticos Identificados:**")
        for area, hora, count in criticos[:5]:
            st.write(f"• {area}: Pico às {hora:02d}:00 ({count} ocorrências)")

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🚔 <b>Sistema de Inteligência Policial - PMMA</b></p>
        <p>Análise desenvolvida com dados reais das ocorrências (2014-2024)</p>
        <p>Total de {len(df) if data_loaded else 0:,} registros analisados</p>
    </div>
    """,
    unsafe_allow_html=True
)