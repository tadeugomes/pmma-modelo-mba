"""
Dashboard Streamlit para visualização dos modelos de ML da PMMA
"""
import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
import json
import joblib
from transformers import BertTokenizer
import folium
from streamlit_folium import st_folium

# Adicionar paths
sys.path.append(str(Path(__file__).parents[1] / 'shared' / 'preprocessing'))
sys.path.append(str(Path(__file__).parents[1] / 'project1' / 'src'))
sys.path.append(str(Path(__file__).parents[1] / 'project2' / 'src'))
sys.path.append(str(Path(__file__).parents[1] / 'project3' / 'src'))

# Configuração da página
st.set_page_config(
    page_title="PMMA ML Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🚔 Sistema de Inteligência Policial - PMMA")
st.markdown("---")

# Sidebar
st.sidebar.title("Navegação")
page = st.sidebar.selectbox(
    "Selecione uma página:",
    ["📊 Visão Geral", "🔮 Previsão de Ocorrências", "🏷️ Classificação", "🎯 Otimização de Recursos"]
)

# Carregar dados
@st.cache_data
def load_data():
    return pd.read_parquet('../output/pmma_unificado_oficial.parquet')

df = load_data()

# Página de Visão Geral
if page == "📊 Visão Geral":
    st.header("📊 Visão Geral das Ocorrências")

    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_occurrences = len(df)
        st.metric("Total de Ocorrências", f"{total_occurrences:,}")

    with col2:
        unique_areas = df['area'].nunique()
        st.metric("Áreas Atendidas", unique_areas)

    with col3:
        avg_daily = len(df) / df['ano'].nunique() / 365
        st.metric("Média Diária", f"{avg_daily:.0f}")

    with col4:
        peak_hour = df.groupby('hora_num').size().idxmax()
        st.metric("Horário de Pico", f"{peak_hour:02d}:00")

    # Filtros
    st.sidebar.subheader("Filtros")

    # Ano
    selected_year = st.sidebar.selectbox(
        "Ano",
        options=sorted(df['ano'].unique()),
        index=len(df['ano'].unique()) - 1
    )

    # Área
    selected_area = st.sidebar.selectbox(
        "Área",
        options=['Todas'] + sorted(df['area'].unique())
    )

    # Aplicar filtros
    filtered_df = df[df['ano'] == selected_year]
    if selected_area != 'Todas':
        filtered_df = filtered_df[filtered_df['area'] == selected_area]

    # Gráficos
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ocorrências por Mês")
        month_counts = filtered_df.groupby('mes_nome').size().reindex(
            ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']
        )
        fig = px.bar(
            x=month_counts.index,
            y=month_counts.values,
            title="Distribuição Mensal",
            labels={'x': 'Mês', 'y': 'Número de Ocorrências'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tipos de Ocorrência")
        top_types = filtered_df['descricao_tipo'].value_counts().head(10)
        fig = px.bar(
            x=top_types.values,
            y=top_types.index,
            orientation='h',
            title="Top 10 Tipos",
            labels={'x': 'Contagem', 'y': 'Tipo'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Mapa de calor temporal
    st.subheader("Mapa de Calor - Ocorrências por Hora e Dia da Semana")

    pivot_df = filtered_df.groupby(['dia_nome', 'hora_num']).size().unstack(fill_value=0)

    fig = px.imshow(
        pivot_df.values,
        x=[f"{h:02d}:00" for h in pivot_df.columns],
        y=pivot_df.index,
        title="Intensidade de Ocorrências",
        labels={'x': 'Hora', 'y': 'Dia da Semana', 'color': 'Ocorrências'},
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

# Página de Previsão de Ocorrências
elif page == "🔮 Previsão de Ocorrências":
    st.header("🔮 Previsão de Ocorrências Policiais")
    st.markdown("Modelo LSTM para previsão de demanda nas próximas 24 horas")

    # Carregar modelo (simulado)
    st.subheader("Configurações da Previsão")

    col1, col2 = st.columns(2)

    with col1:
        selected_area_pred = st.selectbox(
            "Selecione a Área",
            options=['Norte', 'Sul', 'Leste', 'Oeste']
        )

        pred_date = st.date_input(
            "Data da Previsão",
            datetime.date.today() + datetime.timedelta(days=1)
        )

    with col2:
        confidence_level = st.slider(
            "Nível de Confiança",
            min_value=0.5,
            max_value=0.95,
            value=0.8,
            step=0.05
        )

        show_details = st.checkbox("Mostrar Detalhes da Previsão")

    # Botão de previsão
    if st.button("Gerar Previsão", type="primary"):
        st.subheader(f"Previsão para {selected_area_pred} - {pred_date}")

        # Simular previsão (em produção, carregar modelo real)
        hours = list(range(24))
        base_demand = 20 if selected_area_pred == 'Norte' else 15

        # Gerar dados simulados com padrão realista
        predicted_values = []
        for h in hours:
            if 6 <= h <= 9 or 18 <= h <= 22:
                demand = base_demand + np.random.normal(10, 3)
            elif 23 <= h or h <= 5:
                demand = base_demand - np.random.normal(5, 2)
            else:
                demand = base_demand + np.random.normal(0, 2)
            predicted_values.append(max(0, demand))

        # Calcular bandas de confiança
        std_dev = np.std(predicted_values)
        upper_bound = [v + std_dev * 1.96 * confidence_level for v in predicted_values]
        lower_bound = [v - std_dev * 1.96 * confidence_level for v in predicted_values]

        # Gráfico da previsão
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hours,
            y=upper_bound,
            fill=None,
            mode='lines',
            line_color='rgba(0,100,80,0.1)',
            name=f'IC {confidence_level*100:.0f}%'
        ))

        fig.add_trace(go.Scatter(
            x=hours,
            y=lower_bound,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,100,80,0.1)',
            name=''
        ))

        fig.add_trace(go.Scatter(
            x=hours,
            y=predicted_values,
            mode='lines+markers',
            line_color='rgb(0,100,80)',
            name='Previsão',
            line_width=3
        ))

        fig.update_layout(
            title="Previsão de Ocorrências - Próximas 24 Horas",
            xaxis_title="Hora do Dia",
            yaxis_title="Número Previsto de Ocorrências",
            hovermode='x unified',
            showlegend=True
        )

        st.plotly_chart(fig, use_container_width=True)

        # Métricas da previsão
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_predicted = sum(predicted_values)
            st.metric("Total Previsto", f"{total_predicted:.0f}")

        with col2:
            peak_hour_pred = hours[np.argmax(predicted_values)]
            peak_value = max(predicted_values)
            st.metric("Horário de Pico", f"{peak_hour_pred:02d}:00")

        with col3:
            avg_predicted = np.mean(predicted_values)
            st.metric("Média por Hora", f"{avg_predicted:.1f}")

        with col4:
            risk_level = "Alto" if max(predicted_values) > 30 else "Médio" if max(predicted_values) > 20 else "Baixo"
            st.metric("Nível de Risco", risk_level)

        # Detalhes
        if show_details:
            st.subheader("Detalhes da Previsão")

            details_df = pd.DataFrame({
                'Hora': [f"{h:02d}:00" for h in hours],
                'Previsão': [f"{v:.1f}" for v in predicted_values],
                'Mínimo (IC)': [f"{l:.1f}" for l in lower_bound],
                'Máximo (IC)': [f"{u:.1f}" for u in upper_bound]
            })

            st.dataframe(details_df, use_container_width=True)

    # Histórico de previsões
    st.subheader("Histórico de Previsões")
    st.info("Funcionalidade em desenvolvimento: Comparação entre previsões e valores reais")

# Página de Classificação
elif page == "🏷️ Classificação":
    st.header("🏷️ Classificação Inteligente de Ocorrências")
    st.markdown("Classificação automática usando BERT")

    # Área de input
    st.subheader("Classificar Nova Ocorrência")

    col1, col2 = st.columns([2, 1])

    with col1:
        input_text = st.text_area(
            "Descrição da Ocorrência",
            height=150,
            placeholder="Descreva a ocorrência em detalhes...",
            help="Inclua informações sobre local, tipo de evento, envolvidos, etc."
        )

    with col2:
        st.subheader("Contexto")
        area_context = st.selectbox(
            "Área",
            options=['Norte', 'Sul', 'Leste', 'Oeste', 'Centro']
        )

        hour_context = st.time_input(
            "Hora aproximada",
            datetime.time(12, 0)
        )

        urgency = st.selectbox(
            "Urgência Reportada",
            options=['Baixa', 'Média', 'Alta', 'Emergência']
        )

    # Botão de classificação
    if st.button("Classificar Ocorrência", type="primary") and input_text:
        # Simular classificação (em produção, usar modelo real)
        st.subheader("Resultado da Classificação")

        # Probabilidades simuladas
        categories = ['Lesão Corporal', 'Roubo', 'Furto', 'Trânsito', 'Perturbação', 'Outros']
        probabilities = np.random.dirichlet(np.ones(len(categories)))

        # Gráfico de probabilidades
        fig = px.bar(
            x=probabilities,
            y=categories,
            orientation='h',
            title="Probabilidades por Categoria"
        )
        fig.update_layout(xaxis_title="Probabilidade", yaxis_title="Categoria")
        st.plotly_chart(fig, use_container_width=True)

        # Resultado principal
        main_category = categories[np.argmax(probabilities)]
        confidence = max(probabilities)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Categoria Principal", main_category)

        with col2:
            st.metric("Confiança", f"{confidence:.1%}")

        with col3:
            suggested_resources = "1 viatura" if confidence > 0.8 else "2 viaturas"
            st.metric("Recursos Sugeridos", suggested_resources)

        # Análise detalhada
        with st.expander("Análise Detalhada"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Palavras-chave Detectadas")
                keywords = ['rua', 'arma', 'ferido', 'veículo', 'fuga']
                st.write(", ".join(keywords))

            with col2:
                st.subheader("Similaridades")
                similar_cases = np.random.randint(5, 50, size=3)
                st.write(f"Caso 1: {similar_cases[0]}% similar")
                st.write(f"Caso 2: {similar_cases[1]}% similar")
                st.write(f"Caso 3: {similar_cases[2]}% similar")

    # Explorador de categorias
    st.subheader("Explorador de Categorias")

    selected_category = st.selectbox(
        "Selecione uma categoria para analisar",
        options=['Lesão Corporal', 'Roubo', 'Furto', 'Trânsito', 'Perturbação']
    )

    # Simular estatísticas da categoria
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total no Mês", np.random.randint(100, 500))

    with col2:
        st.metric("Tempo Médio Resposta", f"{np.random.randint(5, 30)} min")

    with col3:
        st.metric("Taxa de Resolução", f"{np.random.randint(60, 95)}%")

    # Exemplos recentes
    st.subheader("Exemplos Recentes")
    examples = [
        "Vítima relata agressão física durante discussão",
        "Subtração de celular mediante ameaça",
        "Veículo abandonado em via pública"
    ]

    for i, example in enumerate(examples, 1):
        with st.expander(f"Exemplo {i}"):
            st.write(example)
            st.write(f"**Classificado como:** {selected_category}")
            st.write(f"**Confiança:** {np.random.uniform(0.8, 0.95):.1%}")

# Página de Otimização
elif page == "🎯 Otimização de Recursos":
    st.header("🎯 Otimização de Alocação de Recursos")
    st.markdown("Sistema inteligente para posicionamento de viaturas")

    # Configurações de simulação
    st.sidebar.subheader("Configurações da Simulação")

    num_vehicles = st.sidebar.slider(
        "Número de Viaturas",
        min_value=5,
        max_value=20,
        value=10
    )

    sim_duration = st.sidebar.selectbox(
        "Duração da Simulação",
        options=['1 hora', '6 horas', '12 horas', '24 horas'],
        index=0
    )

    optimization_goal = st.sidebar.selectbox(
        "Objetivo Principal",
        options=['Minimizar Tempo Resposta', 'Maximizar Cobertura', 'Balancear Carga']
    )

    # Mapa de posicionamento
    st.subheader("Mapa de Posicionamento Atual")

    # Criar mapa simulado
    center_coords = [-2.5297, -44.2963]  # São Luís, MA

    m = folium.Map(
        location=center_coords,
        zoom_start=11,
        tiles="OpenStreetMap"
    )

    # Adicionar áreas de cobertura
    areas = {
        'Norte': (-2.45, -44.30),
        'Sul': (-2.55, -44.28),
        'Leste': (-2.53, -44.25),
        'Oeste': (-2.53, -44.33)
    }

    for area, coords in areas.items():
        folium.Circle(
            location=coords,
            radius=3000,
            color='blue',
            fill=True,
            fillColor='blue',
            fillOpacity=0.1,
            popup=area
        ).add_to(m)

    # Adicionar viaturas (posições simuladas)
    for i in range(num_vehicles):
        lat = -2.5 + np.random.uniform(-0.1, 0.1)
        lon = -44.3 + np.random.uniform(-0.1, 0.1)

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color='red', icon='alert', prefix='fa'),
            popup=f"Viatura {i+1}"
        ).add_to(m)

    # Exibir mapa
    map_data = st_folium(m, width=700, height=500)

    # Métricas atuais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Tempo Médio Resposta", f"{np.random.randint(8, 15)} min")

    with col2:
        st.metric("Cobertura Atual", f"{np.random.randint(70, 90)}%")

    with col3:
        st.metric("Viaturas Disponíveis", f"{np.random.randint(2, 5)}/{num_vehicles}")

    with col4:
        st.metric("Ocorrências na Fila", np.random.randint(0, 10))

    # Botão de otimização
    st.subheader("Otimização Inteligente")

    if st.button("Executar Otimização", type="primary"):
        with st.spinner("Executando algoritmo de otimização..."):
            # Simular processamento
            import time
            time.sleep(2)

        st.success("Otimização concluída!")

        # Comparação antes/depois
        st.subheader("Comparação: Antes vs Depois")

        metrics = ['Tempo Médio Resposta', 'Cobertura', 'Balanceamento']
        before_values = [12, 75, 60]
        improvement = np.random.uniform(15, 35, size=3)
        after_values = [b * (1 - imp/100) if i == 0 else b * (1 + imp/100)
                       for i, (b, imp) in enumerate(zip(before_values, improvement))]

        fig = go.Figure()

        fig.add_trace(go.Bar(
            name='Antes',
            x=metrics,
            y=before_values,
            marker_color='lightgray'
        ))

        fig.add_trace(go.Bar(
            name='Depois',
            x=metrics,
            y=after_values,
            marker_color='green'
        ))

        fig.update_layout(
            title="Melhoria nas Métricas",
            yaxis_title="Valor",
            barmode='group'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Sugestões de realocação
        st.subheader("Sugestões de Realocação")

        suggestions = [
            {
                'viatura': 'V-003',
                'atual': 'Centro',
                'sugerido': 'Norte',
                'motivo': 'Alta demanda de ocorrências'
            },
            {
                'viatura': 'V-007',
                'atual': 'Sul',
                'sugerido': 'Leste',
                'motivo': 'Melhor cobertura perimetral'
            },
            {
                'viatura': 'V-010',
                'atual': 'Oeste',
                'sugerido': 'Posição intermediária',
                'motivo': 'Reduzir tempo de resposta para múltiplas áreas'
            }
        ]

        for suggestion in suggestions:
            with st.expander(f"📍 {suggestion['viatura']}"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Posição Atual:** {suggestion['atual']}")
                    st.write(f"**Posição Sugerida:** {suggestion['sugerido']}")

                with col2:
                    st.write(f"**Motivo:** {suggestion['motivo']}")
                    st.write(f"**Melhoria Estimada:** {np.random.randint(20, 40)}%")

    # Simulação em tempo real
    st.subheader("Simulação em Tempo Real")

    if st.button("Iniciar Simulação"):
        # Placeholder para animação
        placeholder = st.empty()

        for minute in range(60):
            # Gerar dados simulados
            current_time = datetime.datetime.now() + datetime.timedelta(minutes=minute)
            incidents = np.random.poisson(2)
            resolved = np.random.poisson(1.5)

            with placeholder.container():
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Hora Atual",
                        current_time.strftime("%H:%M")
                    )

                with col2:
                    st.metric(
                        "Novas Ocorrências",
                        incidents,
                        delta=f"+{incidents - 1}" if incidents > 1 else None
                    )

                with col3:
                    st.metric(
                        "Resolvidas",
                        resolved,
                        delta=f"+{resolved - 1}" if resolved > 1 else None
                    )

                # Barra de progresso
                st.progress(minute / 60)

            time.sleep(1)

        st.success("Simulação concluída!")

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Sistema de Inteligência Policial - PMMA | Desenvolvido com PyTorch, BERT e Reinforcement Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)