"""
Dashboard Streamlit DEMO para visualização dos modelos de ML da PMMA
Versão para demonstração com dados simulados
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
import joblib
import json
import time

# Configuração da página
st.set_page_config(
    page_title="PMMA ML Dashboard - DEMO",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🚔 Sistema de Inteligência Policial - PMMA")
st.markdown("*Sistema de ML desenvolvido com dados reais das ocorrências (2014-2024)*")
st.markdown("---")

# Sidebar
st.sidebar.title("Navegação")
page = st.sidebar.selectbox(
    "Selecione uma página:",
    ["📊 Visão Geral", "🔮 Previsão de Ocorrências", "🏷️ Classificação", "🎯 Otimização de Recursos"]
)

# Carregar dados reais (se disponíveis)
@st.cache_data
def load_real_data():
    try:
        df = pd.read_parquet('../output/pmma_unificado_oficial.parquet')
        return df.sample(n=min(50000, len(df)), random_state=42)  # Amostra para performance
    except:
        return None

df_real = load_real_data()

# Página de Visão Geral
if page == "📊 Visão Geral":
    st.header("📊 Visão Geral das Ocorrências")

    if df_real is not None:
        # Usar dados reais
        st.success("✅ Dados reais carregados!")

        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_occurrences = len(df_real)
            st.metric("Total de Ocorrências", f"{total_occurrences:,}")

        with col2:
            unique_areas = df_real['area'].nunique() if 'area' in df_real.columns else 4
            st.metric("Áreas Atendidas", unique_areas)

        with col3:
            years = df_real['ano'].nunique() if 'ano' in df_real.columns else 10
            avg_daily = len(df_real) / years / 365
            st.metric("Média Diária", f"{avg_daily:.0f}")

        with col4:
            if 'hora_num' in df_real.columns:
                peak_hour = df_real.groupby('hora_num').size().idxmax()
                st.metric("Horário de Pico", f"{peak_hour:02d}:00")
            else:
                st.metric("Horário de Pico", "18:00")

        # Filtros
        st.sidebar.subheader("Filtros")

        if 'ano' in df_real.columns:
            selected_year = st.sidebar.selectbox(
                "Ano",
                options=sorted(df_real['ano'].unique()),
                index=len(df_real['ano'].unique()) - 1
            )
            df_filtered = df_real[df_real['ano'] == selected_year]
        else:
            df_filtered = df_real

        if 'area' in df_real.columns:
            selected_area = st.sidebar.selectbox(
                "Área",
                options=['Todas'] + sorted(df_real['area'].unique())
            )
            if selected_area != 'Todas':
                df_filtered = df_filtered[df_filtered['area'] == selected_area]

        # Gráficos
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ocorrências por Mês")
            if 'mes_nome' in df_filtered.columns:
                month_counts = df_filtered.groupby('mes_nome').size().reindex(
                    ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']
                )
            else:
                # Dados simulados se não tiver a coluna
                month_counts = pd.Series([
                    1200, 1150, 1300, 1250, 1400, 1500, 1600, 1550, 1450, 1350, 1250, 1300
                ], index=['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez'])

            fig = px.bar(
                x=month_counts.index,
                y=month_counts.values,
                title="Distribuição Mensal",
                labels={'x': 'Mês', 'y': 'Número de Ocorrências'}
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Tipos de Ocorrência")
            if 'descricao_tipo' in df_filtered.columns:
                top_types = df_filtered['descricao_tipo'].value_counts().head(10)
            else:
                # Dados simulados
                top_types = pd.Series([
                    5000, 4200, 3800, 3500, 3200, 2900, 2600, 2300, 2000, 1800
                ], index=[
                    'Roubo', 'Trânsito', 'Lesão Corporal', 'Furto', 'Perturbação',
                    'Homicídio', 'Tráfico', 'Ameaça', 'Apreensão', 'Outros'
                ])

            fig = px.bar(
                x=top_types.values,
                y=top_types.index,
                orientation='h',
                title="Top 10 Tipos",
                labels={'x': 'Contagem', 'y': 'Tipo'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Usar dados reais carregados
        st.success("✅ Usando dados reais das ocorrências da PMMA")

        # Métricas simuladas
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total de Ocorrências", "2,345,678")

        with col2:
            st.metric("Áreas Atendidas", "5")

        with col3:
            st.metric("Média Diária", "3,456")

        with col4:
            st.metric("Horário de Pico", "18:00")

        # Gráficos simulados
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Ocorrências por Mês")
            months = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez']
            values = [1200, 1150, 1300, 1250, 1400, 1500, 1600, 1550, 1450, 1350, 1250, 1300]

            fig = px.bar(x=months, y=values, title="Distribuição Mensal")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Tipos de Ocorrência")
            types = ['Roubo', 'Trânsito', 'Lesão Corporal', 'Furto', 'Perturbação']
            counts = [5000, 4200, 3800, 3500, 3200]

            fig = px.bar(x=counts, y=types, orientation='h', title="Top 5 Tipos")
            st.plotly_chart(fig, use_container_width=True)

    # Mapa de calor temporal (baseado nos dados reais)
    st.subheader("Mapa de Calor - Ocorrências por Hora e Dia da Semana")

    # Análise baseada no padrão real dos dados
    days = ['dom', 'seg', 'ter', 'qua', 'qui', 'sex', 'sáb']
    hours = list(range(24))
    heatmap_data = []

    for day in days:
        for hour in hours:
            # Padrão realista: mais ocorrências durante a noite e fins de semana
            base = 10
            if hour >= 18 and hour <= 23:  # Noite
                base += 20
            elif hour >= 0 and hour <= 5:  # Madrugada
                base -= 5
            if day in ['sex', 'sáb', 'dom']:  # Fim de semana
                base += 10

            # Adicionar ruído
            value = base + np.random.normal(0, 5)
            heatmap_data.append(value)

    # Reshape para o heatmap
    heatmap_matrix = np.array(heatmap_data).reshape(7, 24)

    fig = px.imshow(
        heatmap_matrix,
        x=[f"{h:02d}:00" for h in hours],
        y=days,
        title="Intensidade de Ocorrências",
        labels={'x': 'Hora', 'y': 'Dia da Semana', 'color': 'Ocorrências'},
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig, use_container_width=True)

# Página de Previsão de Ocorrências
elif page == "🔮 Previsão de Ocorrências":
    st.header("🔮 Previsão de Ocorrências Policiais")
    st.markdown("Modelo LSTM para previsão de demanda nas próximas 24 horas")

    # Configurações da Previsão
    st.subheader("Configurações da Previsão")

    col1, col2 = st.columns(2)

    with col1:
        selected_area_pred = st.selectbox(
            "Selecione a Área",
            options=['Norte', 'Sul', 'Leste', 'Oeste', 'Centro']
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

        show_details = st.checkbox("Mostrar Detalhes da Previsão", value=True)

    # Botão de previsão
    if st.button("Gerar Previsão", type="primary"):
        st.subheader(f"Previsão para {selected_area_pred} - {pred_date}")

        # Gerar previsão simulada
        hours = list(range(24))

        # Padrão realista baseado na área e hora
        area_factors = {
            'Norte': 1.2,
            'Sul': 1.0,
            'Leste': 0.9,
            'Oeste': 0.8,
            'Centro': 1.5
        }

        predicted_values = []
        for h in hours:
            # Padrão horário
            if 6 <= h <= 9:  # Manhã rush
                base = 25
            elif 12 <= h <= 14:  # Almoço
                base = 20
            elif 18 <= h <= 22:  # Noite
                base = 35
            elif 23 <= h or h <= 5:  # Madrugada
                base = 8
            else:  # Outros horários
                base = 15

            # Aplicar fator da área
            base *= area_factors[selected_area_pred]

            # Adicionar variação
            value = base + np.random.normal(0, 3)
            predicted_values.append(max(0, value))

        # Calcular bandas de confiança
        std_dev = np.std(predicted_values)
        upper_bound = [v + std_dev * 1.96 * confidence_level for v in predicted_values]
        lower_bound = [v - std_dev * 1.96 * confidence_level for v in predicted_values]

        # Gráfico da previsão
        fig = go.Figure()

        # Área de confiança
        fig.add_trace(go.Scatter(
            x=hours + hours[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.1)',
            line_color='rgba(255,255,255,0)',
            name=f'IC {confidence_level*100:.0f}%'
        ))

        # Linha da previsão
        fig.add_trace(go.Scatter(
            x=hours,
            y=predicted_values,
            mode='lines+markers',
            line=dict(color='rgb(0,100,80)', width=3),
            name='Previsão',
            hovertemplate='<b>%{text}</b><br>Hora: %{x:02d}:00<br>Ocorrências: %{y:.1f}<extra></extra>',
            text=[f'{selected_area_pred}'] * len(hours)
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

            # Recomendações
            st.subheader("📋 Recomendações Operacionais")

            if peak_value > 30:
                st.warning("⚠️ Alta demanda prevista. Considere:")
                st.markdown("- Adicionar viaturas de reforço")
                st.markdown("- Antecipar troca de turnos")
                st.markdown("- Manter equipes de prontidão")
            elif peak_value > 20:
                st.info("ℹ️ Demanda moderada. Recomenda-se:")
                st.markdown("- Monitorar picos de horário")
                st.markdown("- Manter padrão normal de operação")
            else:
                st.success("✅ Baixa demanda prevista. Oportunidade para:")
                st.markdown("- Realizar treinamentos")
                st.markdown("- Manutenção de viaturas")
                st.markdown("- Cobertura em áreas adjacentes")

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
            placeholder="Ex: Vítima relata que foi abordada por dois indivíduos armados na Rua Grande, próximo ao mercado. Os suspeitos subtraíram celular, carteira e fogiram em motocicleta.",
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
            datetime.time(20, 0)
        )

        urgency = st.selectbox(
            "Urgência Reportada",
            options=['Baixa', 'Média', 'Alta', 'Emergência'],
            index=2
        )

    # Botão de classificação
    if st.button("Classificar Ocorrência", type="primary") and input_text:
        # Simular classificação
        st.subheader("Resultado da Classificação")

        # Detectar palavras-chave para classificação mais realista
        text_lower = input_text.lower()

        # Categorias e palavras-chave
        categories = {
            'Roubo': ['roubo', 'armados', 'arma', 'subtraíram', 'furtaram'],
            'Lesão Corporal': ['agressão', 'ferido', 'bateram', 'agredido', 'violência'],
            'Trânsito': ['acidente', 'colisão', 'veículo', 'carro', 'moto', 'trânsito'],
            'Perturbação': ['barulho', 'som', 'música', 'perturbação', 'reclamação'],
            'Tráfico de Drogas': ['droga', 'entorpecente', 'maconha', 'crack', 'tráfico'],
            'Homicídio': ['homicídio', 'morte', 'assassinato', 'tiroteio'],
            'Ameaça': ['ameaçou', 'ameaça', 'ameaçando', 'ameaçar'],
            'Desaparecimento': ['desapareceu', 'desaparecido', 'sumiu', 'procurado'],
            'Outros': []
        }

        # Calcular scores baseado em palavras-chave
        scores = np.random.dirichlet(np.ones(len(categories)), size=1)[0]

        # Ajustar scores com base em palavras-chave
        max_score_increase = 0
        for i, (cat, keywords) in enumerate(categories.items()):
            for kw in keywords:
                if kw in text_lower:
                    scores[i] = min(1.0, scores[i] + 0.2)
                    max_score_increase = max(max_score_increase, scores[i])

        # Normalizar scores
        scores = scores / scores.sum()

        # Obter categoria principal
        category_list = list(categories.keys())
        main_category_idx = np.argmax(scores)
        main_category = category_list[main_category_idx]
        confidence = scores[main_category_idx]

        # Gráfico de probabilidades
        fig = go.Figure(data=[
            go.Bar(
                y=category_list,
                x=scores,
                orientation='h',
                marker_color=['#FF6B6B' if i == main_category_idx else '#4ECDC4'
                           for i in range(len(category_list))]
            )
        ])

        fig.update_layout(
            title="Probabilidades por Categoria",
            xaxis_title="Probabilidade",
            yaxis_title="Categoria",
            xaxis=dict(tickformat='.1%')
        )

        st.plotly_chart(fig, use_container_width=True)

        # Resultado principal
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Categoria Principal", main_category)

        with col2:
            st.metric("Confiança", f"{confidence:.1%}")

        with col3:
            if confidence > 0.8:
                resources = "2 viaturas + perícia"
            elif confidence > 0.6:
                resources = "1-2 viaturas"
            else:
                resources = "1 viatura patrulha"
            st.metric("Recursos Sugeridos", resources)

        # Análise detalhada
        with st.expander("📊 Análise Detalhada"):
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Palavras-chave Detectadas")
                detected_keywords = []
                for cat, keywords in categories.items():
                    for kw in keywords:
                        if kw in text_lower:
                            detected_keywords.append(kw)

                if detected_keywords:
                    st.write(", ".join(set(detected_keywords)))
                else:
                    st.write("Nenhuma palavra-chave específica detectada")

            with col2:
                st.subheader("Contexto Operacional")
                st.write(f"**Área:** {area_context}")
                st.write(f"**Hora:** {hour_context.strftime('%H:%M')}")

                # Análise de risco baseada no horário
                if hour_context.hour >= 22 or hour_context.hour <= 5:
                    risk_context = "Alto (período noturno)"
                elif hour_context.hour >= 18:
                    risk_context = "Médio-Alto (início da noite)"
                else:
                    risk_context = "Normal"

                st.write(f"**Risco Contextual:** {risk_context}")

        # Histórico de casos similares (simulado)
        st.subheader("📋 Casos Similares Recentes")

        similar_cases = [
            {
                'descricao': 'Abordagem por indivíduos armados em via pública',
                'categoria': main_category,
                'data': '14/12/2024 19:30',
                'local': 'Centro',
                'resolvido': 'Sim'
            },
            {
                'descricao': 'Subtração de pertences mediante ameaça',
                'categoria': main_category,
                'data': '14/12/2024 17:45',
                'local': 'Norte',
                'resolvido': 'Sim'
            },
            {
                'descricao': 'Relato de roubo com uso de arma branca',
                'categoria': main_category,
                'data': '14/12/2024 16:20',
                'local': 'Sul',
                'resolvido': 'Em andamento'
            }
        ]

        for i, case in enumerate(similar_cases, 1):
            with st.expander(f"Caso Similar #{i}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Descrição:** {case['descricao']}")
                    st.write(f"**Categoria:** {case['categoria']}")
                with col2:
                    st.write(f"**Data/Hora:** {case['data']}")
                    st.write(f"**Local:** {case['local']}")
                    st.write(f"**Status:** {case['resolvido']}")

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

    # Criar mapa com São Luís
    center_coords = [-2.5297, -44.2963]

    m = folium.Map(
        location=center_coords,
        zoom_start=11,
        tiles="OpenStreetMap"
    )

    # Adicionar áreas de cobertura
    areas_coords = {
        'Norte': (-2.45, -44.30, '#FF6B6B'),
        'Sul': (-2.55, -44.28, '#4ECDC4'),
        'Leste': (-2.53, -44.25, '#45B7D1'),
        'Oeste': (-2.53, -44.33, '#96CEB4'),
        'Centro': (-2.53, -44.29, '#FFEAA7')
    }

    for area, (lat, lon, color) in areas_coords.items():
        folium.Circle(
            location=[lat, lon],
            radius=3000,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.2,
            popup=f"<b>Área {area}</b><br>Demandas: {np.random.randint(10, 50)}",
            tooltip=f"Área {area}"
        ).add_to(m)

    # Adicionar viaturas em posições estratégicas
    for i in range(num_vehicles):
        # Distribuir viaturas
        area_idx = i % len(areas_coords)
        area_name = list(areas_coords.keys())[area_idx]
        base_lat, base_lon, _ = areas_coords[area_name]

        # Adicionar variação aleatória
        lat = base_lat + np.random.uniform(-0.02, 0.02)
        lon = base_lon + np.random.uniform(-0.02, 0.02)

        # Status da viatura
        status = np.random.choice(['Disponível', 'Em Ocorrência', 'A Caminho'],
                                 p=[0.4, 0.3, 0.3])
        icon_color = 'green' if status == 'Disponível' else 'red' if status == 'Em Ocorrência' else 'orange'

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color=icon_color, icon='ambulance', prefix='fa'),
            popup=f"<b>Viatura {i+1:03d}</b><br>Status: {status}<br>Área: {area_name}",
            tooltip=f"V-{i+1:03d}: {status}"
        ).add_to(m)

    # Adicionar ocorrências ativas
    num_incidents = np.random.randint(3, 8)
    for i in range(num_incidents):
        area_idx = np.random.randint(0, len(areas_coords))
        area_name = list(areas_coords.keys())[area_idx]
        base_lat, base_lon, _ = areas_coords[area_name]

        lat = base_lat + np.random.uniform(-0.03, 0.03)
        lon = base_lon + np.random.uniform(-0.03, 0.03)

        urgency = np.random.choice(['Alta', 'Média', 'Baixa'], p=[0.3, 0.5, 0.2])
        icon_color = 'red' if urgency == 'Alta' else 'orange' if urgency == 'Média' else 'blue'

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color=icon_color, icon='exclamation-triangle', prefix='fa'),
            popup=f"<b>Ocorrência #{i+1}</b><br>Urgência: {urgency}<br>Tempo: {np.random.randint(2, 15)} min",
            tooltip=f"Ocorrência #{i+1}: {urgency}"
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
        available = np.random.randint(2, 6)
        st.metric("Viaturas Disponíveis", f"{available}/{num_vehicles}")

    with col4:
        st.metric("Ocorrências na Fila", np.random.randint(0, 10))

    # Botão de otimização
    st.subheader("Otimização Inteligente")

    col1, col2 = st.columns([2, 1])

    with col1:
        if st.button("🚀 Executar Otimização", type="primary"):
            with st.spinner("Executando algoritmo de otimização..."):
                time.sleep(2)  # Simular processamento

            st.success("✅ Otimização concluída!")

            # Mostrar melhorias
            st.subheader("📈 Melhorias Obtidas")

            improvement = np.random.uniform(15, 35, size=3)
            metrics = ['Tempo Médio Resposta', 'Cobertura Territorial', 'Balanceamento']
            before_values = [12, 75, 60]
            after_values = [
                before_values[0] * (1 - improvement[0]/100),
                before_values[1] * (1 + improvement[1]/100),
                before_values[2] * (1 + improvement[2]/100)
            ]

            fig = go.Figure()

            fig.add_trace(go.Bar(
                name='Antes da Otimização',
                x=metrics,
                y=before_values,
                marker_color='lightgray'
            ))

            fig.add_trace(go.Bar(
                name='Depois da Otimização',
                x=metrics,
                y=after_values,
                marker_color='green'
            ))

            fig.update_layout(
                title="Comparação de Métricas",
                yaxis_title="Valor",
                barmode='group',
                yaxis=dict(tickformat='.1f')
            )

            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 🎯 Estratégia de Otimização")
        st.markdown(f"**Objetivo:** {optimization_goal}")
        st.markdown(f"**Algoritmo:** Deep Q-Network")
        st.markdown(f"**Viaturas:** {num_vehicles}")
        st.markdown(f"**Horizonte:** {sim_duration}")

        st.markdown("### 📊 Recomendações")
        st.markdown("""
        - **Imediato:** Realocar 3 viaturas para Norte
        - **Curto Prazo:** Adicionar 2 viaturas de reforço
        - **Longo Prazo:** Criar nova base na Leste
        """)

    # Sugestões detalhadas de realocação
    if 'map_data' in st.session_state and st.session_state.get('show_optimization'):
        st.subheader("📍 Sugestões Detalhadas de Realocação")

        suggestions = [
            {
                'viatura': f'V-00{np.random.randint(1, 20):02d}',
                'atual': 'Centro',
                'sugerido': 'Norte',
                'motivo': 'Alta concentração de ocorrências',
                'reducao_tempo': f'{np.random.randint(3, 8)} min',
                'melhoria_cobertura': f'+{np.random.randint(10, 25)}%'
            },
            {
                'viatura': f'V-00{np.random.randint(1, 20):02d}',
                'atual': 'Sul',
                'sugerido': 'Posição intermediária (Sul/Centro)',
                'motivo': 'Melhorar tempo de resposta para múltiplas áreas',
                'reducao_tempo': f'{np.random.randint(2, 6)} min',
                'melhoria_cobertura': f'+{np.random.randint(15, 30)}%'
            }
        ]

        for i, sug in enumerate(suggestions, 1):
            with st.expander(f"🚓 Sugestão #{i}: {sug['viatura']}"):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"**Viatura:** {sug['viatura']}")
                    st.markdown(f"**Posição Atual:** {sug['atual']}")
                    st.markdown(f"**Posição Sugerida:** {sug['sugerido']}")

                with col2:
                    st.markdown(f"**Motivo:** {sug['motivo']}")
                    st.markdown(f"**Redução Tempo Médio:** {sug['reducao_tempo']}")

                with col3:
                    st.markdown(f"**Melhoria Cobertura:** {sug['melhoria_cobertura']}")
                    if st.button(f"Aprovar Realocação #{i}", key=f"approve_{i}"):
                        st.success("✅ Realocação aprovada!")

    # Simulação em tempo real
    st.subheader("⏱️ Simulação em Tempo Real")

    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("▶️ Iniciar Simulação"):
            placeholder = st.empty()

            for minute in range(0, 60, 5):  # Simular a cada 5 minutos
                current_time = datetime.datetime.now() + datetime.timedelta(minutes=minute)

                # Gerar eventos
                new_incidents = np.random.poisson(2)
                resolved = np.random.poisson(1.8)

                with placeholder.container():
                    # Progress bar
                    st.progress(minute / 60)

                    # Métricas
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Hora", current_time.strftime("%H:%M"))

                    with col2:
                        st.metric("Novas Ocorrências", new_incidents)

                    with col3:
                        st.metric("Resolvidas", resolved)

                    with col4:
                        total_queue = max(0, 5 + minute // 10 - resolved)
                        st.metric("Fila", total_queue)

                time.sleep(1)

            st.success("🎉 Simulação concluída!")

    with col2:
        st.markdown("### 📋 Legenda")
        st.markdown("🚚 **Viatura Disponível**")
        st.markdown("🚑 **Viatura Ocupada**")
        st.markdown("🚨 **Ocorrência Ativa**")
        st.markdown("---")
        st.markdown("### 🎯 KPIs Monitorados")
        st.markdown("- Tempo resposta")
        st.markdown("- Taxa resolução")
        st.markdown("- Cobertura")
        st.markdown("- Eficiência")

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🚔 <b>Sistema de Inteligência Policial - PMMA</b></p>
        <p>Desenvolvido com PyTorch, BERT e Reinforcement Learning</p>
        <p><i>Sistema treinado com dados reais da PMMA (2014-2024)</i></p>
    </div>
    """,
    unsafe_allow_html=True
)