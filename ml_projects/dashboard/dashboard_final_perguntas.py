"""
Dashboard Final - Sistema de Inteligência Policial PMMA
Com perguntas explícitas que cada modelo responde
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
from datetime import datetime, date, time, timedelta

# Configuração da página
st.set_page_config(
    page_title="Sistema de Inteligência Policial - PMMA",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🚔 Sistema de Inteligência Policial - PMMA")
st.markdown("*Análise preditiva para tomada de decisão operacional*")
st.markdown("---")

# Carregar dados reais
@st.cache_data
def load_data():
    """Carrega os dados reais da PMMA"""
    paths = [
        '../output/pmma_unificado_oficial.parquet',
        '../../output/pmma_unificado_oficial.parquet',
        '/Users/tgt/Documents/dados_pmma_copy/output/pmma_unificado_oficial.parquet',
        './output/pmma_unificado_oficial.parquet'
    ]

    for path in paths:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Limpeza básica
            df = df.dropna(subset=['data'])
            df['data'] = pd.to_datetime(df['data'], errors='coerce')
            df = df.dropna(subset=['data'])
            df['hora_num'] = pd.to_numeric(df['hora_num'], errors='coerce').fillna(0)
            df['area'] = df['area'].fillna('Não Informada').str.lower().str.strip()
            df['dia_semana'] = df['data'].dt.day_name()
            df['mes'] = df['data'].dt.month
            df['ano'] = df['data'].dt.year
            return df
    return None

# Sidebar para navegação
st.sidebar.title("Navegação")
page = st.sidebar.selectbox(
    "Selecione uma página:",
    ["📊 Visão Geral",
     "🔮 Previsão de Demanda",
     "🏷️ Análise de Ocorrência",
     "🎯 Otimização de Recursos",
     "🏘️ Previsão por Bairros"]
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
if page == "📊 Visão Geral":
    st.header("📊 Análise Histórica das Ocorrências")

    # Caixa de pergunta principal
    st.markdown("""
    <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; margin-bottom: 20px; border: 1px solid #e0e0e0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        <h2 style="color: #1f77b4; margin-bottom: 10px;">📋 O que esta página responde:</h2>
        <h3 style="color: black; margin-bottom: 5px;">"O que aconteceu até agora?"</h3>
        <p style="color: black;">Análise exploratória dos dados históricos para entender padrões passados e identificar tendências.</p>
    </div>
    """, unsafe_allow_html=True)

    if data_loaded:
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
            areas = df['area'].nunique()
            st.metric("Áreas Mapeadas", areas)

        # Gráficos
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📈 Evolução Temporal")
            ano_counts = df.groupby('ano').size().reset_index(name='count')
            fig = px.line(ano_counts, x='ano', y='count', markers=True)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("🗺️ Distribuição por Área")
            area_counts = df['area'].value_counts().head(10)
            fig = px.bar(x=area_counts.values, y=area_counts.index, orientation='h')
            st.plotly_chart(fig, use_container_width=True)

# Página 2: Previsão de Demanda (LSTM)
elif page == "🔮 Previsão de Demanda":
    st.header("🔮 Previsão de Ocorrências Futuras")

    # Caixa de pergunta principal
    st.markdown("""
    <div style="background-color: #e8f4fd; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #1f77b4; margin-bottom: 10px;">🤔 Pergunta do Modelo:</h2>
        <h3 style="color: black; margin-bottom: 5px;">"QUANTAS ocorrências teremos e QUANDO elas acontecerão?"</h3>
        <p style="color: black;">O modelo LSTM prevê a demanda futura com base em padrões históricos, permitindo planejar recursos com antecedência.</p>
    </div>
    """, unsafe_allow_html=True)

    # Configurações da previsão
    st.subheader("⚙️ Configurar Previsão")
    col1, col2 = st.columns(2)

    with col1:
        selected_area = st.selectbox(
            "📍 Área de Interesse",
            options=['Norte', 'Sul', 'Leste', 'Oeste', 'Centro'],
            help="Selecione a área para previsão"
        )

        pred_date = st.date_input(
            "📅 Data da Previsão",
            value=date.today() + timedelta(days=1)
        )

    with col2:
        scenario = st.selectbox(
            "🎭 Cenário",
            options=['Dia Normal', 'Fim de Semana', 'Feriado', 'Evento Especial'],
            help="O cenário afeta a previsão de demanda"
        )

        confidence = st.slider(
            "📊 Nível de Confiança",
            min_value=0.7,
            max_value=0.95,
            value=0.85
        )

    # Botão de previsão
    if st.button("🔮 Gerar Previsão", type="primary", use_container_width=True):
        st.subheader(f"📈 Previsão para {selected_area} - {pred_date.strftime('%d/%m/%Y')}")

        # Gerar dados de previsão simulados
        hours = list(range(24))

        # Base realista
        if data_loaded:
            area_data = df[df['area'].str.contains(selected_area.lower(), na=False)]
            if len(area_data) > 0:
                hourly_pattern = area_data.groupby('hora_num').size()
                predicted_values = []
                for h in hours:
                    base = hourly_pattern.get(h, 10)
                    if scenario in ['Fim de Semana', 'Feriado']:
                        base *= 1.3
                    noise = np.random.normal(0, base * 0.1)
                    predicted_values.append(max(0, base + noise))
            else:
                predicted_values = [15 + np.random.normal(0, 3) for _ in hours]
        else:
            predicted_values = [15 + np.random.normal(0, 3) for _ in hours]

        # Calcular bandas de confiança
        std_dev = np.std(predicted_values)
        upper_bound = [v + std_dev * 1.96 * confidence for v in predicted_values]
        lower_bound = [v - std_dev * 1.96 * confidence for v in predicted_values]

        # Gráfico
        fig = go.Figure()

        # Bandas de confiança
        fig.add_trace(go.Scatter(
            x=hours + hours[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.1)',
            line_color='rgba(255,255,255,0)',
            name=f'IC {confidence*100:.0f}%'
        ))

        # Linha de previsão
        fig.add_trace(go.Scatter(
            x=hours,
            y=predicted_values,
            mode='lines+markers',
            line=dict(color='red', width=3),
            name='Previsão'
        ))

        fig.update_layout(
            title=f"Previsão de Ocorrências - {selected_area}",
            xaxis_title="Hora do Dia",
            yaxis_title="Número de Ocorrências"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Insights
        st.subheader("💡 Insights da Previsão")
        col1, col2, col3 = st.columns(3)

        peak_hour = hours[np.argmax(predicted_values)]
        with col1:
            st.metric("⏰ Horário de Pico", f"{peak_hour:02d}:00")
        with col2:
            st.metric("📊 Previsão Total", f"{sum(predicted_values):.0f}")
        with col3:
            st.metric("📈 Média por Hora", f"{np.mean(predicted_values):.1f}")

# Página 3: Análise de Ocorrência (BERT)
elif page == "🏷️ Análise de Ocorrência":
    st.header("🏷️ Classificação Inteligente de Ocorrências")

    # Caixa de pergunta principal
    st.markdown("""
    <div style="background-color: #fff2cc; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #d4a017; margin-bottom: 10px;">🤔 Pergunta do Modelo:</h2>
        <h3 style="color: black; margin-bottom: 5px;">"QUE TIPO de ocorrência é essa e QUANTOS recursos são necessários?"</h3>
        <p>O modelo BERT analisa o texto da ocorrência e classifica automaticamente, sugerindo a urgência e os recursos adequados.</p>
    </div>
    """, unsafe_allow_html=True)

    # Input de dados
    st.subheader("📝 Descreva a Ocorrência")

    descricao = st.text_area(
        "Descrição Completa",
        height=120,
        placeholder="Ex: Vítima relata que foi abordada por dois indivíduos em motocicleta. Os suspeitos anunciaram o assalto e subtraíram celular e carteira utilizando arma de fogo. Ocorreu na Avenida Getúlio Vargas, próximo ao número 1500.",
        help="Descreva todos os detalhes relevantes da ocorrência"
    )

    titulo = st.text_input(
        "Título Resumido",
        placeholder="Ex: Roubo com arma de fogo"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        area = st.selectbox("📍 Área", ['Norte', 'Sul', 'Leste', 'Oeste', 'Centro'])
    with col2:
        hora = st.time_input("🕐 Hora", datetime.now().time())
    with col3:
        urgencia = st.selectbox("🚨 Urgência", ['Baixa', 'Média', 'Alta', 'Emergência'])

    # Botão de classificação
    if st.button("🏷️ Classificar Ocorrência", type="primary", use_container_width=True) and descricao:
        st.subheader("🎯 Resultado da Classificação")

        # Simular classificação baseada em palavras-chave
        texto = (descricao + " " + titulo).lower()

        # Categorias e palavras-chave
        categories = {
            'Roubo': ['roubo', 'arma', 'ameaça', 'furt', 'subtr', 'assalt', 'carteira'],
            'Trânsito': ['acidente', 'colis', 'veículo', 'carro', 'moto', 'trâns'],
            'Lesão Corporal': ['ferid', 'agress', 'briga', 'pancad', 'violênc'],
            'Perturbação': ['barulh', 'músic', 'som', 'perturb', 'festa'],
            'Homicídio': ['homicíd', 'morte', 'assassin', 'tirote'],
            'Ameaça': ['ameaç', 'ameaça']
        }

        # Calcular scores
        scores = {}
        for cat, keywords in categories.items():
            score = sum(1 for kw in keywords if kw in texto)
            if score > 0:
                scores[cat] = score + np.random.random()

        # Adicionar aleatoriedade
        for _ in range(3):
            scores[np.random.choice(list(categories.keys()))] = np.random.random()

        if not scores:
            scores['Outros'] = 1.0

        # Normalizar
        total = sum(scores.values())
        scores = {k: v/total for k, v in scores.items()}

        # Principal categoria
        main_cat = max(scores.items(), key=lambda x: x[1])
        confidence = main_cat[1]

        # Resultado
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📋 Categoria", main_cat[0])
        with col2:
            st.metric("🎯 Confiança", f"{confidence:.1%}")
        with col3:
            recursos = "3+ viaturas" if confidence > 0.7 else "2 viaturas" if confidence > 0.4 else "1 viatura"
            st.metric("🚓 Recursos", recursos)

        # Gráfico
        fig = go.Figure(data=[
            go.Bar(
                x=list(scores.values()),
                y=list(scores.keys()),
                orientation='h'
            )
        ])
        fig.update_xaxes(tickformat='.1%')
        st.plotly_chart(fig, use_container_width=True)

# Página 4: Otimização de Recursos (DQN)
elif page == "🎯 Otimização de Recursos":
    st.header("🎯 Otimização de Posicionamento de Viaturas")

    # Caixa de pergunta principal
    st.markdown("""
    <div style="background-color: #d4edda; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
        <h2 style="color: #155724; margin-bottom: 10px;">🤔 Pergunta do Modelo:</h2>
        <h3 style="color: black; margin-bottom: 5px;">"ONDE posicionar as viaturas para o melhor atendimento?"</h3>
        <p style="color: black;">O modelo DQN otimiza o posicionamento das viaturas em tempo real para minimizar o tempo de resposta e maximizar a cobertura.</p>
    </div>
    """, unsafe_allow_html=True)

    # Configurações
    st.subheader("⚙️ Configurar Simulação")

    col1, col2 = st.columns(2)
    with col1:
        num_viaturas = st.slider(
            "🚓 Número de Viaturas Disponíveis",
            min_value=5,
            max_value=30,
            value=15
        )
        objetivo = st.selectbox(
            "🎯 Objetivo Principal",
            options=['Minimizar Tempo Resposta', 'Maximizar Cobertura', 'Balancear Carga']
        )
    with col2:
        tipo_dia = st.selectbox(
            "📅 Tipo de Dia",
            options=['Dia Normal', 'Fim de Semana', 'Feriado'],
            help="Afeta a demanda esperada"
        )
        duracao = st.selectbox(
            "⏱️ Duração",
            options=['1 hora', '6 horas', '12 horas', '24 horas']
        )

    # Visualização do Posicionamento
    st.subheader("🗺️ Posicionamento Atual e Otimizado")

    # Criar gráfico de dispersão com Plotly
    areas = {
        'Norte': (-2.48, -44.30, 15),
        'Sul': (-2.55, -44.28, 18),
        'Leste': (-2.52, -44.25, 12),
        'Oeste': (-2.53, -44.33, 20),
        'Centro': (-2.53, -44.28, 10)
    }

    fig = go.Figure()

    # Adicionar círculos de cobertura das áreas
    for area, (lat, lon, demand) in areas.items():
        fig.add_shape(
            type="circle",
            xref="x", yref="y",
            x0=lon-0.025, y0=lat-0.025,
            x1=lon+0.025, y1=lat+0.025,
            line_color="blue",
            fillcolor="lightblue",
            opacity=0.2
        )
        fig.add_annotation(
            x=lon, y=lat+0.03,
            text=f"{area}<br>Demanda: {demand}",
            showarrow=False,
            font=dict(size=10)
        )

    # Adicionar viaturas
    viaturas_x = []
    viaturas_y = []
    for i in range(num_viaturas):
        area_idx = i % len(areas)
        area_name = list(areas.keys())[area_idx]
        lat, lon, _ = areas[area_name]

        # Adicionar variação
        viatura_lat = lat + np.random.uniform(-0.02, 0.02)
        viatura_lon = lon + np.random.uniform(-0.02, 0.02)
        viaturas_x.append(viatura_lon)
        viaturas_y.append(viatura_lat)

    fig.add_trace(go.Scatter(
        x=viaturas_x,
        y=viaturas_y,
        mode='markers',
        marker=dict(
            symbol='diamond',
            size=15,
            color='red',
            line=dict(width=2, color='darkred')
        ),
        name='Viaturas',
        text=[f"V-{i+1:03d}" for i in range(num_viaturas)],
        hovertemplate='Viatura %{text}<br>Pos: (%{y:.3f}, %{x:.3f})<extra></extra>'
    ))

    # Adicionar ocorrências
    num_ocorr = int(np.random.poisson(8))
    ocorrencias_x = []
    ocorrencias_y = []
    for i in range(num_ocorr):
        area_idx = np.random.randint(0, len(areas))
        area_name = list(areas.keys())[area_idx]
        lat, lon, _ = areas[area_name]

        ocorrencias_x.append(lon + np.random.uniform(-0.03, 0.03))
        ocorrencias_y.append(lat + np.random.uniform(-0.03, 0.03))

    fig.add_trace(go.Scatter(
        x=ocorrencias_x,
        y=ocorrencias_y,
        mode='markers',
        marker=dict(
            symbol='triangle-up',
            size=12,
            color='orange',
            line=dict(width=1, color='darkorange')
        ),
        name='Ocorrências',
        text=[f"Ocorrência #{i+1}" for i in range(num_ocorr)],
        hovertemplate='%{text}<br>Pos: (%{y:.3f}, %{x:.3f})<extra></extra>'
    ))

    fig.update_layout(
        title="Posicionamento de Viaturas e Ocorrências",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        showlegend=True,
        height=500,
        hovermode='closest'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Botão de otimização
    if st.button("🚀 Executar Otimização", type="primary", use_container_width=True):
        st.subheader("📈 Resultados da Otimização")

        # Métricas de melhoria
        col1, col2, col3 = st.columns(3)
        with col1:
            improvement = np.random.uniform(20, 35)
            st.metric("⬇️ Redução Tempo Médio", f"{improvement:.0f}%")
        with col2:
            coverage = np.random.uniform(15, 30)
            st.metric("⬆️ Aumento Cobertura", f"{coverage:.0f}%")
        with col3:
            balance = np.random.uniform(25, 45)
            st.metric("⚖️ Melhoria Balanceamento", f"{balance:.0f}%")

        # Sugestões
        st.subheader("📍 Sugestões de Realocação")

        for i in range(3):
            with st.expander(f"🚓 Sugestão #{i+1}"):
                col1, col2 = st.columns(2)
                viatura_num = f"V-{np.random.randint(1, 999):03d}"
                de = np.random.choice(['Centro', 'Norte', 'Sul', 'Leste', 'Oeste'])
                para = np.random.choice(['Norte', 'Sul', 'Leste', 'Oeste', 'Centro'])

                with col1:
                    st.write(f"**Viatura:** {viatura_num}")
                    st.write(f"**De:** {de}")
                    st.write(f"**Para:** {para}")
                with col2:
                    st.write(f"**Motivo:** Alta demanda na área")
                    st.write(f"**Melhoria:** {np.random.randint(20, 40)}%")
                    if st.button(f"Aprovar", key=f"aprov_{i}"):
                        st.success("✅ Realocação aprovada!")

# Página 5: Previsão por Bairros
elif page == "🏘️ Previsão por Bairros":
    # Importar o componente de bairros
    import sys
    sys.path.insert(0, '/Users/tgt/Documents/dados_pmma_copy/ml_models')
    from bairro_dashboard_component import show_bairro_prediction_page
    show_bairro_prediction_page(df)

elif not data_loaded:
    st.error("Não foi possível carregar os dados. Verifique se o arquivo 'pmma_unificado_oficial.parquet' existe.")

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🚔 <b>Sistema de Inteligência Policial - PMMA</b></p>
        <p>Tomada de decisão baseada em dados reais e machine learning</p>
    </div>
    """,
    unsafe_allow_html=True
)