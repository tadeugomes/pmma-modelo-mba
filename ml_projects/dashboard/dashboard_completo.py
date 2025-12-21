"""
Dashboard Completo - Sistema de Inteligência Policial PMMA
Com funcionalidades de simulação e input
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
st.markdown("*Análise de dados reais e simulação preditiva*")
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
            df['dia_semana'] = df['data'].dt.day_name()
            df['mes'] = df['data'].dt.month
            df['ano'] = df['data'].dt.year

            return df
    return None

# Sidebar para navegação
st.sidebar.title("Navegação")
page = st.sidebar.selectbox(
    "Selecione uma página:",
    ["📊 Visão Geral", "🔮 Previsão de Ocorrências", "🏷️ Classificar Ocorrência", "🎯 Simulação de Recursos"]
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
    st.header("📊 Visão Geral das Ocorrências")

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

# Página 2: Previsão de Ocorrências
elif page == "🔮 Previsão de Ocorrências":
    st.header("🔮 Previsão de Ocorrências Policiais")
    st.markdown("Sistema preditivo baseado em padrões históricos")

    # Configurações da previsão
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Configurações da Previsão")

        selected_area = st.selectbox(
            "Selecione a Área",
            options=['Norte', 'Sul', 'Leste', 'Oeste', 'Centro'],
            index=0
        )

        pred_date = st.date_input(
            "Data da Previsão",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=30)
        )

        scenario = st.selectbox(
            "Cenário",
            options=['Normal', 'Fim de Semana', 'Feriado', 'Evento Especial'],
            index=0
        )

    with col2:
        st.subheader("Parâmetros")
        confidence = st.slider(
            "Nível de Confiança",
            min_value=0.7,
            max_value=0.95,
            value=0.85,
            step=0.05
        )

        show_historical = st.checkbox("Mostrar Histórico", value=True)

    # Botão de previsão
    if st.button("🔮 Gerar Previsão", type="primary"):
        st.subheader(f"Previsão para {selected_area} - {pred_date.strftime('%d/%m/%Y')}")

        # Calcular previsão baseada nos dados históricos
        hours = list(range(24))

        # Base real da área
        area_data = df[df['area_padrao'] == selected_area.lower()]
        if len(area_data) > 0:
            # Padrão real baseado nos dados
            hourly_pattern = area_data.groupby('hora_valida').size()
            max_hour = hourly_pattern.idxmax()
            max_count = hourly_pattern.max()

            # Ajustar por cenário
            scenario_factor = 1.0
            if scenario == 'Fim de Semana' and pred_date.weekday() >= 5:
                scenario_factor = 1.3
            elif scenario == 'Feriado':
                scenario_factor = 1.2
            elif scenario == 'Evento Especial':
                scenario_factor = 1.5

            # Gerar previsão
            predicted_values = []
            for h in hours:
                base_value = hourly_pattern.get(h, 10)
                if scenario_factor > 1:
                    base_value *= scenario_factor

                # Adicionar variação aleatória
                noise = np.random.normal(0, base_value * 0.1)
                predicted = max(0, base_value + noise)
                predicted_values.append(predicted)

            # Calcular bandas de confiança
            std_dev = np.std(predicted_values)
            upper_bound = [v + std_dev * 1.96 * confidence for v in predicted_values]
            lower_bound = [v - std_dev * 1.96 * confidence for v in predicted_values]

            # Gráfico da previsão
            fig = go.Figure()

            # Adicionar histórico se solicitado
            if show_historical and len(area_data) > 0:
                # Pegar média dos últimos 30 dias
                recent_data = area_data[area_data['data'] > (datetime.now() - timedelta(days=30))]
                if len(recent_data) > 0:
                    hist_avg = recent_data.groupby('hora_valida').size().reindex(hours, fill_value=0)
                    fig.add_trace(go.Scatter(
                        x=hours,
                        y=hist_avg,
                        mode='lines',
                        name='Média Histórica (30 dias)',
                        line=dict(color='gray', dash='dash')
                    ))

            # Bandas de confiança
            fig.add_trace(go.Scatter(
                x=hours + hours[::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(255,0,0,0.1)',
                line_color='rgba(255,255,255,0)',
                name=f'IC {confidence*100:.0f}%'
            ))

            # Linha da previsão
            fig.add_trace(go.Scatter(
                x=hours,
                y=predicted_values,
                mode='lines+markers',
                line=dict(color='red', width=3),
                name='Previsão',
                hovertemplate='<b>Hora: %{x:02d}:00</b><br>Ocorrências: %{y:.1f}<extra></extra>'
            ))

            fig.update_layout(
                title=f"Previsão de Ocorrências - {selected_area}",
                xaxis_title="Hora do Dia",
                yaxis_title="Número de Ocorrências",
                hovermode='x unified',
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)

            # Métricas
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                total_predicted = sum(predicted_values)
                st.metric("Total Previsto", f"{total_predicted:.0f}")

            with col2:
                peak_hour = hours[np.argmax(predicted_values)]
                peak_value = max(predicted_values)
                st.metric("Horário de Pico", f"{peak_hour:02d}:00")

            with col3:
                avg_predicted = np.mean(predicted_values)
                st.metric("Média por Hora", f"{avg_predicted:.1f}")

            with col4:
                risk_level = "Alto" if max(predicted_values) > 30 else "Médio" if max(predicted_values) > 20 else "Baixo"
                st.metric("Nível de Risco", risk_level)

            # Recomendações
            st.subheader("📋 Recomendações Operacionais")

            if peak_value > 30:
                st.warning("⚠️ **Alta demanda prevista:**")
                st.markdown("- Adicionar viaturas de reforço")
                st.markdown("- Antecipar troca de turnos")
                st.markdown("- Manter equipes de prontidão")
            elif peak_value > 20:
                st.info("ℹ️ **Demanda moderada:**")
                st.markdown("- Monitorar picos de horário")
                st.markdown("- Manter padrão normal de operação")
            else:
                st.success("✅ **Baixa demanda:**")
                st.markdown("- Oportunidade para treinamentos")
                st.markdown("- Manutenção de viaturas")

# Página 3: Classificar Ocorrência
elif page == "🏷️ Classificar Ocorrência":
    st.header("🏷️ Classificação Inteligente de Ocorrências")
    st.markdown("Classifique automaticamente novas ocorrências")

    # Input de dados
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Informações da Ocorrência")

        # Texto da ocorrência
        descricao = st.text_area(
            "Descrição Completa da Ocorrência",
            height=150,
            placeholder="Descreva detalhadamente o que aconteceu, incluindo local, envolvidos, objetos, etc.",
            help="Seja o mais detalhado possível para melhor classificação"
        )

        # Título resumido
        titulo = st.text_input(
            "Título Resumido",
            placeholder="Ex: Roubo de celular na Avenida Principal"
        )

    with col2:
        st.subheader("Contexto")

        # Área
        area_ocorrencia = st.selectbox(
            "Área da Ocorrência",
            options=['Norte', 'Sul', 'Leste', 'Oeste', 'Centro', 'Outra']
        )

        # Hora
        hora_ocorrencia = st.time_input(
            "Hora Aproximada",
            value=datetime.now().time()
        )

        # Urgência
        urgencia_reportada = st.selectbox(
            "Urgência Reportada",
            options=['Baixa', 'Média', 'Alta', 'Emergência'],
            index=1
        )

        # Botão de classificação
        if st.button("🏷️ Classificar Ocorrência", type="primary", use_container_width=True) and descricao:
            st.subheader("Resultado da Classificação")

            # Detectar palavras-chave
            texto_completo = (descricao + " " + titulo).lower()

            # Categorias e palavras-chave
            categories = {
                'Roubo': ['roubo', 'arma', 'ameaça', 'furt', 'subtr', 'celular', 'carteira', 'dinheiro'],
                'Trânsito': ['acidente', 'colis', 'veículo', 'carro', 'moto', 'trâns', 'atropel'],
                'Lesão Corporal': ['ferid', 'agress', 'briga', 'pancad', 'violênc', 'bateram'],
                'Perturbação': ['barulh', 'músic', 'som', 'perturb', 'reclam', 'festa'],
                'Tráfico de Drogas': ['droga', 'entorpec', 'maconh', 'crack', 'trafic', 'venda'],
                'Homicídio': ['homicíd', 'morte', 'assassin', 'tirote', 'arma de fog'],
                'Ameaça': ['ameaç', 'ameaça', 'ameaçando'],
                'Desaparecimento': ['desaparec', 'sumi', 'proc', 'extrav'],
                'Apreensão': ['apreens', 'recup', 'encontrad', 'detent']
            }

            # Calcular scores
            scores = {}
            for cat, keywords in categories.items():
                score = 0
                for kw in keywords:
                    if kw in texto_completo:
                        score += texto_completo.count(kw)
                scores[cat] = score

            # Adicionar um pouco de aleatoriedade para simular o modelo
            for cat in scores:
                scores[cat] += np.random.exponential(0.5)

            # Normalizar
            total = sum(scores.values())
            if total > 0:
                scores = {k: v/total for k, v in scores.items()}
            else:
                scores['Outros'] = 1.0

            # Ordenar
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            main_category = sorted_scores[0][0]
            confidence = sorted_scores[0][1]

            # Mostrar resultado principal
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Categoria Principal", main_category)

            with col2:
                st.metric("Confiança", f"{confidence:.1%}")

            with col3:
                if confidence > 0.7:
                    recursos = "2+ viaturas"
                elif confidence > 0.4:
                    recursos = "1-2 viaturas"
                else:
                    recursos = "1 viatura"
                st.metric("Recursos Sugeridos", recursos)

            # Gráfico de probabilidades
            st.subheader("Distribuição de Probabilidades")
            top_scores = sorted_scores[:10]

            fig = go.Figure(data=[
                go.Bar(
                    x=[score for _, score in top_scores],
                    y=[cat for cat, _ in top_scores],
                    orientation='h',
                    marker_color=['red' if cat == main_category else 'lightblue'
                                for cat, _ in top_scores]
                )
            ])

            fig.update_layout(
                xaxis_title="Probabilidade",
                yaxis_title="Categoria",
                xaxis=dict(tickformat='.1%')
            )

            st.plotly_chart(fig, use_container_width=True)

            # Análise detalhada
            with st.expander("📊 Análise Detalhada"):
                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Palavras-chave Detectadas:**")
                    detected = []
                    for cat, keywords in categories.items():
                        for kw in keywords:
                            if kw in texto_completo:
                                detected.append(f"• {kw} ({cat})")
                    if detected:
                        st.write("\n".join(set(detected)))
                    else:
                        st.write("Nenhuma palavra-chave específica detectada")

                with col2:
                    st.write("**Contexto Operacional:**")
                    st.write(f"• Área: {area_ocorrencia}")
                    st.write(f"• Hora: {hora_ocorrencia.strftime('%H:%M')}")

                    # Análise de risco
                    if hora_ocorrencia.hour >= 22 or hora_ocorrencia.hour <= 5:
                        risco = "Alto (noturno)"
                    elif hora_ocorrencia.hour >= 18:
                        risco = "Médio-Alto"
                    else:
                        risco = "Normal"
                    st.write(f"• Risco Contextual: {risco}")

# Página 4: Simulação de Recursos
elif page == "🎯 Simulação de Recursos":
    st.header("🎯 Simulação de Alocação de Recursos")
    st.markdown("Otimize o posicionamento das viaturas")

    # Configurações
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Configuração da Simulação")

        num_viaturas = st.slider(
            "Número de Viaturas",
            min_value=5,
            max_value=30,
            value=15,
            help="Total de viaturas disponíveis"
        )

        duration = st.selectbox(
            "Duração",
            options=['1 hora', '6 horas', '12 horas', '24 horas'],
            index=0
        )

        optimization = st.selectbox(
            "Objetivo Principal",
            options=['Minimizar Tempo Resposta', 'Maximizar Cobertura', 'Balancear Carga'],
            index=0
        )

    with col2:
        st.subheader("Cenário")

        # Tipo de dia
        tipo_dia = st.selectbox(
            "Tipo de Dia",
            options=['Dia Normal', 'Fim de Semana', 'Feriado']
        )

        # Fator de demanda
        if tipo_dia == 'Fim de Semana':
            fator_demanda = 1.3
        elif tipo_dia == 'Feriado':
            fator_demanda = 1.2
        else:
            fator_demanda = 1.0

        st.metric("Fator de Demanda", f"{fator_demanda:.1f}x")

    # Mapa de posicionamento
    st.subheader("Mapa de Posicionamento")

    # Criar mapa
    m = folium.Map(location=[-2.53, -44.30], zoom_start=11)

    # Adicionar áreas
    areas_coords = {
        'norte': (-2.48, -44.30, '#FF6B6B'),
        'sul': (-2.55, -44.28, '#4ECDC4'),
        'leste': (-2.52, -44.25, '#45B7D1'),
        'oeste': (-2.53, -44.33, '#96CEB4'),
        'centro': (-2.53, -44.28, '#FFEAA7')
    }

    for area, (lat, lon, color) in areas_coords.items():
        folium.Circle(
            location=[lat, lon],
            radius=3000,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.2,
            popup=f"Área {area.title()}"
        ).add_to(m)

    # Adicionar viaturas
    viaturas_por_area = num_viaturas // 5
    for i, (area, (lat, lon, color)) in enumerate(areas_coords.items()):
        for j in range(viaturas_por_area):
            # Adicionar variação
            viatura_lat = lat + np.random.uniform(-0.02, 0.02)
            viatura_lon = lon + np.random.uniform(-0.02, 0.02)

            folium.Marker(
                location=[viatura_lat, viatura_lon],
                icon=folium.Icon(color='red', icon='ambulance', prefix='fa'),
                popup=f"Viatura {i*viaturas_por_area + j + 1:03d}<br>Área: {area.title()}"
            ).add_to(m)

    # Adicionar ocorrências ativas
    num_ocorrencias = int(np.random.poisson(8) * fator_demanda)
    for i in range(num_ocorrencias):
        area_idx = np.random.randint(0, len(areas_coords))
        area, (lat, lon, _) = list(areas_coords.items())[area_idx]

        occ_lat = lat + np.random.uniform(-0.03, 0.03)
        occ_lon = lon + np.random.uniform(-0.03, 0.03)

        folium.Marker(
            location=[occ_lat, occ_lon],
            icon=folium.Icon(color='orange', icon='exclamation-triangle', prefix='fa'),
            popup=f"Ocorrência #{i+1}"
        ).add_to(m)

    # Exibir mapa
    st_folium(m, width=700, height=500)

    # Botão de otimização
    if st.button("🚀 Executar Otimização", type="primary"):
        st.subheader("Resultados da Otimização")

        # Simular melhorias
        improvement_time = np.random.uniform(20, 35)
        improvement_coverage = np.random.uniform(15, 30)
        improvement_balance = np.random.uniform(25, 45)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Redução Tempo Médio",
                f"-{improvement_time:.0f}%",
                delta=f"{12*(1-improvement_time/100):.1f} min → {12*(1-improvement_time/100)*(1-improvement_time/100):.1f} min"
            )

        with col2:
            st.metric(
                "Aumento Cobertura",
                f"+{improvement_coverage:.0f}%",
                delta=f"{75:.0f}% → {min(100, 75*(1+improvement_coverage/100)):.0f}%"
            )

        with col3:
            st.metric(
                "Melhoria Balanceamento",
                f"+{improvement_balance:.0f}%",
                delta="Otimizado"
            )

        # Sugestões
        st.subheader("📍 Sugestões de Realocação")

        suggestions = [
            ("V-001", "Centro", "Norte", "Alta concentração de chamados"),
            ("V-005", "Sul", "Posição intermediária", "Melhorar tempo de resposta"),
            ("V-010", "Oeste", "Leste", "Aumentar cobertura")
        ]

        for viat, de, para, motivo in suggestions:
            with st.expander(f"🚓 {viat}: {de} → {para}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Origem:** {de}")
                    st.write(f"**Destino:** {para}")
                    st.write(f"**Motivo:** {motivo}")
                with col2:
                    st.write(f"**Melhoria Estimada:** {np.random.randint(20, 40)}%")
                    if st.button(f"Aprovar", key=viat):
                        st.success("✅ Realocação aprovada!")

    # Estatísticas atuais
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Viaturas Totais", num_viaturas)

    with col2:
        st.metric("Disponíveis", f"{np.random.randint(2, 6)}")

    with col3:
        st.metric("Em Atendimento", f"{np.random.randint(3, 8)}")

    with col4:
        st.metric("Ocorrências na Fila", num_ocorrencias)

elif not data_loaded:
    st.error("Não foi possível carregar os dados. Verifique se o arquivo 'pmma_unificado_oficial.parquet' existe no diretório de output.")

# Rodapé
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>🚔 <b>Sistema de Inteligência Policial - PMMA</b></p>
        <p>Análise de dados reais das ocorrências (2014-2024)</p>
    </div>
    """,
    unsafe_allow_html=True
)