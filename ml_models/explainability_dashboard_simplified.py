"""
Dashboard de Explicabilidade PMMA - Versão Simplificada
Funciona com dados reais e menor consumo de memória
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
from datetime import datetime

def check_pmma_data():
    """Verifica se os dados PMMA estão disponíveis"""
    data_paths = [
        '/Users/tgt/Documents/dados_pmma_copy/output/pmma_unificado_oficial.parquet',
        './pmma_unificado_oficial.parquet'
    ]

    for path in data_paths:
        if os.path.exists(path):
            return True, path

    return False, None

def load_pmma_data():
    """Carrega dados PMMA com amostragem para economizar memória"""
    data_available, data_path = check_pmma_data()

    if not data_available:
        return None, None

    try:
        # Carregar apenas uma amostra dos dados para economizar memória
        df = pd.read_parquet(data_path)

        # Amostrar 10% dos dados se for muito grande
        if len(df) > 200000:
            df = df.sample(n=200000, random_state=42)
            st.warning(f"Usando amostra de 200.000 registros para melhor performance")

        # Limpeza básica
        df = df.dropna(subset=['data', 'bairro'])
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df = df.dropna(subset=['data'])

        return df, data_path

    except Exception as e:
        st.error(f"Erro ao carregar dados: {str(e)}")
        return None, None

def show_attention_weights():
    """Versão simplificada de attention weights"""

    st.markdown("""
    ### 🧠 **Análise de Attention Weights**
    *Análise simplificada para melhor performance*
    """)

    df, _ = load_pmma_data()
    if df is None:
        st.stop()

    # Obter bairros
    bairros_reais = df['bairro'].value_counts().head(10).index.tolist()
    bairro_selecionado = st.selectbox("Selecione um bairro:", bairros_reais)

    # Mostrar informações básicas
    bairro_data = df[df['bairro'] == bairro_selecionado]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📊 Ocorrências", len(bairro_data))
    with col2:
        st.metric("📅 Período", f"{bairro_data['data'].dt.year.min()}-{bairro_data['data'].dt.year.max()}")
    with col3:
        st.metric("🏘️ Bairros Totais", df['bairro'].nunique())

    if st.button("🔍 Analisar Padrões Horários"):
        with st.spinner("Analisando..."):
            # Análise horária simples
            bairro_data_sorted = bairro_data.sort_values('data')
            hourly_data = bairro_data_sorted.groupby(
                bairro_data_sorted['data'].dt.hour
            ).size()

            # Garantir que temos todas as horas do dia
            all_hours = range(24)
            hourly_data = hourly_data.reindex(all_hours, fill_value=0)

            # Criar gráfico simples
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(all_hours),
                y=hourly_data.values,
                name='Ocorrências por Hora',
                marker_color='lightblue',
                hovertemplate='<b>Hora: %{x}h</b><br>Ocorrências: %{y}<extra></extra>'
            ))

            fig.update_layout(
                title=f'📊 Distribuição Horária - {bairro_selecionado}',
                xaxis_title='Hora do Dia',
                yaxis_title='Número de Ocorrências',
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

            # Identificar horas críticas
            peak_hours = hourly_data.nlargest(3)
            st.markdown("#### 🚨 Horas Críticas")
            for hour, count in peak_hours.items():
                st.markdown(f"• **{hour}h**: {count} ocorrências")

def show_feature_importance():
    """Versão simplificada de feature importance"""

    st.markdown("""
    ### 🎯 **Análise de Features Principais**
    *Análise estatística básica dos dados*
    """)

    df, _ = load_pmma_data()
    if df is None:
        st.stop()

    # Análise de features básicas
    st.markdown("#### 📊 **Estatísticas Descritivas**")

    # Contagens por categoria
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Top 5 Bairros:**")
        top_bairros = df['bairro'].value_counts().head()
        for bairro, count in top_bairros.items():
            st.write(f"• {bairro}: {count:,} ocorrências")

    with col2:
        st.markdown("**Top 5 Tipos:**")
        if 'tipo' in df.columns:
            top_tipos = df['tipo'].value_counts().head()
            for tipo, count in top_tipos.items():
                st.write(f"• {tipo}: {count:,} ocorrências")
        else:
            st.write("Coluna 'tipo' não encontrada")

    # Distribuição por hora
    st.markdown("#### ⏰ **Padrão por Hora do Dia**")
    df['hora'] = df['data'].dt.hour
    hourly_dist = df['hora'].value_counts().sort_index()

    fig = px.line(
        x=hourly_dist.index,
        y=hourly_dist.values,
        title='Distribuição de Ocorrências por Hora',
        labels={'x': 'Hora do Dia', 'y': 'Número de Ocorrências'}
    )
    st.plotly_chart(fig, use_container_width=True)

    # Métricas do dataset
    st.markdown("#### 📈 **Informações do Dataset**")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📊 Registros", f"{len(df):,}")
    with col2:
        st.metric("🏘️ Bairros", f"{df['bairro'].nunique():,}")
    with col3:
        st.metric("📅 Período", f"{df['data'].min().year}-{df['data'].max().year}")
    with col4:
        st.metric("📍 Média/Dia", f"{len(df)/365:.0f}")

def main():
    """Função principal simplificada"""

    st.set_page_config(
        page_title="Explicabilidade PMMA - Simplificado",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Verificar dados
    data_available, data_path = check_pmma_data()

    if not data_available:
        st.error("## 🚫 **Dados PMMA Não Encontrados**")
        st.error("Arquivo procurado: /Users/tgt/Documents/dados_pmma_copy/output/pmma_unificado_oficial.parquet")
        return

    st.title("🔍 **Dashboard de Explicabilidade PMMA**")
    st.markdown("*Versão simplificada para melhor performance*")

    st.success(f"✅ **Dados PMMA Carregados**: {data_path}")

    # Carregar dados para mostrar informações básicas
    df, _ = load_pmma_data()
    if df is not None:
        st.sidebar.markdown(f"""
        ### 📊 **Dataset**

        - **Registros**: {len(df):,}
        - **Bairros**: {df['bairro'].nunique():,}
        - **Período**: {df['data'].min().year}-{df['data'].max().year}
        """)

    # Navegação
    st.sidebar.title("📋 Análises")
    page = st.sidebar.selectbox(
        "Selecione uma análise:",
        [
            "🧠 Attention Weights",
            "🎯 Feature Importance"
        ]
    )

    # Informações
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **Versão Simplificada**

    • Menor consumo de memória
    • Processamento mais rápido
    • Dados em amostra se necessário
    • Foco nas análises principais
    """)

    # Renderizar página
    if page == "🧠 Attention Weights":
        show_attention_weights()
    elif page == "🎯 Feature Importance":
        show_feature_importance()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
        🔍 Dashboard PMMA - Versão Simplificada |
        Otimizado para performance e estabilidade
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()