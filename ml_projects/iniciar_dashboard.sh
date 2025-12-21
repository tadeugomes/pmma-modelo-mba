#!/bin/bash

echo "🚔 Iniciando Sistema de Inteligência Policial - PMMA"
echo "=================================================="

# Verificar se estamos no diretório correto
if [ ! -f "dashboard/real_app.py" ]; then
    echo "❌ Erro: Navegue até o diretório ml_projects"
    exit 1
fi

# Verificar se os dados existem
if [ ! -f "../output/pmma_unificado_oficial.parquet" ]; then
    echo "❌ Erro: Arquivo de dados não encontrado"
    exit 1
fi

# Iniciar o dashboard
echo "✅ Iniciando dashboard com dados reais..."
echo "📊 Total de ocorrências: 2,262,405"
echo "📅 Período: 2014-2024"
echo ""
echo "Acessando em:"
echo "➡️  Local: http://localhost:8506"
echo "➡️  Rede: http://192.168.1.100:8506"
echo ""
echo "Pressione Ctrl+C para parar"
echo ""

cd dashboard
streamlit run real_app.py --server.port 8506 --server.headless false