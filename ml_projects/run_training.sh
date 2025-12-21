#!/bin/bash

# Script para executar o treinamento de todos os modelos

echo "🚔 Iniciando treinamento dos modelos PMMA ML"
echo "=========================================="

# Verificar se os dados existem
if [ ! -f "../output/pmma_unificado_oficial.parquet" ]; then
    echo "❌ Erro: Arquivo de dados não encontrado em ../output/pmma_unificado_oficial.parquet"
    exit 1
fi

# Criar diretórios necessários
mkdir -p project1/models
mkdir -p project2/models
mkdir -p project3/models
mkdir -p logs

# Projeto 1 - LSTM
echo ""
echo "📍 Projeto 1: Treinando modelo LSTM para previsão de ocorrências..."
cd project1
python train_model.py \
    --data_path ../output/pmma_unificado_oficial.parquet \
    --hidden_size 128 \
    --num_layers 2 \
    --batch_size 32 \
    --epochs 100 \
    --save_dir models 2>&1 | tee ../logs/project1_training.log

if [ $? -eq 0 ]; then
    echo "✅ Projeto 1 concluído com sucesso!"
else
    echo "❌ Erro no treinamento do Projeto 1"
fi
cd ..

# Projeto 2 - BERT
echo ""
echo "📍 Projeto 2: Treinando classificador BERT..."
cd project2
python train_classifier.py \
    --data_path ../output/pmma_unificado_oficial.parquet \
    --max_length 128 \
    --batch_size 16 \
    --epochs 5 \
    --learning_rate 2e-5 \
    --save_dir models 2>&1 | tee ../logs/project2_training.log

if [ $? -eq 0 ]; then
    echo "✅ Projeto 2 concluído com sucesso!"
else
    echo "❌ Erro no treinamento do Projeto 2"
fi
cd ..

# Projeto 3 - DQN
echo ""
echo "📍 Projeto 3: Treinando agente DQN para otimização..."
cd project3
python train_dqn.py \
    --data_path ../output/pmma_unificado_oficial.parquet \
    --num_vehicles 10 \
    --grid_size 10 10 \
    --num_episodes 500 \
    --lr 1e-3 \
    --save_dir models 2>&1 | tee ../logs/project3_training.log

if [ $? -eq 0 ]; then
    echo "✅ Projeto 3 concluído com sucesso!"
else
    echo "❌ Erro no treinamento do Projeto 3"
fi
cd ..

echo ""
echo "=========================================="
echo "🎉 Treinamento concluído!"
echo ""
echo "Para executar o dashboard:"
echo "cd dashboard && streamlit run app.py"
echo ""
echo "Para verificar os logs:"
echo "ls -la logs/"
echo "=========================================="