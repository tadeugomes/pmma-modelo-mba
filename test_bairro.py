#!/usr/bin/env python3
"""Teste simples para verificar se o componente de bairros funciona"""

import sys
import os
import pandas as pd

# Adicionar o caminho do módulo
sys.path.insert(0, '/Users/tgt/Documents/dados_pmma_copy/ml_models')

# Testar importação
try:
    from bairro_dashboard_component import show_bairro_prediction_page
    print("✅ Importação bem-sucedida")
except Exception as e:
    print(f"❌ Erro na importação: {e}")
    sys.exit(1)

# Testar carregamento de dados
print("\nTestando carregamento de dados...")
data_path = '/Users/tgt/Documents/dados_pmma_copy/output/pmma_unificado_oficial.parquet'

if os.path.exists(data_path):
    print(f"✅ Arquivo existe: {data_path}")

    try:
        df = pd.read_parquet(data_path)
        print(f"✅ Arquivo lido com sucesso!")
        print(f"   Shape: {df.shape}")
        print(f"   Colunas: {len(df.columns)}")
        print(f"   Bairros únicos: {df['bairro'].nunique()}")
    except Exception as e:
        print(f"❌ Erro ao ler arquivo: {e}")
else:
    print(f"❌ Arquivo não existe: {data_path}")

print("\nTeste concluído!")