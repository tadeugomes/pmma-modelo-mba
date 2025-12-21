"""
Script para treinar o modelo de previsão por bairros
"""

import pandas as pd
import numpy as np
from bairro_prediction_model import BairroPredictionModel
import torch
import matplotlib.pyplot as plt
import os


def main():
    print("🚔 Treinando Modelo de Previsão por Bairros - PMMA")
    print("=" * 50)

    # Configurações
    DATA_PATH = '/Users/tgt/Documents/dados_pmma_copy/output/pmma_unificado_oficial.parquet'
    MODEL_PATH = './bairro_model.pth'

    # Verificar se o arquivo existe
    if not os.path.exists(DATA_PATH):
        print(f"❌ Erro: Arquivo não encontrado em {DATA_PATH}")
        return

    # Criar instância do modelo
    model = BairroPredictionModel(
        hidden_size=128,
        num_layers=2,
        sequence_length=24  # 24 horas de histórico
    )

    try:
        # Treinar o modelo
        train_losses, test_losses = model.train(
            df_path=DATA_PATH,
            epochs=50,
            batch_size=32,
            learning_rate=0.001
        )

        # Salvar o modelo
        model.save_model(MODEL_PATH)

        # Opcional: Plotar curvas de aprendizado
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.title('Curvas de Aprendizado')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('bairro_model_learning_curves.png')
        plt.close()

        print("\n✅ Modelo treinado e salvo com sucesso!")

        # Exemplo de previsão
        print("\n🔮 Exemplo de Previsão:")
        df = pd.read_parquet(DATA_PATH)
        top_bairro = df['bairro'].value_counts().index[0]
        print(f"Bairro: {top_bairro}")

        # Obter dados recentes do bairro
        bairro_data = df[df['bairro'] == top_bairro].sort_values('data', ascending=False)
        recent_data = bairro_data.head(model.sequence_length)

        # Simular previsão (note: isso é apenas ilustrativo)
        print("Previsões para as próximas 24 horas (simulado):")
        for i in range(24):
            hora = (pd.Timestamp.now().hour + i) % 24
            base_pred = len(bairro_data) / (bairro_data['data'].max() - bairro_data['data'].min()).days
            print(f"  Hora {hora:02d}:00 - ~{base_pred:.1f} ocorrências")

    except Exception as e:
        print(f"\n❌ Erro durante o treinamento: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()