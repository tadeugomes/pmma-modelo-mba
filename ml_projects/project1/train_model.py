"""
Script de treinamento para o Projeto 1 - Previsão de Ocorrências
"""
import argparse
import sys
from pathlib import Path

# Adicionar o path para importar módulos
sys.path.append(str(Path(__file__).parents[1]))

from src.lstm_model import PMMALSTM, PMMALSTMTrainer, create_time_series_dataloaders
from shared.preprocessing.data_preparation import PMMADataPreparator

import torch
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Treinar modelo LSTM para previsão de ocorrências')
    parser.add_argument('--data_path', type=str,
                       default='../output/pmma_unificado_oficial.parquet',
                       help='Caminho para os dados')
    parser.add_argument('--hidden_size', type=int, default=128,
                       help='Tamanho da camada oculta LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Número de camadas LSTM')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Taxa de dropout')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Tamanho do batch')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Número de épocas')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Diretório para salvar o modelo')

    args = parser.parse_args()

    # Criar diretório de modelos
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Preparar dados
    logger.info("Preparando dados...")
    preparator = PMMADataPreparator(args.data_path)
    df = preparator.load_data()
    df = preparator.clean_data()
    df = preparator.feature_engineering()

    # Preparar para séries temporais
    ts_df, metadata = preparator.prepare_for_time_series()

    # Salvar metadata
    with open(save_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # Criar dataloaders
    train_loader, val_loader, (target_mean, target_std) = create_time_series_dataloaders(
        ts_df, metadata, batch_size=args.batch_size
    )

    # Salvar estatísticas de normalização
    with open(save_dir / 'target_stats.json', 'w') as f:
        json.dump({'mean': float(target_mean), 'std': float(target_std)}, f)

    # Criar modelo
    model = PMMALSTM(
        input_size=len(metadata['features']),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_areas=len(metadata['areas']),
        forecast_horizon=24
    )

    logger.info(f"Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")

    # Treinar
    trainer = PMMALSTMTrainer(model)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=str(save_dir / 'best_model.pth')
    )

    # Salvar histórico
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Salvar encoders
    preparator.save_encoders(str(save_dir / 'encoders.pkl'))

    logger.info(f"Modelo treinado e salvo em {save_dir}")


if __name__ == '__main__':
    main()