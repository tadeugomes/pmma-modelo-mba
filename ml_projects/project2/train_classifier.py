"""
Script de treinamento para o Projeto 2 - Classificação BERT
"""
import argparse
import sys
from pathlib import Path

# Adicionar o path para importar módulos
sys.path.append(str(Path(__file__).parents[1]))

from src.bert_classifier import PMMABERTClassifier, BERTTrainer, create_classification_dataloaders
from shared.preprocessing.data_preparation import PMMADataPreparator

import torch
import json
import logging
from transformers import BertTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Treinar classificador BERT para ocorrências')
    parser.add_argument('--data_path', type=str,
                       default='../output/pmma_unificado_oficial.parquet',
                       help='Caminho para os dados')
    parser.add_argument('--model_name', type=str,
                       default='neuralmind/bert-base-portuguese-cased',
                       help='Modelo BERT pré-treinado')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Comprimento máximo das sequências')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Tamanho do batch')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Número de épocas')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Taxa de aprendizado')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Taxa de dropout')
    parser.add_argument('--save_dir', type=str, default='models',
                       help='Diretório para salvar o modelo')

    args = parser.parse_args()

    # Criar diretório de modelos
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Carregar tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    # Preparar dados
    logger.info("Preparando dados...")
    preparator = PMMADataPreparator(args.data_path)
    df = preparator.load_data()
    df = preparator.clean_data()
    df = preparator.feature_engineering()

    # Preparar para classificação
    class_df, metadata = preparator.prepare_for_classification()

    # Salvar metadata
    with open(save_dir / 'metadata.json', 'w') as f:
        metadata_dict = {
            'num_classes': metadata['num_classes'],
            'classes': metadata['classes']
        }
        json.dump(metadata_dict, f, indent=2)

    logger.info(f"Categorias encontradas: {metadata['classes']}")

    # Criar dataloaders
    train_loader, val_loader, test_loader = create_classification_dataloaders(
        class_df,
        tokenizer,
        metadata,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

    # Criar modelo
    model = PMMABERTClassifier(
        num_classes=metadata['num_classes'],
        model_name=args.model_name,
        dropout=args.dropout
    )

    logger.info(f"Modelo criado com {sum(p.numel() for p in model.parameters())} parâmetros")

    # Treinar
    trainer = BERTTrainer(model)
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        save_path=str(save_dir / 'best_model.pth')
    )

    # Salvar histórico
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Salvar tokenizer
    tokenizer.save_pretrained(save_dir / 'tokenizer')

    # Salvar configuração
    config = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'dropout': args.dropout,
        'num_classes': metadata['num_classes']
    }

    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Salvar encoders
    preparator.save_encoders(str(save_dir / 'encoders.pkl'))

    logger.info(f"Modelo treinado e salvo em {save_dir}")


if __name__ == '__main__':
    main()