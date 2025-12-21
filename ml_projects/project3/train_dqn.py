"""
Script de treinamento para o Projeto 3 - DQN
"""
import argparse
import sys
from pathlib import Path

# Adicionar o path para importar módulos
sys.path.append(str(Path(__file__).parents[1]))

from src.dqn_agent import DQNAgent, PoliceEnvironment
from shared.preprocessing.data_preparation import PMMADataPreparator

import torch
import pandas as pd
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Treinar agente DQN para alocação de recursos')
    parser.add_argument('--data_path', type=str,
                       default='../output/pmma_unificado_oficial.parquet',
                       help='Caminho para os dados')
    parser.add_argument('--num_vehicles', type=int, default=10,
                       help='Número de viaturas policiais')
    parser.add_argument('--grid_size', type=int, nargs=2, default=[10, 10],
                       help='Tamanho da grade de simulação')
    parser.add_argument('--num_episodes', type=int, default=1000,
                       help='Número de episódios de treinamento')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Taxa de aprendizado')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Tamanho do batch')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Fator de desconto')
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

    # Amostrar dados para treinamento (dados recentes são mais relevantes)
    sample_df = df.sample(n=min(10000, len(df)), random_state=42)

    # Criar ambiente
    env = PoliceEnvironment(
        occurrence_data=sample_df,
        num_vehicles=args.num_vehicles,
        grid_size=tuple(args.grid_size),
        max_time_steps=1440,  # 24 horas
        response_time_weight=0.5,
        coverage_weight=0.3,
        workload_weight=0.2
    )

    # Calcular tamanhos de estado e ação
    state_size = len(env.reset())
    action_size = args.num_vehicles * args.grid_size[0] * args.grid_size[1]

    logger.info(f"Tamanho do estado: {state_size}")
    logger.info(f"Tamanho da ação: {action_size}")

    # Criar agente
    agent = DQNAgent(
        state_size=state_size,
        action_size=action_size,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=5000,
        target_update=100
    )

    # Treinar
    results = agent.train(
        env=env,
        num_episodes=args.num_episodes,
        max_steps_per_episode=1440,
        batch_size=args.batch_size,
        save_path=str(save_dir / 'dqn_model.pth')
    )

    # Salvar resultados
    with open(save_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Salvar configuração
    config = {
        'num_vehicles': args.num_vehicles,
        'grid_size': args.grid_size,
        'state_size': state_size,
        'action_size': action_size,
        'hyperparameters': {
            'lr': args.lr,
            'gamma': args.gamma,
            'batch_size': args.batch_size
        }
    }

    with open(save_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Avaliar modelo
    logger.info("Avaliando modelo treinado...")
    eval_results = agent.evaluate(env, num_episodes=10)

    logger.info(f"Resultados da avaliação:")
    logger.info(f"Recompensa média: {eval_results['mean_reward']:.2f}")
    logger.info(f"Tempo médio de resposta: {eval_results['mean_response_time']:.2f} min")
    logger.info(f"Taxa de cobertura média: {eval_results['mean_coverage_rate']:.2%}")

    with open(save_dir / 'evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)

    logger.info(f"Modelo treinado e salvo em {save_dir}")


if __name__ == '__main__':
    main()