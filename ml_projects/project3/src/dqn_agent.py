"""
Projeto 3: Otimização de Alocação de Recursos usando Deep Q-Network
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging
from pathlib import Path
import networkx as nx
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar o path para importar módulos compartilhados
import sys
sys.path.append(str(Path(__file__).parents[2]))

from shared.preprocessing.data_preparation import PMMADataPreparator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Transição para o replay buffer
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'done')
)


class DQN(nn.Module):
    """Deep Q-Network para alocação de recursos"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] = [256, 256, 128]
    ):
        super(DQN, self).__init__()

        # Camadas fully connected
        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size

        layers.append(nn.Linear(input_size, action_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Buffer para experiência replay"""

    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition: Transition):
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class PoliceEnvironment:
    """Ambiente de simulação para alocação de recursos policiais"""

    def __init__(
        self,
        occurrence_data: pd.DataFrame,
        num_vehicles: int = 10,
        grid_size: Tuple[int, int] = (10, 10),
        max_time_steps: int = 1440,  # 24 horas em minutos
        response_time_weight: float = 0.5,
        coverage_weight: float = 0.3,
        workload_weight: float = 0.2
    ):
        self.df = occurrence_data.copy()
        self.num_vehicles = num_vehicles
        self.grid_size = grid_size
        self.max_time_steps = max_time_steps

        # Pesos para a função de recompensa
        self.response_time_weight = response_time_weight
        self.coverage_weight = coverage_weight
        self.workload_weight = workload_weight

        # Inicializar posições das viaturas
        self.vehicle_positions = self._initialize_vehicle_positions()
        self.vehicle_workload = np.zeros(num_vehicles)  # Carga de trabalho atual
        self.vehicle_available = np.ones(num_vehicles, dtype=bool)

        # Ocorrências pendentes
        self.pending_occurrences = []
        self.resolved_occurrences = []

        # Métricas
        self.current_step = 0
        self.total_response_time = 0
        self.covered_areas = set()

        # Criar grafo de distâncias
        self.distance_matrix = self._create_distance_matrix()

    def _initialize_vehicle_positions(self) -> np.ndarray:
        """Inicializa posições das viaturas de forma distribuída"""
        positions = []
        for i in range(self.num_vehicles):
            # Distribuir viaturas pela grade
            row = i // self.grid_size[1]
            col = i % self.grid_size[1]
            positions.append([row, col])
        return np.array(positions)

    def _create_distance_matrix(self) -> np.ndarray:
        """Cria matriz de distâncias entre pontos da grade"""
        total_cells = self.grid_size[0] * self.grid_size[1]
        matrix = np.zeros((total_cells, total_cells))

        for i in range(total_cells):
            for j in range(total_cells):
                row1, col1 = i // self.grid_size[1], i % self.grid_size[1]
                row2, col2 = j // self.grid_size[1], j % self.grid_size[1]
                matrix[i, j] = np.sqrt((row1 - row2)**2 + (col1 - col2)**2)

        return matrix

    def reset(self) -> np.ndarray:
        """Reinicia o ambiente"""
        self.vehicle_positions = self._initialize_vehicle_positions()
        self.vehicle_workload = np.zeros(self.num_vehicles)
        self.vehicle_available = np.ones(self.num_vehicles, dtype=bool)
        self.pending_occurrences = []
        self.resolved_occurrences = []
        self.current_step = 0
        self.total_response_time = 0
        self.covered_areas = set()

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """Obtém o estado atual do ambiente"""
        state = []

        # Posições das viaturas (normalizadas)
        normalized_positions = self.vehicle_positions.flatten() / max(self.grid_size)
        state.extend(normalized_positions)

        # Disponibilidade das viaturas
        state.extend(self.vehicle_available.astype(float))

        # Carga de trabalho das viaturas (normalizada)
        normalized_workload = self.vehicle_workload / 10.0  # Máximo 10 ocorrências
        state.extend(normalized_workload)

        # Número de ocorrências pendentes
        state.append(len(self.pending_occurrences) / 20.0)  # Normalizado

        # Hora do dia (normalizada)
        hour_of_day = (self.current_step % (24 * 60)) / (24 * 60)
        state.append(hour_of_day)

        # Dia da semana (normalizado)
        day_of_week = ((self.current_step // (24 * 60)) % 7) / 6.0
        state.append(day_of_week)

        # Áreas cobertas (one-hot)
        area_coverage = np.zeros(self.grid_size[0] * self.grid_size[1])
        for vehicle_pos in self.vehicle_positions:
            cell_idx = vehicle_pos[0] * self.grid_size[1] + vehicle_pos[1]
            # Cobrir área vizinha
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    r, c = vehicle_pos[0] + dr, vehicle_pos[1] + dc
                    if 0 <= r < self.grid_size[0] and 0 <= c < self.grid_size[1]:
                        area_coverage[r * self.grid_size[1] + c] = 1
        state.extend(area_coverage)

        return np.array(state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """Executa uma ação no ambiente"""
        # Ação: qual viatura mover para qual posição
        vehicle_id = action // (self.grid_size[0] * self.grid_size[1])
        target_cell = action % (self.grid_size[0] * self.grid_size[1])

        # Calcular recompensa
        reward = self._calculate_reward(vehicle_id, target_cell)

        # Executar movimento se a viatura estiver disponível
        if self.vehicle_available[vehicle_id]:
            target_row = target_cell // self.grid_size[1]
            target_col = target_cell % self.grid_size[1]
            self.vehicle_positions[vehicle_id] = [target_row, target_col]

        # Atualizar ocorrências
        self._update_occurrences()

        # Próximo passo
        self.current_step += 1

        # Verificar se terminou
        done = self.current_step >= self.max_time_steps

        return self._get_state(), reward, done

    def _calculate_reward(self, vehicle_id: int, target_cell: int) -> float:
        """Calcula a recompensa para uma ação"""
        reward = 0

        # 1. Tempo de resposta
        if self.pending_occurrences:
            target_row = target_cell // self.grid_size[1]
            target_col = target_cell % self.grid_size[1]

            min_response_time = float('inf')
            for occ in self.pending_occurrences:
                dist = abs(target_row - occ['location'][0]) + \
                       abs(target_col - occ['location'][1])
                response_time = dist * 2  # 2 minutos por unidade de distância
                min_response_time = min(min_response_time, response_time)

            if min_response_time < float('inf'):
                reward -= self.response_time_weight * min_response_time

        # 2. Cobertura de áreas
        target_area = target_cell
        if target_area not in self.covered_areas:
            reward += self.coverage_weight * 10
            self.covered_areas.add(target_area)

        # 3. Balanceamento de carga de trabalho
        if self.vehicle_workload[vehicle_id] > 5:
            reward -= self.workload_weight * 5
        elif self.vehicle_workload[vehicle_id] < 2:
            reward += self.workload_weight * 2

        return reward

    def _update_occurrences(self):
        """Atualiza ocorrências (adiciona novas e resolve as atendidas)"""
        # Adicionar novas ocorrências baseado no padrão histórico
        if self.current_step % 30 == 0:  # A cada 30 minutos
            # Probabilidade baseada na hora do dia
            hour = (self.current_step % (24 * 60)) / 60
            prob = 0.3 + 0.4 * np.sin((hour - 6) * np.pi / 12)

            if random.random() < prob:
                # Gerar ocorrência aleatória
                location = [
                    random.randint(0, self.grid_size[0] - 1),
                    random.randint(0, self.grid_size[1] - 1)
                ]
                self.pending_occurrences.append({
                    'location': location,
                    'time': self.current_step,
                    'urgency': random.choice(['baixa', 'media', 'alta'])
                })

        # Verificar ocorrências resolvidas
        for occ in self.pending_occurrences[:]:
            for i, vehicle_pos in enumerate(self.vehicle_positions):
                if self.vehicle_available[i]:
                    dist = abs(vehicle_pos[0] - occ['location'][0]) + \
                           abs(vehicle_pos[1] - occ['location'][1])
                    if dist <= 1:  # Viatura próxima o suficiente
                        self.resolved_occurrences.append(occ)
                        self.pending_occurrences.remove(occ)
                        self.vehicle_workload[i] += 1
                        self.total_response_time += dist * 2
                        break


class DQNAgent:
    """Agente DQN para otimização"""

    def __init__(
        self,
        state_size: int,
        action_size: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 1000,
        target_update: int = 100
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update

        # Redes neural principal e alvo
        self.q_network = DQN(state_size, action_size)
        self.target_network = DQN(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Otimizador
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer()

        # Estatísticas
        self.steps_done = 0
        self.episode_rewards = []

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Seleciona uma ação usando epsilon-greedy"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, *args):
        """Armazena uma transição no buffer"""
        self.replay_buffer.push(Transition(*args))

    def train_step(self, batch_size: int = 64) -> float:
        """Realiza um passo de treinamento"""
        if len(self.replay_buffer) < batch_size:
            return 0

        # Amostrar transições
        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Converter para tensores
        state_batch = torch.FloatTensor(batch.state)
        action_batch = torch.LongTensor(batch.action)
        reward_batch = torch.FloatTensor(batch.reward)
        next_state_batch = torch.FloatTensor(batch.next_state)
        done_batch = torch.BoolTensor(batch.done)

        # Calcular Q-values
        current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))

        # Calcular Q-values alvo
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)

        # Calcular loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        # Otimizar
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Atualizar epsilon
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (self.epsilon - self.epsilon_end) / self.epsilon_decay
        )

        self.steps_done += 1

        # Atualizar rede alvo
        if self.steps_done % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def train(
        self,
        env: PoliceEnvironment,
        num_episodes: int = 1000,
        max_steps_per_episode: int = 1440,
        batch_size: int = 64,
        save_path: str = None
    ) -> Dict:
        """Treina o agente"""
        logger.info(f"Iniciando treinamento por {num_episodes} episódios")

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            for step in range(max_steps_per_episode):
                # Selecionar e executar ação
                action = self.select_action(state)
                next_state, reward, done = env.step(action)

                # Armazenar transição
                self.store_transition(state, action, reward, next_state, done)

                # Treinar
                loss = self.train_step(batch_size)

                state = next_state
                episode_reward += reward

                if done:
                    break

            # Salvar recompensa do episódio
            self.episode_rewards.append(episode_reward)

            # Log
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                logger.info(
                    f"Episódio {episode}/{num_episodes} - "
                    f"Recompensa média: {avg_reward:.2f} - "
                    f"Epsilon: {self.epsilon:.3f} - "
                    f"Loss: {loss:.4f}"
                )

            # Salvar modelo
            if save_path and episode % 500 == 0:
                torch.save({
                    'episode': episode,
                    'q_network_state_dict': self.q_network.state_dict(),
                    'target_network_state_dict': self.target_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'episode_rewards': self.episode_rewards,
                    'epsilon': self.epsilon
                }, save_path)

        logger.info("Treinamento concluído!")
        return {
            'episode_rewards': self.episode_rewards,
            'final_epsilon': self.epsilon
        }

    def plot_training_results(self, save_path: str = None):
        """Plota resultados do treinamento"""
        plt.figure(figsize=(12, 4))

        # Recompensas por episódio
        plt.subplot(1, 2, 1)
        plt.plot(self.episode_rewards)
        plt.title('Recompensa por Episódio')
        plt.xlabel('Episódio')
        plt.ylabel('Recompensa')
        plt.grid(True)

        # Média móvel
        plt.subplot(1, 2, 2)
        window = 100
        if len(self.episode_rewards) > window:
            moving_avg = pd.Series(self.episode_rewards).rolling(window).mean()
            plt.plot(moving_avg)
            plt.title(f'Média Móvel ({window} episódios)')
            plt.xlabel('Episódio')
            plt.ylabel('Recompensa Média')
            plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def evaluate(self, env: PoliceEnvironment, num_episodes: int = 10) -> Dict:
        """Avalia o agente treinado"""
        self.q_network.eval()

        total_rewards = []
        response_times = []
        coverage_rates = []

        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_response_times = []

            for step in range(env.max_time_steps):
                # Selecionar ação greedy (sem exploração)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    q_values = self.q_network(state_tensor)
                    action = q_values.argmax().item()

                next_state, reward, done = env.step(action)
                episode_reward += reward

                # Coletar métricas
                if env.pending_occurrences:
                    for occ in env.pending_occurrences:
                        for vehicle_pos in env.vehicle_positions:
                            dist = abs(vehicle_pos[0] - occ['location'][0]) + \
                                   abs(vehicle_pos[1] - occ['location'][1])
                            if dist <= 2:
                                episode_response_times.append(dist * 2)

                state = next_state
                if done:
                    break

            total_rewards.append(episode_reward)
            if episode_response_times:
                response_times.extend(episode_response_times)
            coverage_rates.append(len(env.covered_areas) / (env.grid_size[0] * env.grid_size[1]))

        return {
            'mean_reward': np.mean(total_rewards),
            'mean_response_time': np.mean(response_times) if response_times else 0,
            'mean_coverage_rate': np.mean(coverage_rates)
        }