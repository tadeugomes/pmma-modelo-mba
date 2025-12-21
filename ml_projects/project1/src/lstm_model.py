"""
Projeto 1: Previsão de Ocorrências Policiais usando LSTM Bidirecional
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import logging
from pathlib import Path

# Adicionar o path para importar módulos compartilhados
import sys
sys.path.append(str(Path(__file__).parents[2]))

from shared.preprocessing.data_preparation import PMMADataPreparator, TimeSeriesDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMMALSTM(nn.Module):
    """Modelo LSTM para previsão de ocorrências policiais"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_areas: int = 5,
        forecast_horizon: int = 24
    ):
        super(PMMALSTM, self).__init__()

        self.hidden_size = hidden_size
        num_layers = num_layers
        self.num_areas = num_areas
        self.forecast_horizon = forecast_horizon

        # Embeddings para áreas
        self.area_embedding = nn.Embedding(num_areas, 16)

        # LSTM Bidirecional
        self.lstm = nn.LSTM(
            input_size=input_size + 16,  # +16 do embedding da área
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Mecanismo de atenção
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )

        # Camadas densas
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, forecast_horizon)

        # Ativação
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x, area_ids):
        batch_size, seq_len, _ = x.size()

        # Embedding da área
        area_emb = self.area_embedding(area_ids)
        area_emb = area_emb.unsqueeze(1).repeat(1, seq_len, 1)

        # Concatenar features com embedding
        x = torch.cat([x, area_emb], dim=-1)

        # Passar pela LSTM
        lstm_out, (hidden, cell) = self.lstm(x)

        # Aplicar atenção
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )

        # Global average pooling com atenção
        attended = torch.mean(attn_out, dim=1)

        # Passar pelas camadas densas
        out = self.relu(self.fc1(attended))
        out = self.dropout(out)
        out = self.fc2(out)

        # Aplicar tanh para manter valores positivos
        out = self.tanh(out) * 50  # Escala máxima de 50 ocorrências/hora

        return out


class PMMALSTMTrainer:
    """Classe para treinar o modelo LSTM"""

    def __init__(
        self,
        model: PMMALSTM,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        self.history = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Treina por uma época"""
        self.model.train()
        total_loss = 0

        for batch_idx, (sequences, targets, area_ids) in enumerate(dataloader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            area_ids = area_ids.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(sequences, area_ids)
            loss = self.criterion(outputs, targets)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> Tuple[float, Dict]:
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets, area_ids in dataloader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                area_ids = area_ids.to(self.device)

                outputs = self.model(sequences, area_ids)
                loss = self.criterion(outputs, targets)

                total_loss += loss.item()
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calcular métricas
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)

        mae = mean_absolute_error(all_targets, all_predictions)
        rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
        r2 = r2_score(all_targets, all_predictions)

        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }

        return total_loss / len(dataloader), metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        save_path: str = None
    ) -> Dict:
        """Treina o modelo completo"""
        best_val_loss = float('inf')
        best_epoch = 0

        logger.info(f"Iniciando treinamento por {epochs} épocas")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, metrics = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            self.scheduler.step(val_loss)

            logger.info(
                f"Época {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"MAE: {metrics['mae']:.2f}, "
                f"RMSE: {metrics['rmse']:.2f}, "
                f"R²: {metrics['r2']:.3f}"
            )

            # Salvar melhor modelo
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_loss': val_loss,
                        'metrics': metrics
                    }, save_path)

            # Early stopping
            if epoch - best_epoch > 20:
                logger.info(f"Early stopping na época {epoch+1}")
                break

        return {
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'history': self.history
        }

    def predict(
        self,
        dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Faz previsões"""
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for sequences, target, area_ids in dataloader:
                sequences = sequences.to(self.device)
                area_ids = area_ids.to(self.device)

                outputs = self.model(sequences, area_ids)

                predictions.extend(outputs.cpu().numpy())
                targets.extend(target.numpy())

        return np.array(predictions), np.array(targets)

    def plot_training_history(self, save_path: str = None):
        """Plota histórico de treinamento"""
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Treino')
        plt.plot(self.history['val_loss'], label='Validação')
        plt.title('Loss por Época')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(
            np.array(self.history['val_loss']) - np.array(self.history['train_loss'])
        )
        plt.title('Overfitting (Val - Train)')
        plt.xlabel('Época')
        plt.ylabel('Diferença de Loss')
        plt.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_time_series_dataloaders(
    df: pd.DataFrame,
    metadata: Dict,
    batch_size: int = 32,
    sequence_length: int = 24,
    forecast_horizon: int = 24
) -> Tuple[DataLoader, DataLoader]:
    """Cria DataLoaders para treinamento"""

    # Preparar dados
    areas = metadata['areas']
    features = metadata['features']

    # Mapear áreas para IDs
    area_to_id = {area: i for i, area in enumerate(areas)}

    sequences = []
    targets = []
    area_ids = []

    for area in areas:
        area_data = df[df['area'] == area].sort_values('timestamp')
        area_features = area_data[features].values
        area_target = area_data['num_ocorrencias'].values

        # Criar sequências
        for i in range(len(area_data) - sequence_length - forecast_horizon + 1):
            seq = area_features[i:i+sequence_length]
            target = area_target[i+sequence_length:i+sequence_length+forecast_horizon]

            sequences.append(seq)
            targets.append(target)
            area_ids.append(area_to_id[area])

    # Converter para arrays
    sequences = np.array(sequences)
    targets = np.array(targets)
    area_ids = np.array(area_ids)

    # Normalizar targets
    target_mean = targets.mean()
    target_std = targets.std()
    targets = (targets - target_mean) / target_std

    # Dividir em treino/validação (80/20)
    split_idx = int(len(sequences) * 0.8)

    train_sequences = torch.FloatTensor(sequences[:split_idx])
    train_targets = torch.FloatTensor(targets[:split_idx])
    train_area_ids = torch.LongTensor(area_ids[:split_idx])

    val_sequences = torch.FloatTensor(sequences[split_idx:])
    val_targets = torch.FloatTensor(targets[split_idx:])
    val_area_ids = torch.LongTensor(area_ids[split_idx:])

    # Criar datasets
    train_dataset = torch.utils.data.TensorDataset(
        train_sequences, train_targets, train_area_ids
    )
    val_dataset = torch.utils.data.TensorDataset(
        val_sequences, val_targets, val_area_ids
    )

    # Criar dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, (target_mean, target_std)