"""
Modelo de Predição de Ocorrências por Bairros
Utilizando LSTM para previsão no nível granular dos bairros
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
from datetime import datetime, timedelta


class BairroDataset(Dataset):
    """Dataset personalizado para dados de ocorrências por bairro"""

    def __init__(self, sequences, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class BairroLSTM(nn.Module):
    """Modelo LSTM para previsão por bairros"""

    def __init__(self, input_size, hidden_size, num_layers, num_bairros):
        super(BairroLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_bairros = num_bairros

        # Embedding para bairros
        self.bairro_embedding = nn.Embedding(num_bairros, 50)

        # LSTM com input dinâmico (features temporais + embedding)
        self.lstm = nn.LSTM(input_size + 50, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)

        # Camada de atenção
        self.attention = nn.Linear(hidden_size, 1)

        # Camadas de saída
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size // 2, 1)

    def forward(self, x_temporal, bairro_ids):
        # Embedding do bairro
        bairro_emb = self.bairro_embedding(bairro_ids)
        bairro_emb = bairro_emb.unsqueeze(1).expand(-1, x_temporal.size(1), -1)

        # Concatenar features temporais com embedding
        x = torch.cat([x_temporal, bairro_emb], dim=2)

        # Passar pela LSTM
        lstm_out, _ = self.lstm(x)

        # Aplicar atenção
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # Camadas fully connected
        out = self.fc1(context_vector)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out.squeeze(), attention_weights.squeeze(-1)


class BairroPredictionModel:
    """Classe principal para predição por bairros"""

    def __init__(self, hidden_size=128, num_layers=2, sequence_length=24):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # Encoders
        self.bairro_encoder = LabelEncoder()
        self.scaler_temporal = MinMaxScaler()
        self.scaler_target = MinMaxScaler()

        # Dicionários para salvar mapeamentos
        self.bairro_mapping = None
        self.feature_columns = ['hora_num', 'dia_num', 'mes_num', 'turno']

        # Modelo
        self.model = None

    def prepare_data(self, df_path):
        """Prepara os dados para treinamento"""
        print("🏘️ Preparando dados por bairros...")

        # Carregar dados
        df = pd.read_parquet(df_path)

        # Limpeza básica
        df = df.dropna(subset=['data', 'bairro'])
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df = df.dropna(subset=['data'])

        # Filtrar bairros com dados suficientes
        bairro_counts = df['bairro'].value_counts()
        bairros_validos = bairro_counts[bairro_counts >= 100].index
        df = df[df['bairro'].isin(bairros_validos)]

        # Codificar bairros
        df['bairro_encoded'] = self.bairro_encoder.fit_transform(df['bairro'])
        self.num_bairros = len(self.bairro_encoder.classes_)

        # Preparar features temporais
        df['hora_num'] = pd.to_numeric(df['hora_num'], errors='coerce').fillna(0)
        df['dia_num'] = df['data'].dt.day
        df['mes_num'] = df['data'].dt.month
        df['turno'] = df['hora'].apply(lambda x: self._get_turno(x))

        # Agrupar por bairro e hora
        df['data_hora'] = df['data'].dt.floor('H')
        grouped = df.groupby(['bairro_encoded', 'data_hora']).agg({
            'id_ocorrencia': 'count',
            'hora_num': 'first',
            'dia_num': 'first',
            'mes_num': 'first',
            'turno': 'first'
        }).reset_index()

        grouped.columns = ['bairro_encoded', 'data_hora', 'ocorrencias',
                         'hora_num', 'dia_num', 'mes_num', 'turno']

        # Criar sequências temporais
        sequences = []
        targets = []
        bairro_ids = []

        for bairro_id in grouped['bairro_encoded'].unique():
            bairro_data = grouped[grouped['bairro_encoded'] == bairro_id].sort_values('data_hora')

            if len(bairro_data) > self.sequence_length + 1:
                features = bairro_data[self.feature_columns + ['ocorrencias']].values

                # Normalizar features
                features[:, :-1] = self.scaler_temporal.fit_transform(features[:, :-1])
                target_scaled = self.scaler_target.fit_transform(features[:, [-1]])

                # Criar sequências
                for i in range(len(features) - self.sequence_length):
                    sequences.append(features[i:i+self.sequence_length, :-1])
                    targets.append(target_scaled[i+self.sequence_length, 0])
                    bairro_ids.append(bairro_id)

        return np.array(sequences), np.array(targets), np.array(bairro_ids)

    def _get_turno(self, hora_str):
        """Determina o turno based na hora"""
        try:
            hora = int(str(hora_str)[:2])
            if 6 <= hora < 12:
                return 1  # Manhã
            elif 12 <= hora < 18:
                return 2  # Tarde
            elif 18 <= hora < 24:
                return 3  # Noite
            else:
                return 0  # Madrugada
        except:
            return 0

    def train(self, df_path, epochs=50, batch_size=32, learning_rate=0.001):
        """Treina o modelo"""
        print("\n🚀 Iniciando treinamento do modelo por bairros...")

        # Preparar dados
        sequences, targets, bairro_ids = self.prepare_data(df_path)

        # Dividir treino/teste
        split = int(0.8 * len(sequences))
        train_seq, test_seq = sequences[:split], sequences[split:]
        train_target, test_target = targets[:split], targets[split:]
        train_bairros, test_bairros = bairro_ids[:split], bairro_ids[split:]

        # Criar datasets
        train_dataset = BairroDataset(train_seq, train_target)
        test_dataset = BairroDataset(test_seq, test_target)

        # DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Inicializar modelo
        self.model = BairroLSTM(
            input_size=len(self.feature_columns),
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            num_bairros=self.num_bairros
        )

        # Treinamento
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        print(f"Total de bairros: {self.num_bairros}")
        print(f"Sequências de treinamento: {len(train_seq)}")
        print(f"Sequências de teste: {len(test_seq)}")

        train_losses = []
        test_losses = []

        for epoch in range(epochs):
            # Treino
            self.model.train()
            train_loss = 0
            for batch_seq, batch_target in train_loader:
                # Obter IDs dos bairros para este batch
                start_idx = len(train_loss) // batch_size
                end_idx = start_idx + len(batch_target)
                batch_bairros = torch.LongTensor(train_bairros[start_idx:end_idx])

                optimizer.zero_grad()
                outputs = self.model(batch_seq, batch_bairros)
                loss = criterion(outputs, batch_target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Teste
            self.model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch_seq, batch_target in test_loader:
                    start_idx = len(test_loss) // batch_size
                    end_idx = start_idx + len(batch_target)
                    batch_bairros = torch.LongTensor(test_bairros[start_idx:end_idx])

                    outputs = self.model(batch_seq, batch_bairros)
                    loss = criterion(outputs, batch_target)
                    test_loss += loss.item()

            train_losses.append(train_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], '
                      f'Train Loss: {train_losses[-1]:.4f}, '
                      f'Test Loss: {test_losses[-1]:.4f}')

        print("\n✅ Treinamento concluído!")

        # Avaliação final
        self.model.eval()
        predictions = []
        actuals = []

        with torch.no_grad():
            for batch_seq, batch_target in test_loader:
                start_idx = len(predictions) // batch_size
                end_idx = start_idx + len(batch_target)
                batch_bairros = torch.LongTensor(test_bairros[start_idx:end_idx])

                outputs = self.model(batch_seq, batch_bairros)
                predictions.extend(outputs.numpy())
                actuals.extend(batch_target.numpy())

        # Desnormalizar para métricas reais
        predictions = self.scaler_target.inverse_transform(np.array(predictions).reshape(-1, 1))
        actuals = self.scaler_target.inverse_transform(np.array(actuals).reshape(-1, 1))

        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)

        print(f'\n📊 Métricas de Avaliação:')
        print(f'  MAE: {mae:.2f} ocorrências')
        print(f'  RMSE: {rmse:.2f} ocorrências')
        print(f'  R²: {r2:.3f}')

        return train_losses, test_losses

    def predict(self, bairro_id, last_hours_data, steps=24):
        """Faz previsão para um bairro específico"""
        if self.model is None:
            raise ValueError("Modelo não treinado!")

        self.model.eval()

        # Preparar sequência
        sequence = last_hours_data[self.feature_columns].values
        sequence[:, :-1] = self.scaler_temporal.transform(sequence)

        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
        bairro_tensor = torch.LongTensor([bairro_id])

        predictions = []
        attention_weights = []

        with torch.no_grad():
            for _ in range(steps):
                pred, attn_weights = self.model(sequence_tensor, bairro_tensor)
                pred_unscaled = self.scaler_target.inverse_transform(pred.numpy().reshape(-1, 1))
                predictions.append(pred_unscaled[0, 0])
                attention_weights.append(attn_weights.squeeze().numpy())

                # Atualizar sequência (simplificado)
                sequence_tensor = torch.roll(sequence_tensor, -1, dims=1)

        return predictions, attention_weights

    def save_model(self, path):
        """Salva o modelo e os encoders"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'bairro_encoder': self.bairro_encoder,
            'scaler_temporal': self.scaler_temporal,
            'scaler_target': self.scaler_target,
            'num_bairros': self.num_bairros,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'sequence_length': self.sequence_length
        }, path)
        print(f"Modelo salvo em: {path}")

    def load_model(self, path):
        """Carrega o modelo e os encoders"""
        checkpoint = torch.load(path)

        self.model = BairroLSTM(
            input_size=len(self.feature_columns),
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            num_bairros=checkpoint['num_bairros']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])

        self.bairro_encoder = checkpoint['bairro_encoder']
        self.scaler_temporal = checkpoint['scaler_temporal']
        self.scaler_target = checkpoint['scaler_target']
        self.num_bairros = checkpoint['num_bairros']
        self.hidden_size = checkpoint['hidden_size']
        self.num_layers = checkpoint['num_layers']
        self.sequence_length = checkpoint['sequence_length']

        print(f"Modelo carregado de: {path}")


def get_bairro_hotspots(df, top_n=20):
    """Identifica os bairros com mais ocorrências (hotspots)"""
    bairro_counts = df['bairro'].value_counts().head(top_n)
    return bairro_counts


def get_bairro_predictions(model, df_path, bairro_name, hours_ahead=24):
    """Gera previsões para um bairro específico"""
    try:
        # Carregar dados
        df = pd.read_parquet(df_path)

        # Obter ID do bairro
        bairro_id = model.bairro_encoder.transform([bairro_name])[0]

        # Obter últimas horas do bairro
        bairro_data = df[df['bairro'] == bairro_name].sort_values('data', ascending=False)
        recent_data = bairro_data.head(model.sequence_length)

        # Fazer previsão
        predictions, attention_weights = model.predict(bairro_id, recent_data, hours_ahead)

        return predictions, attention_weights
    except Exception as e:
        print(f"Erro ao gerar previsão para {bairro_name}: {str(e)}")
        return None

    def explain_prediction(self, bairro_name, last_hours_data):
        """
        Gera explicação da previsão usando attention weights

        Returns:
        - dict: {
            'attention_weights': array de pesos por timestep,
            'important_hours': horas mais importantes,
            'importance_scores': scores de importância,
            'temporal_pattern': padrão temporal identificado
        }
        """
        try:
            if self.model is None:
                raise ValueError("Modelo não treinado!")

            # Obter ID do bairro
            bairro_id = self.bairro_encoder.transform([bairro_name])[0]

            # Preparar sequência
            sequence = last_hours_data[self.feature_columns].values
            sequence[:, :-1] = self.scaler_temporal.transform(sequence)

            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)
            bairro_tensor = torch.LongTensor([bairro_id])

            self.model.eval()

            with torch.no_grad():
                # Obter pesos de atenção
                pred, attention_weights = self.model(sequence_tensor, bairro_tensor)
                attention_weights = attention_weights.squeeze().numpy()

                # Identificar horas mais importantes (top 5)
                top_hours_idx = np.argsort(attention_weights)[-5:]
                important_hours = []
                importance_scores = []

                for idx in top_hours_idx:
                    hour = last_hours_data.iloc[idx]['hora_num'] if 'hora_num' in last_hours_data.columns else idx
                    importance = attention_weights[idx]

                    important_hours.append(int(hour))
                    importance_scores.append(float(importance))

                # Analisar padrão temporal
                temporal_pattern = self._analyze_temporal_pattern(attention_weights, important_hours)

                return {
                    'attention_weights': attention_weights.tolist(),
                    'important_hours': important_hours,
                    'importance_scores': importance_scores,
                    'temporal_pattern': temporal_pattern,
                    'prediction': float(pred.item()),
                    'bairro': bairro_name
                }

        except Exception as e:
            print(f"Erro na explicação: {str(e)}")
            return None

    def _analyze_temporal_pattern(self, attention_weights, important_hours):
        """Analisa o padrão temporal baseado nos pesos de atenção"""

        # Verificar padrões comuns
        pattern_explanations = []

        # Padrão de pico noturno
        if any(hour >= 22 or hour <= 4 for hour in important_hours):
            pattern_explanations.append("Pico de atividade noturna detectado")

        # Padrão de rush
        if any(7 <= hour <= 9 or 17 <= hour <= 19 for hour in important_hours):
            pattern_explanations.append("Padrão de horário de rush identificado")

        # Padrão de fim de semana
        if any(hour >= 12 and hour <= 14 for hour in important_hours):
            pattern_explanations.append("Período de almoço/madrugada")

        # Tendência crescente/decrescente
        if len(attention_weights) > 1:
            trend = np.polyfit(range(len(attention_weights)), attention_weights, 1)[0]
            if trend > 0.01:
                pattern_explanations.append("Tendência crescente de importância")
            elif trend < -0.01:
                pattern_explanations.append("Tendência decrescente de importância")
            else:
                pattern_explanations.append("Padrão estável de importância")

        return pattern_explanations if pattern_explanations else ["Padrão temporal variável"]