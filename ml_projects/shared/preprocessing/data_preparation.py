"""
Módulo de preparação de dados para os projetos de ML da PMMA
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import holidays
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import geopandas as gpd
from typing import Dict, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PMMADataPreparator:
    """Classe principal para preparação dos dados da PMMA"""

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.label_encoders = {}
        self.scalers = {}
        self.brazil_holidays = holidays.BR()

    def load_data(self) -> pd.DataFrame:
        """Carrega os dados unificados"""
        logger.info(f"Carregando dados de {self.data_path}")
        self.df = pd.read_parquet(self.data_path)

        # Converter timestamp para datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        else:
            # Criar timestamp a partir de data e hora
            self.df['timestamp'] = pd.to_datetime(
                self.df['data'].astype(str) + ' ' + self.df['hora'].astype(str),
                errors='coerce'
            )

        # Ordenar por timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Dados carregados: {len(self.df)} registros")
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Realiza limpeza dos dados"""
        logger.info("Iniciando limpeza dos dados")

        # Remover duplicatas
        antes = len(self.df)
        self.df = self.df.drop_duplicates(subset=['id_ocorrencia'])
        logger.info(f"Removidos {antes - len(self.df)} duplicatas")

        # Tratar valores nulos críticos
        self.df = self.df.dropna(subset=['timestamp', 'municipio'])

        # Preencher valores nulos categóricos
        categorical_cols = ['bairro', 'area', 'descricao_tipo', 'descricao_subtipo']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('desconhecido')

        # Preencher horas nulas
        self.df['hora_num'] = pd.to_numeric(self.df['hora_num'], errors='coerce').fillna(0)

        return self.df

    def feature_engineering(self) -> pd.DataFrame:
        """Cria features para os modelos"""
        logger.info("Criando features")

        # Features temporais
        self.df['dia_semana_num'] = self.df['timestamp'].dt.dayofweek
        self.df['semana_ano'] = self.df['timestamp'].dt.isocalendar().week
        self.df['trimestre'] = self.df['timestamp'].dt.quarter
        self.df['fim_de_semana'] = (self.df['dia_semana_num'] >= 5).astype(int)

        # Feriados
        self.df['e_feriado'] = self.df['data'].apply(
            lambda x: x in self.brazil_holidays if pd.notnull(x) else False
        ).astype(int)

        # Períodos do dia
        self.df['periodo_dia'] = pd.cut(
            self.df['hora_num'],
            bins=[0, 6, 12, 18, 24],
            labels=['madrugada', 'manha', 'tarde', 'noite'],
            include_lowest=True
        )

        # Features espaciais
        self.df['area_numerica'] = self.df['area'].map({
            'norte': 1, 'sul': 2, 'leste': 3, 'oeste': 4, 'f': 5
        }).fillna(0)

        # Codificar categorias importantes
        self._encode_categories()

        # Features agregadas por região
        self._create_regional_features()

        return self.df

    def _encode_categories(self):
        """Codifica variáveis categóricas"""
        categorical_to_encode = [
            'area', 'periodo_dia', 'grupo', 'cpam',
            'descricao_tipo', 'descricao_subtipo'
        ]

        for col in categorical_to_encode:
            if col in self.df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                # Tratar valores novos
                self.df[col] = self.df[col].astype(str)
                self.df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                    self.df[col]
                )

    def _create_regional_features(self):
        """Cria features agregadas por região"""
        # Contagem de ocorrências por área nas últimas 24h
        self.df = self.df.sort_values('timestamp')

        for area in self.df['area'].unique():
            if pd.notna(area):
                mask = self.df['area'] == area
                self.df.loc[mask, 'ocorrencias_24h_area'] = (
                    self.df[mask]
                    .rolling('24h', on='timestamp')['id_ocorrencia']
                    .count()
                )

        # Preencher NaN com 0
        self.df['ocorrencias_24h_area'] = self.df['ocorrencias_24h_area'].fillna(0)

    def prepare_for_time_series(self, target_col: str = 'id_ocorrencia') -> Tuple[pd.DataFrame, Dict]:
        """Prepara dados para séries temporais (Projeto 1)"""
        logger.info("Preparando dados para séries temporais")

        # Agregar por hora e área
        ts_df = self.df.groupby([
            pd.Grouper(key='timestamp', freq='H'),
            'area'
        ]).agg({
            target_col: 'count',
            'descricao_tipo': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'outro',
            'fim_de_semana': 'first',
            'e_feriado': 'first',
            'periodo_dia': 'first',
        }).reset_index()

        ts_df = ts_df.rename(columns={target_col: 'num_ocorrencias'})

        # Criar features de lag
        for lag in [1, 24, 168]:  # 1 hora, 1 dia, 1 semana
            ts_df[f'lag_{lag}h'] = ts_df.groupby('area')['num_ocorrencias'].shift(lag)

        # Médias móveis
        for window in [6, 24, 168]:
            ts_df[f'ma_{window}h'] = (
                ts_df.groupby('area')['num_ocorrencia            ]
            .rolling(window=window)
            .mean()
            .reset_index(level=0, drop=True)
        )

        # Remover linhas com NaN
        ts_df = ts_df.dropna()

        # Normalizar features numéricas
        numeric_cols = [col for col in ts_df.columns
                       if ts_df[col].dtype in ['int64', 'float64']
                       and col != 'num_ocorrencias']

        if numeric_cols:
            scaler = StandardScaler()
            ts_df[numeric_cols] = scaler.fit_transform(ts_df[numeric_cols])
            self.scalers['time_series'] = scaler

        metadata = {
            'sequence_length': 24,  # 24 horas de histórico
            'features': [col for col in ts_df.columns
                        if col not in ['timestamp', 'area', 'num_ocorrencias']],
            'target': 'num_ocorrencias',
            'areas': ts_df['area'].unique().tolist()
        }

        return ts_df, metadata

    def prepare_for_classification(self) -> Tuple[pd.DataFrame, Dict]:
        """Prepara dados para classificação (Projeto 2)"""
        logger.info("Preparando dados para classificação")

        # Filtrar ocorrências com descrição
        class_df = self.df[
            self.df['descricao_tipo'].notna() &
            (self.df['descricao_tipo'] != 'desconhecido')
        ].copy()

        # Combinar texto das colunas
        class_df['texto_completo'] = (
            class_df['titulo'].fillna('') + ' ' +
            class_df['descricao_tipo'].fillna('') + ' ' +
            class_df['descricao_subtipo'].fillna('') + ' ' +
            class_df['logradouro'].fillna('')
        ).str.strip()

        # Mapear categorias principais
        top_categories = class_df['descricao_tipo'].value_counts().head(20).index
        class_df['categoria_principal'] = class_df['descricao_tipo'].apply(
            lambda x: x if x in top_categories else 'outros'
        )

        # Criar labels
        if 'categoria_principal' not in self.label_encoders:
            self.label_encoders['categoria_principal'] = LabelEncoder()
        class_df['label'] = self.label_encoders['categoria_principal'].fit_transform(
            class_df['categoria_principal']
        )

        # Remover textos muito curtos
        class_df = class_df[class_df['texto_completo'].str.len() > 10]

        metadata = {
            'num_classes': len(class_df['categoria_principal'].unique()),
            'classes': class_df['categoria_principal'].unique().tolist(),
            'label_encoder': self.label_encoders['categoria_principal']
        }

        return class_df, metadata

    def create_sequences(self, df: pd.DataFrame, sequence_length: int = 24) -> np.ndarray:
        """Cria sequências para LSTM"""
        sequences = []

        for area in df['area'].unique():
            area_data = df[df['area'] == area].sort_values('timestamp')

            for i in range(len(area_data) - sequence_length):
                seq = area_data.iloc[i:i+sequence_length]
                sequences.append(seq.drop(['timestamp', 'area'], axis=1).values)

        return np.array(sequences)

    def save_encoders(self, path: str):
        """Salva os encoders e scalers"""
        import joblib
        joblib.dump({
            'label_encoders': self.label_encoders,
            'scalers': self.scalers
        }, path)
        logger.info(f"Encoders salvos em {path}")

    def load_encoders(self, path: str):
        """Carrega os encoders e scalers"""
        import joblib
        data = joblib.load(path)
        self.label_encoders = data['label_encoders']
        self.scalers = data['scalers']
        logger.info(f"Encoders carregados de {path}")


class TimeSeriesDataset(Dataset):
    """Dataset PyTorch para séries temporais"""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class TextDataset(Dataset):
    """Dataset PyTorch para classificação de texto"""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }