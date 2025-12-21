"""
Projeto 2: Classificação de Ocorrências usando BERT
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import logging
from pathlib import Path

# Adicionar o path para importar módulos compartilhados
import sys
sys.path.append(str(Path(__file__).parents[2]))

from shared.preprocessing.data_preparation import PMMADataPreparator, TextDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PMMABERTClassifier(nn.Module):
    """Classificador BERT para ocorrências policiais"""

    def __init__(
        self,
        num_classes: int,
        model_name: str = 'neuralmind/bert-base-portuguese-cased',
        dropout: float = 0.3,
        hidden_size: int = 768
    ):
        super(PMMABERTClassifier, self).__init__()

        self.num_classes = num_classes
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)

        # Camadas adicionais
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_classes)
        )

        # Features adicionais (contexto)
        self.context_features = nn.Sequential(
            nn.Linear(4, 32),  # hora_num, dia_semana, area, turno
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Camada final combinando BERT e contexto
        self.final_classifier = nn.Linear(hidden_size // 2 + 16, num_classes)

    def forward(
        self,
        input_ids,
        attention_mask,
        hour_of_day=None,
        day_of_week=None,
        area_id=None
    ):
        # Saída do BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Usar o token [CLS] como representação
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)

        # Classificador base
        logits = self.classifier(pooled_output)

        # Adicionar contexto se disponível
        if hour_of_day is not None:
            # Normalizar features de contexto
            context = torch.stack([
                hour_of_day.float() / 24,
                day_of_week.float() / 6,
                area_id.float() / 5,
                torch.zeros_like(area_id.float())  # placeholder para turno
            ], dim=1)

            context_features = self.context_features(context)

            # Combinar com features do BERT
            bert_features = self.classifier[0](pooled_output)
            combined = torch.cat([bert_features, context_features], dim=1)

            # Classificação final
            logits = self.final_classifier(combined)

        return logits


class BERTTrainer:
    """Classe para treinar o classificador BERT"""

    def __init__(
        self,
        model: PMMABERTClassifier,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}

    def prepare_optimizer(
        self,
        train_loader: DataLoader,
        epochs: int,
        learning_rate: float = 2e-5
    ):
        """Prepara otimizador e scheduler"""
        total_steps = len(train_loader) * epochs

        # Parâmetros com e sem weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            eps=1e-8
        )

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Treina por uma época"""
        self.model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader: DataLoader) -> Tuple[float, float, np.ndarray]:
        """Valida o modelo"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                _, predictions = torch.max(outputs, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_predictions, average='weighted')

        return total_loss / len(dataloader), f1, np.array(all_labels), np.array(all_predictions)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        save_path: str = None
    ) -> Dict:
        """Treina o modelo completo"""
        best_f1 = 0
        best_epoch = 0

        self.prepare_optimizer(train_loader, epochs)

        logger.info(f"Iniciando treinamento por {epochs} épocas")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_f1, _, _ = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_f1'].append(val_f1)

            logger.info(
                f"Época {epoch+1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"Val F1: {val_f1:.3f}"
            )

            # Salvar melhor modelo
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_epoch = epoch
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_f1': val_f1
                    }, save_path)

            # Early stopping
            if epoch - best_epoch > 5:
                logger.info(f"Early stopping na época {epoch+1}")
                break

        return {
            'best_epoch': best_epoch,
            'best_f1': best_f1,
            'history': self.history
        }

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Faz previsões"""
        self.model.eval()
        predictions = []
        probabilities = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)

                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        return np.array(predictions), np.array(probabilities)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        save_path: str = None
    ):
        """Plota matriz de confusão"""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title('Matriz de Confusão')
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

    def plot_training_history(self, save_path: str = None):
        """Plota histórico de treinamento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        ax1.plot(self.history['train_loss'], label='Treino')
        ax1.plot(self.history['val_loss'], label='Validação')
        ax1.set_title('Loss por Época')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # F1 Score
        ax2.plot(self.history['val_f1'], label='F1 Score', color='green')
        ax2.set_title('F1 Score por Época')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def create_classification_dataloaders(
    df: pd.DataFrame,
    tokenizer: BertTokenizer,
    metadata: Dict,
    batch_size: int = 16,
    max_length: int = 128,
    test_size: float = 0.2
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Cria DataLoaders para classificação"""

    # Dividir em treino, validação e teste (60/20/20)
    train_df, temp_df = train_test_split(
        df,
        test_size=0.4,
        stratify=df['label'],
        random_state=42
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['label'],
        random_state=42
    )

    # Criar datasets
    train_dataset = TextDataset(
        texts=train_df['texto_completo'].tolist(),
        labels=train_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    val_dataset = TextDataset(
        texts=val_df['texto_completo'].tolist(),
        labels=val_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    test_dataset = TextDataset(
        texts=test_df['texto_completo'].tolist(),
        labels=test_df['label'].tolist(),
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Criar dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader