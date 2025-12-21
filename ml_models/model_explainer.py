"""
Módulo de Explicabilidade para Modelos de Machine Learning
Implementa SHAP, LIME e Feature Importance para os modelos do PMMA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, accuracy_score, f1_score
import shap
import warnings
warnings.filterwarnings('ignore')

class ModelExplainer:
    """Classe principal para explicar modelos de ML"""

    def __init__(self):
        self.traditional_models = {}
        self.explainers = {}
        self.feature_names = []
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def prepare_features(self, df, target_column='ocorrencias'):
        """
        Prepara features para modelos tradicionais

        Features criadas:
        - Temporais: hora, dia_semana, mes, fim_de_semana, feriado
        - Espaciais: area_encoded, bairro_encoded
        - Históricas: media_3h, tendencia, ocorrencias_anteriores
        """
        try:
            # Verificar se dataframe tem as colunas necessárias
            if 'data' not in df.columns:
                # Criar data simulada para teste
                df = df.copy()
                df['data'] = pd.date_range('2023-01-01', periods=len(df), freq='H')

            # Criar features temporais
            df['data'] = pd.to_datetime(df['data'])
            df['hora'] = df['data'].dt.hour
            df['dia_semana'] = df['data'].dt.dayofweek
            df['mes'] = df['data'].dt.month
            df['fim_de_semana'] = (df['dia_semana'] >= 5).astype(int)
            df['feriado'] = 0  # Simplificado

            # Features espaciais
            if 'area' not in self.label_encoders:
                self.label_encoders['area'] = LabelEncoder()
                df['area_encoded'] = self.label_encoders['area'].fit_transform(df['area'].fillna('desconhecido'))
            else:
                df['area_encoded'] = self.label_encoders['area'].transform(df['area'].fillna('desconhecido'))

            if 'bairro' not in self.label_encoders:
                self.label_encoders['bairro'] = LabelEncoder()
                df['bairro_encoded'] = self.label_encoders['bairro'].fit_transform(df['bairro'].fillna('desconhecido'))
            else:
                df['bairro_encoded'] = self.label_encoders['bairro'].transform(df['bairro'].fillna('desconhecido'))

            # Features temporais derivadas
            df['seno_hora'] = np.sin(2 * np.pi * df['hora'] / 24)
            df['cosseno_hora'] = np.cos(2 * np.pi * df['hora'] / 24)
            df['seno_mes'] = np.sin(2 * np.pi * df['mes'] / 12)
            df['cosseno_mes'] = np.cos(2 * np.pi * df['mes'] / 12)

            # Features históricas (requer dados ordenados)
            df = df.sort_values(['bairro', 'data'])
            df['ocorrencias_anteriores'] = df.groupby('bairro')['ocorrencias'].shift(1).fillna(0)
            df['media_3h'] = df.groupby('bairro')['ocorrencias'].rolling(3, min_periods=1).mean().reset_index(0, drop=True)
            df['tendencia'] = df.groupby('bairro')['ocorrencias'].rolling(3, min_periods=1).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0).reset_index(0, drop=True)

            # Selecionar features finais
            feature_columns = [
                'hora', 'dia_semana', 'mes', 'fim_de_semana', 'feriado',
                'area_encoded', 'bairro_encoded',
                'seno_hora', 'cosseno_hora', 'seno_mes', 'cosseno_mes',
                'ocorrencias_anteriores', 'media_3h', 'tendencia'
            ]

            # Remover linhas com NaN
            df_clean = df[feature_columns + [target_column]].dropna()

            self.feature_names = feature_columns
            X = df_clean[feature_columns]
            y = df_clean[target_column]

            # Normalizar features
            X_scaled = self.scaler.fit_transform(X)

            return X_scaled, y

        except Exception as e:
            print(f"Erro ao preparar features: {str(e)}")
            return None, None

    def train_traditional_models(self, X, y, task_type='regression'):
        """
        Treina modelos tradicionais para feature importance

        Parameters:
        - task_type: 'regression' ou 'classification'
        """

        try:
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if task_type == 'regression':
                # Modelos de regressão
                models = {
                    'RandomForest_Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
                    'Linear_Regression': LinearRegression()
                }
            else:
                # Modelos de classificação
                models = {
                    'RandomForest_Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
                    'Logistic_Regression': LogisticRegression(max_iter=1000, random_state=42)
                }

            # Treinar modelos
            results = {}
            for name, model in models.items():
                print(f"Treinando {name}...")

                # Treinar
                model.fit(X_train, y_train)

                # Avaliar
                if task_type == 'regression':
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    mse = np.mean((y_test - y_pred) ** 2)

                    results[name] = {
                        'model': model,
                        'r2_score': r2,
                        'mse': mse,
                        'predictions': y_pred,
                        'X_test': X_test,
                        'y_test': y_test
                    }
                    print(f"  R²: {r2:.3f}, MSE: {mse:.3f}")

                else:
                    y_pred = model.predict(X_test)
                    accuracy = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')

                    results[name] = {
                        'model': model,
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'predictions': y_pred,
                        'X_test': X_test,
                        'y_test': y_test
                    }
                    print(f"  Accuracy: {accuracy:.3f}, F1: {f1:.3f}")

            self.traditional_models = results
            return results

        except Exception as e:
            print(f"Erro ao treinar modelos: {str(e)}")
            return None

    def calculate_feature_importance(self):
        """Calcula feature importance para todos os modelos treinados"""

        importance_results = {}

        for model_name, model_data in self.traditional_models.items():
            model = model_data['model']

            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_)
            else:
                continue

            # Ordenar por importância
            indices = np.argsort(importances)[::-1]

            importance_results[model_name] = {
                'importances': importances,
                'indices': indices,
                'sorted_features': [self.feature_names[i] for i in indices],
                'sorted_importances': importances[indices]
            }

        return importance_results

    def create_shap_explainer(self, model_name='RandomForest_Regressor'):
        """Cria SHAP explainer para o modelo especificado"""

        try:
            if model_name not in self.traditional_models:
                raise ValueError(f"Modelo {model_name} não encontrado")

            model_data = self.traditional_models[model_name]
            model = model_data['model']
            X_test = model_data['X_test']

            # Criar explainer baseado no tipo de modelo
            if 'RandomForest' in model_name:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, X_test)

            # Calcular SHAP values
            shap_values = explainer.shap_values(X_test)

            self.explainers[model_name] = {
                'explainer': explainer,
                'shap_values': shap_values,
                'X_test': X_test
            }

            return explainer, shap_values

        except Exception as e:
            print(f"Erro ao criar SHAP explainer: {str(e)}")
            return None, None

    def generate_feature_importance_report(self):
        """Gera relatório completo de feature importance"""

        importance_data = self.calculate_feature_importance()

        report = {
            'summary': {},
            'detailed_analysis': {},
            'recommendations': []
        }

        for model_name, data in importance_data.items():
            features = data['sorted_features']
            importances = data['sorted_importances']

            # Top 5 features
            top_5_features = list(zip(features[:5], importances[:5]))

            report['detailed_analysis'][model_name] = {
                'top_5_features': top_5_features,
                'feature_importance_dict': dict(zip(features, importances))
            }

        # Análise consolidada
        all_features = set()
        feature_scores = {}

        for model_data in importance_data.values():
            for feature, importance in zip(model_data['sorted_features'], model_data['sorted_importances']):
                if feature not in feature_scores:
                    feature_scores[feature] = []
                feature_scores[feature].append(importance)
                all_features.add(feature)

        # Calcular importância média
        avg_importance = {}
        for feature in all_features:
            avg_importance[feature] = np.mean(feature_scores[feature])

        # Ordenar por importância média
        sorted_avg = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

        report['summary'] = {
            'top_10_features_global': sorted_avg[:10],
            'total_features_analyzed': len(all_features)
        }

        # Recomendações
        top_features_global = [feat for feat, _ in sorted_avg[:5]]
        report['recommendations'] = [
            f"Focar em '{top_features_global[0]}' - feature mais importante",
            f"Monitorar '{top_features_global[1]}' para ajustes finos",
            f"Considerar engenharia de features para '{top_features_global[2]}'"
        ]

        return report

    def create_visualizations(self, save_path='./explainability_plots/'):
        """Cria visualizações de explicabilidade"""

        import os
        os.makedirs(save_path, exist_ok=True)

        importance_data = self.calculate_feature_importance()

        # 1. Feature Importance Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Feature Importance - Comparação entre Modelos', fontsize=16)

        for idx, (model_name, data) in enumerate(importance_data.items()):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]

            features = data['sorted_features'][:10]
            importances = data['sorted_importances'][:10]

            ax.barh(range(len(features)), importances)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_title(model_name)
            ax.set_xlabel('Importância')

        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. SHAP Summary Plot (se disponível)
        for model_name in self.explainers:
            explainer_data = self.explainers[model_name]
            shap_values = explainer_data['shap_values']
            X_test = explainer_data['X_test']

            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, feature_names=self.feature_names, show=False)
            plt.title(f'SHAP Summary - {model_name}')
            plt.savefig(f'{save_path}/shap_summary_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Visualizações salvas em: {save_path}")

    def explain_single_prediction(self, model_name, X_instance):
        """
        Explica uma previsão individual usando SHAP

        Parameters:
        - model_name: Nome do modelo treinado
        - X_instance: Instância individual para explicar (numpy array)

        Returns:
        - dict: Explicação detalhada
        """

        try:
            if model_name not in self.explainers:
                raise ValueError(f"SHAP explainer não encontrado para {model_name}")

            explainer_data = self.explainers[model_name]
            explainer = explainer_data['explainer']
            shap_values = explainer.shap_values(X_instance.reshape(1, -1))

            # Calcular contribuições
            contributions = {}
            for i, feature in enumerate(self.feature_names):
                contributions[feature] = {
                    'shap_value': float(shap_values[0][i]),
                    'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0,
                    'feature_value': float(X_instance[i])
                }

            # Ordenar por impacto
            sorted_contributions = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]['shap_value']),
                reverse=True
            )

            return {
                'model': model_name,
                'prediction_explanation': {
                    'base_value': float(explainer.expected_value) if hasattr(explainer, 'expected_value') else 0,
                    'final_prediction': float(np.sum(shap_values[0]) + (explainer.expected_value if hasattr(explainer, 'expected_value') else 0)),
                    'feature_contributions': dict(sorted_contributions[:10]),  # Top 10
                    'most_influential_feature': sorted_contributions[0][0] if sorted_contributions else None
                }
            }

        except Exception as e:
            print(f"Erro ao explicar previsão: {str(e)}")
            return None

    def get_explainability_summary(self):
        """Retorna resumo das capacidades de explicabilidade"""

        summary = {
            'available_models': list(self.traditional_models.keys()),
            'shap_explainers': list(self.explainers.keys()),
            'total_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'capabilities': [
                'Feature Importance (MDI, coeficientes)',
                'SHAP values (TreeExplainer, LinearExplainer)',
                'Individual prediction explanations',
                'Model comparison analysis'
            ]
        }

        return summary


if __name__ == "__main__":
    # Exemplo de uso
    print("🔍 Módulo de Explicabilidade de Modelos PMMA")
    print("=" * 50)

    # Criar instância
    explainer = ModelExplainer()

    print("Capacidades disponíveis:")
    for capability in explainer.get_explainability_summary()['capabilities']:
        print(f"  ✅ {capability}")