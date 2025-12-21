# 🚔 Sistema de Machine Learning - PMMA

Este projeto implementa 4 soluções de machine learning usando redes neurais para otimizar as operações da Polícia Militar do Maranhão (PMMA).

## 📁 Estrutura do Projeto

```
ml_projects/
├── shared/                     # Módulos compartilhados
│   ├── preprocessing/          # Preparação de dados
│   ├── models/                 # Modelos base
│   └── utils/                  # Utilitários
├── project1/                   # Previsão de Ocorrências (LSTM)
│   ├── src/                    # Código fonte
│   ├── models/                 # Modelos treinados
│   └── notebooks/              # Análises
├── project2/                   # Classificação (BERT)
│   ├── src/                    # Código fonte
│   ├── models/                 # Modelos treinados
│   └── notebooks/              # Análises
├── project3/                   # Otimização (DQN)
│   ├── src/                    # Código fonte
│   ├── models/                 # Modelos treinados
│   └── notebooks/              # Análises
├── project4/                   # Previsão por Bairros (LSTM+Embedding)
│   ├── src/                    # Código fonte
│   ├── models/                 # Modelos treinados
│   └── notebooks/              # Análises
├── ml_models/                  # Modelos de ML adicionais
│   ├── bairro_prediction_model.py   # Modelo de previsão por bairros
│   ├── bairro_dashboard_component.py # Componente de visualização
│   └── train_bairro_model.py        # Script de treinamento
└── dashboard/                  # Dashboard Streamlit
    ├── app.py                  # Aplicação principal
    ├── pages/                  # Páginas do dashboard
    └── components/             # Componentes reutilizáveis
```

## 🎯 Projetos

### 1. 🔮 Previsão de Ocorrências Policiais
- **Técnica**: LSTM Bidirecional com Attention
- **Objetivo**: Prever demanda por área nas próximas 24 horas
- **Features**: Histórico, dia da semana, feriados, padrões sazonais

### 2. 🏷️ Classificação Inteligente de Ocorrências
- **Técnica**: BERT pré-treinado em português com fine-tuning
- **Objetivo**: Classificar automaticamente ocorrências por tipo e urgência
- **Features**: Texto da ocorrência, contexto temporal e espacial

### 3. 🎯 Otimização de Alocação de Recursos
- **Técnica**: Deep Q-Network (Reinforcement Learning)
- **Objetivo**: Otimizar posicionamento de viaturas em tempo real
- **Features**: Posição atual, demanda prevista, restrições operacionais

### 4. 🏘️ Previsão por Bairros
- **Técnica**: LSTM com Attention e Embedding de Bairros
- **Objetivo**: Prever demanda em nível granular por bairros
- **Features**: Histórico temporal, características específicas dos bairros, padrões sazonais
- **Abrangência**: 3.906 bairros únicos com mais de 100 ocorrências cada
- **Visualizações**: Mapa de calor geográfico, análise de tipos de ocorrência, padrões horários
- **Dados geográficos**: 300.066 registros com coordenadas GPS válidas
- **Tipos analisados**: Descrições detalhadas (limpeza automática de códigos)

## 🚀 Instalação

1. Clone o repositório:
```bash
git clone <URL>
cd ml_projects
```

2. Crie um ambiente virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Baixe o modelo BERT em português:
```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')
tokenizer.save_pretrained('./project2/models/bert-tokenizer')
model.save_pretrained('./project2/models/bert-model')
```

## 📊 Treinamento dos Modelos

### Projeto 1 - LSTM
```bash
cd project1
python train_model.py \
    --data_path ../output/pmma_unificado_oficial.parquet \
    --hidden_size 128 \
    --num_layers 2 \
    --batch_size 32 \
    --epochs 100
```

### Projeto 2 - BERT
```bash
cd project2
python train_classifier.py \
    --data_path ../output/pmma_unificado_oficial.parquet \
    --max_length 128 \
    --batch_size 16 \
    --epochs 10 \
    --learning_rate 2e-5
```

### Projeto 3 - DQN
```bash
cd project3
python train_dqn.py \
    --data_path ../output/pmma_unificado_oficial.parquet \
    --num_vehicles 10 \
    --num_episodes 1000 \
    --lr 1e-3
```

### Projeto 4 - Previsão por Bairros
```bash
cd ml_models
python train_bairro_model.py
```

O modelo será treinado com:
- **Sequência temporal**: 24 horas de histórico
- **Embedding size**: 50 dimensões por bairro
- **Hidden layers**: 128 neurônios
- **Bairros considerados**: 3.906 (com >100 ocorrências)
- **Output**: Previsão para as próximas 24-48 horas

## 📱 Dashboard Streamlit

Execute o dashboard interativo:

```bash
cd dashboard
streamlit run app.py
```

O dashboard inclui:
- **Visão Geral**: Estatísticas e visualizações dos dados
- **Previsão**: Interface para gerar previsões de demanda
- **Classificação**: Classificador de ocorrências em tempo real
- **Otimização**: Visualização e simulação de alocação de recursos
- **Previsão por Bairros**: Análise granular e previsões no nível dos bairros

### Funcionalidades da Página de Bairros:
- **🔥 Hotspots**: Identificação dos bairros com mais ocorrências
- **📊 Análise Temporal**: Padrões diários e semanais por bairro
- **🗺️ Mapa de Calor**: Visualização geográfica com marcadores proporcionais
- **🏷️ Tipos de Ocorrência**: Top 10 tipos com descrições detalhadas por bairro
- **📈 Padrões Horários**: Distribuição das ocorrências por hora do dia
- **💡 Recomendações**: Diretrizes operacionais baseadas nos dados

## 📈 Performance Esperada

### Projeto 1 - Previsão
- **MAE**: < 5 ocorrências/hora
- **RMSE**: < 8 ocorrências/hora
- **R²**: > 0.85

### Projeto 2 - Classificação
- **F1-Score**: > 0.90 (macro)
- **Acurácia Top-3**: > 0.95
- **Latência**: < 500ms

### Projeto 3 - Otimização
- **Redução Tempo Resposta**: 20-30%
- **Aumento Cobertura**: 15-25%
- **Balanceamento Carga**: Melhorias significativas

### Projeto 4 - Previsão por Bairros
- **MAE**: < 3 ocorrências/hora/bairro
- **RMSE**: < 5 ocorrências/hora/bairro
- **R²**: > 0.80
- **Cobertura**: 3.906 bairros
- **Latência**: < 100ms por previsão

## 🔧 Configuração

### Variáveis de Ambiente
```bash
# .env
DATA_PATH=../output/pmma_unificado_oficial.parquet
MODEL_PATH=./models
DEVICE=cuda  # ou cpu
LOG_LEVEL=INFO
```

### Configuração dos Modelos

Cada projeto tem seu próprio arquivo de configuração em `models/config.json`:

```json
{
  "model_type": "LSTM",
  "hyperparameters": {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "learning_rate": 0.001
  }
}
```

## 🧪 Testes

Execute os testes unitários:

```bash
pytest tests/
```

Execute o linting:

```bash
flake8 .
black .
```

## 📝 logging

O sistema gera logs detalhados em `logs/`:
- `training.log`: Logs de treinamento
- `inference.log`: Logs de inferência
- `error.log`: Logs de erros

## 🚀 Deploy

### Via Docker
```bash
docker build -t pmma-ml .
docker run -p 8501:8501 pmma-ml
```

### Via Kubernetes
```bash
kubectl apply -f k8s/
```

## 🤝 Contribuição

1. Fork o projeto
2. Crie uma branch feature (`git checkout -b feature/NovaFuncionalidade`)
3. Commit suas mudanças (`git commit -m 'Adiciona NovaFuncionalidade'`)
4. Push para a branch (`git push origin feature/NovaFuncionalidade`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 👥 Equipe

- [Seu Nome] - Arquiteto de ML
- [Outro Nome] - Cientista de Dados
- [Outro Nome] - Engenheiro de Software

## 📞 Contato

- Email: contato@pmma.ma.gov.br
- Issues: [GitHub Issues](URL/issues)

## 🙏 Agradecimentos

- Polícia Militar do Maranhão (PMMA)
- NeuralMind (BERTimbau)
- Comunidade PyTorch
- Streamlit Team