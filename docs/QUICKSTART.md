# 🚀 **Guia de Início Rápido - PMMA ML**

### *Instalação e execução em 5 minutos*

---

## ⚡ **Setup Ultra Rápido**

### **1. Pré-requisitos**
- Python 3.9+ instalado
- Git configurado
- 8GB+ RAM recomendado

### **2. Clonar e Configurar**
```bash
# Clonar repositório
git clone git@github.com:tadeugomes/pmma_dados_ciops.git
cd pmma_dados_ciops

# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

### **3. Verificar Dados**
```bash
# Verificar dataset principal
ls -la pmma_unificado_oficial.parquet
# Esperado: ~136MB
```

### **4. Executar Sistema**
```bash
# Dashboard principal
streamlit run dashboard/dashboard_final_perguntas.py
# Acesso: http://localhost:8508

# Dashboard explicabilidade (requer dados)
streamlit run ml_models/explainability_dashboard_real.py
# Acesso: http://localhost:8501
```

---

## 🎯 **Teste Rápido de Funcionalidades**

### **Verificar Modelos**
```bash
python ml_models/test_explainability.py
# Esperado: 5/5 testes passando
```

### **Previsão Simples**
```python
import pandas as pd
from ml_models.bairro_prediction_model import BairroPredictionModel

# Carregar modelo
model = BairroPredictionModel()

# Dados de exemplo (substituir com dados reais)
data = pd.DataFrame({
    'data': pd.date_range('2024-01-01', periods=24, freq='H'),
    'ocorrencias': [5] * 24
})

# Fazer previsão
predictions = model.predict('Centro', data, hours_ahead=24)
print(f"Previsão para próximas 24h: {predictions[:5]}")
```

---

## 🔍 **Verificação de Instalação**

### **Scripts de Verificação**
```bash
# Verificar Python
python --version  # Esperado: 3.9+

# Verificar pacotes principais
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Verificar dados
python -c "
import pandas as pd
df = pd.read_parquet('pmma_unificado_oficial.parquet')
print(f'Dataset: {len(df):,} registros')
print(f'Período: {df[\"data\"].min()} a {df[\"data\"].max()}')
"
```

### **Resultado Esperado**
```
PyTorch: 2.x.x
Streamlit: 1.x.x
Transformers: 4.x.x
Dataset: 2,262,405 registros
Período: 2014-XX-XX a 2024-XX-XX
```

---

## 🚨 **Problemas Comuns**

### **"Comando não encontrado"**
```bash
# Adicionar Python ao PATH (Windows)
# Usar python3 em vez de python (Linux)
python3 --version
```

### **"ModuleNotFoundError"**
```bash
# Reinstalar dependências
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### **"Arquivo .parquet não encontrado"**
```bash
# Verificar localização do arquivo
find . -name "*.parquet" -type f
# Mover para local correto se necessário
mv caminho/do/arquivo.parquet ./pmma_unificado_oficial.parquet
```

### **"Streamlit não inicia"**
```bash
# Verificar porta
lsof -i :8508
# Matar processo se necessário
kill -9 <PID>
# Tentar porta diferente
streamlit run dashboard/dashboard_final_perguntas.py --server.port 8509
```

---

## ✅ **Checklist de Funcionalidades**

### **Básico**
- [ ] Python 3.9+ funcionando
- [ ] Ambiente virtual ativado
- [ ] Dependências instaladas sem erros
- [ ] Dataset .parquet encontrado

### **Dashboards**
- [ ] Dashboard principal carrega em http://localhost:8508
- [ ] Dashboard explicabilidade carrega em http://localhost:8501
- [ ] Navegação entre páginas funciona
- [ ] Visualizações carregam corretamente

### **Modelos**
- [ ] Testes passam (5/5)
- [ ] Previsões funcionam
- [ ] Explicabilidade operacional
- [ ] Performance aceitável (<1s)

---

## 🔗 **Links Rápidos**

- **Dashboard Principal**: http://localhost:8508
- **Explicabilidade**: http://localhost:8501
- **Documentação**: [DOCUMENTATION.md](../DOCUMENTATION.md)
- **Issues**: [GitHub Issues](https://github.com/tadeugomes/pmma_dados_ciops/issues)

---

## 📞 **Ajuda Rápida**

### **Comandos Essenciais**
```bash
# Verificar status
python ml_models/test_explainability.py

# Reiniciar dashboard
streamlit run dashboard/dashboard_final_perguntas.py

# Verificar logs
tail -f ~/.streamlit/logs/streamlit_stderr_2024-XX-XX.log
```

### **Busca de Ajuda**
1. Verifique [DOCUMENTATION.md](../DOCUMENTATION.md)
2. Busque no [README.md](../README.md)
3. Abra issue no GitHub
4. Contate equipe do projeto

---

*Guia atualizado: 2024-12-21 | Versão: 1.0*