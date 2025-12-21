# 📋 Nota Metodológica - Resultados dos Modelos

## ⚠️ Importante Esclarecimento sobre os Resultados

### Resultados Baseados em Simulação

Os resultados apresentados neste projeto (redução de 28% no tempo de resposta, aumento de 32% na cobertura, etc.) são **estimativas baseadas em simulações e benchmarks da indústria**, não em medições reais em produção na PMMA.

### Origem das Métricas:

#### 1. **Redução de Tempo de Resposta: 28%**
- **Fonte:** Simulação do algoritmo DQN com dados históricos
- **Base:** Benchmark de sistemas de otimização similar
- **Cálculo:** Comparação entre posicionamento atual vs otimizado

#### 2. **Aumento de Cobertura: 32%**
- **Fonte:** Análise geométrica do raio de cobertura
- **Base:** Simulação de reposicionamento otimizado
- **Cálculo:** Área coberta com viaturas otimizadas vs posição atual

#### 3. **Acurácia de Classificação: 93%**
- **Fonte:** Teste do modelo BERT com dados de validação
- **Base:** Validação cruzada temporal
- **Cálculo:** (VP + VN) / Total nas classificações

#### 4. **Acurácia de Previsão: R²=0.87**
- **Fonte:** Modelo LSTM validado com holdout temporal
- **Base:** Comparação previsto vs real (20% dos dados)
- **Cálculo:** Coeficiente de determinação padrão

### Limitações:

1. **Sem deployment em produção** - Todos os testes foram offline
2. **Dados limitados** - Apenas dados históricos, sem validação operacional
3. **Fatores externos não considerados** - Trânsito, clima, eventos imprevistos
4. **Aceitação humana não testada** - Reação de operadores não avaliada

### Para Validação Real:

1. **Piloto Controlado:** 1 mês com subset de viaturas
2. **Coleta de Métricas Reais:** Tempo resposta real vs previsto
3. **A/B Testing:** Operações com e sem sistema
4. **Avaliação Qualitativa:** Feedback dos usuários
5. **Análise de Custo-Benefício:** Com dados financeiros reais

### Recomendação:

Os resultados devem ser vistos como **potencial estimado** do sistema, garantindo assim expectativas realistas sobre os benefícios que podem ser alcançados quando o sistema for efetivamente implementado em ambiente de produção.

*Transparência é fundamental para a credibilidade do projeto.*