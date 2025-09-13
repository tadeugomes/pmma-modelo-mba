# Processo de unificação dos dados PMMA

Este documento descreve, de ponta a ponta, o pipeline usado para perfilar, padronizar e unificar as planilhas anuais, gerando um único dataset em Parquet e um arquivo "oficial" para consumo.

## 1) Preparação do ambiente

- Criar e ativar o ambiente:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas openpyxl pyarrow fastparquet
```

## 2) Perfilamento dos arquivos Excel

- Script: `scripts/pmma_profile_and_union.py`
- O script lista todos os `.xlsx/.xls` no diretório raiz, identifica as abas e coleta os nomes de colunas.
- Saídas:
  - Perfis por arquivo: `profiles/*.profile.json`
  - Perfil combinado: `profiles/_combined_profiles.json`

## 3) Padronização de nomes de colunas

- Regras gerais:
  - Remoção de acentos, `lowercase`, substituição de espaços por `_`.
  - Dicionário de sinônimos em `scripts/pmma_profile_and_union.py:81` com alinhamento semântico (ex.: `cidade`→`municipio`, `rua`/`endereco`→`logradouro`).
  - Deduplicação de nomes após a normalização (sufixo `__1`, `__2` quando necessário).

## 4) Carga e harmonização

- Para cada arquivo, é carregada a primeira aba (ajustável), os nomes são harmonizados e são aplicadas rotinas de:
  - Consolidação de IDs em `id_ocorrencia` (primeiro valor não nulo entre `id_incidente`/`nr_protocolo_ocorrencia`/`ocorrencia`/`numero`).
  - Criação de `timestamp` quando houver `data` e `hora` (formato flexível, `dayfirst=True`).
  - Conversão de colunas textuais para `lowercase` mantendo nulos.
  - Geocoordenadas: `x`→`longitude`, `y`→`latitude`, com reescala automática (e.g., divisão por `1e6/1e7`) para graus decimais plausíveis.
  - Colunas de proveniência: `_source_file`, `_source_sheet` em minúsculas.

## 5) União vertical e coerção de tipos

- União por superconjunto de colunas (outer union) e alinhamento de dtypes para escrita em Parquet.
- Datas/horas são convertidas para datetime sempre que possível; textos para `string`.

## 6) Exportação

- Saídas datadas:
  - Parquet: `output/pmma_unificado_YYYYMMDD_HHMMSS.parquet`
  - Mapeamento: `output/column_mapping_YYYYMMDD_HHMMSS.json`

## 7) Arquivo oficial (latest)

- Script: `scripts/set_official.py`
- Escolhe o Parquet mais recente e copia para:
  - `output/pmma_unificado_oficial.parquet`
  - `output/column_mapping_oficial.json`
  - Registro de versão: `output/official_version.txt`
  - Também gera automaticamente o relatório de minúsculas em nome estável:
    - `output/lowercase_report_oficial.txt`

## 8) Validações auxiliares

- Lowercase: `scripts/check_lowercase.py` gera `output/lowercase_report_oficial.txt` (se apontar para o oficial) ou `output/_lowercase_report.txt` (se apontar para um Parquet datado) com percentuais de valores não-minúsculos por coluna (depois do pipeline deve estar 0%).

## 9) Execução completa (exemplo)

```bash
source .venv/bin/activate
python scripts/pmma_profile_and_union.py
python scripts/set_official.py  # gera o oficial e o relatório lowercase estável
```

## 10) Ajustes comuns

- Sinônimos: editar `SYNONYMS` em `scripts/pmma_profile_and_union.py:81`.
- Seleção de abas: alterar a lógica que hoje pega a primeira aba por arquivo.
- Lowercase: já habilitado para todas as colunas textuais.
