#+ PMMA Dados CIOPS — Documentação do Projeto

+ Repositório de dados e scripts para padronizar, unificar e documentar planilhas do CIOPS/PMMA (anos 2014–2024), produzindo um dataset único em Parquet, artefatos de mapeamento/qualidade e dicionário de dados.

+ Repositório remoto: `git@github.com:tadeugomes/pmma_dados_ciops.git`
+ Branch principal: `main`
+ Aviso de tamanho: alguns `.xlsx` > 50MB (recomendação GitHub é 50MB; limite duro: 100MB). Ver seção “Git LFS”.

## Sumário

- Visão geral e objetivos
- Estrutura de pastas
- Requisitos e setup
- Execução rápida (end-to-end)
- Pipeline detalhado (perfilamento → normalização → união → exportação → oficial)
- Scripts utilitários e como usar
- Saídas geradas
- Dicionário de dados
- Git LFS e arquivos grandes
- Solução de problemas e próximos passos

## Visão Geral e Objetivos

Padronizar nomes de colunas, harmonizar tipos e unir as planilhas anuais em um único conjunto de dados, com reprodutibilidade e artefatos auxiliares para auditoria:

- Perfis por arquivo/aba com nomes originais e normalizados.
- Regras de normalização e sinônimos centralizadas.
- União “outer” com coerção de tipos para Parquet.
- Artefatos versionados: mapeamento de colunas, relatório de qualidade (lowercase), data dictionary.

## Estrutura de Pastas

- `scripts/` — pipeline e utilitários (Python).
- `profiles/` — perfis por arquivo (`*.profile.json`) e combinado (`_combined_profiles.json`).
- `docs/` — documentação adicional; ver `docs/PROCESSO.md:1`.
- `output/` — saídas geradas (ignorado no Git): Parquet/CSV, mapeamentos, relatórios e dicionário de dados.
- Raiz — arquivos `.xlsx` originais do CIOPS/PMMA.

## Requisitos e Setup

Recomendado Python 3.10+ com ambiente virtual.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install pandas openpyxl pyarrow fastparquet
```

## Execução Rápida (End-to-End)

```bash
source .venv/bin/activate

# 1) Perfilamento + união e exportação
python scripts/pmma_profile_and_union.py

# 2) Marcar artefato oficial (copia o mais recente)
python scripts/set_official.py

# 3) (Opcional) Gerar dicionário de dados
python scripts/make_data_dictionary.py
```

Após isso, consulte em `output/`:
- `pmma_unificado_YYYYMMDD_HHMMSS.parquet` e `column_mapping_YYYYMMDD_HHMMSS.json` (última execução)
- `pmma_unificado_oficial.parquet`, `column_mapping_oficial.json`, `official_version.txt`, `lowercase_report_oficial.txt`
- `data_dictionary.csv` e `data_dictionary.md` (se gerado)

## Pipeline Detalhado

1) Perfilamento — `scripts/pmma_profile_and_union.py:1`
- Lista `*.xlsx/*.xls` na raiz, lê as abas e captura cabeçalhos.
- Gera `profiles/<arquivo>.profile.json` e `profiles/_combined_profiles.json`.

2) Normalização de nomes
- Regras: remover acentos, `lowercase`, substituir espaços/“-”/“/” por `_`, filtrar caracteres não `[0-9a-z_]`, colapsar múltiplos `_`.
- Sinônimos (exemplos):
  - Localização: `cidade`/`municipio_ou_cidade` → `municipio`; `rua`/`endereco` → `logradouro`; `x`/`y` → `longitude`/`latitude`.
  - Datas/horas: `data_fato`/`data_ocorrencia` → `data`; `hora_fato` → `hora`; `dt_registro_*` → `dt_registro`.
  - IDs: `id_incidente`/`nr_protocolo_ocorrencia`/`ocorrencia` → `id_ocorrencia` (primeiro não-nulo).

3) Carga e harmonização
- Por padrão, utiliza a primeira aba de cada planilha (ajustável no código).
- Concatena colunas de data/hora em `timestamp` quando aplicável.
- Converte textos para `string` minúscula; datas/horas para `datetime`.
- Reescala coordenadas para graus decimais plausíveis; grava `_source_file` e `_source_sheet`.

4) União e coerção de tipos
- União “outer” por superconjunto de colunas, preenchendo ausentes com `NA`.
- Coerção amigável a Parquet (pyarrow, fallback fastparquet; se ambos falharem, gera CSV).

5) Exportação e artefatos
- Parquet datado: `output/pmma_unificado_YYYYMMDD_HHMMSS.parquet`.
- CSV fallback: `output/pmma_unificado_YYYYMMDD_HHMMSS.csv`.
- Mapeamento por arquivo/aba: `output/column_mapping_YYYYMMDD_HHMMSS.json`.

6) Oficialização — `scripts/set_official.py:1`
- Copia o Parquet mais recente para `pmma_unificado_oficial.parquet` e o mapping correspondente para `column_mapping_oficial.json`.
- Registra `output/official_version.txt`.
- Gera `output/lowercase_report_oficial.txt` (usa lógica de `check_lowercase`).

## Scripts e Como Usar

- `scripts/pmma_profile_and_union.py:1`
  - Perfilamento, normalização, união e exportação (principal).
  - Execução: `python scripts/pmma_profile_and_union.py`

- `scripts/set_official.py:1`
  - Torna o artefato mais recente o “oficial” e gera relatório lowercase.
  - Execução: `python scripts/set_official.py`

- `scripts/make_data_dictionary.py:1`
  - Lê `output/pmma_unificado_oficial.parquet` e gera `data_dictionary.csv` e `data_dictionary.md` com tipos, contagens, min/max e amostras de valores.
  - Execução: `python scripts/make_data_dictionary.py`
  - Variáveis: `DICT_SAMPLE_SIZE` (linhas amostradas; padrão 200000).

- `scripts/check_lowercase.py:1`
  - Verifica colunas textuais que não estão em minúsculas; salva relatório em `output/`.
  - Execução: `python scripts/check_lowercase.py [opcional: caminho.parquet]`

- `scripts/list_columns.py:1`
  - Utilitário para listar colunas/tipos de um Parquet.

- `scripts/query_profiles.py:1`, `scripts/summarize_profiles.py:1`
  - Auxiliam a explorar `profiles/*.profile.json` e o combinado.

## Saídas Geradas (pasta `output/`)

- `pmma_unificado_*.parquet` — dataset unificado (preferencial para consumo).
- `pmma_unificado_*.csv` — fallback quando Parquet não é possível.
- `column_mapping_*.json` — mapeamento original→normalizado por arquivo/aba.
- `pmma_unificado_oficial.parquet` — cópia do mais recente para referência estável.
- `column_mapping_oficial.json`, `official_version.txt` — metadados do oficial.
- `lowercase_report_oficial.txt` — auditoria de minúsculas em colunas textuais.
- `data_dictionary.csv` e `data_dictionary.md` — dicionário de dados do artefato oficial.

## Como Atualizar os Dados

1) Adicione novas planilhas `.xlsx/.xls` na raiz do repositório.
2) Execute:

```bash
source .venv/bin/activate
python scripts/pmma_profile_and_union.py
python scripts/set_official.py
python scripts/make_data_dictionary.py  # opcional
```

3) Faça o commit/push dos metadados (scripts, perfis, mapeamentos, dicionário). A pasta `output/` é ignorada no Git por padrão; publique artefatos em release se necessário.

## Git LFS (Arquivos Grandes)

Para rastrear `.xlsx` grandes em novos commits:

```bash
# Instalar e habilitar (no macOS: brew install git-lfs)
git lfs install
git lfs track "*.xlsx"
git add .gitattributes && git commit -m "chore: track xlsx with LFS"
git push
```

Para migrar o histórico existente (reescreve commits; coordene com a equipe):

```bash
git lfs migrate import --include="*.xlsx"
git push --force-with-lease
```

## Solução de Problemas

- “pandas não está disponível”: ative a venv e `pip install pandas openpyxl pyarrow fastparquet`.
- Erro ao escrever Parquet: o script já tenta `pyarrow` e `fastparquet`, e gera CSV se ambos falharem.
- Colunas sem minúsculas no relatório: revise a etapa de normalização e a origem; o esperado é 0% pós-pipeline.
- Aba incorreta escolhida: por padrão pega a primeira aba; ajuste em `scripts/pmma_profile_and_union.py` se necessário.

## Próximos Passos (sugestões)

- Definir conjunto “golden” de colunas padrão e validações obrigatórias.
- Publicar artefatos oficiais via GitHub Releases com changelog.
- Adicionar testes simples para funções de normalização e mapeamento.

---
Esta documentação consolida o fluxo do projeto. Para detalhes adicionais do pipeline, veja `docs/PROCESSO.md:1`.
