# PMMA Dados CIOPS

Repositório de dados e scripts relacionados ao conjunto de dados do PMMA/CIOPS.

## Visão Geral

Este repositório foi convertido para Git e publicado no GitHub em `main` com controle de versão dos arquivos de dados (.xlsx), perfis gerados e scripts auxiliares.

- Repositório remoto: `git@github.com:tadeugomes/pmma_dados_ciops.git`
- Branch principal: `main`
- Aviso de tamanho: alguns `.xlsx` têm > 50MB (limite recomendado do GitHub), mas < 100MB (limite duro). Avalie usar Git LFS.

## Estrutura do Repositório

- `scripts/` — scripts Python de apoio (consulta, perfilamento, dicionário, etc.).
- `profiles/` — perfis/estatísticas por arquivo, agregados e resumos.
- `docs/` — documentação do processo (`PROCESSO.md`).
- `output/` — saídas geradas (ignorado pelo Git via `.gitignore`).
- Arquivos `.xlsx` — dados brutos do CIOPS/PMMA (2014–2024).

## O que foi feito

1. Inicialização do Git (na pasta `/Users/tgt/Documents/dados_pmma`).
   - Criado `.gitignore` com entradas para macOS, Python/venv, caches de ferramentas, IDEs e `output/`.
2. Commit inicial
   - Adicionados arquivos `.xlsx`, `scripts/`, `profiles/` e `docs/`.
3. Configuração de autenticação por SSH
   - Gerada chave ED25519 (`~/.ssh/id_ed25519`) e adicionada ao agente/Keychain.
   - Chave pública cadastrada em GitHub (Settings → SSH and GPG keys).
4. Configuração do remoto e push
   - `origin` ajustado para SSH: `git@github.com:tadeugomes/pmma_dados_ciops.git`.
   - Push da `main` realizado e branch configurada para rastrear `origin/main`.

## Comandos Utilizados (referência)

```bash
# Inicialização e commit inicial
git init
printf "..." > .gitignore
git add .
git commit -m "chore: initial commit of dataset and scripts"

# Configuração SSH (macOS)
ssh-keygen -t ed25519 -C "tadeugomes@github" -f ~/.ssh/id_ed25519 -N ""
ssh-add --apple-use-keychain ~/.ssh/id_ed25519  # ou: ssh-add ~/.ssh/id_ed25519
# Adicionar a pública (~/.ssh/id_ed25519.pub) no GitHub

# Remoto e push
git remote add origin git@github.com:tadeugomes/pmma_dados_ciops.git
git push -u origin main
```

## Git LFS (opcional, recomendado para arquivos grandes)

Se desejar otimizar o versionamento de arquivos grandes como `.xlsx`:

```bash
# Instalar e habilitar
# macOS: brew install git-lfs
git lfs install

# Rastrear apenas futuros commits de .xlsx
git lfs track "*.xlsx"
git add .gitattributes
git commit -m "chore: track xlsx with LFS"
git push
```

Para retroativamente migrar os arquivos já versionados para LFS (reescreve histórico, requer força no push e coordenação com colaboradores):

```bash
git lfs migrate import --include="*.xlsx"
# Verifique resultados e FAÇA BACKUP antes
# Push forçado (após confirmar)
git push --force-with-lease
```

## Dicas/Próximos Passos

- Padronizar nomes de arquivos e colunas conforme scripts em `scripts/`.
- Registrar no README alterações relevantes do pipeline/dados.
- Considerar criar releases (tags) por “vagas” de dados (p. ex., por ano).

---
Criado automaticamente como registro do processo de criação e publicação do repositório.
