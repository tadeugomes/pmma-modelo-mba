#!/usr/bin/env python3
"""
Script para organizar arquivos de documentação duplicados
Move e reorganiza arquivos espalhados pelo projeto
"""

import os
import shutil
from pathlib import Path

def mover_arquivos_duplicados():
    """Move arquivos duplicados para locais apropriados"""

    # Mapeamento de arquivos duplicados
    movimentos = {
        # ml_projects/EXPLICACAO_MODELOS.md -> já existe em docs/
        'ml_projects/EXPLICACAO_MODELOS.md': 'DELETE',

        # Manter apenas os mais recentes/completos
        'ml_projects/DOCUMENTACAO_COMPLETA.md': 'KEEP',
        'ml_projects/RESUMO_IMPLEMENTACAO.md': 'KEEP',
        'ml_projects/CHANGELOG.md': 'KEEP',
        'ml_projects/NOTA_METODOLOGICA.md': 'KEEP',
        'ml_projects/README.md': 'RENAME_TO_ml_projects_README.md',

        # Mover testes para pasta docs
        'ml_models/explainability_test_report.md': 'docs/explainability_test_report.md',
    }

    base_path = Path('.')

    for arquivo, acao in movimentos.items():
        caminho_completo = base_path / arquivo

        if not caminho_completo.exists():
            print(f"⚠️  Arquivo não encontrado: {arquivo}")
            continue

        if acao == 'DELETE':
            # Arquivo duplicado - pode ser removido
            try:
                os.remove(caminho_completo)
                print(f"🗑️  Removido: {arquivo}")
            except Exception as e:
                print(f"❌ Erro ao remover {arquivo}: {e}")

        elif acao == 'KEEP':
            # Manter arquivo onde está
            print(f"✅ Mantido: {arquivo}")

        elif acao.startswith('RENAME_TO_'):
            # Renomear arquivo
            novo_nome = acao.replace('RENAME_TO_', '')
            novo_caminho = base_path / novo_nome

            try:
                shutil.move(str(caminho_completo), str(novo_caminho))
                print(f"📝 Renomeado: {arquivo} -> {novo_nome}")
            except Exception as e:
                print(f"❌ Erro ao renomear {arquivo}: {e}")

        elif acao.startswith('docs/'):
            # Mover para pasta docs
            novo_caminho = base_path / acao

            try:
                # Criar pasta docs se não existir
                novo_caminho.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(caminho_completo), str(novo_caminho))
                print(f"📁 Movido: {arquivo} -> {acao}")
            except Exception as e:
                print(f"❌ Erro ao mover {arquivo}: {e}")

def criar_indice_docs():
    """Cria um índice automático da pasta docs"""

    docs_path = Path('docs')
    if not docs_path.exists():
        docs_path.mkdir(exist_ok=True)
        return

    arquivos_docs = list(docs_path.glob('*.md')) + list(docs_path.glob('*.html'))

    indice = """# 📚 **Índice da Documentação**

## 📁 Arquivos em `docs/`

"""

    for arquivo in sorted(arquivos_docs):
        nome = arquivo.name
        descricao = obter_descricao_arquivo(nome)
        indice += f"- **[{nome}](./{nome})** - {descricao}\n"

    # Salvar índice
    with open(docs_path / 'INDEX.md', 'w', encoding='utf-8') as f:
        f.write(indice)

    print("📋 Índice criado: docs/INDEX.md")

def obter_descricao_arquivo(nome):
    """Retorna descrição do arquivo baseado no nome"""

    descricoes = {
        'slides_tecnicos.html': 'Apresentação técnica (13 slides)',
        'slides_modelos.html': 'Apresentação explicativa (16 slides)',
        'detalhes_tecnicos.md': 'Especificações técnicas completas',
        'explicacao_modelos.md': 'Explicações para leigos',
        'PROCESSO.md': 'Metodologia de processamento ETL',
        'QUICKSTART.md': 'Guia de instalação rápida',
        'INDEX.md': 'Índice de documentação',
        'explainability_test_report.md': 'Relatório de testes de explicabilidade'
    }

    return descricoes.get(nome, 'Documentação')

def limpar_arquivos_vazios():
    """Remove pastas vazias"""

    for pasta, subpastas, arquivos in os.walk('.', topdown=False):
        if pasta == '.':
            continue

        if not subpastas and not arquivos:
            try:
                os.rmdir(pasta)
                print(f"🗑️  Removida pasta vazia: {pasta}")
            except:
                pass

def gerar_resumo():
    """Gera resumo da organização"""

    print("\n" + "="*60)
    print("📚 **RESUMO DA ORGANIZAÇÃO DE DOCUMENTAÇÃO**")
    print("="*60)

    # Contar arquivos por pasta
    pastas_contagem = {}

    for pasta, subpastas, arquivos in os.walk('.'):
        if '.git' in pasta or '.venv' in pasta or '__pycache__' in pasta:
            continue

        arquivos_md = [f for f in arquivos if f.endswith('.md')]
        arquivos_html = [f for f in arquivos if f.endswith('.html')]

        if arquivos_md or arquivos_html:
            pastas_contagem[pasta] = {
                'md': len(arquivos_md),
                'html': len(arquivos_html)
            }

    print("\n📊 **Arquivos por Pasta:**")
    for pasta, contagem in sorted(pastas_contagem.items()):
        print(f"  {pasta}: {contagem['md']} .md, {contagem['html']} .html")

    # Arquivos principais
    print(f"\n📄 **Arquivos Principais:**")
    principais = [
        'README.md',
        'DOCUMENTATION.md',
        'docs/INDEX.md',
        'docs/detalhes_tecnicos.md',
        'ml_projects/README.md'
    ]

    for arquivo in principais:
        if os.path.exists(arquivo):
            print(f"  ✅ {arquivo}")
        else:
            print(f"  ❌ {arquivo} (não encontrado)")

if __name__ == "__main__":
    print("🔧 **Organizando Documentação do Projeto PMMA**")
    print("="*50)

    # 1. Mover arquivos duplicados
    print("\n1️⃣ **Movendo arquivos duplicados...**")
    mover_arquivos_duplicados()

    # 2. Criar índice da pasta docs
    print("\n2️⃣ **Criando índice de docs...**")
    criar_indice_docs()

    # 3. Limpar pastas vazias
    print("\n3️⃣ **Limpando pastas vazias...**")
    limpar_arquivos_vazios()

    # 4. Gerar resumo
    print("\n4️⃣ **Gerando resumo...**")
    gerar_resumo()

    print("\n✅ **Organização concluída!**")
    print("\n📋 **Próximos passos:**")
    print("1. Verifique os arquivos movidos")
    print("2. Atualize links nos READMEs se necessário")
    print("3. Revise o índice gerado em docs/INDEX.md")
    print("4. Commit as mudanças organizadas")