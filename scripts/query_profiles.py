#!/usr/bin/env python3
import json
import re
from pathlib import Path

def normalize(s: str) -> str:
    return re.sub(r"[^0-9a-z_]", "", s.lower().strip().replace(" ", "_")).replace("__", "_")

def main():
    p = Path('profiles/_combined_profiles.json')
    if not p.exists():
        print('profiles/_combined_profiles.json não encontrado.')
        return 1
    prof = json.loads(p.read_text(encoding='utf-8'))

    target_norms = {"no_regiao_atuacao", "regiao_atuacao"}
    target_raws = {"NO_REGIAO_ATUACAO"}

    results = []
    for f in prof:
        fname = f.get('file') or ''
        year = None
        m = re.match(r"(\d{4})", fname)
        if m:
            year = int(m.group(1))
        has = False
        found_as = []
        for s in f.get('sheets', []) or []:
            cols_raw = s.get('columns') or []
            cols_norm = s.get('columns_normalized') or []
            # check normalized
            for c in cols_norm:
                if c in target_norms:
                    has = True
                    found_as.append(c)
            # check raw
            for c in cols_raw:
                if c in target_raws:
                    has = True
                    found_as.append(c)
        if has:
            results.append((year, fname, sorted(set(found_as))))

    results.sort()
    if not results:
        print('Nenhum arquivo contém a coluna regiao de atuacao.')
        return 0

    print('Arquivos com a coluna regiao de atuacao:')
    for year, fname, forms in results:
        ys = str(year) if year is not None else '?'
        print(f"- {ys} | {fname} | formas: {', '.join(forms)}")

    # Print only years, unique
    years = sorted({y for y,_,_ in results if y is not None})
    print('\nAnos identificados:', ", ".join(map(str, years)) if years else '(indeterminado)')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

