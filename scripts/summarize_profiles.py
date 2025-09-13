#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

def main():
    p = Path('profiles/_combined_profiles.json')
    if not p.exists():
        print('profiles/_combined_profiles.json não encontrado.')
        return 1
    prof = json.loads(p.read_text(encoding='utf-8'))
    raw = Counter()
    norm = Counter()
    for f in prof:
        for s in f.get('sheets', []) or []:
            for c in (s.get('columns') or []):
                raw[c] += 1
            for c in (s.get('columns_normalized') or []):
                norm[c] += 1
    outp = Path('profiles/_columns_summary.txt')
    with outp.open('w', encoding='utf-8') as w:
        w.write(f'TOTAL UNIQUE RAW: {len(raw)}\n')
        w.write(f'TOTAL UNIQUE NORMALIZED: {len(norm)}\n\n')
        w.write('RAW (nome original):\n')
        for k, v in raw.most_common():
            w.write(f'{v}\t{k}\n')
        w.write('\nNORMALIZED (normalizado):\n')
        for k, v in norm.most_common():
            w.write(f'{v}\t{k}\n')
    print('Resumo salvo em profiles/_columns_summary.txt')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

