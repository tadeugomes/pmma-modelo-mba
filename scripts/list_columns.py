#!/usr/bin/env python3
import os
import sys
import pyarrow.parquet as pq

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..', 'output', 'pmma_unificado_oficial.parquet')
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print('arquivo oficial não encontrado:', path)
        return 1
    pf = pq.ParquetFile(path)
    for n in sorted(pf.schema_arrow.names):
        print(n)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())

