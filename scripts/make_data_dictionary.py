#!/usr/bin/env python3
import os
import sys
import math
import json
from collections import Counter
from datetime import datetime

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa


OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))


def arrow_type_to_kind(t: pa.DataType) -> str:
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return 'string'
    if pa.types.is_integer(t):
        return 'integer'
    if pa.types.is_floating(t) or pa.types.is_decimal(t):
        return 'number'
    if pa.types.is_boolean(t):
        return 'boolean'
    if pa.types.is_timestamp(t) or pa.types.is_date(t) or pa.types.is_time(t):
        return 'datetime'
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return 'binary'
    if pa.types.is_list(t):
        return 'list'
    if pa.types.is_struct(t):
        return 'struct'
    return str(t)


def summarize_column(name: str, series: pd.Series, kind: str) -> dict:
    s = series
    total = int(len(s))
    non_null = int(s.notna().sum())
    nulls = total - non_null
    out = {
        'column': name,
        'kind': kind,
        'count': total,
        'non_null': non_null,
        'nulls': nulls,
    }

    # samples (top 5 most frequent non-null)
    try:
        top = s.dropna().astype('string').value_counts().head(5).index.tolist()
    except Exception:
        top = []
    out['sample_values'] = top

    # numeric/datetime min/max
    if kind in ('number', 'integer'):
        try:
            sn = pd.to_numeric(s, errors='coerce')
            out['min'] = float(sn.min()) if sn.notna().any() else None
            out['max'] = float(sn.max()) if sn.notna().any() else None
        except Exception:
            out['min'] = out['max'] = None
    elif kind == 'datetime':
        try:
            sd = pd.to_datetime(s, errors='coerce')
            mn = sd.min()
            mx = sd.max()
            out['min'] = mn.isoformat() if pd.notna(mn) else None
            out['max'] = mx.isoformat() if pd.notna(mx) else None
        except Exception:
            out['min'] = out['max'] = None
    elif kind == 'string':
        try:
            sl = s.dropna().astype('string').str.len()
            out['avg_len'] = float(sl.mean()) if not sl.empty else None
            out['max_len'] = int(sl.max()) if not sl.empty else None
        except Exception:
            out['avg_len'] = out['max_len'] = None
    return out


def main():
    src = os.path.join(OUTPUT_DIR, 'pmma_unificado_oficial.parquet')
    if not os.path.exists(src):
        print('Parquet oficial não encontrado em', src)
        return 1

    sample_size = int(os.environ.get('DICT_SAMPLE_SIZE', '200000'))
    dataset = ds.dataset(src, format='parquet')

    # Pull a sample table to pandas
    # Note: for very large datasets, consider scanning in batches
    table = dataset.to_table()
    if sample_size and len(table) > sample_size:
        table = table.slice(0, sample_size)
    df = table.to_pandas()

    schema = pq.ParquetFile(src).schema_arrow
    kinds = {name: arrow_type_to_kind(schema.field(name).type) for name in schema.names}

    summaries = []
    for name in df.columns:
        kind = kinds.get(name, 'unknown')
        summaries.append(summarize_column(name, df[name], kind))

    # write CSV
    csv_path = os.path.join(OUTPUT_DIR, 'data_dictionary.csv')
    md_path = os.path.join(OUTPUT_DIR, 'data_dictionary.md')
    now = datetime.now().isoformat(timespec='seconds')

    df_out = pd.DataFrame(summaries)
    # normalize sample values to JSON strings
    df_out['sample_values'] = df_out['sample_values'].apply(lambda v: json.dumps(v, ensure_ascii=False))
    df_out.to_csv(csv_path, index=False)

    # write Markdown
    with open(md_path, 'w', encoding='utf-8') as w:
        w.write(f"# Dicionário de Dados — PMMA (gerado em {now})\n\n")
        w.write(f"Fonte: {os.path.basename(src)} — amostra usada: {len(df)} linhas.\n\n")
        w.write("| coluna | tipo | não nulos | nulos | min | max | avg_len | max_len | amostras |\n")
        w.write("|---|---|---:|---:|---|---|---:|---:|---|\n")
        for row in summaries:
            w.write(
                f"| {row['column']} | {row['kind']} | {row['non_null']} | {row['nulls']} | "
                f"{row.get('min', '')} | {row.get('max', '')} | {row.get('avg_len', '')} | {row.get('max_len', '')} | "
                f"{', '.join(map(str, row.get('sample_values', [])[:5]))} |\n"
            )

    print('Dicionário salvo em:')
    print('-', csv_path)
    print('-', md_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

