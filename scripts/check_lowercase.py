#!/usr/bin/env python3
import os
import sys
from glob import glob
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))


def latest_parquet(path: str) -> str | None:
    files = sorted(glob(os.path.join(path, "pmma_unificado_*.parquet")))
    return files[-1] if files else None


def generate_lowercase_report(src_parquet: str, report_path: str) -> None:
    pf = pq.ParquetFile(src_parquet)
    schema = pf.schema_arrow
    str_cols = []
    for name in schema.names:
        f = schema.field(name)
        t = f.type
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            str_cols.append(name)

    counts_non_lower = defaultdict(int)
    counts_non_null = defaultdict(int)
    samples = defaultdict(set)

    if str_cols:
        for rg in range(pf.num_row_groups):
            table = pf.read_row_group(rg, columns=str_cols)
            df = table.to_pandas(types_mapper=pd.ArrowDtype)
            for c in str_cols:
                s = df[c].astype("string")
                nn = s.notna()
                counts_non_null[c] += int(nn.sum())
                mask = nn & (s != s.str.lower())
                nbad = int(mask.sum())
                if nbad:
                    counts_non_lower[c] += nbad
                    bad_vals = s[mask].head(5).tolist()
                    for v in bad_vals:
                        if len(samples[c]) < 5:
                            samples[c].add(str(v))

    lines = []
    lines.append(f"Arquivo: {os.path.basename(src_parquet)}")
    if str_cols:
        lines.append("Colunas textuais verificadas: " + ", ".join(str_cols))
        lines.append("")
        lines.append("Resumo por coluna (não-minúsculas / não-nulas | %):")
        for c in sorted(str_cols):
            nn = counts_non_null.get(c, 0)
            nbad = counts_non_lower.get(c, 0)
            pct = (nbad / nn * 100.0) if nn else 0.0
            lines.append(f"- {c}: {nbad} / {nn} | {pct:.2f}%")
            if samples.get(c):
                lines.append(f"  amostras: {', '.join(list(samples[c]))}")
    else:
        lines.append("Nenhuma coluna textual para verificar.")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def main():
    # fonte: arg1, senão oficial, senão mais recente
    if len(sys.argv) > 1:
        src = sys.argv[1]
    else:
        official = os.path.join(OUTPUT_DIR, "pmma_unificado_oficial.parquet")
        src = official if os.path.exists(official) else latest_parquet(OUTPUT_DIR)
    if not src:
        print("Parquet unificado não encontrado em", OUTPUT_DIR)
        return 1

    # nome estável: se for o oficial, gravar em lowercase_report_oficial.txt
    if os.path.basename(src) == "pmma_unificado_oficial.parquet":
        report_path = os.path.join(OUTPUT_DIR, "lowercase_report_oficial.txt")
    else:
        report_path = os.path.join(OUTPUT_DIR, "_lowercase_report.txt")

    generate_lowercase_report(src, report_path)
    print("Relatório salvo em:", report_path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
