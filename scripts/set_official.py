#!/usr/bin/env python3
import os
import re
import shutil
from glob import glob
from dataclasses import dataclass

# permitir import local de check_lowercase
import sys
sys.path.append(os.path.dirname(__file__))
try:
    from check_lowercase import generate_lowercase_report
except Exception:
    generate_lowercase_report = None

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))


@dataclass
class Candidate:
    parquet: str
    mapping: str | None
    ts: str


def find_candidates() -> list[Candidate]:
    parquets = sorted(glob(os.path.join(OUTPUT_DIR, "pmma_unificado_*.parquet")))
    maps = sorted(glob(os.path.join(OUTPUT_DIR, "column_mapping_*.json")))
    maps_by_ts = {}
    for m in maps:
        mname = os.path.basename(m)
        mm = re.search(r"column_mapping_(\d{8}_\d{6})\.json$", mname)
        if mm:
            maps_by_ts[mm.group(1)] = m
    cands: list[Candidate] = []
    for p in parquets:
        pname = os.path.basename(p)
        m = re.search(r"pmma_unificado_(\d{8}_\d{6})\.parquet$", pname)
        if not m:
            continue
        ts = m.group(1)
        cands.append(Candidate(parquet=p, mapping=maps_by_ts.get(ts), ts=ts))
    return cands


def main() -> int:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cands = find_candidates()
    if not cands:
        print("Nenhum arquivo pmma_unificado_*.parquet encontrado em", OUTPUT_DIR)
        return 1
    cand = sorted(cands, key=lambda c: c.ts)[-1]
    dst_parquet = os.path.join(OUTPUT_DIR, "pmma_unificado_oficial.parquet")
    dst_mapping = os.path.join(OUTPUT_DIR, "column_mapping_oficial.json")
    shutil.copy2(cand.parquet, dst_parquet)
    print("Oficial (Parquet):", dst_parquet)
    if cand.mapping:
        shutil.copy2(cand.mapping, dst_mapping)
        print("Oficial (Mapping):", dst_mapping)
    else:
        print("[AVISO] Não há mapping com o mesmo timestamp:", cand.ts)
    with open(os.path.join(OUTPUT_DIR, "official_version.txt"), 'w', encoding='utf-8') as f:
        f.write(f"timestamp={cand.ts}\n")
        f.write(f"source_parquet={os.path.basename(cand.parquet)}\n")
        f.write(f"source_mapping={os.path.basename(cand.mapping) if cand.mapping else ''}\n")
    print("Versão registrada em output/official_version.txt")

    # gerar relatório de lowercase com nome estável
    if generate_lowercase_report is not None:
        report_path = os.path.join(OUTPUT_DIR, "lowercase_report_oficial.txt")
        try:
            generate_lowercase_report(dst_parquet, report_path)
            print("Relatório lowercase oficial:", report_path)
        except Exception as e:
            print("[AVISO] Falha ao gerar relatório lowercase:", e)
    else:
        print("[AVISO] Módulo check_lowercase indisponível; relatório não gerado.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
