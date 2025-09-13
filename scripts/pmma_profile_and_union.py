#!/usr/bin/env python3
import os
import sys
import json
import re
import unicodedata
from datetime import datetime

try:
    import pandas as pd
except Exception as e:
    sys.stderr.write("[ERRO] pandas não está disponível: %s\n" % e)
    sys.exit(2)


DATA_DIR = os.path.abspath(os.path.dirname(__file__) + "/../")
PROFILE_DIR = os.path.join(DATA_DIR, "profiles")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")


def ensure_dirs():
    os.makedirs(PROFILE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def list_excel_files(base_dir: str):
    files = []
    for name in os.listdir(base_dir):
        if name.startswith('.'):
            continue
        lower = name.lower()
        if lower.endswith('.xlsx') or lower.endswith('.xls'):
            files.append(os.path.join(base_dir, name))
    files.sort()
    return files


def normalize_text(s: str) -> str:
    if s is None:
        return s
    # strip accents, lowercase, collapse spaces and punctuation to underscores
    s = unicodedata.normalize('NFKD', str(s))
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = s.strip()
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"[^0-9a-z_]+", "", s)
    s = re.sub(r"_+", "_", s)
    return s


def profile_excel(path: str):
    prof = {
        "file": os.path.basename(path),
        "sheets": [],
    }
    try:
        xls = pd.ExcelFile(path)
    except Exception as e:
        prof["error"] = f"Falha ao abrir: {e}"
        return prof

    for sheet in xls.sheet_names:
        try:
            df_head = xls.parse(sheet_name=sheet, nrows=5)
            cols = list(df_head.columns)
        except Exception as e:
            cols = []
            err = str(e)
        else:
            err = None
        prof["sheets"].append({
            "name": sheet,
            "columns": cols,
            "columns_normalized": [normalize_text(c) for c in cols],
            "error": err,
        })
    return prof


SYNONYMS = {
    # Localização
    "municipio": "municipio",
    "municipio_ou_cidade": "municipio",
    "cidade": "municipio",
    "no_municipio": "municipio",
    "mun": "municipio",
    "bairro": "bairro",
    "ed_bairro": "bairro",
    "endereco": "logradouro",
    "ed_logradouro": "logradouro",
    "rua": "logradouro",
    "ed_numero": "numero",
    "ed_complemento": "complemento",
    "ed_ponto_referencia": "ponto_referencia",
    "no_rodovia": "rodovia",
    "nr_km": "km",
    "tx_trecho": "trecho",
    "sg_uf": "uf",
    "tipo_localizacao": "tipo_localizacao",
    "tipo_local": "tipo_local",
    "regiao": "regiao",
    "regiao_metropolitana": "regiao",
    "no_regiao_atuacao": "regiao_atuacao",

    # Georreferência
    "latitude": "latitude",
    "lat": "latitude",
    "longitude": "longitude",
    "lon": "longitude",
    "long": "longitude",
    "x": "longitude",
    "y": "latitude",

    # Datas/horas e calendários
    "data": "data",
    "data_fato": "data",
    "data_ocorrencia": "data",
    "dt_registro_ocorrencia": "dt_registro",
    "dt_registro_incidente": "dt_registro",
    "dt_finalizacao_despacho": "dt_finalizacao",
    "cricaodespacho": "dt_criacao_despacho",
    "criacaochegada": "dt_criacao_chegada",
    "despachochegada": "dt_despacho_chegada",
    "hora": "hora",
    "horario": "hora",
    "hora_fato": "hora",
    "hora_ocorrencia": "hora",
    "hora_int": "hora_num",
    "dia": "dia",
    "dia_int": "dia_num",
    "txt_dia": "dia_nome",
    "dia_sem": "dia_nome",
    "mes": "mes",
    "mes_int": "mes_num",
    "txt_mes": "mes_nome",
    "mes_tx": "mes_nome",
    "ano": "ano",
    "periodo": "turno",
    "turno": "turno",

    # Identificação
    "id": "id_ocorrencia",
    "id_incidente": "id_ocorrencia",
    "nr_protocolo_ocorrencia": "id_ocorrencia",
    "ocorrencia": "id_ocorrencia",
    "numero": "id_ocorrencia",
    "n_ocorrencia": "id_ocorrencia",

    # Órgãos/Unidades e despacho
    "agencia": "agencia",
    "no_agencia": "agencia",
    "unidade": "unidade",
    "upm": "unidade",
    "cia": "unidade",
    "bpm": "unidade",
    "viatura": "viatura",
    "viaturas": "viaturas",
    "cpam": "cpam",
    "gds": "grupo_despacho",
    "gd": "grupo_despacho",
    "grupodedespacho": "grupo_despacho",
    "giro": "giro",

    # Classificação e natureza
    "natureza": "natureza",
    "tipo": "tipo",
    "tipo_ocorrencia": "tipo",
    "subtipo": "subtipo",
    "codigotipo": "codigo_tipo",
    "codtipo": "codigo_tipo",
    "codigosubtipo": "codigo_subtipo",
    "codsubtipo": "codigo_subtipo",
    "codigofinalizacao": "codigo_finalizacao",
    "codfin": "codigo_finalizacao",
    "classificacao": "classificacao",
    "descricao": "descricao",
    "descricao_fato": "descricao",
    "descricao_tipo": "descricao_tipo",
    "descricao_subtipo": "descricao_subtipo",
    "descricao_finalizacao": "descricao_finalizacao",
    "motivo_finalizacao": "motivo_finalizacao",
    "sub_motivo_finalizacao": "sub_motivo_finalizacao",
    "tx_motivo_finalizacao": "motivo_finalizacao_descricao",
    "nome_finalaizacao_sub_motivo_finalizado": "sub_motivo_finalizacao_nome",
    "status": "status",
    "situacao": "status",
    "finalizacao": "finalizacao",

    # Textos/narrativas e diversos
    "tx_narrativa": "narrativa",
    "tx_atividade": "atividade",
    "titulo": "titulo",
    "personalizar": "personalizar",
    "area": "area",
    "quadrante": "quadrante",
    "prioridade": "prioridade",
}


def harmonize_columns(cols):
    norm = [normalize_text(c) for c in cols]
    mapped = []
    for c in norm:
        mapped.append(SYNONYMS.get(c, c))
    return mapped


def dedupe_columns(columns):
    seen = {}
    out = []
    for c in columns:
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}__{seen[c]}")
    return out


def parse_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Try to consolidate datetime from common patterns
    df = df.copy()
    cols = {c: normalize_text(c) for c in df.columns}
    inv = {}
    for k, v in cols.items():
        inv.setdefault(v, []).append(k)

    date_cols = inv.get('data', [])
    time_cols = inv.get('hora', [])
    dt_col_name = 'timestamp'
    if date_cols:
        dcol = date_cols[0]
        try:
            d = pd.to_datetime(df[dcol], errors='coerce', dayfirst=True)
        except Exception:
            d = pd.to_datetime(df[dcol].astype(str), errors='coerce', dayfirst=True)
        if time_cols:
            tcol = time_cols[0]
            try:
                t = pd.to_datetime(df[tcol].astype(str), errors='coerce').dt.time
            except Exception:
                t = pd.to_datetime(df[tcol].astype(str), errors='coerce').dt.time
            dt = pd.to_datetime(d.dt.date.astype(str) + ' ' + pd.Series(t).astype(str), errors='coerce')
        else:
            dt = d
        df[dt_col_name] = dt
    return df


def consolidate_alias_groups(df: pd.DataFrame, bases: list[str]) -> pd.DataFrame:
    """Merge duplicate alias columns created by dedupe (e.g., id_ocorrencia, id_ocorrencia__1).
    Keeps the first non-null value across duplicates and drops the extras.
    """
    out = df.copy()
    for base in bases:
        cols = [c for c in out.columns if c == base or c.startswith(base + "__")]
        if not cols:
            continue
        main = base if base in cols else cols[0]
        combined = None
        for c in cols:
            if combined is None:
                combined = out[c]
            else:
                combined = combined.combine_first(out[c])
        # ensure column named exactly as base
        out[base] = combined
        # drop extras
        for c in cols:
            if c != base:
                out.drop(columns=[c], inplace=True)
    return out


def _scale_value_to_range(v, max_abs):
    try:
        if v is None:
            return v
        import math
        if isinstance(v, str):
            v = v.strip().replace(" ", "")
        v = float(v)
        a = abs(v)
        if a <= max_abs:
            return v
        for scale in (1e6, 1e7, 1e5, 1e8):
            vv = v / scale
            if abs(vv) <= max_abs:
                return vv
        return v
    except Exception:
        return v


def coerce_geocoordinates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Try to build latitude/longitude if coord_x/coord_y survived
    if 'latitude' in out.columns or 'longitude' in out.columns:
        pass
    # Coerce to numeric and rescale to plausible degrees
    if 'latitude' in out.columns:
        out['latitude'] = pd.to_numeric(out['latitude'], errors='coerce').apply(lambda v: _scale_value_to_range(v, 90))
    if 'longitude' in out.columns:
        out['longitude'] = pd.to_numeric(out['longitude'], errors='coerce').apply(lambda v: _scale_value_to_range(v, 180))
    return out


def load_and_harmonize(path: str, sheet: str | int | None = None) -> pd.DataFrame:
    try:
        df = pd.read_excel(path, sheet_name=sheet)
    except Exception as e:
        raise RuntimeError(f"Falha ao ler {os.path.basename(path)}: {e}")

    # harmonize columns
    new_cols = harmonize_columns(df.columns.tolist())
    # After harmonization, ensure unique names to avoid Series/DataFrame ambiguity
    if len(set(new_cols)) != len(new_cols):
        new_cols = dedupe_columns(new_cols)
    df.columns = new_cols

    # minimally standardize text columns and enforce lowercase
    for c in df.select_dtypes(include=['object']).columns:
        ser = df[c]
        try:
            s = ser.astype('string')
            s = s.str.strip().str.lower()
            df[c] = s
        except Exception:
            # best effort fallback
            df[c] = ser.apply(lambda x: str(x).strip().lower() if pd.notna(x) else pd.NA)

    # attempt datetime consolidation
    df = parse_datetime_columns(df)

    # consolidate key aliases (id, latitude/longitude)
    df = consolidate_alias_groups(df, [
        'id_ocorrencia', 'latitude', 'longitude'
    ])

    # coerce geocoordinates to decimal degrees
    df = coerce_geocoordinates(df)

    # attach provenance (lowercased to cumprir padrão geral)
    df["_source_file"] = os.path.basename(path).lower()
    if isinstance(sheet, str):
        df["_source_sheet"] = str(sheet).lower()
    else:
        df["_source_sheet"] = None
    return df


def main():
    base = DATA_DIR
    ensure_dirs()
    files = list_excel_files(base)
    if not files:
        print("Nenhum arquivo .xlsx encontrado em", base)
        return 1

    print(f"Encontrados {len(files)} arquivos Excel.")
    all_profiles = []
    for fp in files:
        prof = profile_excel(fp)
        all_profiles.append(prof)
        # save per-file profile
        prof_out = os.path.join(PROFILE_DIR, os.path.basename(fp) + ".profile.json")
        with open(prof_out, 'w', encoding='utf-8') as f:
            json.dump(prof, f, ensure_ascii=False, indent=2)

    # write combined profile
    combined_path = os.path.join(PROFILE_DIR, "_combined_profiles.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_profiles, f, ensure_ascii=False, indent=2)
    print("Perfis gravados em:", PROFILE_DIR)

    # Decide sheets: default to first sheet per file
    unified_frames = []
    mapping_info = []
    for prof in all_profiles:
        if prof.get('error'):
            print("[AVISO] Pulando", prof['file'], "erro:", prof['error'])
            continue
        if not prof['sheets']:
            print("[AVISO] Sem abas em", prof['file'])
            continue
        sheet_entry = prof['sheets'][0]
        chosen_sheet = sheet_entry['name']
        try:
            df = load_and_harmonize(os.path.join(base, prof['file']), sheet=chosen_sheet)
        except Exception as e:
            print("[AVISO] Não foi possível carregar", prof['file'], ":", e)
            continue
        # record mapping from original -> harmonized
        mapping_info.append({
            "file": prof['file'],
            "sheet": chosen_sheet,
            "original_columns": sheet_entry.get('columns', []),
            "harmonized_columns": harmonize_columns(sheet_entry.get('columns', [])),
        })
        unified_frames.append(df)

    if not unified_frames:
        print("[ERRO] Nenhum dataframe unificado foi carregado.")
        return 2

    # align columns (outer union)
    master_cols = set()
    for df in unified_frames:
        master_cols.update(df.columns.tolist())
    master_cols = sorted(master_cols)
    aligned = []
    for df in unified_frames:
        for c in master_cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[master_cols]
        aligned.append(df)

    union_df = pd.concat(aligned, ignore_index=True)

    # Coerce columns for robust Parquet writing
    def coerce_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in out.columns:
            cname = normalize_text(c)
            if cname == 'timestamp' or 'data' in cname or cname.startswith('dt_'):
                try:
                    out[c] = pd.to_datetime(out[c], errors='coerce', dayfirst=True)
                except Exception:
                    pass
            else:
                if out[c].dtype == 'object':
                    try:
                        out[c] = out[c].astype('string')
                    except Exception:
                        out[c] = out[c].astype(str)
        return out

    union_df = coerce_for_parquet(union_df)

    # write parquet and mapping artifacts
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parquet_path = os.path.join(OUTPUT_DIR, f"pmma_unificado_{timestamp}.parquet")
    csv_path = os.path.join(OUTPUT_DIR, f"pmma_unificado_{timestamp}.csv")
    mapping_path = os.path.join(OUTPUT_DIR, f"column_mapping_{timestamp}.json")

    # Try pyarrow first, fallback to fastparquet, else CSV only
    wrote_parquet = False
    try:
        union_df.to_parquet(parquet_path, engine='pyarrow', index=False)
        wrote_parquet = True
    except Exception:
        try:
            union_df.to_parquet(parquet_path, engine='fastparquet', index=False)
            wrote_parquet = True
        except Exception as e2:
            print("[AVISO] Não foi possível escrever Parquet (pyarrow/fastparquet):", e2)

    if not wrote_parquet:
        # write CSV as fallback
        union_df.to_csv(csv_path, index=False)
        print("[INFO] CSV gerado em:", csv_path)
    else:
        print("[INFO] Parquet gerado em:", parquet_path)

    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping_info, f, ensure_ascii=False, indent=2)
    print("[INFO] Mapeamento de colunas gravado em:", mapping_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
