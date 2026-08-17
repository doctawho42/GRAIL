#!/usr/bin/env python3
"""
Строит рамку выборки для цитатного аудита из forward citations OpenAlex.

Зачем: премисса статьи — «coverage-цифру публикуют как свойство правил, а процедуру
применения не пишут». Это надо не утверждать, а измерить на выборке, построенной
воспроизводимо. Скрипт строит рамку, применяет inclusion-фильтр и тянет seeded-выборку.

Запуск:
    python build_frame.py --snapshot 2026-08-20 --n 120 --seed 20260820 --out frame/
    python build_frame.py --self-test          # проверка логики без сети

Требуется: requests. Ключ OpenAlex не нужен, но вежливо указать --mailto.

ВАЖНО: рамка замораживается снапшотом. Все файлы в out/ коммитятся ДО начала кодирования,
вместе с preregistration.md. Кто кодирует — рамку после этого не трогает.
"""

import argparse
import csv
import json
import hashlib
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------- seed works

# DOI посевных работ. Три группы: рулбейзд-метаболизм, ретросинтез-шаблоны,
# базы правил/движки. Список фиксируется в preregistration и больше не меняется.
SEEDS = {
    # --- metabolism rule sets ---
    "10.1002/cmdc.200700312":            "SyGMa (Ridder & Wagener 2008)",
    "10.1186/s13321-018-0324-5":         "BioTransformer (Djoumbou-Feunang 2019)",
    "10.1021/acs.chemrestox.0c00224":    "GLORYx (de Bruyn Kops 2021)",
    "10.3389/fchem.2019.00402":          "GLORY (de Bruyn Kops 2019)",
    "10.1093/bib/bbae374":               "MetaPredictor (Zhu 2024)",
    "10.1039/d0sc02639e":                "MetaTrans (Litsa 2020)",
    # NAR 2015, not the ES&T paper: the DOI first written here resolves to nothing and the
    # seed was silently dropped, taking its citing works with it
    "10.1093/nar/gkv1229":               "enviPath / EAWAG-BBD (Wicker 2016)",
    # --- retrosynthesis template libraries / appliers ---
    "10.1021/acscentsci.7b00355":        "RetroSim (Coley 2017)",
    "10.1002/chem.201605499":            "Neuralsym (Segler & Waller 2017)",
    "10.1021/acs.jcim.9b00286":          "RDChiral (Coley 2019)",
    "10.1021/acs.jcim.0c00174":          "AiZynthFinder (Genheden 2020)",
    "10.1021/jacsau.1c00246":            "LocalRetro (Chen & Jung 2021)",   # JACS Au, not JCIM
    # --- rule bases / engines ---
    "10.1093/nar/gky940":                "RetroRules (Duigou 2019)",
    "10.1186/s13321-018-0295-6":         "Ambit-SMIRKS (Kochev 2018)",
    "10.3390/data3020014":               "RetroTransformDB (Avramova 2018)",
    "10.1021/acs.jcim.1c01192":          "Template size/canonicalisation (Heid 2021)",
}

OPENALEX = "https://api.openalex.org/works"

# ---------------------------------------------------------------- inclusion

# Работа попадает в рамку, если title+abstract содержит хотя бы один термин
# ИЗ КАЖДОЙ группы. Оба списка фиксируются заранее.
RULE_TERMS = [
    r"\brule[- ]based\b", r"\brules?\b", r"\btemplates?\b", r"\bsmirks\b",
    r"\bsmarts\b", r"\btransformation(s)?\b", r"\bbiotransformation(s)?\b",
    r"\breaction rules?\b",
]
REACH_TERMS = [
    r"\bcoverage\b", r"\brecall\b", r"\breach\b", r"\bapplicab", r"\bsensitivity\b",
    r"\bhit rate\b", r"\bretriev", r"\bfraction of (known|reference|observed)\b",
]

# Явные исключения: обзоры без собственных чисел, препринты-дубликаты, ретракции.
EXCLUDE_TYPES = {"review", "editorial", "erratum", "letter", "paratext"}

MIN_YEAR = 2015          # окно; фиксируется заранее
MAX_YEAR = 2026


def _rx(patterns):
    return re.compile("|".join(patterns), re.IGNORECASE)


RULE_RX = _rx(RULE_TERMS)
REACH_RX = _rx(REACH_TERMS)


def inverted_index_to_text(idx):
    """OpenAlex отдаёт abstract как inverted index. Собираем обратно."""
    if not idx:
        return ""
    positions = []
    for word, pos_list in idx.items():
        for p in pos_list:
            positions.append((p, word))
    positions.sort()
    return " ".join(w for _, w in positions)


def passes_inclusion(work):
    """Единственная точка решения о включении. Тестируется в --self-test."""
    if work.get("type") in EXCLUDE_TYPES:
        return False, "type"
    if work.get("is_retracted"):
        return False, "retracted"
    year = work.get("publication_year")
    if year is None or not (MIN_YEAR <= year <= MAX_YEAR):
        return False, "year"

    title = work.get("title") or ""
    abstract = inverted_index_to_text(work.get("abstract_inverted_index"))
    text = f"{title} {abstract}"
    if not text.strip():
        return False, "no_text"

    if not RULE_RX.search(text):
        return False, "no_rule_term"
    if not REACH_RX.search(text):
        return False, "no_reach_term"
    return True, "included"


# ---------------------------------------------------------------- fetching

def fetch_citations(seed_doi, mailto, session):
    """Все работы, цитирующие seed. Курсорная пагинация OpenAlex."""
    import requests  # локальный импорт: --self-test работает без него

    meta = session.get(f"{OPENALEX}/doi:{seed_doi}",
                       params={"mailto": mailto}, timeout=30)
    meta.raise_for_status()
    seed_id = meta.json()["id"].rsplit("/", 1)[-1]

    out, cursor = [], "*"
    while cursor:
        r = session.get(OPENALEX, params={
            "filter": f"cites:{seed_id}",
            "per-page": 200,
            "cursor": cursor,
            "mailto": mailto,
            "select": ("id,doi,title,publication_year,type,is_retracted,"
                       "abstract_inverted_index,primary_location,authorships"),
        }, timeout=60)
        r.raise_for_status()
        page = r.json()
        out.extend(page["results"])
        cursor = page["meta"].get("next_cursor")
        if not page["results"]:
            break
    return seed_id, out


def first_author(work):
    auths = work.get("authorships") or []
    if not auths:
        return ""
    name = (auths[0].get("author") or {}).get("display_name", "")
    return name.split()[-1] if name else ""


def venue(work):
    loc = work.get("primary_location") or {}
    src = loc.get("source") or {}
    return src.get("display_name", "") or ""


# ---------------------------------------------------------------- sampling

def draw_sample(pool, n, seed):
    """Детерминированная выборка. Сортировка по хешу id, потом seeded shuffle —
    так порядок не зависит от того, в каком порядке OpenAlex вернул страницы."""
    pool = sorted(pool, key=lambda w: hashlib.sha256(w["id"].encode()).hexdigest())
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


CODING_COLUMNS = [
    "paper_id", "doi", "year", "venue", "first_author", "seeds_cited",
    "C1_role",
    "C2_reach_reported", "C2_value", "C2_quote",
    "C3_attribution", "C3_quote",
    "C4a_hydrogen", "C4b_toolkit_version", "C4c_sanitisation",
    "C4d_normalisation", "C4e_depth", "C4f_matching_key", "C4_score",
    "C5_comparison", "C5_quote",
    "C6_diff_acknowledged",
    "C7_rules_released", "C7_code_released",
    "coder", "coding_date", "minutes", "notes",
]


def write_outputs(sample, frame, out_dir, args, excl_counts, resolved=None, missing=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "frame_full.jsonl", "w") as f:
        for w in frame:
            f.write(json.dumps(w) + "\n")

    with open(out / "coding_sheet.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=CODING_COLUMNS)
        wr.writeheader()
        for w in sample:
            wr.writerow({
                "paper_id": w["id"].rsplit("/", 1)[-1],
                "doi": (w.get("doi") or "").replace("https://doi.org/", ""),
                "year": w.get("publication_year", ""),
                "venue": venue(w),
                "first_author": first_author(w),
                "seeds_cited": ";".join(sorted(w.get("_seeds", []))),
            })

    manifest = {
        "snapshot_date": args.snapshot,
        "sample_seed": args.seed,
        "n_requested": args.n,
        "n_sampled": len(sample),
        "frame_size": len(frame),
        "seed_works": SEEDS,
        "inclusion": {
            "min_year": MIN_YEAR, "max_year": MAX_YEAR,
            "rule_terms": RULE_TERMS, "reach_terms": REACH_TERMS,
            "excluded_types": sorted(EXCLUDE_TYPES),
        },
        "exclusion_counts": excl_counts,
        # which seeds actually contributed, so the manifest cannot claim a seed set it did not use
        "seeds_resolved": resolved or {},
        "seeds_missing": missing or [],
    }
    with open(out / "frame_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"рамка: {len(frame)} работ, выборка: {len(sample)}")
    print("отсев:", ", ".join(f"{k}={v}" for k, v in sorted(excl_counts.items())))
    print(f"записано в {out}/")


# ---------------------------------------------------------------- self-test

FIXTURE = [
    # включается: есть rule-термин и reach-термин, тип article, год в окне
    {"id": "https://openalex.org/W1", "type": "article", "publication_year": 2022,
     "title": "A rule-based metabolite predictor",
     "abstract_inverted_index": {"Our": [0], "rules": [1], "achieve": [2],
                                 "coverage": [3], "of": [4], "80%": [5]}},
    # отсев: нет reach-термина
    {"id": "https://openalex.org/W2", "type": "article", "publication_year": 2022,
     "title": "Reaction templates for synthesis planning",
     "abstract_inverted_index": {"We": [0], "extract": [1], "templates": [2]}},
    # отсев: нет rule-термина
    {"id": "https://openalex.org/W3", "type": "article", "publication_year": 2021,
     "title": "Graph neural networks for property prediction",
     "abstract_inverted_index": {"We": [0], "improve": [1], "recall": [2]}},
    # отсев: обзор
    {"id": "https://openalex.org/W4", "type": "review", "publication_year": 2023,
     "title": "A review of rule-based coverage", "abstract_inverted_index": {}},
    # отсев: год вне окна
    {"id": "https://openalex.org/W5", "type": "article", "publication_year": 2009,
     "title": "SMIRKS rules and coverage", "abstract_inverted_index": {}},
    # отсев: ретракция
    {"id": "https://openalex.org/W6", "type": "article", "publication_year": 2020,
     "is_retracted": True, "title": "Rules and recall",
     "abstract_inverted_index": {"coverage": [0], "rules": [1]}},
    # включается: reach-термин только в abstract
    {"id": "https://openalex.org/W7", "type": "article", "publication_year": 2024,
     "title": "SMARTS transformations in silico",
     "abstract_inverted_index": {"The": [0], "applicability": [1], "domain": [2]}},
]


def self_test():
    ok = True

    got = {w["id"]: passes_inclusion(w) for w in FIXTURE}
    expect = {
        "https://openalex.org/W1": (True, "included"),
        "https://openalex.org/W2": (False, "no_reach_term"),
        "https://openalex.org/W3": (False, "no_rule_term"),
        "https://openalex.org/W4": (False, "type"),
        "https://openalex.org/W5": (False, "year"),
        "https://openalex.org/W6": (False, "retracted"),
        "https://openalex.org/W7": (True, "included"),
    }
    for wid, exp in expect.items():
        if got[wid] != exp:
            print(f"FAIL inclusion {wid}: получили {got[wid]}, ждали {exp}")
            ok = False

    txt = inverted_index_to_text({"metabolite": [1], "A": [0], "rule": [2]})
    if txt != "A metabolite rule":
        print(f"FAIL inverted index: {txt!r}")
        ok = False

    pool = [{"id": f"https://openalex.org/W{i}"} for i in range(500)]
    a = [w["id"] for w in draw_sample(pool, 50, 42)]
    b = [w["id"] for w in draw_sample(list(reversed(pool)), 50, 42)]
    if a != b:
        print("FAIL sampling: выборка зависит от порядка входа")
        ok = False
    c = [w["id"] for w in draw_sample(pool, 50, 43)]
    if a == c:
        print("FAIL sampling: разные seed дают одну выборку")
        ok = False
    if len(set(a)) != 50:
        print("FAIL sampling: дубликаты в выборке")
        ok = False

    print("self-test: OK" if ok else "self-test: ЕСТЬ ОШИБКИ")
    return 0 if ok else 1


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", help="дата снапшота YYYY-MM-DD, идёт в манифест")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--out", default="frame")
    ap.add_argument("--mailto", default="", help="почта для polite pool OpenAlex")
    ap.add_argument("--allow-missing-seeds", action="store_true",
                    help="build the frame even if a seed DOI does not resolve; the gap is "
                         "then recorded in the manifest instead of stopping the run")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.snapshot:
        ap.error("--snapshot обязателен (он замораживает рамку)")

    import requests
    session = requests.Session()

    by_id, excl, resolved, missing = {}, {}, {}, []
    for doi, label in SEEDS.items():
        try:
            seed_id, cits = fetch_citations(doi, args.mailto, session)
        except Exception as e:                       # noqa: BLE001
            print(f"! seed {label} ({doi}): {e}", file=sys.stderr)
            missing.append({"doi": doi, "label": label, "error": str(e)})
            continue
        resolved[doi] = seed_id
        print(f"{label}: {len(cits)} цитирующих")
        for w in cits:
            keep, why = passes_inclusion(w)
            if not keep:
                excl[why] = excl.get(why, 0) + 1
                continue
            rec = by_id.setdefault(w["id"], w)
            rec.setdefault("_seeds", set()).add(label)

    if missing and not args.allow_missing_seeds:
        print(f"\n{len(missing)} of {len(SEEDS)} seeds did not resolve:", file=sys.stderr)
        for m in missing:
            print(f"  {m['label']} ({m['doi']})", file=sys.stderr)
        print("The manifest records the seed list, so a frame built from the rest would be "
              "recorded as having used all of them. Fix the DOI, or pass --allow-missing-seeds "
              "to record the gap deliberately.", file=sys.stderr)
        return 2

    frame = list(by_id.values())
    for w in frame:
        w["_seeds"] = sorted(w.get("_seeds", []))
    sample = draw_sample(frame, args.n, args.seed)
    write_outputs(sample, frame, args.out, args, excl, resolved, missing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
