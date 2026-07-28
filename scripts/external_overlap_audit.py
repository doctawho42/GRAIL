#!/usr/bin/env python3
"""Audit every set this paper calls external for overlap with GRAIL's training data.

The GLORYx contamination -- 19 of 37 drugs inside the training split -- was found by accident while
measuring something else. A reviewer's next question is what else was not checked, so this runs the
same check over every external or third-party set the paper reports on, and prints the answer for
each rather than for the one that happened to be caught.
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rdkit import Chem, RDLogger
from grail_metabolism.metrics import _tautomer_inchikey
RDLogger.DisableLog('rdApp.*')
R = Path(__file__).resolve().parents[1]


def tk(s):
    try:
        return _tautomer_inchikey(s)
    except Exception:
        return None


def trained_keys():
    keys = set()
    for split in ('train', 'val'):
        sdf, tri = R/f'grail_metabolism/data/{split}.sdf', R/f'grail_metabolism/data/{split}_triples_clean.txt'
        if not sdf.exists():
            continue
        ids = set()
        with open(tri) as fh:
            for line in fh:
                a = line.split()
                if len(a) == 3:
                    ids.add(a[0])
        for m in Chem.SDMolSupplier(str(sdf)):
            if m is None:
                continue
            p = m.GetPropsAsDict()
            if str(p.get('Index', '')) in ids:
                s = p.get('SMILES') or Chem.MolToSmiles(m)
                k = tk(s) if s else None
                if k:
                    keys.add(k)
    return keys


def gloryx():
    raw = (R/'docs/benchmark/data/gloryx_test.json').read_text()
    d = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw))
    return [p['smiles'] for p in d if p.get('smiles')]


def shared_subset():
    """The 150-substrate multi-method subset: drawn from the test split, so external only to training."""
    f = R/'artifacts/tier2/metatrans_preds.json'
    return list(json.loads(f.read_text())) if f.exists() else []


def main():
    seen = trained_keys()
    print(f'{len(seen)} substrates seen in training or validation\n')
    sets = {'GLORYx external set': gloryx(), 'shared 150-substrate subset': shared_subset()}
    out = {}
    for name, subs in sets.items():
        if not subs:
            print(f'{name}: not available'); continue
        keys = [tk(s) for s in subs]
        n_ok = sum(1 for k in keys if k)
        hit = sum(1 for k in keys if k and k in seen)
        out[name] = {'n': len(subs), 'keyed': n_ok, 'in_train_or_val': hit,
                     'fraction': round(hit/n_ok, 3) if n_ok else None}
        print(f'{name}: {hit}/{n_ok} substrates were in training or validation ({hit/n_ok:.1%})')
    (R/'results/external_overlap_audit.json').write_text(json.dumps(out, indent=1))
    print('\nwrote results/external_overlap_audit.json')


if __name__ == '__main__':
    main()
