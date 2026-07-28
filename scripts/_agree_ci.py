"""Bootstrap the inter-curation agreement and check the population it is compared against."""
import json, re, sys, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from rdkit import Chem, RDLogger
from grail_metabolism.metrics import _tautomer_inchikey
RDLogger.DisableLog('rdApp.*')
R = Path(__file__).resolve().parents[1]

raw = (R/'docs/benchmark/data/gloryx_test.json').read_text()
data = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw))
def flat(ns):
    o = []
    for n in ns or []:
        if n.get('smiles'): o.append(n['smiles'])
        o.extend(flat(n.get('metabolites')))
    return o
gl = {p['smiles']: flat(p.get('metabolites')) for p in data if p.get('smiles')}

def tk(s):
    try: return _tautomer_inchikey(s)
    except Exception: return None

corp = {}
for split in ('train', 'val', 'test'):
    sdf, tri = R/f'grail_metabolism/data/{split}.sdf', R/f'grail_metabolism/data/{split}_triples_clean.txt'
    if not sdf.exists(): continue
    by = {}
    for m in Chem.SDMolSupplier(str(sdf)):
        if m is None: continue
        p = m.GetPropsAsDict(); i = str(p.get('Index','')); s = p.get('SMILES') or Chem.MolToSmiles(m)
        if i and s: by[i] = s
    for line in open(tri):
        a = line.split()
        if len(a) == 3 and a[2] == '1':
            s, mm = by.get(a[0]), by.get(a[1])
            if s and mm: corp.setdefault(s, set()).add(mm)

cbk = {}
for s in corp:
    k = tk(s)
    if k: cbk.setdefault(k, s)

truth = json.loads((R/'results/test_references.json').read_text())
test_keys = {tk(s) for s in truth}
test_keys.discard(None)

vals, in_test = [], 0
for g in gl:
    k = tk(g)
    if not k or k not in cbk: continue
    if k in test_keys: in_test += 1
    a = {x for x in (tk(y) for y in gl[g]) if x}
    b = {x for x in (tk(y) for y in corp[cbk[k]]) if x}
    if a and b: vals.append((len(a & b)/len(a), len(a & b)/len(b), len(a & b)/len(a | b)))

v = np.array(vals)
print(f'{len(v)} shared parents; {in_test} of them lie in the TEST split')
print('  -> the 0.585 headline is a full-test figure on 1,170 substrates, so it is NOT computed')
print('     on these; the comparison is between populations and is reported as such')
rng = np.random.default_rng(0)
idx = rng.integers(0, len(v), (10000, len(v)))
out = {}
for j, lab in enumerate(('gloryx_recovered_by_corpus', 'corpus_recovered_by_gloryx', 'jaccard')):
    bt = v[:, j][idx].mean(axis=1)
    lo, hi = float(np.quantile(bt, .025)), float(np.quantile(bt, .975))
    out[lab] = {'mean': round(float(v[:, j].mean()), 4), 'ci95': [round(lo, 4), round(hi, 4)]}
    print(f'  {lab:28} {v[:,j].mean():.3f} [{lo:.3f}, {hi:.3f}]')
out['n_shared'] = len(v); out['n_in_test_split'] = in_test
(R/'results/annotation_agreement_ci.json').write_text(json.dumps(out, indent=1))
print('\nwrote results/annotation_agreement_ci.json')
