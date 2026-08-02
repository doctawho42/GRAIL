"""Is similarity-to-training confounded with chemistry difficulty?

The transfer stratification found SyGMa degrading as fast as GRAIL, which it cannot do by transfer
since it has no training set. This checks the confound directly on a quantity that no learning
touches: the fraction of a substrate's references its rule bank can reach at all. If bank coverage
falls with distance from the training set, the axis is chemistry difficulty and no transfer claim
can rest on it.

This printed its table to stdout and wrote nothing for as long as the appendix quoted four of its
numbers, so none of them was checkable against the record. It writes the table now. Note what the
written table shows that the paper's two-endpoint contrast does not: mean reference-set size across
the five strata is not monotone in similarity, so "reference sets shrink the same way" is a
statement about the endpoints and the appendix now says so.
"""
import json, numpy as np
from pathlib import Path
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
R = Path(__file__).resolve().parents[1]
fac = {r['sub']: r for r in json.loads((R/'results/recall_factorization.json').read_text())['per_substrate']}
truth = json.loads((R/'results/test_references.json').read_text())
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
def fp(s):
    m = Chem.MolFromSmiles(s)
    return gen.GetFingerprint(m) if m else None
sup = Chem.SDMolSupplier(str(R/'grail_metabolism/data/train.sdf'))
tr = [fp(m.GetPropsAsDict().get('SMILES') or Chem.MolToSmiles(m)) for m in sup
      if m is not None and str(m.GetPropsAsDict().get('State', '')) == 'Substrate']
tr = [f for f in tr if f is not None]
rows = []
for s, r in fac.items():
    if s not in truth or not truth[s] or not r['U']:
        continue
    f = fp(s)
    if f is None:
        continue
    rows.append((max(DataStructs.BulkTanimotoSimilarity(f, tr)), r['Cfull']/r['U'], r['U']))
sim = np.array([x[0] for x in rows]); cov = np.array([x[1] for x in rows]); nref = np.array([x[2] for x in rows])
print(f'{len(rows)} substrates\n')
print(f"{'stratum':>16}{'n':>6}{'bank coverage':>16}{'|refs|':>9}")
B = [0, .3, .4, .5, .6, 1.01]
strata = []
for i in range(len(B)-1):
    m = (sim >= B[i]) & (sim < B[i+1])
    if m.sum() < 20: continue
    strata.append({'lo': B[i], 'hi': B[i+1], 'n': int(m.sum()),
                   'bank_coverage': round(float(cov[m].mean()), 4),
                   'mean_n_refs': round(float(nref[m].mean()), 2)})
    print(f'  [{B[i]:.2f},{B[i+1]:.2f}){m.sum():6}{cov[m].mean():16.4f}{nref[m].mean():9.2f}')
slope = float(np.polyfit(sim, cov, 1)[0]); corr = float(np.corrcoef(sim, nref)[0, 1])
print(f'\nslope of BANK COVERAGE on similarity: {slope:+.4f}')
print(f'corr(similarity, |refs|): {corr:+.3f}')

import subprocess
def _git(*a):
    try:
        return subprocess.run(['git', *a], cwd=R, capture_output=True, text=True,
                              timeout=10).stdout.strip() or None
    except Exception:
        return None
OUT = R / 'results' / 'transfer_confound.json'
OUT.write_text(json.dumps({
    'config': {'script': 'transfer_confound.py', 'git_commit': _git('rev-parse', 'HEAD'),
               'git_dirty': bool(_git('status', '--porcelain')),
               'substrate_source': 'results/recall_factorization.json per_substrate',
               'fingerprint': 'Morgan r=2, 2048 bits, max Tanimoto to any train substrate',
               'bins': B},
    'n': len(rows), 'strata': strata,
    'slope_bank_coverage_on_similarity': round(slope, 4),
    'corr_similarity_n_refs': round(corr, 3),
    'note': 'mean |refs| is not monotone across the strata; the appendix contrasts the endpoints',
}, indent=1))
print(f'wrote {OUT}')
