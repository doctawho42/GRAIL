"""Is similarity-to-training confounded with chemistry difficulty?

The transfer stratification found SyGMa degrading as fast as GRAIL, which it cannot do by transfer
since it has no training set. This checks the confound directly on a quantity that no learning
touches: the fraction of a substrate's references its rule bank can reach at all. If bank coverage
falls with distance from the training set, the axis is chemistry difficulty and no transfer claim
can rest on it.
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
for i in range(len(B)-1):
    m = (sim >= B[i]) & (sim < B[i+1])
    if m.sum() < 20: continue
    print(f'  [{B[i]:.2f},{B[i+1]:.2f}){m.sum():6}{cov[m].mean():16.4f}{nref[m].mean():9.2f}')
print(f'\nslope of BANK COVERAGE on similarity: {np.polyfit(sim, cov, 1)[0]:+.4f}')
print(f'corr(similarity, |refs|): {np.corrcoef(sim, nref)[0,1]:+.3f}')
