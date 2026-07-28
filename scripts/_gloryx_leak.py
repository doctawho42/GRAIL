"""Is the GLORYx external set actually external to GRAIL's training data?

The paper treats the 37 GLORYx drugs as an external hold-out. If those drugs appear in the training
split of this repository's corpus, GRAIL was trained on them and the external replication is not
external. Checks each GLORYx parent against each split under the tautomer-aware key.
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rdkit import Chem, RDLogger
from grail_metabolism.metrics import _tautomer_inchikey
RDLogger.DisableLog('rdApp.*')
R = Path(__file__).resolve().parents[1]

raw = (R/'docs/benchmark/data/gloryx_test.json').read_text()
data = json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw))
parents = [p['smiles'] for p in data if p.get('smiles')]

def tk(s):
    try: return _tautomer_inchikey(s)
    except Exception: return None

gk = {}
for p in parents:
    k = tk(p)
    if k: gk.setdefault(k, p)
print(f'GLORYx parents: {len(parents)}, keyed: {len(gk)}')

for split in ('train', 'val', 'test'):
    sdf, tri = R/f'grail_metabolism/data/{split}.sdf', R/f'grail_metabolism/data/{split}_triples_clean.txt'
    if not sdf.exists():
        print(f'  {split}: missing'); continue
    subs = set()
    with open(tri) as fh:
        for line in fh:
            a = line.split()
            if len(a) == 3: subs.add(a[0])
    by = {}
    for m in Chem.SDMolSupplier(str(sdf)):
        if m is None: continue
        p = m.GetPropsAsDict(); i = str(p.get('Index', ''))
        if i in subs:
            s = p.get('SMILES') or Chem.MolToSmiles(m)
            k = tk(s) if s else None
            if k: by[k] = s
    hit = sorted(gk.keys() & by.keys())
    print(f'  {split}: {len(by)} substrates, {len(hit)} of the {len(gk)} GLORYx parents present')
