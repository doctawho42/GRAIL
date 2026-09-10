# Training the support-gated rule identity on Kaggle

The intervention (docs/RULE_REPRESENTATION_PREREGISTRATION.md) needs retraining: id_gate_lambda = 0
is the deployed model, and the gate is a positive lambda. This trains the registered pair --
lambda = 0 and lambda = 8, identical but for the gate -- on a Kaggle GPU, so the comparison is the
gate alone.

## What the two configs are

`configs/rulegate/rulegate_baseline.yaml`  (id_gate_lambda = 0.0, the retrained baseline)
`configs/rulegate/rulegate_lambda8.yaml`   (id_gate_lambda = 8.0, the intervention)

They share seed 42 and every other setting, so a difference between their results is the gate. The
baseline is retrained rather than read from the deployed checkpoint, because the deployed one was a
different run and would confound the gate with training noise.

## One-time: upload the data as a private Kaggle dataset

The splits are gitignored and ~1.4 GB. Create a private dataset (call it `grail-data`) holding, at
the paths the configs name relative to a data root:

    train.sdf  train_triples_clean.txt
    val.sdf    val_triples_clean.txt
    test.sdf   test_triples_clean.txt
    extended_smirks.txt        (the rule bank, from grail_metabolism/resources/)

## The notebook

GPU on. Kaggle sessions are <= 12 h and each arm is a pretrain (30 ep) plus a generator (20 ep) and
a filter (20 ep), so run one arm per session if they do not both fit; the second arm reuses nothing
from the first, so order does not matter.

```python
# 1. code
!git clone https://github.com/doctawho42/GRAIL && cd GRAIL && pip install -e . -q

# 2. data: link the uploaded dataset to where the configs look
import os, pathlib
root = pathlib.Path("GRAIL/grail_metabolism/data"); root.mkdir(parents=True, exist_ok=True)
src = pathlib.Path("/kaggle/input/grail-data")
for a, b in [("train.sdf","train.sdf"), ("train_triples_clean.txt","train_triples_clean.txt"),
             ("val.sdf","val.sdf"), ("val_triples_clean.txt","val_triples_clean.txt"),
             ("test.sdf","test.sdf"), ("test_triples_clean.txt","test_triples_clean.txt")]:
    os.symlink(src/a, root/b)
os.symlink(src/"extended_smirks.txt", "GRAIL/grail_metabolism/resources/extended_smirks.txt")

# 3. train -- one arm, or both if the session allows
%cd GRAIL
!grail run-config configs/rulegate/rulegate_baseline.yaml
!grail run-config configs/rulegate/rulegate_lambda8.yaml

# 4. persist the checkpoints and the metrics run-config printed
!mkdir -p /kaggle/working/rulegate
!cp -r artifacts/rulegate_baseline/checkpoints /kaggle/working/rulegate/baseline_ckpt
!cp -r artifacts/rulegate_lambda8/checkpoints  /kaggle/working/rulegate/lambda8_ckpt
!find artifacts -name 'metrics.json' -path '*rulegate*' -exec cp --parents {} /kaggle/working/rulegate/ \;
```

Download `/kaggle/working/rulegate/`.

## Back here, after the run

Put the two `generator.pt` under `artifacts/rulegate_baseline/checkpoints/` and
`artifacts/rulegate_lambda8/checkpoints/`, then check the prediction:

```bash
# mechanism check: the id's variance share must fall and the graph's must rise
python scripts/rule_embed_decomposition.py --gen artifacts/rulegate_baseline/checkpoints/generator.pt --out results/decomp_baseline.json
python scripts/rule_embed_decomposition.py --gen artifacts/rulegate_lambda8/checkpoints/generator.pt  --out results/decomp_lambda8.json
```

The primary test (recall@5 on the comparison set, gate vs baseline) and the guardrail (recall@15
not below baseline) are read from the two `metrics.json` for the first signal on the eval split;
the strict comparison-set reading, against the five published comparators, is a separate pooling
run once the eval-split signal is positive, using the same producers as the rest of this work.

What the preregistration commits to, in one line each: recall@5(lambda=8) > recall@5(lambda=0);
the graph variance share rises from ~0.17; recall@15 does not fall. All three are reported whether
or not they hold.
