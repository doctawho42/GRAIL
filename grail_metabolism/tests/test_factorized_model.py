import pandas as pd
import torch
from rdkit import Chem

from grail_metabolism.utils.transform import from_rdmol
from grail_metabolism.model.factorized import FactorizedGenerator
from grail_metabolism.model.factorized_data import build_factorized_dataset
from grail_metabolism.utils.preparation import MolFrame

RULE = "[CH2:1][OH:2]>>[CH:1]=[O:2]"


def test_head_shapes():
    data = from_rdmol(Chem.MolFromSmiles("CCO"))
    model = FactorizedGenerator(num_types=50)
    tl = model.type_logits(data)
    sl = model.site_logits(data)
    assert tl.shape[-1] == 50
    assert sl.numel() == data.x.size(0)


def test_factorized_fit_reduces_loss():
    # Mirrors test_puloss_trains_on_logits: a tiny 2-substrate toy dataset, assert the
    # combined (type + site) BCE loss actually moves over a few epochs.
    frame = MolFrame(pd.DataFrame([
        {"sub": "CCO", "prod": "CC=O", "real": 1},
        {"sub": "CCCO", "prod": "CCC=O", "real": 1},
    ]))
    frame.full_setup(rules=[RULE], include_pair_graphs=False, include_morgan=False)
    rule_to_type = {RULE: 0}
    catalog = {
        RULE: {
            "count": 2,
            "source_pairs": [["CCO", "CC=O"], ["CCCO", "CCC=O"]],
        }
    }
    dataset = build_factorized_dataset(frame, rule_to_type, catalog=catalog)
    assert len(dataset) == 2

    torch.manual_seed(0)
    model = FactorizedGenerator(num_types=1, hidden_dims=(16, 16), out_dim=8)
    history = model.fit(dataset, epochs=15, lr=1e-2, batch_size=2)

    assert len(history) == 15
    assert history[-1] < history[0]
