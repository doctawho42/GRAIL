import torch
from rdkit import Chem

from grail_metabolism.utils.transform import from_rdmol
from grail_metabolism.model.factorized import FactorizedGenerator


def test_head_shapes():
    data = from_rdmol(Chem.MolFromSmiles("CCO"))
    model = FactorizedGenerator(num_types=50)
    tl = model.type_logits(data)
    sl = model.site_logits(data)
    assert tl.shape[-1] == 50
    assert sl.numel() == data.x.size(0)
