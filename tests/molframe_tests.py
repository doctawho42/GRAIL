import sys
sys.path.append('..')
from grail.utils.preparation import MolFrame
import signal
from grail.model.filter import Filter
import torch

import logging

logging.basicConfig(level=logging.DEBUG)

def handler(signum, frame):
    raise TimeoutError

signal.signal(signal.SIGALRM, handler)
triples = MolFrame.read_triples('../grail/data/val_triples.txt')
test = MolFrame.from_file('../grail/data/val.sdf', triples)

def test_setup() -> None:
    logging.debug("Starting full setup")
    test.full_setup()
    logging.debug("Completed full setup")

@torch.no_grad()
def test_model_pairs() -> None:
    logging.debug("Initializing model")
    model = Filter(12)
    model.eval()
    signal.alarm(10)
    try:
        logging.debug("Training model pairs")
        test.train_pairs(model, test)
        logging.debug("Completed training model pairs")
    except TimeoutError:
        pass

def test_permute() -> None:
    logging.debug("Starting permute augmentation")
    test.permute_augmentation()
    logging.debug("Completed permute augmentation")

def test_metabolize() -> None:
    test_rule = ['[c:1][CH1:2]([H])=[O:3]>>[c:1][C:2](O)=[O:3]']
    logging.debug("Starting metabolize with opt mode")
    test.metabolize(test_rule, mode='opt')
    logging.debug("Completed metabolize with opt mode")
    logging.debug("Starting metabolize with gen mode")
    test.metabolize(test_rule, mode='gen')
    logging.debug("Completed metabolize with gen mode")
