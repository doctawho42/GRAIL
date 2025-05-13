from grail_metabolism.utils.transform import from_rule
import typing as tp
from .generator import Generator
from .filter import Filter
from .wrapper import ModelWrapper

def summon_the_grail(rules: tp.List[str], node_dim, edge_dim):
    rule_dict = {}
    for rule in rules:
        rule_dict[rule] = from_rule(rule)
    generator = Generator(rule_dict)
    arg_vec = [400] * 6
    filter = Filter(node_dim, edge_dim, arg_vec, mode='pair')
    return ModelWrapper(filter, generator)
