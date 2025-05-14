from grail_metabolism.utils.transform import from_rule
import typing as tp
from .generator import Generator
from .filter import Filter
from .wrapper import ModelWrapper

def summon_the_grail(rules: tp.List[str], node_dim: tp.Tuple[int, int], edge_dim: tp.Tuple[int, int]) -> ModelWrapper:
    r"""
    Create the ready-to-use model with generator and filter parts
    :param rules: list of SMARTS rules to use
    :param node_dim: a tuple with two integers: (node_dim_generator, node_dim_filter)
    :param edge_dim: a tuple with two integers: (edge_dim_generator, edge_dim_filter)
    :return: ready-to-use model
    """
    rule_dict = {}
    for rule in rules:
        rule_dict[rule] = from_rule(rule)
    generator = Generator(rule_dict, node_dim[0], edge_dim[0])
    arg_vec = [400] * 6
    filter = Filter(node_dim[1], edge_dim[1], arg_vec, mode='pair')
    return ModelWrapper(filter, generator)
