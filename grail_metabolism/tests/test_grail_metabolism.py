import unittest
import torch
from grail_metabolism.utils.preparation import MolFrame
from grail_metabolism.model.generator import Generator, generate_vectors
from grail_metabolism.model.filter import Filter
from grail_metabolism.model.grail import summon_the_grail
from torch_geometric.data import Data


class TestGrailMetabolism(unittest.TestCase):

    def test_molframe_initialization(self):
        """Test initializing MolFrame with valid and invalid data."""
        valid_data = {'sub': ['C(C(=O)O)N', 'CC(=O)O'], 'prod': ['CC(=O)N', 'CCO'], 'real': [1, 0]}
        molframe = MolFrame(map=valid_data)
        self.assertIsInstance(molframe, MolFrame)

        invalid_data = {'sub': [], 'prod': [], 'real': []}
        with self.assertRaises(ValueError):
            MolFrame(map=invalid_data)

    def test_summon_the_grail(self):
        """Test the summon_the_grail function."""
        rules = ["[C:1][H:2]>>[C:1][O][H:2]", "[c:1][H:2]>>[c:1][O][H:2]"]
        node_dim = (10, 10)
        edge_dim = (5, 5)
        model = summon_the_grail(rules, node_dim, edge_dim)
        self.assertIsNotNone(model.generator, "Generator should not be None")
        self.assertIsNotNone(model.filter, "Filter should not be None")
        self.assertEqual(len(model.generator.rules), len(rules), "Rules count mismatch")

    def test_generate_vectors(self):
        """Test the generate_vectors function."""
        reaction_dict = {
            "substrate1": {"product1": [0, 1], "product2": [2]},
            "substrate2": {"product3": [3]}
        }
        real_products_dict = {"substrate1": {"product1"}, "substrate2": {"product3"}}
        len_of_rules = 5

        vectors = generate_vectors(reaction_dict, real_products_dict, len_of_rules)

        self.assertEqual(len(vectors), 2, "Vectors should be generated for 2 substrates")
        self.assertEqual(vectors["substrate1"], [1, 1, 0, 0, 0], "Vector for substrate1 is incorrect")
        self.assertEqual(vectors["substrate2"], [0, 0, 0, 1, 0], "Vector for substrate2 is incorrect")

    def test_filter_initialization(self):
        """Test the initialization of the Filter class."""
        in_channels = 10
        edge_dim = 6
        arg_vec = [32, 64, 128, 256, 512, 1024]
        filter_model = Filter(in_channels, edge_dim, arg_vec, mode='pair')
        self.assertIsNotNone(filter_model.module, "Filter module should be initialized")

    def test_filter_forward_pair(self):
        """Test the forward method of the Filter class in 'pair' mode."""
        in_channels = 10
        edge_dim = 6
        arg_vec = [32, 64, 128, 256, 512, 1024]
        filter_model = Filter(in_channels, edge_dim, arg_vec, mode='pair')

        # Mock batch data
        data = Data(
            x=torch.rand(10, in_channels),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.rand(20, edge_dim),
            batch=torch.zeros(10, dtype=torch.long),
            fp=torch.rand(10, 256 * 2)
        )
        output = filter_model(data, 'pass')
        self.assertEqual(output.shape, (10, 1), "Output shape mismatch for 'pair' mode")


if __name__ == "__main__":
    unittest.main()