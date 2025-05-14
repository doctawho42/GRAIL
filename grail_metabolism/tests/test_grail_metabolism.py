import unittest
from grail_metabolism.utils.preparation import MolFrame
from grail_metabolism.model.generator import Generator
from grail_metabolism.model.filter import Filter

class TestGrailMetabolism(unittest.TestCase):
    
    def test_molframe_initialization(self):
        """Test initializing MolFrame with valid and invalid data."""
        valid_data = {'sub': ['C(C(=O)O)N', 'CC(=O)O'], 'prod': ['CC(=O)N', 'CCO'], 'real': [1, 0]}
        molframe = MolFrame(map=valid_data)
        self.assertIsInstance(molframe, MolFrame)
        
        invalid_data = {'sub': [], 'prod': [], 'real': []}
        with self.assertRaises(ValueError):
            MolFrame(map=invalid_data)

    def test_generator_class(self):
        """Basic test for the generator class."""
        # This would require mock data or using predefined molecular structures
        # Assuming a setup of a known generator here
        pass

    def test_filter_class(self):
        """Basic test for the filter class."""
        # Similar to the generator class, would require mock data or a predefined condition
        pass

if __name__ == "__main__":
    unittest.main()

