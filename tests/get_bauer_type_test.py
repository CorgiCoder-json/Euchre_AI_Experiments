import unittest
from src.utils import get_bauer_type

class TestBauerType(unittest.TestCase):
    def test_right_bauer(self):
        self.assertEqual(get_bauer_type('JH', 'H'), 3)
        self.assertEqual(get_bauer_type('JD', 'D'), 3)
        self.assertEqual(get_bauer_type('JS', 'S'), 3)
        self.assertEqual(get_bauer_type('JC', 'C'), 3)
    
    def test_left_bauer(self):
        self.assertEqual(get_bauer_type('JH', 'D'), 2)
        self.assertEqual(get_bauer_type('JD', 'H'), 2)
        self.assertEqual(get_bauer_type('JS', 'C'), 2)
        self.assertEqual(get_bauer_type('JC', 'S'), 2)
        
    def test_no_bauer(self):
        self.assertEqual(get_bauer_type('JH', 'S'), 1)
        self.assertEqual(get_bauer_type('JD', 'C'), 1)
        self.assertEqual(get_bauer_type('JS', 'H'), 1)
        self.assertEqual(get_bauer_type('JC', 'D'), 1)

if __name__ == "__main__":
    unittest.main()