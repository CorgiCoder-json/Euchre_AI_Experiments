import unittest
from src.utils import score_card

class TestScoreCard(unittest.TestCase):
    
    def test_right_bauer(self):
        self.assertEqual(score_card('JH', 'H', 'H'), 13)
        self.assertEqual(score_card('JS', 'S', 'S'), 13)
        self.assertEqual(score_card('JD', 'D', 'D'), 13)
        self.assertEqual(score_card('JC', 'C', 'C'), 13)
        self.assertEqual(score_card('JH', 'H', 'D'), 13)
        self.assertEqual(score_card('JS', 'S', 'C'), 13)
        self.assertEqual(score_card('JD', 'D', 'H'), 13)
        self.assertEqual(score_card('JC', 'C', 'S'), 13)
        
if __name__ == '__main__':
    unittest.main()