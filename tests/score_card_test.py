import unittest
from src.utils import score_card

class TestScoreCard(unittest.TestCase):
    
    def test_right_bauer(self):
        self.assertEqual(score_card('JH', 'H', 'H'), 13)
        self.assertEqual(score_card('JS', 'S', 'S'), 13)
        self.assertEqual(score_card('JD', 'D', 'D'), 13)
        self.assertEqual(score_card('JC', 'C', 'C'), 13)
    
    def test_right_bauer_no_suite(self):
        self.assertEqual(score_card('JH', 'H', 'D'), 13)
        self.assertEqual(score_card('JS', 'S', 'C'), 13)
        self.assertEqual(score_card('JD', 'D', 'H'), 13)
        self.assertEqual(score_card('JC', 'C', 'S'), 13)
    
    def test_left_bauer_suite(self):
        self.assertEqual(score_card('JD', 'H', 'H'), 12)
        self.assertEqual(score_card('JC', 'S', 'S'), 12)
        self.assertEqual(score_card('JH', 'D', 'D'), 12)
        self.assertEqual(score_card('JS', 'C', 'C'), 12)
    
    def test_left_bauer_no_suite(self):
        self.assertEqual(score_card('JD', 'H', 'D'), 12)
        self.assertEqual(score_card('JC', 'S', 'C'), 12)
        self.assertEqual(score_card('JH', 'D', 'H'), 12)
        self.assertEqual(score_card('JS', 'C', 'S'), 12)
    
    def test_non_trump_spade_start(self):
        self.assertEqual(score_card('9S', 'H', 'S'), 1)
        self.assertEqual(score_card('0S', 'H', 'S'), 2)
        self.assertEqual(score_card('JS', 'H', 'S'), 3)
        self.assertEqual(score_card('QS', 'H', 'S'), 4)
        self.assertEqual(score_card('KS', 'H', 'S'), 5)
        self.assertEqual(score_card('AS', 'H', 'S'), 6)
    
    def test_non_trump_club_start(self):
        self.assertEqual(score_card('9C', 'H', 'C'), 1)
        self.assertEqual(score_card('0C', 'H', 'C'), 2)
        self.assertEqual(score_card('JC', 'H', 'C'), 3)
        self.assertEqual(score_card('QC', 'H', 'C'), 4)
        self.assertEqual(score_card('KC', 'H', 'C'), 5)
        self.assertEqual(score_card('AC', 'H', 'C'), 6)
    
    def test_non_trump_heart_start(self):
        self.assertEqual(score_card('9H', 'S', 'H'), 1)
        self.assertEqual(score_card('0H', 'S', 'H'), 2)
        self.assertEqual(score_card('JH', 'S', 'H'), 3)
        self.assertEqual(score_card('QH', 'S', 'H'), 4)
        self.assertEqual(score_card('KH', 'S', 'H'), 5)
        self.assertEqual(score_card('AH', 'S', 'H'), 6)
        
    def test_non_trump_diamond_start(self):
        self.assertEqual(score_card('9D', 'C', 'D'), 1)
        self.assertEqual(score_card('0D', 'C', 'D'), 2)
        self.assertEqual(score_card('JD', 'C', 'D'), 3)
        self.assertEqual(score_card('QD', 'C', 'D'), 4)
        self.assertEqual(score_card('KD', 'C', 'D'), 5)
        self.assertEqual(score_card('AD', 'C', 'D'), 6)
    
    def test_non_trump_non_suite(self):
        pass
    
    def test_trump_non_suite(self):
        pass
    
    def test_trump_suite(self):
        pass
        
if __name__ == '__main__':
    unittest.main()