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
        self.assertEqual(score_card('9D', 'S', 'D'), 1)
        self.assertEqual(score_card('0D', 'S', 'D'), 2)
        self.assertEqual(score_card('JD', 'S', 'D'), 3)
        self.assertEqual(score_card('QD', 'S', 'D'), 4)
        self.assertEqual(score_card('KD', 'S', 'D'), 5)
        self.assertEqual(score_card('AD', 'S', 'D'), 6)
    
    def test_non_trump_non_suite_heart(self):
        self.assertEqual(score_card('9H', 'S', 'C'), 0)
        self.assertEqual(score_card('0H', 'S', 'C'), 0)
        self.assertEqual(score_card('JH', 'S', 'C'), 0)
        self.assertEqual(score_card('QH', 'S', 'C'), 0)
        self.assertEqual(score_card('KH', 'S', 'C'), 0)
        self.assertEqual(score_card('AH', 'S', 'C'), 0)
    
    def test_non_trump_non_suite_diamond(self):
        self.assertEqual(score_card('9D', 'S', 'C'), 0)
        self.assertEqual(score_card('0D', 'S', 'C'), 0)
        self.assertEqual(score_card('JD', 'S', 'C'), 0)
        self.assertEqual(score_card('QD', 'S', 'C'), 0)
        self.assertEqual(score_card('KD', 'S', 'C'), 0)
        self.assertEqual(score_card('AD', 'S', 'C'), 0)
    
    def test_non_trump_non_suite_club(self):
        self.assertEqual(score_card('9C', 'H', 'D'), 0)
        self.assertEqual(score_card('0C', 'H', 'D'), 0)
        self.assertEqual(score_card('JC', 'H', 'D'), 0)
        self.assertEqual(score_card('QC', 'H', 'D'), 0)
        self.assertEqual(score_card('KC', 'H', 'D'), 0)
        self.assertEqual(score_card('AC', 'H', 'D'), 0)
    
    def test_non_trump_non_suite_spade(self):
        self.assertEqual(score_card('9S', 'H', 'D'), 0)
        self.assertEqual(score_card('0S', 'H', 'D'), 0)
        self.assertEqual(score_card('JS', 'H', 'D'), 0)
        self.assertEqual(score_card('QS', 'H', 'D'), 0)
        self.assertEqual(score_card('KS', 'H', 'D'), 0)
        self.assertEqual(score_card('AS', 'H', 'D'), 0)
    
    def test_trump_heart_non_suite(self):
        self.assertEqual(score_card('9H', 'H', 'D'), 7)
        self.assertEqual(score_card('0H', 'H', 'D'), 8)
        self.assertEqual(score_card('QH', 'H', 'D'), 9)
        self.assertEqual(score_card('KH', 'H', 'D'), 10)
        self.assertEqual(score_card('AH', 'H', 'D'), 11)
    
    def test_trump_diamond_non_suite(self):
        self.assertEqual(score_card('9D', 'D', 'H'), 7)
        self.assertEqual(score_card('0D', 'D', 'H'), 8)
        self.assertEqual(score_card('QD', 'D', 'H'), 9)
        self.assertEqual(score_card('KD', 'D', 'H'), 10)
        self.assertEqual(score_card('AD', 'D', 'H'), 11)
    
    def test_trump_club_non_suite(self):
        self.assertEqual(score_card('9C', 'C', 'S'), 7)
        self.assertEqual(score_card('0C', 'C', 'S'), 8)
        self.assertEqual(score_card('QC', 'C', 'S'), 9)
        self.assertEqual(score_card('KC', 'C', 'S'), 10)
        self.assertEqual(score_card('AC', 'C', 'S'), 11)
    
    def test_trump_spade_non_suite(self):
        self.assertEqual(score_card('9S', 'S', 'C'), 7)
        self.assertEqual(score_card('0S', 'S', 'C'), 8)
        self.assertEqual(score_card('QS', 'S', 'C'), 9)
        self.assertEqual(score_card('KS', 'S', 'C'), 10)
        self.assertEqual(score_card('AS', 'S', 'C'), 11)
    
    def test_trump_diamond_suite(self):
        self.assertEqual(score_card('9D', 'D', 'D'), 7)
        self.assertEqual(score_card('0D', 'D', 'D'), 8)
        self.assertEqual(score_card('QD', 'D', 'D'), 9)
        self.assertEqual(score_card('KD', 'D', 'D'), 10)
        self.assertEqual(score_card('AD', 'D', 'D'), 11)
    
    def test_trump_heart_suite(self):
        self.assertEqual(score_card('9H', 'H', 'H'), 7)
        self.assertEqual(score_card('0H', 'H', 'H'), 8)
        self.assertEqual(score_card('QH', 'H', 'H'), 9)
        self.assertEqual(score_card('KH', 'H', 'H'), 10)
        self.assertEqual(score_card('AH', 'H', 'H'), 11)
    
    def test_trump_club_suite(self):
        self.assertEqual(score_card('9C', 'C', 'C'), 7)
        self.assertEqual(score_card('0C', 'C', 'C'), 8)
        self.assertEqual(score_card('QC', 'C', 'C'), 9)
        self.assertEqual(score_card('KC', 'C', 'C'), 10)
        self.assertEqual(score_card('AC', 'C', 'C'), 11)
        
    def test_trump_spade_suite(self):
        self.assertEqual(score_card('9S', 'S', 'S'), 7)
        self.assertEqual(score_card('0S', 'S', 'S'), 8)
        self.assertEqual(score_card('QS', 'S', 'S'), 9)
        self.assertEqual(score_card('KS', 'S', 'S'), 10)
        self.assertEqual(score_card('AS', 'S', 'S'), 11)
        
if __name__ == '__main__':
    unittest.main()