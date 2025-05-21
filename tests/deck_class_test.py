import unittest
from src.Cards import Deck

class TestDeckClass(unittest.TestCase):
    def test_shuffle(self):
        test_instance = Deck()
        previous = test_instance._master_copy
        test_instance.shuffle()
        new = test_instance.deck
        self.assertNotEqual(previous, new)
        
    def test_draw_single(self):
        test_instance = Deck()
        cards = test_instance.draw(1)
        self.assertEqual(len(cards), 1)
    
    def test_draw_multiple(self):
        test_instance = Deck()
        cards = test_instance.draw(3)
        self.assertEqual(len(cards), 3)
        
    def test_reset_from_shuffle(self):
        test_instance = Deck()
        previous = test_instance._master_copy
        test_instance.shuffle()
        test_instance.reset()
        new = test_instance.deck
        self.assertEqual(previous, new)
    
    def test_reset_from_draw(self):
        test_instance = Deck()
        previous = test_instance._master_copy
        test_instance.shuffle()
        test_instance.draw(5)
        test_instance.reset()
        new = test_instance.deck
        self.assertEqual(previous, new)

if __name__ == '__main__':
    unittest.main()