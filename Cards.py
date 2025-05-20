import numpy as np
import random
import copy

class Deck:
    def __init__(self) -> None:
        self.deck = ['AS','KS','QS','JS','0S','9S','AC','KC','QC','JC','0C','9C',
                     'AH','KH','QH','JH','0H','9H','AD','KD','QD','JD','0D','9D']
        self._master_copy = ['AS','KS','QS','JS','0S','9S','AC','KC','QC','JC','0C','9C',
                     'AH','KH','QH','JH','0H','9H','AD','KD','QD','JD','0D','9D']
    
    def shuffle(self) -> None:
        np.random.shuffle(self.deck)
    
    def draw(self, num: int) -> list[str]:
        cards: list[str] = [''] 
        for i in range(num):
            cards.append(self.deck.pop(0))
        return cards
    
    def reset(self) -> None:
        self.deck = copy.deepcopy(self._master_copy)
        
    