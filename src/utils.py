
def score_card(card: str, trump: str, starting_suite: str):
    trump_convert = {'9': 7, '0': 8, 'Q': 9, 'K': 10, 'A': 11}
    non_trump_convert = {'9': 1, '0': 2, 'J': 3, 'Q': 4, 'K': 5, 'A': 6}
    
    #score the bauer
    if card[0] == 'J':
        return 13 if get_bauer == 0 else 12
    
    #score any other card
    if not (card[1] == starting_suite or card[1] == trump):
        return 0 
    elif card[1] == trump:
        return trump_convert[card[0]]
    elif card[1] == starting_suite:
        return non_trump_convert[card[0]]
        

def get_bauer(card, trump):
    if card[]