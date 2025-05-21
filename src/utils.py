
def score_card(card: str, trump: str, starting_suite: str):
    trump_convert = {'9': 7, '0': 8, 'Q': 9, 'K': 10, 'A': 11}
    non_trump_convert = {'9': 1, '0': 2, 'J': 3, 'Q': 4, 'K': 5, 'A': 6}
    
    #score Jack card seperate, since it has the possibillity of being big points
    if card[0] == 'J':
        bauer = get_bauer_type(card, trump)
        if bauer == 3:
            return 13
        elif bauer == 2:
            return 12
        elif bauer == 1:
            return non_trump_convert[card[0]] if card[1] == starting_suite else 0
    
    #score any other card that is not a jack
    if not (card[1] == starting_suite or card[1] == trump):
        return 0 
    elif card[1] == trump:
        return trump_convert[card[0]]
    elif card[1] == starting_suite:
        return non_trump_convert[card[0]]
        

def get_bauer_type(card: str, trump: str):
    other_bauer = {'H': 'D', 'S': 'C', 'C': 'S', 'D': 'H'}
    if card[1] == trump:
        return 3
    elif other_bauer[card[1]] == trump:
        return 2
    else:
        return 1
        