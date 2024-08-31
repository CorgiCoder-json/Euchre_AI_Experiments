#########################################################################################
#
# Programmer: Jacob Maurer
# Date: 7/25/2024
# Description: AIs play the card game Euchre in pytorch and deck of cards api
#
#########################################################################################
import requests
import copy
from DQN import *

deck_id = "rc5o65e15oms"

# Player class represents a singular player of Euchre
class Player:
    def __init__(self, id, team_id, starter):
        self.self_id = id
        self.team = team_id
        self.cards = []
        self.suites = []
        self.values = []
        self.is_starter = starter
        self.brain = DQNManager(128, .99, .9, .05, 1000,
                                0.005, 1e-4, 5, [0,1,2,3,4], 'cuda', 10000)
    # sets the cards of the player
    def set_cards(self, new_cards):
        self.cards = new_cards
        self.suites = [card[1] for card in new_cards]
        self.values = [card[0] for card in new_cards]

    # returns a card that the player has chosen to play
    def play_card(self, suite):
        # implement brain system
        print(self.cards)
        card = int(input("Enter Card: ")) - 1
        while card >= len(self.cards) or card < 0:
            print("Invalid Index!")
            card = int(input("Enter Card: ")) - 1
        plays_card = self.cards[card]
        while suite in self.suites and not (plays_card[1] == suite):
            print("Play the Correct Suite!")
            card = int(input("Enter Card: ")) - 1
            plays_card = self.cards[card]
        print("Player " + str(self.self_id) + " played the card: " + str(plays_card))
        self.cards.remove(plays_card)
        self.values.pop(card)
        self.suites.pop(card)
        return plays_card


# Round Manager class that manages each round of Euchre
class RoundManager:
    def __init__(self, players):
        self.team_one_score = 0
        self.team_two_score = 0
        self.current_pile = []
        self.team_called = 0
        self.players = players
        self.trump = None
        self.winner = None
        self.card_conversion_no_trump = {"9": 1, "0": 2, "J": 3, "Q": 4, "K": 5, "A": 6}
        self.card_conversion_trump = {"9": 7, "0": 8, "Q": 9, "K": 10, "A": 11}
        self.starter_index = 0

    def draw_phase(self):
        # loop through all players and give them cards
        for p in self.players:
            p.set_cards([card["code"] for card in
                         requests.get("https://www.deckofcardsapi.com/api/deck/" + deck_id + "/draw/?count=5").json()[
                             "cards"]])
            self.trump = \
            requests.get("https://www.deckofcardsapi.com/api/deck/" + deck_id + "/draw/?count=1").json()["cards"][0][
                "code"]

    def bet_phase(self):
        possible_bets = ['S', 'D', 'C', 'H']

        # Check to see if anyone wants to pick up the card
        for p in range(len(self.players)):
            hand_info = self.has_trump(self.players[p].cards, self.trump)
            if hand_info["Number"] >= 2:
                index_delete = self.players[0].cards.index(random.choice(self.players[0].cards))
                self.players[0].cards[index_delete] = self.trump
                self.trump = self.trump
                self.team_called = self.players[p].team
                return self.trump

        # remove the flipped over card suite
        possible_bets.remove(self.trump[1])

        # check to see for other bets
        for p in self.players:
            for bet in possible_bets:
                hand_info = self.has_trump(p.cards, bet)
                if hand_info["Number"] >= 3:
                    self.trump = bet
                    self.team_called = p.team
                    return self.trump
        self.trump = ''
        return self.trump

    def trick_phase(self):
        turn_order = copy.deepcopy(self.players)
        tricks_team_one = 0
        tricks_team_two = 0
        for i in range(5):
            starting_suite = ''

            # Play the cards in order
            for p in range(len(turn_order)):
                played_card = turn_order[p].play_card(starting_suite)
                if len(self.current_pile) == 0:
                    starting_suite = played_card[1]
                self.add_card(played_card, turn_order[p], p)

            winning_player = None
            max_card = 0

            # check who won
            for card in self.current_pile:
                card_val = self.card_to_value(card["Card"], self.has_trump([card["Card"]], self.trump)["Trump"],
                                              starting_suite)
                if card_val > max_card:
                    winning_player = card["Player"]
                    max_card = card_val
            tricks_team_one += 1 if winning_player.team == 1 else 0
            tricks_team_two += 1 if winning_player.team == 2 else 0

            # make new turn order
            turn_order[0].is_starter = False
            while not (turn_order[0].self_id == winning_player.self_id):
                turn_order.append(turn_order.pop(0))
            turn_order[0].is_starter = True

            # reset current pile
            self.current_pile.clear()
        return {"Team_1": tricks_team_one, "Team_2": tricks_team_two,
                "Winner": 1 if tricks_team_one > tricks_team_two else 2}

    def score_round(self, trick_scores):
        if trick_scores["Winner"] == 1:
            self.team_one_score += 2 if trick_scores["Team_1"] == 5 or self.team_called == 2 else 1
        else:
            self.team_two_score += 2 if trick_scores["Team_2"] == 5 or self.team_called == 1 else 1

    def reset(self):
        self.players[self.starter_index].is_starter = False
        self.starter_index += 1
        self.players[self.starter_index % 5].is_starter = True
        self.players.append(self.players.pop(0))
        self.current_pile.clear()

    def add_card(self, new_card, player, order):
        self.current_pile.append({"Card": new_card, "Player": player, "Order": order})

    # converts a card code to a value for easier scoring
    def card_to_value(self, card, is_trump, suit):
        if is_trump:
            if card[0] == 'J' and card[1] == self.trump:
                return 13
            elif self.other_bauer(card[1], self.trump):
                return 12
            else:
                return self.card_conversion_trump[str(card[0])]
        elif card[1] == suit:
            return self.card_conversion_no_trump[str(card[0])]
        return 0

    # checks if the jack is the other bauer
    def other_bauer(self, jack_code, trump_id):
        if trump_id == 'D' and jack_code == 'H':
            return True
        elif trump_id == 'H' and jack_code == 'D':
            return True
        elif trump_id == 'D' and jack_code == 'H':
            return True
        elif trump_id == 'H' and jack_code == 'D':
            return True
        return False

    # checks a set of cards to see if it has trump, and returns how much
    def has_trump(self, cards, trump_id):
        num = 0
        trump = False
        for card in cards:
            if card[1] == trump_id:
                num += 1
                trump = True
            if card[0] == 'J':
                if self.other_bauer(trump_id, card[1]):
                    num += 1
                    trump = True
        return {"Trump": trump, "Number": num}


def run_game():
    p_1 = Player(1, 1, True)
    p_2 = Player(2, 2, False)
    p_3 = Player(3, 1, False)
    p_4 = Player(4, 2, False)
    teams = [p_1, p_2, p_3, p_4]
    manager = RoundManager(teams)
    for i in range(3):
        requests.get("https://www.deckofcardsapi.com/api/deck/" + deck_id + "/shuffle/")
        manager.draw_phase()
        for p in manager.players:
            print(p.cards)
        manager.bet_phase()
        if manager.trump == '':
            print("Bad Hand! Return")
        else:
            print(manager.trump)
            results = manager.trick_phase()
            manager.score_round(results)
        manager.reset()
        requests.get("https://www.deckofcardsapi.com/api/deck/" + deck_id + "/return/")
    print(manager.team_one_score)
    print(manager.team_two_score)

    return 0


