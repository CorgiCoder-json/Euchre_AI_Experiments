class Player:
    def __init__(self, self_id: int, team_id: int, brain):
        self.id: int = self_id
        self.team: int = team_id
        self.brain = brain
        self.hand: list[str] = ['']
    def play_card(self):
        pass
        