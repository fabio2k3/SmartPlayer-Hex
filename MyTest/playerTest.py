class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def play(self, board) -> tuple:
        raise NotImplementedError("¡Implementa este método!")
