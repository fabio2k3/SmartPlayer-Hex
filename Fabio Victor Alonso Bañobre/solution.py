from player import Player
from board import HexBoard
import time
import heapq
import random
from collections import deque

REGIME_A_MAX_N = 9    
REGIME_B_MAX_N = 13   
REGIME_C_MIN_N = 14   

TT_MAX_SIZE = 200_000   

BEAM_A = 8   
BEAM_B = 7   
BEAM_C = 6    

W_PATH      = 1.0   
W_VIRTUAL   = 0.4   
W_VIRTUAL_O = 0.3   

def get_neighbors(r: int, c: int, n: int) -> list:
    neighbors = [(r, c - 1), (r, c + 1)]
    if r % 2 == 0:
        neighbors += [(r - 1, c - 1), (r - 1, c),
                      (r + 1, c - 1), (r + 1, c)]
    else:
        neighbors += [(r - 1, c),     (r - 1, c + 1),
                      (r + 1, c),     (r + 1, c + 1)]

class PlayerProfile:

    def __init__(self, player_id: int):
        self.player_id = player_id
        self.opponent  = 2 if player_id == 1 else 1

        if player_id == 1:
            self.time_budget = 4.0
            self.center_bias = 1.5
            self.aggression  = 0.7
        else:
            self.time_budget = 4.2
            self.center_bias = 1.2
            self.aggression  = 0.5

class SmartPlayer(Player):

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.profile     = PlayerProfile(player_id)
        self._start_time = 0.0
        self._time_limit = 4.0   
        self._n          = 0
        self._eval_mode  = 'FAST'
        self._max_depth  = 2

        self._tt: dict = {}

        self._zobrist_n = 0
        self._zobrist   = None
