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

    def play(self, board: HexBoard) -> tuple:
        self._start_time = time.time()
        self._time_limit = self.profile.time_budget   #
        self._n          = board.size

        self._configure_regime()
        self._init_zobrist()   

        occupied = sum(
            board.board[r][c] != 0
            for r in range(self._n) for c in range(self._n)
        )
        if occupied <= 1:
            self._tt.clear()

        opening = self._opening_move(board)
        if opening:
            return opening

        candidates = self._order_moves(board)
        if not candidates:
            for r in range(self._n):
                for c in range(self._n):
                    if board.board[r][c] == 0:
                        return (r, c)

        return self._iterative_deepening_ab(board, candidates, candidates[0])

    def _configure_regime(self):
        n = self._n
        if n <= REGIME_A_MAX_N:
            self._eval_mode = 'RICH'
            self._max_depth = 4
        elif n <= REGIME_B_MAX_N:
            self._eval_mode = 'RICH' if n <= 11 else 'FAST'
            self._max_depth = 3
        else:
            self._eval_mode = 'FAST'
            self._max_depth = 2


    def _init_zobrist(self):
        if self._zobrist_n == self._n:
            return
        n   = self._n
        rng = random.Random(0xDEADBEEF)   
        self._zobrist = [
            [[rng.getrandbits(64) for _ in range(n)] for _ in range(n)]
            for _ in range(3)
        ]
        self._zobrist_n = n
        self._tt.clear()

    def _board_hash(self, board: HexBoard) -> int:
        h   = 0
        zob = self._zobrist
        for r, row in enumerate(board.board):
            for c, val in enumerate(row):
                if val:
                    h ^= zob[val][r][c]
        return h
