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
    return [(nr, nc) for nr, nc in neighbors if 0 <= nr < n and 0 <= nc < n]

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

        self._move_count = 0

        self._tt: dict = {}

        self._zobrist_n    = 0
        self._zobrist      = None
        self._current_hash = 0   #

    def play(self, board: HexBoard) -> tuple:
        self._start_time = time.time()
        self._time_limit = self.profile.time_budget
        self._n          = board.size

        self._configure_regime()
        self._init_zobrist()   

        self._current_hash = self._compute_hash(board)
        if self._move_count <= 1 or self._current_hash == 0:
            self._tt.clear()
            self._move_count = 0
        self._move_count += 1

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

        self._zobrist_n    = n
        self._current_hash = 0
        self._tt.clear()

    def _compute_hash(self, board: HexBoard) -> int:
        h   = 0
        zob = self._zobrist
        for r, row in enumerate(board.board):
            for c, val in enumerate(row):
                if val:
                    h ^= zob[val][r][c]
        return h

    def _update_hash(self, h: int, r: int, c: int, player_id: int) -> int:
        return h ^ self._zobrist[player_id][r][c]

    def _tt_get(self, h: int, depth: int):
        entry = self._tt.get(h)
        if entry and entry[1] >= depth:
            return entry[0]
        return None

    def _tt_put(self, h: int, score: float, depth: int):
        if len(self._tt) >= TT_MAX_SIZE:
            evict_n = TT_MAX_SIZE // 4
            shallow = [k for k, v in self._tt.items() if v[1] <= 1]
            if len(shallow) >= evict_n:
                victims = random.sample(shallow, evict_n)
            else:
                victims = random.sample(list(self._tt.keys()),
                                        min(evict_n, len(self._tt)))
            for k in victims:
                del self._tt[k]
        self._tt[h] = (score, depth)

    def _opening_move(self, board: HexBoard):
        center = self._n // 2
        if self._move_count <= 2 and board.board[center][center] == 0:
            return (center, center)
        return None

    def _iterative_deepening_ab(self, board: HexBoard, candidates: list, fallback) -> tuple:
        best_move = fallback

        n = self._n
        if n <= REGIME_A_MAX_N:
            beam = BEAM_A
        elif n <= REGIME_B_MAX_N:
            beam = BEAM_B
        else:
            beam = BEAM_C
        root_candidates = candidates[:beam]

        for depth in range(1, self._max_depth + 1):
            if self._time_remaining() < self._time_limit * 0.15:
                break

            best_val    = float('-inf')
            depth_best  = best_move
            alpha, beta = float('-inf'), float('inf')

            for move in root_candidates:
                if self._time_remaining() < self._time_limit * 0.10:
                    break

                clone = board.clone()
                clone.place_piece(move[0], move[1], self.player_id)

                h_child = self._update_hash(
                    self._current_hash, move[0], move[1], self.player_id
                )
                val = self._alphabeta(clone, h_child, depth - 1,
                                      alpha, beta, False)

                if val > best_val:
                    best_val   = val
                    depth_best = move
                alpha = max(alpha, best_val)

            best_move = depth_best

        return best_move

    def _alphabeta(self, board: HexBoard, h: int, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        if board.check_connection(self.player_id):
            return 1_000_000.0
        if board.check_connection(self.profile.opponent):
            return -1_000_000.0

        if depth == 0 or self._time_remaining() < 0.05:
            return self._evaluate(board)

        cached = self._tt_get(h, depth)
        if cached is not None:
            return cached

        moves = self._order_moves(board)
        if not moves:
            return self._evaluate(board)

        if maximizing:
            val = float('-inf')
            for move in moves:
                clone = board.clone()
                clone.place_piece(move[0], move[1], self.player_id)
                h_child = self._update_hash(h, move[0], move[1], self.player_id)
                val   = max(val, self._alphabeta(clone, h_child, depth - 1,
                                                 alpha, beta, False))
                alpha = max(alpha, val)
                if val >= beta:
                    break   
        else:
            val = float('inf')
            for move in moves:
                clone = board.clone()
                clone.place_piece(move[0], move[1], self.profile.opponent)
                h_child = self._update_hash(h, move[0], move[1],
                                            self.profile.opponent)
                val  = min(val, self._alphabeta(clone, h_child, depth - 1,
                                                alpha, beta, True))
                beta = min(beta, val)
                if val <= alpha:
                    break   

        self._tt_put(h, val, depth)
        return val

    def _evaluate(self, board: HexBoard) -> float:
        my_id    = self.player_id
        opp_id   = self.profile.opponent
        max_dist = float(self._n * self._n)   

        raw_mine = self._bfs_distance(board, my_id)
        raw_opp  = self._bfs_distance(board, opp_id)

        dist_mine = raw_mine if raw_mine != float('inf') else max_dist
        dist_opp  = raw_opp  if raw_opp  != float('inf') else max_dist

        score = W_PATH * (dist_opp - dist_mine)

        if self._eval_mode == 'RICH':
            vc_mine = self._count_virtual_connections(board, my_id)
            vc_opp  = self._count_virtual_connections(board, opp_id)
            score  += W_VIRTUAL * vc_mine - W_VIRTUAL_O * vc_opp

        return score * (0.5 + self.profile.aggression)

    def _bfs_distance(self, board: HexBoard, player_id: int) -> float:
        n   = self._n
        INF = float('inf')
        opp = 2 if player_id == 1 else 1

        dist = [[INF] * n for _ in range(n)]
        dq   = deque()

        if player_id == 1:
            sources = [(r, 0) for r in range(n)]
            goal_fn = lambda r, c: c == n - 1
        else:
            sources = [(0, c) for c in range(n)]
            goal_fn = lambda r, c: r == n - 1

        for r, c in sources:
            cell = board.board[r][c]
            if cell == opp:
                continue
            cost = 0 if cell == player_id else 1
            if cost < dist[r][c]:
                dist[r][c] = cost
                if cost == 0:
                    dq.appendleft((cost, r, c))
                else:
                    dq.append((cost, r, c))

        while dq:
            cost, r, c = dq.popleft()
            if cost > dist[r][c]:
                continue
            if goal_fn(r, c):
                return cost   
            for nr, nc in get_neighbors(r, c, n):
                cell = board.board[nr][nc]
                if cell == opp:
                    continue
                step     = 0 if cell == player_id else 1
                new_cost = cost + step
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    if step == 0:
                        dq.appendleft((new_cost, nr, nc))
                    else:
                        dq.append((new_cost, nr, nc))

        return INF

    def _count_virtual_connections(self, board: HexBoard, player_id: int) -> int:
        n            = self._n
        grid         = board.board
        count        = 0
        seen_bridges: set = set()

        own_cells = [
            (r, c)
            for r in range(n) for c in range(n)
            if grid[r][c] == player_id
        ]

        for (r1, c1) in own_cells:
            nb1 = set(get_neighbors(r1, c1, n))
            for (m1r, m1c) in nb1:
                if grid[m1r][m1c] != 0:
                    continue
                for (r2, c2) in get_neighbors(m1r, m1c, n):
                    if (r2, c2) == (r1, c1):
                        continue
                    if grid[r2][c2] != player_id:
                        continue
                    bridge = frozenset([(r1, c1), (r2, c2)])
                    if bridge in seen_bridges:
                        continue
                    nb2    = set(get_neighbors(r2, c2, n))
                    shared = nb1 & nb2
                    for (m2r, m2c) in shared:
                        if (m2r, m2c) != (m1r, m1c) and grid[m2r][m2c] == 0:
                            seen_bridges.add(bridge)
                            count += 1
                            break

        return count

    def _order_moves(self, board: HexBoard) -> list:
        n    = self._n
        grid = board.board
        mid  = n / 2.0

        scored = []
        for r in range(n):
            for c in range(n):
                if grid[r][c] != 0:
                    continue

                score          = 0.0
                own_nb = opp_nb = 0

                for nr, nc in get_neighbors(r, c, n):
                    if grid[nr][nc] == self.player_id:
                        own_nb += 1
                    elif grid[nr][nc] == self.profile.opponent:
                        opp_nb += 1

                score += own_nb * 3.0
                score += opp_nb * 2.0
                score += self.profile.center_bias * (n - abs(r - mid) - abs(c - mid))

                scored.append((score, r, c))

        scored.sort(key=lambda x: -x[0])
        return [(r, c) for (_, r, c) in scored]

    def _time_remaining(self) -> float:
        return self._time_limit - (time.time() - self._start_time)