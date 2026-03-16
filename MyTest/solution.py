from playerTest import Player
from boardTest import HexBoard
import time
import math
import random
from collections import deque

MINIMAX_MAX_N  = 9      
TIME_BUDGET_P1 = 4.3    
TIME_BUDGET_P2 = 4.4  
EMERGENCY_CUT  = 0.03   
UCB_C          = 1.414  
TT_MAX_SIZE    = 200_000 

W_PATH      = 1.0
W_VIRTUAL   = 0.4
W_VIRTUAL_O = 0.3

def mcts_threshold(n):
    if n > 13: return 2
    if n > 11: return 3
    return 4

def get_neighbors(r: int, c: int, n: int) -> list:
    neighbors = [(r, c-1), (r, c+1)]
    if r % 2 == 0:
        neighbors += [(r-1, c-1), (r-1, c), (r+1, c-1), (r+1, c)]
    else:
        neighbors += [(r-1, c), (r-1, c+1), (r+1, c), (r+1, c+1)]
    return [(nr, nc) for nr, nc in neighbors if 0 <= nr < n and 0 <= nc < n]

class MCTSNode:
    __slots__ = ["move", "parent", "children", "wins", "visits",
                 "untried_moves", "player_who_moved"]

    def __init__(self, move, parent, untried_moves, player_who_moved):
        self.move             = move
        self.parent           = parent
        self.children         = []
        self.wins             = 0.0
        self.visits           = 0
        self.untried_moves    = untried_moves
        self.player_who_moved = player_who_moved

    def ucb1(self, c=UCB_C) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits +
                c * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self, c=UCB_C):
        return max(self.children, key=lambda n: n.ucb1(c))

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

class SmartPlayer(Player):

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.player_id   = player_id
        self._opponent   = 2 if player_id == 1 else 1
        self._start_time = 0.0
        self._time_limit = TIME_BUDGET_P1
        self._n          = 0
        self._move_count = 0

        self._tt           = {}
        self._zobrist_n    = 0
        self._zobrist      = None
        self._current_hash = 0

    def play(self, board: HexBoard) -> tuple:
        self._start_time = time.time()
        self._time_limit = (TIME_BUDGET_P1 if self.player_id == 1
                            else TIME_BUDGET_P2)
        self._n = board.size

        self._init_zobrist()
        self._current_hash = self._compute_hash(board)

        if self._move_count <= 1 or self._current_hash == 0:
            self._tt.clear()
            self._move_count = 0
        self._move_count += 1

        center = self._n // 2
        if self._move_count <= 2 and board.board[center][center] == 0:
            return (center, center)

        if self._n <= MINIMAX_MAX_N:
            return self._play_minimax(board)
        else:
            return self._play_mcts(board)

    def _play_minimax(self, board: HexBoard) -> tuple:
        early = self._move_count <= 3
        self._max_depth = 4 if early else 6
        beam            = 8 if early else 10

        candidates = self._order_moves(board)
        if not candidates:
            return self._fallback(board)

        return self._idab(board, candidates[:beam], candidates[0])

    def _idab(self, board: HexBoard, candidates: list, fallback) -> tuple:
        best_move = fallback

        for depth in range(1, self._max_depth + 1):
            if self._time_remaining() < self._time_limit * 0.15:
                break

            best_val    = float('-inf')
            depth_best  = best_move
            alpha, beta = float('-inf'), float('inf')

            for move in candidates:
                if self._time_remaining() < self._time_limit * 0.10:
                    break
                clone = board.clone()
                clone.place_piece(move[0], move[1], self.player_id)
                h_child = self._update_hash(
                    self._current_hash, move[0], move[1], self.player_id)
                val = self._alphabeta(clone, h_child, depth-1,
                                      alpha, beta, False)
                if val > best_val:
                    best_val = val; depth_best = move
                alpha = max(alpha, best_val)

            best_move = depth_best

        return best_move

    def _alphabeta(self, board: HexBoard, h: int, depth: int, alpha: float, beta: float, maximizing: bool) -> float:
        if board.check_connection(self.player_id):  return  1_000_000.0
        if board.check_connection(self._opponent):  return -1_000_000.0

        if depth == 0 or self._time_remaining() < EMERGENCY_CUT:
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
                h_c = self._update_hash(h, move[0], move[1], self.player_id)
                val   = max(val, self._alphabeta(clone, h_c, depth-1,
                                                 alpha, beta, False))
                alpha = max(alpha, val)
                if val >= beta: break
        else:
            val = float('inf')
            for move in moves:
                clone = board.clone()
                clone.place_piece(move[0], move[1], self._opponent)
                h_c = self._update_hash(h, move[0], move[1], self._opponent)
                val  = min(val, self._alphabeta(clone, h_c, depth-1,
                                                alpha, beta, True))
                beta = min(beta, val)
                if val <= alpha: break

        self._tt_put(h, val, depth)
        return val

    def _play_mcts(self, board: HexBoard) -> tuple:
        moves = self._order_moves(board)
        if not moves:
            return self._fallback(board)
        if len(moves) == 1:
            return moves[0]

        root = MCTSNode(
            move             = None,
            parent           = None,
            untried_moves    = list(moves),
            player_who_moved = self._opponent
        )

        simulations = 0

        while self._time_remaining() > EMERGENCY_CUT:

            node       = root
            sim_board  = board.clone()
            sim_player = self.player_id

            while node.is_fully_expanded() and node.children:
                node = node.best_child(UCB_C)
                sim_board.place_piece(node.move[0], node.move[1],
                                      node.player_who_moved)
                sim_player = 2 if node.player_who_moved == 1 else 1
                if (sim_board.check_connection(1) or
                        sim_board.check_connection(2)):
                    break

            if (node.untried_moves and not
                    (sim_board.check_connection(1) or
                     sim_board.check_connection(2))):
                move = node.untried_moves.pop(
                    random.randint(0, len(node.untried_moves) - 1))
                sim_board.place_piece(move[0], move[1], sim_player)
                child = MCTSNode(
                    move             = move,
                    parent           = node,
                    untried_moves    = self._get_empty_cells(sim_board),
                    player_who_moved = sim_player
                )
                node.children.append(child)
                node       = child
                sim_player = 2 if sim_player == 1 else 1

            result = self._bfs_evaluate(sim_board)
            simulations += 1

            while node is not None:
                node.visits += 1
                if result == self.player_id:
                    node.wins += 1.0
                elif result == 0:
                    node.wins += 0.5
                node = node.parent

        if not root.children:
            return moves[0]

        best = max(root.children, key=lambda n: n.visits)
        return best.move

    def _bfs_evaluate(self, board: HexBoard) -> int:
        if board.check_connection(self.player_id): return self.player_id
        if board.check_connection(self._opponent): return self._opponent

        max_dist  = float(self._n * self._n)
        dist_mine = self._bfs_distance(board, self.player_id)
        dist_opp  = self._bfs_distance(board, self._opponent)

        if dist_mine == float('inf'): dist_mine = max_dist
        if dist_opp  == float('inf'): dist_opp  = max_dist

        diff      = dist_opp - dist_mine   
        threshold = mcts_threshold(self._n)

        if diff >= threshold:   return self.player_id
        elif diff <= -threshold: return self._opponent
        else:
            if dist_mine < dist_opp:   return self.player_id
            elif dist_opp < dist_mine: return self._opponent
            else:                      return 0

    def _evaluate(self, board: HexBoard) -> float:
        max_dist = float(self._n * self._n)
        raw_mine = self._bfs_distance(board, self.player_id)
        raw_opp  = self._bfs_distance(board, self._opponent)

        dist_mine = raw_mine if raw_mine != float('inf') else max_dist
        dist_opp  = raw_opp  if raw_opp  != float('inf') else max_dist

        score = W_PATH * (dist_opp - dist_mine)

        if self._n <= 11:
            vc_mine = self._count_vc(board, self.player_id)
            vc_opp  = self._count_vc(board, self._opponent)
            score  += W_VIRTUAL * vc_mine - W_VIRTUAL_O * vc_opp

        aggression = 0.7 if self.player_id == 1 else 0.5
        return score * (0.5 + aggression)

    def _bfs_distance(self, board: HexBoard, player_id: int) -> float:
        n   = self._n
        INF = float('inf')
        opp = 2 if player_id == 1 else 1

        dist = [[INF]*n for _ in range(n)]
        dq   = deque()

        if player_id == 1:
            sources = [(r, 0) for r in range(n)]
            goal_fn = lambda r, c: c == n-1
        else:
            sources = [(0, c) for c in range(n)]
            goal_fn = lambda r, c: r == n-1

        for r, c in sources:
            cell = board.board[r][c]
            if cell == opp: continue
            cost = 0 if cell == player_id else 1
            if cost < dist[r][c]:
                dist[r][c] = cost
                if cost == 0: dq.appendleft((cost, r, c))
                else:         dq.append((cost, r, c))

        while dq:
            cost, r, c = dq.popleft()
            if cost > dist[r][c]: continue
            if goal_fn(r, c): return cost
            for nr, nc in get_neighbors(r, c, n):
                cell = board.board[nr][nc]
                if cell == opp: continue
                step     = 0 if cell == player_id else 1
                new_cost = cost + step
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    if step == 0: dq.appendleft((new_cost, nr, nc))
                    else:         dq.append((new_cost, nr, nc))

        return INF

    def _count_vc(self, board: HexBoard, player_id: int) -> int:
        n    = self._n
        grid = board.board
        count = 0
        seen  = set()

        own = [(r,c) for r in range(n) for c in range(n)
               if grid[r][c] == player_id]

        for (r1, c1) in own:
            nb1 = set(get_neighbors(r1, c1, n))
            for (m1r, m1c) in nb1:
                if grid[m1r][m1c] != 0: continue
                for (r2, c2) in get_neighbors(m1r, m1c, n):
                    if (r2,c2)==(r1,c1) or grid[r2][c2]!=player_id: continue
                    bridge = frozenset([(r1,c1),(r2,c2)])
                    if bridge in seen: continue
                    nb2    = set(get_neighbors(r2, c2, n))
                    shared = nb1 & nb2
                    for (m2r, m2c) in shared:
                        if (m2r,m2c)!=(m1r,m1c) and grid[m2r][m2c]==0:
                            seen.add(bridge); count+=1; break

        return count

    def _order_moves(self, board: HexBoard) -> list:
        n    = self._n
        grid = board.board
        mid  = n / 2.0

        scored = []
        for r in range(n):
            for c in range(n):
                if grid[r][c] != 0: continue
                own_nb = opp_nb = 0
                for nr, nc in get_neighbors(r, c, n):
                    if grid[nr][nc] == self.player_id:  own_nb += 1
                    elif grid[nr][nc] == self._opponent: opp_nb += 1
                score  = own_nb * 3.0 + opp_nb * 2.0
                score += 1.5 * (n - abs(r-mid) - abs(c-mid))
                scored.append((score, r, c))

        scored.sort(key=lambda x: -x[0])
        return [(r, c) for (_, r, c) in scored]

    def _init_zobrist(self):
        if self._zobrist_n == self._n: return
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
        h = 0
        for r, row in enumerate(board.board):
            for c, val in enumerate(row):
                if val: h ^= self._zobrist[val][r][c]
        return h

    def _update_hash(self, h: int, r: int, c: int, pid: int) -> int:
        return h ^ self._zobrist[pid][r][c]

    def _tt_get(self, h: int, depth: int):
        e = self._tt.get(h)
        return e[0] if e and e[1] >= depth else None

    def _tt_put(self, h: int, score: float, depth: int):
        if len(self._tt) >= TT_MAX_SIZE:
            evict_n = TT_MAX_SIZE // 4
            shallow = [k for k, v in self._tt.items() if v[1] <= 1]
            victims = (random.sample(shallow, evict_n)
                       if len(shallow) >= evict_n
                       else random.sample(list(self._tt.keys()),
                                          min(evict_n, len(self._tt))))
            for k in victims: del self._tt[k]
        self._tt[h] = (score, depth)

    def _get_empty_cells(self, board: HexBoard) -> list:
        n = self._n
        return [(r,c) for r in range(n) for c in range(n)
                if board.board[r][c] == 0]

    def _fallback(self, board: HexBoard) -> tuple:
        for r in range(self._n):
            for c in range(self._n):
                if board.board[r][c] == 0: return (r, c)

    def _time_remaining(self) -> float:
        return self._time_limit - (time.time() - self._start_time)
