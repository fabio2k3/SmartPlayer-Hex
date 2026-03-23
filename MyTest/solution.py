from player import Player
from board import HexBoard

import math
import random
import time
from collections import deque

MINIMAX_MAX_N = 11

TT_MAX_SIZE = 180_000

EXACT       = 0
LOWER_BOUND = 1
UPPER_BOUND = 2

W_PATH      = 1.00
W_VIRTUAL   = 0.35
W_CONNECT   = 0.20
W_BRIDGE    = 0.15
W_CENTER    = 0.08

HISTORY_SCALE = 0.0007

UCB_C    = 0.8

BONUS_MY_LM    = 10_000
BONUS_OPP_LM   =  9_000
BONUS_MY_PATH  =  5_000
BONUS_OPP_PATH =  4_000

TIME_MARGIN        = 0.03
EARLY_STOP_MARGIN  = 0.12

def get_neighbors(r: int, c: int, n: int) -> list:
    neighbors = [(r, c - 1), (r, c + 1)]
    if r % 2 == 0:
        neighbors += [(r-1, c-1), (r-1, c), (r+1, c-1), (r+1, c)]
    else:
        neighbors += [(r-1, c), (r-1, c+1), (r+1, c), (r+1, c+1)]
    return [(nr, nc) for nr, nc in neighbors if 0 <= nr < n and 0 <= nc < n]

class PlayerProfile:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.opponent  = 2 if player_id == 1 else 1
        if player_id == 1:
            self.time_budget = 4.70
            self.center_bias = 1.45
            self.aggression  = 0.72
        else:
            self.time_budget = 4.75
            self.center_bias = 1.20
            self.aggression  = 0.58

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

    def ucb1(self, c: float = UCB_C) -> float:
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits
                + c * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self, c: float = UCB_C) -> 'MCTSNode':
        return max(self.children, key=lambda nd: nd.ucb1(c))

    def is_fully_expanded(self) -> bool:
        return len(self.untried_moves) == 0

class SmartPlayer(Player):

    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.profile     = PlayerProfile(player_id)
        self._start_time = 0.0
        self._time_limit = self.profile.time_budget
        self._n          = 0
        self._move_count = 0

        self._max_depth  = 4
        self._beam       = 12
        self._eval_mode  = "RICH"

        self._zobrist_n  = 0
        self._zobrist    = None
        self._tt         = {}
        self._current_hash = 0

        self._killers: dict = {}
        self._history: dict = {}

        self._my_count    = 0
        self._opp_count   = 0
        self._known_cells: set = set()

    def play(self, board: HexBoard) -> tuple:
        self._start_time = time.time()
        self._time_limit = self.profile.time_budget
        self._n          = board.size

        self._init_zobrist()

        if self._move_count <= 1:

            self._tt.clear()
            self._killers.clear()
            self._move_count   = 0
            self._my_count     = 0
            self._opp_count    = 0
            self._known_cells  = set()
            self._current_hash = self._compute_hash(board)

            n = board.size
            for r in range(n):
                for c in range(n):
                    if board.board[r][c] != 0:
                        self._known_cells.add((r, c))
        else:

            opp_move = self._find_new_piece(board)
            if opp_move is not None:
                r, c = opp_move
                self._current_hash = self._update_hash(
                    self._current_hash, r, c, self.profile.opponent)
                self._known_cells.add(opp_move)
                self._opp_count += 1

        self._move_count += 1

        opening = self._opening_move(board)
        if opening is not None:
            self._current_hash = self._update_hash(
                self._current_hash, opening[0], opening[1], self.player_id)
            self._known_cells.add(opening)
            self._my_count += 1
            return opening

        immediate = self._immediate_move_csp(board)
        if immediate is not None:
            self._current_hash = self._update_hash(
                self._current_hash, immediate[0], immediate[1], self.player_id)
            self._known_cells.add(immediate)
            self._my_count += 1
            return immediate

        if self._n > MINIMAX_MAX_N:
            move = self._play_mcts(board)
        else:
            move = self._play_minimax(board)

        if move is not None:
            self._current_hash = self._update_hash(
                self._current_hash, move[0], move[1], self.player_id)
            self._known_cells.add(move)
        self._my_count += 1
        return move

    def _bfs_full(self, board: HexBoard, player_id: int,
                  reverse: bool = False) -> list:
        n   = self._n
        INF = float('inf')
        opp = 3 - player_id
        dist = [[INF] * n for _ in range(n)]
        dq   = deque()

        if player_id == 1:
            sources = [(r, n-1 if reverse else 0) for r in range(n)]
        else:
            sources = [(n-1 if reverse else 0, c) for c in range(n)]

        for r, c in sources:
            cell = board.board[r][c]
            if cell == opp:
                continue
            cost = 0 if cell == player_id else 1
            if cost < dist[r][c]:
                dist[r][c] = cost
                (dq.appendleft if cost == 0 else dq.append)((cost, r, c))

        while dq:
            cost, r, c = dq.popleft()
            if cost > dist[r][c]:
                continue
            for nr, nc in get_neighbors(r, c, n):
                cell = board.board[nr][nc]
                if cell == opp:
                    continue
                step     = 0 if cell == player_id else 1
                new_cost = cost + step
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    (dq.appendleft if step == 0 else dq.append)((new_cost, nr, nc))

        return dist

    def _get_path_and_landmarks(self, board: HexBoard,
                                 player_id: int) -> tuple:
        dist_f  = self._bfs_full(board, player_id, reverse=False)
        dist_b  = self._bfs_full(board, player_id, reverse=True)
        n       = self._n
        opp     = 3 - player_id
        INF     = float('inf')

        if player_id == 1:
            optimal = min((dist_f[r][n-1] for r in range(n)
                           if dist_f[r][n-1] < INF), default=INF)
        else:
            optimal = min((dist_f[n-1][c] for c in range(n)
                           if dist_f[n-1][c] < INF), default=INF)

        if optimal >= INF:
            return set(), set()

        path_cells = set()
        for r in range(n):
            for c in range(n):
                if board.board[r][c] == opp:
                    continue
                if dist_f[r][c] >= INF or dist_b[r][c] >= INF:
                    continue
                cost = 0 if board.board[r][c] == player_id else 1
                if dist_f[r][c] + dist_b[r][c] - cost == optimal:
                    path_cells.add((r, c))

        landmark_cells = set()
        for r, c in path_cells:
            if board.board[r][c] != 0:
                continue
            board.board[r][c] = opp
            new_dist = self._bfs_distance(board, player_id)
            board.board[r][c] = 0
            if new_dist > optimal:
                landmark_cells.add((r, c))

        return path_cells, landmark_cells

    def _immediate_move_csp(self, board: HexBoard):
        n         = self._n

        my_count  = self._my_count
        opp_count = self._opp_count

        if my_count >= max(1, n - 2):
            for r in range(n):
                for c in range(n):
                    if board.board[r][c] != 0:
                        continue
                    clone = board.clone()
                    clone.place_piece(r, c, self.player_id)
                    if clone.check_connection(self.player_id):
                        return (r, c)

        if opp_count >= max(1, n - 2):
            for r in range(n):
                for c in range(n):
                    if board.board[r][c] != 0:
                        continue
                    clone = board.clone()
                    clone.place_piece(r, c, self.profile.opponent)
                    if clone.check_connection(self.profile.opponent):
                        return (r, c)

        if self._n <= MINIMAX_MAX_N and self._time_remaining() > 0.8:
            mid_game_threshold = n // 2
            if opp_count >= mid_game_threshold:
                _, opp_lm = self._get_path_and_landmarks(
                    board, self.profile.opponent)
                if len(opp_lm) == 1:
                    lm = next(iter(opp_lm))
                    if board.board[lm[0]][lm[1]] == 0:
                        return lm

        return None

    def _order_moves_landmark(self, board: HexBoard) -> list:
        n    = self._n
        grid = board.board
        mid  = n / 2.0

        my_path,  my_lm  = self._get_path_and_landmarks(board, self.player_id)
        opp_path, opp_lm = self._get_path_and_landmarks(board, self.profile.opponent)

        tt_move = None
        entry   = self._tt.get(self._current_hash)
        if entry is not None:
            tt_move = entry[3]

        scored = {}
        for r in range(n):
            for c in range(n):
                if grid[r][c] != 0:
                    continue
                own_nb = opp_nb = 0
                for nr, nc in get_neighbors(r, c, n):
                    if   grid[nr][nc] == self.player_id:        own_nb += 1
                    elif grid[nr][nc] == self.profile.opponent:  opp_nb += 1

                base  = own_nb * 3.0 + opp_nb * 2.0
                base += self.profile.center_bias * (n - abs(r - mid) - abs(c - mid))
                base += HISTORY_SCALE * self._history.get((r, c), 0)

                if   (r, c) in my_lm:    bonus = BONUS_MY_LM
                elif (r, c) in opp_lm:   bonus = BONUS_OPP_LM
                elif (r, c) in my_path:  bonus = BONUS_MY_PATH
                elif (r, c) in opp_path: bonus = BONUS_OPP_PATH
                else:                    bonus = 0

                scored[(r, c)] = bonus + base

        if not scored:
            return []

        ordered = sorted(scored, key=lambda m: -scored[m])

        result: list = []
        used:   set  = set()

        if tt_move is not None and tt_move in scored:
            result.append(tt_move)
            used.add(tt_move)

        for km in self._killers.get(0, []):
            if km in scored and km not in used:
                result.append(km)
                used.add(km)

        for m in ordered:
            if m not in used:
                result.append(m)

        return result

    def _order_moves_fast(self, board: HexBoard,
                          droot: int, tt_move=None) -> list:
        n    = self._n
        grid = board.board
        mid  = n / 2.0

        if n > MINIMAX_MAX_N:
            candidates = self._get_frontier(board, self._known_cells)
        else:
            candidates = [(r, c) for r in range(n) for c in range(n)
                          if grid[r][c] == 0]

        scored = {}
        for r, c in candidates:
            own_nb = opp_nb = 0
            for nr, nc in get_neighbors(r, c, n):
                if   grid[nr][nc] == self.player_id:        own_nb += 1
                elif grid[nr][nc] == self.profile.opponent:  opp_nb += 1
            base  = own_nb * 3.0 + opp_nb * 2.0
            base += self.profile.center_bias * (n - abs(r - mid) - abs(c - mid))
            base += HISTORY_SCALE * self._history.get((r, c), 0)
            scored[(r, c)] = base

        if not scored:
            return []

        ordered = sorted(scored, key=lambda m: -scored[m])

        result: list = []
        used:   set  = set()

        if tt_move is not None and tt_move in scored:
            result.append(tt_move)
            used.add(tt_move)

        for km in self._killers.get(droot, []):
            if km in scored and km not in used:
                result.append(km)
                used.add(km)

        for m in ordered:
            if m not in used:
                result.append(m)

        return result

    def _play_minimax(self, board: HexBoard) -> tuple:
        self._configure_search()
        root_moves = self._order_moves_landmark(board)
        if not root_moves:
            return self._fallback(board)
        return self._iterative_deepening(board, root_moves)

    def _configure_search(self):
        n = self._n; early = self._move_count <= 4
        if   n <= 5:  self._eval_mode="RICH"; self._max_depth=7 if early else 9;  self._beam=18
        elif n <= 7:  self._eval_mode="RICH"; self._max_depth=6 if early else 8;  self._beam=16
        elif n <= 9:  self._eval_mode="RICH"; self._max_depth=5 if early else 7;  self._beam=14 if early else 16
        elif n <= 11: self._eval_mode="RICH"; self._max_depth=4 if early else 6;  self._beam=12 if early else 14
        else:         self._eval_mode="FAST"; self._max_depth=3;                   self._beam=10

    def _iterative_deepening(self, board: HexBoard,
                              root_moves: list) -> tuple:
        best_move = root_moves[0]
        best_val  = float('-inf')

        for depth in range(1, self._max_depth + 1):
            if self._time_remaining() < self._time_limit * EARLY_STOP_MARGIN:
                break

            alpha, beta    = float('-inf'), float('inf')
            depth_best_val = float('-inf')
            depth_best_move = best_move
            self._killers.clear()

            for move in root_moves[:self._beam]:
                if self._time_remaining() < TIME_MARGIN:
                    break

                clone   = board.clone()
                clone.place_piece(move[0], move[1], self.player_id)
                h_child = self._update_hash(
                    self._current_hash, move[0], move[1], self.player_id)

                val = self._alphabeta(
                    clone, h_child, depth - 1, alpha, beta, False, droot=1)

                if val > depth_best_val:
                    depth_best_val  = val
                    depth_best_move = move

                alpha = max(alpha, depth_best_val)

            if depth_best_val > best_val or depth == 1:
                best_val  = depth_best_val
                best_move = depth_best_move

            if best_move in root_moves:
                root_moves.remove(best_move)
                root_moves.insert(0, best_move)

        return best_move

    def _alphabeta(self, clone: HexBoard, h: int, depth: int,
                   alpha: float, beta: float,
                   maximizing: bool, droot: int) -> float:

        if clone.check_connection(self.player_id):
            return  1_000_000.0
        if clone.check_connection(self.profile.opponent):
            return -1_000_000.0
        if depth == 0 or self._time_remaining() < TIME_MARGIN:
            return self._evaluate(clone)

        alpha0 = alpha
        beta0  = beta

        tt_move = None
        entry   = self._tt.get(h)
        if entry is not None and entry[1] >= depth:
            cached_score, _, flag, cached_best = entry
            tt_move = cached_best
            if flag == EXACT:
                return cached_score
            if flag == LOWER_BOUND:
                alpha = max(alpha, cached_score)
            elif flag == UPPER_BOUND:
                beta  = min(beta, cached_score)
            if alpha >= beta:
                return cached_score

        moves = self._order_moves_fast(clone, droot, tt_move)
        if not moves:
            return self._evaluate(clone)

        best_local = moves[0]
        opp        = self.profile.opponent

        if maximizing:
            value = float('-inf')
            for move in moves:
                if self._time_remaining() < TIME_MARGIN:
                    break
                child      = clone.clone()
                child.place_piece(move[0], move[1], self.player_id)
                child_hash = self._update_hash(h, move[0], move[1], self.player_id)
                score      = self._alphabeta(
                    child, child_hash, depth - 1, alpha, beta, False, droot + 1)
                if score > value:
                    value      = score
                    best_local = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    self._register_cutoff(move, droot, depth)
                    break
        else:
            value = float('inf')
            for move in moves:
                if self._time_remaining() < TIME_MARGIN:
                    break
                child      = clone.clone()
                child.place_piece(move[0], move[1], opp)
                child_hash = self._update_hash(h, move[0], move[1], opp)
                score      = self._alphabeta(
                    child, child_hash, depth - 1, alpha, beta, True, droot + 1)
                if score < value:
                    value      = score
                    best_local = move
                beta = min(beta, value)
                if alpha >= beta:
                    self._register_cutoff(move, droot, depth)
                    break

        if   value <= alpha0: flag = UPPER_BOUND
        elif value >= beta0:  flag = LOWER_BOUND
        else:                 flag = EXACT
        self._tt_put(h, value, depth, flag, best_local)

        return value

    def _register_cutoff(self, move: tuple, droot: int, depth: int):
        if move is None:
            return
        self._add_killer(droot, move)
        self._history[move] = self._history.get(move, 0) + depth * depth

    def _add_killer(self, droot: int, move: tuple):
        lst = self._killers.setdefault(droot, [])
        if move in lst:
            return
        lst.insert(0, move)
        if len(lst) > 2:
            lst.pop()

    def _play_mcts(self, board: HexBoard) -> tuple:
        moves = self._order_moves_fast(board, droot=0)
        if not moves:
            return self._fallback(board)
        if len(moves) == 1:
            return moves[0]

        mcts_wins   = {m: 0.0 for m in moves}
        mcts_visits = {m: 0   for m in moves}
        WARMUP      = max(20, len(moves))

        root = MCTSNode(
            move             = None,
            parent           = None,
            untried_moves    = list(moves),
            player_who_moved = self.profile.opponent
        )

        iters = 0
        while self._time_remaining() > 0.05:
            node       = root
            sim_board  = board.clone()
            sim_player = self.player_id
            first_move = None

            while node.is_fully_expanded() and node.children:
                node = node.best_child(UCB_C)
                sim_board.place_piece(
                    node.move[0], node.move[1], node.player_who_moved)
                if first_move is None and node.player_who_moved == self.player_id:
                    first_move = node.move
                sim_player = 3 - node.player_who_moved
                if sim_board.check_connection(1) or sim_board.check_connection(2):
                    break

            if node.untried_moves and not (
                sim_board.check_connection(1) or sim_board.check_connection(2)
            ):
                move = node.untried_moves.pop(0)
                sim_board.place_piece(move[0], move[1], sim_player)
                child = MCTSNode(
                    move             = move,
                    parent           = node,
                    untried_moves    = self._get_frontier(sim_board),
                    player_who_moved = sim_player
                )
                node.children.append(child)
                if first_move is None and sim_player == self.player_id:
                    first_move = move
                node       = child
                sim_player = 3 - sim_player

            result = self._rollout_bfs(sim_board)

            win_val = 1.0 if result == self.player_id else (0.5 if result == 0 else 0.0)
            while node is not None:
                node.visits += 1
                if result == self.player_id:
                    node.wins += 1.0
                elif result == 0:
                    node.wins += 0.5
                node = node.parent

            if first_move is not None and first_move in mcts_wins:
                mcts_wins[first_move]   += win_val
                mcts_visits[first_move] += 1

            iters += 1

            if iters == WARMUP and root.untried_moves:
                root.untried_moves.sort(
                    key=lambda m: (
                        -(mcts_wins[m] / mcts_visits[m]) if mcts_visits[m] > 0
                        else 0.0
                    )
                )

        if not root.children:
            return moves[0]
        return max(root.children, key=lambda nd: nd.visits).move

    def _rollout_bfs(self, board: HexBoard) -> int:
        if board.check_connection(self.player_id):
            return self.player_id
        if board.check_connection(self.profile.opponent):
            return self.profile.opponent

        d_mine = self._bfs_distance(board, self.player_id)
        d_opp  = self._bfs_distance(board, self.profile.opponent)
        max_d  = float(self._n * self._n)
        if d_mine == float('inf'): d_mine = max_d
        if d_opp  == float('inf'): d_opp  = max_d

        threshold = 2 if self._n > 13 else 3
        diff = d_opp - d_mine

        if   diff >=  threshold: return self.player_id
        elif diff <= -threshold: return self.profile.opponent
        elif d_mine < d_opp:     return self.player_id
        elif d_opp  < d_mine:    return self.profile.opponent
        else:                    return 0

    def _evaluate(self, board: HexBoard) -> float:
        my_id  = self.player_id
        opp_id = self.profile.opponent
        max_d  = float(self._n * self._n + 5)

        raw_mine = self._bfs_distance(board, my_id)
        raw_opp  = self._bfs_distance(board, opp_id)
        dist_mine = raw_mine if raw_mine != float('inf') else max_d
        dist_opp  = raw_opp  if raw_opp  != float('inf') else max_d

        score = W_PATH * (dist_opp - dist_mine)

        if self._eval_mode == "RICH":
            vc_mine = self._count_virtual_connections(board, my_id)
            vc_opp  = self._count_virtual_connections(board, opp_id)
            score  += W_VIRTUAL * (vc_mine - vc_opp)

            cc_mine = self._largest_connected_component(board, my_id)
            cc_opp  = self._largest_connected_component(board, opp_id)
            score  += W_CONNECT * (cc_mine - cc_opp)

            bp_mine = self._bridge_potential(board, my_id)
            bp_opp  = self._bridge_potential(board, opp_id)
            score  += W_BRIDGE * (bp_mine - bp_opp)

            score  += (self._center_control(board, my_id)
                       - self._center_control(board, opp_id))

        return score * (0.5 + self.profile.aggression)

    def _bfs_distance(self, board: HexBoard, player_id: int) -> float:
        n   = self._n
        INF = float('inf')
        opp = 3 - player_id
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
                (dq.appendleft if cost == 0 else dq.append)((cost, r, c))

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
                    (dq.appendleft if step == 0 else dq.append)((new_cost, nr, nc))

        return INF

    def _count_virtual_connections(self, board: HexBoard, player_id: int) -> int:
        n = self._n; grid = board.board; count = 0; seen: set = set()
        own = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == player_id]
        for r1, c1 in own:
            nb1 = set(get_neighbors(r1, c1, n))
            for m1r, m1c in nb1:
                if grid[m1r][m1c] != 0: continue
                for r2, c2 in get_neighbors(m1r, m1c, n):
                    if (r2, c2) == (r1, c1) or grid[r2][c2] != player_id: continue
                    br = frozenset(((r1, c1), (r2, c2)))
                    if br in seen: continue
                    nb2 = set(get_neighbors(r2, c2, n)); shared = nb1 & nb2
                    for m2r, m2c in shared:
                        if (m2r, m2c) != (m1r, m1c) and grid[m2r][m2c] == 0:
                            seen.add(br); count += 1; break
        return count

    def _largest_connected_component(self, board: HexBoard, player_id: int) -> int:
        n = self._n; grid = board.board; visited: set = set(); largest = 0
        for r in range(n):
            for c in range(n):
                if grid[r][c] != player_id or (r, c) in visited: continue
                size = 0; stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited: continue
                    visited.add((cr, cc)); size += 1
                    for nr, nc in get_neighbors(cr, cc, n):
                        if grid[nr][nc] == player_id and (nr, nc) not in visited:
                            stack.append((nr, nc))
                largest = max(largest, size)
        return largest

    def _bridge_potential(self, board: HexBoard, player_id: int) -> float:
        n = self._n; grid = board.board
        cells = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == player_id]
        score = 0.0
        for idx, (r1, c1) in enumerate(cells):
            s1 = set(get_neighbors(r1, c1, n))
            for r2, c2 in cells[idx + 1:]:
                if (r2, c2) in s1:
                    score += 2.0
                else:
                    for nr, nc in s1:
                        if (r2, c2) in get_neighbors(nr, nc, n):
                            score += 1.0
                            break
        return score

    def _center_control(self, board: HexBoard, player_id: int) -> float:
        n = self._n; mid = n / 2.0; grid = board.board; total = 0.0
        for r in range(n):
            for c in range(n):
                if grid[r][c] == player_id:
                    total += n - abs(r - mid) - abs(c - mid)
        return total * W_CENTER

    def _init_zobrist(self):
        if self._zobrist_n == self._n and self._zobrist is not None:
            return
        rng = random.Random(0xDEADBEEF)
        n   = self._n
        self._zobrist = [
            [[rng.getrandbits(64) for _ in range(n)] for _ in range(n)]
            for _ in range(3)
        ]
        self._zobrist_n    = n
        self._tt.clear()
        self._current_hash = 0

    def _compute_hash(self, board: HexBoard) -> int:
        h = 0; zob = self._zobrist
        for r, row in enumerate(board.board):
            for c, val in enumerate(row):
                if val: h ^= zob[val][r][c]
        return h

    def _update_hash(self, h: int, r: int, c: int, player_id: int) -> int:
        return h ^ self._zobrist[player_id][r][c]

    def _tt_put(self, h: int, score: float, depth: int, flag: int, best_move):
        if len(self._tt) >= TT_MAX_SIZE:
            shallow = [k for k, v in self._tt.items() if v[1] <= 1]
            victims = shallow[:TT_MAX_SIZE // 4]
            if len(victims) < TT_MAX_SIZE // 4:
                extra = random.sample(
                    list(self._tt.keys()),
                    min(TT_MAX_SIZE // 4 - len(victims), len(self._tt)))
                for k in extra:
                    if k not in victims:
                        victims.append(k)
            for k in victims:
                self._tt.pop(k, None)
        self._tt[h] = (score, depth, flag, best_move)

    def _opening_move(self, board: HexBoard):
        center = self._n // 2
        if self._move_count <= 2 and board.board[center][center] == 0:
            return (center, center)
        return None

    def _fallback(self, board: HexBoard):
        for r in range(self._n):
            for c in range(self._n):
                if board.board[r][c] == 0:
                    return (r, c)
        return None

    def _find_new_piece(self, board: HexBoard):
        grid = board.board
        n    = self._n

        for r, c in self._known_cells:
            for nr, nc in get_neighbors(r, c, n):
                if (grid[nr][nc] == self.profile.opponent
                        and (nr, nc) not in self._known_cells):
                    return (nr, nc)

        for r in range(n):
            for c in range(n):
                if (grid[r][c] == self.profile.opponent
                        and (r, c) not in self._known_cells):
                    return (r, c)

        return None

    def _get_empty_cells(self, board: HexBoard) -> list:
        n = self._n
        return [(r, c) for r in range(n) for c in range(n)
                if board.board[r][c] == 0]

    def _get_frontier(self, board: HexBoard,
                       occupied: set = None) -> list:
        n    = self._n
        grid = board.board
        seen: set = set()
        frontier  = []

        if occupied:

            for r, c in occupied:
                for nr, nc in get_neighbors(r, c, n):
                    if grid[nr][nc] == 0 and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        frontier.append((nr, nc))
        else:

            for r in range(n):
                for c in range(n):
                    if grid[r][c] == 0:
                        continue
                    for nr, nc in get_neighbors(r, c, n):
                        if grid[nr][nc] == 0 and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            frontier.append((nr, nc))

        if not frontier:
            mid = n // 2
            for r in range(max(0, mid-1), min(n, mid+2)):
                for c in range(max(0, mid-1), min(n, mid+2)):
                    if grid[r][c] == 0 and (r, c) not in seen:
                        seen.add((r, c))
                        frontier.append((r, c))
        return frontier

    def _time_remaining(self) -> float:
        return self._time_limit - (time.time() - self._start_time)