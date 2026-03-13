from collections import deque

def _get_neighbors_board(r, c, n):
    neighbors = [(r, c - 1), (r, c + 1)]
    if r % 2 == 0:
        neighbors += [(r-1, c-1), (r-1, c), (r+1, c-1), (r+1, c)]
    else:
        neighbors += [(r-1, c),   (r-1, c+1), (r+1, c), (r+1, c+1)]
    return [(nr, nc) for nr, nc in neighbors if 0 <= nr < n and 0 <= nc < n]

class HexBoard:
    def __init__(self, size):
        self.size  = size
        self.board = [[0]*size for _ in range(size)]

    def clone(self):
        new = HexBoard(self.size)
        new.board = [row[:] for row in self.board]
        return new

    def place_piece(self, row, col, player_id):
        if self.board[row][col] != 0:
            return False
        self.board[row][col] = player_id
        return True

    def check_connection(self, player_id):
        n = self.size
        if player_id == 1:
            sources = [(r, 0) for r in range(n) if self.board[r][0] == 1]
            goal_fn = lambda r, c: c == n - 1
        else:
            sources = [(0, c) for c in range(n) if self.board[0][c] == 2]
            goal_fn = lambda r, c: r == n - 1
        visited = set()
        queue   = deque()
        for s in sources:
            if goal_fn(*s): return True
            queue.append(s); visited.add(s)
        while queue:
            r, c = queue.popleft()
            for nr, nc in _get_neighbors_board(r, c, n):
                if (nr, nc) in visited or self.board[nr][nc] != player_id:
                    continue
                if goal_fn(nr, nc): return True
                visited.add((nr, nc)); queue.append((nr, nc))
        return False
