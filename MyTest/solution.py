from playerTest import Player
from boardTest import HexBoard

import math
import random
import time
from collections import deque

# Umbral de tamaño: N <= 11 usa Minimax, N > 11 usa MCTS
MINIMAX_MAX_N = 11

# Tamaño máximo de la tabla de transposición (número de entradas)
TT_MAX_SIZE = 180_000

# Tipos de cota para la tabla de transposición (Bloque 2)
EXACT       = 0  # Valor minimax exacto, ventana [alpha, beta] no saturada
LOWER_BOUND = 1  # Corte beta: el valor real es >= al almacenado
UPPER_BOUND = 2  # All-node: el valor real es <= al almacenado

# Pesos de los componentes de la función de evaluación (Bloque 1)
W_PATH      = 1.00  # Peso de la diferencia de distancias BFS (componente principal)
W_VIRTUAL   = 0.35  # Peso de las conexiones virtuales propias
W_CONNECT   = 0.20  # Peso de la diferencia de la mayor componente conexa
W_BRIDGE    = 0.15  # Peso del potencial de puente entre piezas
W_CENTER    = 0.08  # Peso del control del centro del tablero

# Factor de escala de la History Heuristic en el ordenamiento de movimientos
HISTORY_SCALE = 0.0007

# Constante de exploración UCB1 para MCTS (Bloque 4)
# C = 0.8 < sqrt(2): favorece explotación porque el rollout BFS ya es informado
UCB_C = 0.8

# Bonuses de tier para el ordenamiento de acciones por landmarks STRIPS (Bloque 5)
BONUS_MY_LM    = 10_000  # Mi landmark: acción necesaria en todo plan ganador propio
BONUS_OPP_LM   =  9_000  # Landmark del oponente: bloquear su acción necesaria
BONUS_MY_PATH  =  5_000  # Mi path cell: celda que avanza mi plan actual
BONUS_OPP_PATH =  4_000  # Path cell del oponente: celda que interfiere su plan

# Márgenes de tiempo para evitar descalificación (5s límite del torneo)
TIME_MARGIN        = 0.03  # Corte de emergencia: detener búsqueda si queda < 0.03s
EARLY_STOP_MARGIN  = 0.12  # No iniciar nueva iteración IDAB si queda < 12% del budget


def get_neighbors(r: int, c: int, n: int) -> list:
    """
    Devuelve los vecinos válidos de la celda (r, c) en un tablero hexagonal
    con representación even-r offset layout.

    Recibe:
        r (int): fila de la celda.
        c (int): columna de la celda.
        n (int): tamaño del tablero (N x N).

    Retorna:
        list: lista de tuplas (nr, nc) con las coordenadas de los vecinos
              válidos dentro de los límites del tablero.
    """
    # Vecinos horizontales comunes a todas las filas
    neighbors = [(r, c - 1), (r, c + 1)]

    # Vecinos diagonales según paridad de la fila (even-r offset)
    if r % 2 == 0:
        neighbors += [(r-1, c-1), (r-1, c), (r+1, c-1), (r+1, c)]
    else:
        neighbors += [(r-1, c), (r-1, c+1), (r+1, c), (r+1, c+1)]

    # Filtrar vecinos fuera de los límites del tablero
    return [(nr, nc) for nr, nc in neighbors if 0 <= nr < n and 0 <= nc < n]


class PlayerProfile:
    """
    Almacena los parámetros de comportamiento diferenciados según el rol
    del jugador (P1 o P2): presupuesto de tiempo, sesgo de centro y agresividad.
    """

    def __init__(self, player_id: int):
        """
        Recibe:
            player_id (int): identificador del jugador (1 o 2).

        Inicializa:
            player_id (int): id propio del jugador.
            opponent  (int): id del oponente (complementario).
            time_budget (float): segundos disponibles por turno.
            center_bias (float): peso del sesgo hacia el centro en el ordenamiento.
            aggression  (float): parámetro que escala la función de evaluación.
        """
        self.player_id = player_id
        self.opponent  = 2 if player_id == 1 else 1

        if player_id == 1:
            # P1 juega primero: perfil más agresivo, menor margen de tiempo
            self.time_budget = 4.70
            self.center_bias = 1.45
            self.aggression  = 0.72
        else:
            # P2 juega segundo: perfil más conservador, mayor margen de tiempo
            self.time_budget = 4.75
            self.center_bias = 1.20
            self.aggression  = 0.58


class MCTSNode:
    """
    Nodo del árbol MCTS. Almacena estadísticas de victorias/visitas
    y la lista de movimientos aún no expandidos.
    """

    # __slots__ optimiza memoria evitando el diccionario __dict__ por instancia
    __slots__ = ["move", "parent", "children", "wins", "visits",
                 "untried_moves", "player_who_moved"]

    def __init__(self, move, parent, untried_moves, player_who_moved):
        """
        Recibe:
            move             (tuple|None): jugada (r,c) que llevó a este nodo;
                                           None en la raíz.
            parent           (MCTSNode):   nodo padre; None en la raíz.
            untried_moves    (list):       movimientos aún no expandidos desde
                                           este nodo.
            player_who_moved (int):        jugador que realizó 'move' (1 o 2).
        """
        self.move             = move
        self.parent           = parent
        self.children         = []    # Nodos hijo ya expandidos
        self.wins             = 0.0   # Victorias acumuladas en simulaciones
        self.visits           = 0     # Veces que este nodo fue visitado
        self.untried_moves    = untried_moves
        self.player_who_moved = player_who_moved

    def ucb1(self, c: float = UCB_C) -> float:
        """
        Calcula el valor UCB1 del nodo para la selección en MCTS.
        Nodos no visitados retornan infinito para garantizar exploración inicial.

        Recibe:
            c (float): constante de exploración (por defecto UCB_C = 0.8).

        Retorna:
            float: UCB1 = wins/visits + C * sqrt(ln(visitas_padre) / visitas).
        """
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits
                + c * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self, c: float = UCB_C) -> 'MCTSNode':
        """
        Retorna el hijo con mayor valor UCB1 (fase de selección MCTS).

        Recibe:
            c (float): constante de exploración.

        Retorna:
            MCTSNode: hijo con el mayor UCB1.
        """
        return max(self.children, key=lambda nd: nd.ucb1(c))

    def is_fully_expanded(self) -> bool:
        """
        Retorna True si todos los movimientos del nodo ya fueron expandidos.
        """
        return len(self.untried_moves) == 0


class SmartPlayer(Player):
    """
    Agente autónomo para HEX con estrategia híbrida:
        N <= 11: Minimax + Alpha-Beta + IDAB + TT con cotas + Killers + History
        N >  11: MCTS + rollout BFS + frontera + win-rate warmup
    """

    def __init__(self, player_id: int):
        """
        Inicializa el agente y todos sus componentes de estado.

        Recibe:
            player_id (int): identificador del jugador (1 o 2).
        """
        super().__init__(player_id)

        # Perfil de parámetros según rol del jugador
        self.profile     = PlayerProfile(player_id)

        # Control de tiempo por turno
        self._start_time = 0.0   # Marca de inicio del turno actual
        self._time_limit = self.profile.time_budget  # Budget en segundos
        self._n          = 0     # Tamaño del tablero actual (se actualiza en play)
        self._move_count = 0     # Contador de turnos jugados en la partida actual

        # Parámetros adaptativos de búsqueda Minimax (se configuran en _configure_search)
        self._max_depth  = 4     # Profundidad máxima del IDAB
        self._beam       = 12    # Número máximo de candidatos raíz a explorar
        self._eval_mode  = "RICH"  # Modo de evaluación: RICH (todos los componentes)
                                   # o FAST (solo BFS, para tableros grandes)

        # Tabla de transposición Zobrist (Bloque 2)
        self._zobrist_n    = 0     # Tamaño del tablero para el que se generó Zobrist
        self._zobrist      = None  # Tabla 3D de números aleatorios de 64 bits
        self._tt           = {}    # Tabla de transposición: hash -> (score, depth, flag, best_move)
        self._current_hash = 0     # Hash Zobrist incremental del estado actual del tablero

        # Heurísticas de ordenamiento para Alpha-Beta (Bloque 2)
        self._killers: dict = {}   # droot -> [move1, move2]: jugadas que causaron cortes
        self._history: dict = {}   # (r,c) -> score: acumulado de depth² por cortes

        # Estado incremental O(1) para evitar scans O(N²) por turno
        self._my_count    = 0          # Número de piezas propias colocadas
        self._opp_count   = 0          # Número de piezas del oponente colocadas
        self._known_cells: set = set() # Conjunto de todas las celdas ocupadas conocidas


    # PUNTO DE ENTRADA
    def play(self, board: HexBoard) -> tuple:
        """
        Punto de entrada del agente. Decide la mejor jugada para el turno actual.

        Recibe:
            board (HexBoard): copia del tablero actual (0=vacío, 1=P1, 2=P2).

        Retorna:
            tuple: (fila, columna) de la jugada seleccionada.
        """
        # Paso 1: Registrar inicio del turno y configurar parámetros básicos
        self._start_time = time.time()
        self._time_limit = self.profile.time_budget
        self._n          = board.size

        # Paso 2: Asegurar que la tabla Zobrist esté inicializada para este N
        self._init_zobrist()

        # Paso 3: Inicializar o actualizar el estado incremental de la partida
        if self._move_count <= 1:
            # Inicio de partida: reiniciar todos los componentes de estado
            self._tt.clear()
            self._killers.clear()
            self._move_count   = 0
            self._my_count     = 0
            self._opp_count    = 0
            self._known_cells  = set()

            # Calcular hash completo O(N²) solo una vez al inicio
            self._current_hash = self._compute_hash(board)

            # Registrar piezas ya presentes (reanudación de partida)
            n = board.size
            for r in range(n):
                for c in range(n):
                    if board.board[r][c] != 0:
                        self._known_cells.add((r, c))
        else:
            # Turno N>1: localizar la jugada nueva del oponente en O(k)
            # usando la frontera de celdas conocidas (sin scan O(N²))
            opp_move = self._find_new_piece(board)
            if opp_move is not None:
                r, c = opp_move
                # Actualizar hash incremental O(1) con la pieza del oponente
                self._current_hash = self._update_hash(
                    self._current_hash, r, c, self.profile.opponent)
                self._known_cells.add(opp_move)
                self._opp_count += 1

        self._move_count += 1

        # Paso 4: Intentar jugada de apertura (centro del tablero)
        opening = self._opening_move(board)
        if opening is not None:
            self._current_hash = self._update_hash(
                self._current_hash, opening[0], opening[1], self.player_id)
            self._known_cells.add(opening)
            self._my_count += 1
            return opening

        # Paso 5: Detectar jugadas forzadas con razonamiento CSP (Bloque 3)
        immediate = self._immediate_move_csp(board)
        if immediate is not None:
            self._current_hash = self._update_hash(
                self._current_hash, immediate[0], immediate[1], self.player_id)
            self._known_cells.add(immediate)
            self._my_count += 1
            return immediate

        # Paso 6: Seleccionar algoritmo según tamaño del tablero
        if self._n > MINIMAX_MAX_N:
            move = self._play_mcts(board)    # N > 11: MCTS
        else:
            move = self._play_minimax(board) # N <= 11: Minimax

        # Paso 7: Actualizar hash y estado incremental con la jugada elegida
        if move is not None:
            self._current_hash = self._update_hash(
                self._current_hash, move[0], move[1], self.player_id)
            self._known_cells.add(move)
        self._my_count += 1
        return move


    # BFS BIDIRECCIONAL: PATH CELLS Y LANDMARKS (Bloques 1, 3 y 5)
    def _bfs_full(self, board: HexBoard, player_id: int,
                  reverse: bool = False) -> list:
        """
        0-1 BFS completo que devuelve la MATRIZ de distancias mínimas.
        Usado en el análisis bidireccional para identificar path cells y landmarks.

        Recibe:
            board     (HexBoard): estado actual del tablero.
            player_id (int):      jugador para el que se calcula la distancia.
            reverse   (bool):     si True, BFS desde el borde objetivo;
                                  si False, BFS desde el borde fuente.

        Retorna:
            list[list[float]]: matriz N x N con la distancia mínima desde
                               la fuente (o el objetivo) hasta cada celda.
        """
        n   = self._n
        INF = float('inf')
        opp = 3 - player_id  # Id del oponente (complementario: 1->2, 2->1)

        # Paso 1: Inicializar matriz de distancias con infinito
        dist = [[INF] * n for _ in range(n)]
        dq   = deque()  # Deque para 0-1 BFS (nodos coste 0 al frente)

        # Paso 2: Determinar fuentes según jugador y dirección
        # P1 conecta columna 0 -> columna N-1; P2 conecta fila 0 -> fila N-1
        if player_id == 1:
            sources = [(r, n-1 if reverse else 0) for r in range(n)]
        else:
            sources = [(n-1 if reverse else 0, c) for c in range(n)]

        # Paso 3: Insertar fuentes en la deque con su coste inicial
        for r, c in sources:
            cell = board.board[r][c]
            if cell == opp:
                continue  # Celda del oponente: bloqueada
            cost = 0 if cell == player_id else 1  # Propia=0, Vacía=1
            if cost < dist[r][c]:
                dist[r][c] = cost
                (dq.appendleft if cost == 0 else dq.append)((cost, r, c))

        # Paso 4: Expandir la deque propagando distancias mínimas
        while dq:
            cost, r, c = dq.popleft()
            if cost > dist[r][c]:
                continue  # Entrada obsoleta, ya se encontró un camino mejor
            for nr, nc in get_neighbors(r, c, n):
                cell = board.board[nr][nc]
                if cell == opp:
                    continue  # Celda bloqueada por el oponente
                step     = 0 if cell == player_id else 1
                new_cost = cost + step
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    (dq.appendleft if step == 0 else dq.append)((new_cost, nr, nc))

        return dist

    def _get_path_and_landmarks(self, board: HexBoard,
                                 player_id: int) -> tuple:
        """
        Identifica las path cells (celdas en algún camino óptimo) y
        las landmark cells (celdas en todo camino óptimo) del jugador.

        Recibe:
            board     (HexBoard): estado actual del tablero.
            player_id (int):      jugador para el que se analizan los caminos.

        Retorna:
            tuple: (path_cells: set, landmark_cells: set)
                   path_cells:     celdas que pertenecen a algún camino óptimo.
                   landmark_cells: celdas que aparecen en TODOS los caminos
                                   óptimos (bloquearlas aumenta h(n)).
        """
        # Paso 1: BFS bidireccional — dist_f desde fuente, dist_b desde objetivo
        dist_f  = self._bfs_full(board, player_id, reverse=False)
        dist_b  = self._bfs_full(board, player_id, reverse=True)
        n       = self._n
        opp     = 3 - player_id
        INF     = float('inf')

        # Paso 2: Calcular distancia óptima d* desde dist_f en el borde objetivo
        if player_id == 1:
            optimal = min((dist_f[r][n-1] for r in range(n)
                           if dist_f[r][n-1] < INF), default=INF)
        else:
            optimal = min((dist_f[n-1][c] for c in range(n)
                           if dist_f[n-1][c] < INF), default=INF)

        if optimal >= INF:
            return set(), set()  # Jugador bloqueado, sin camino viable

        # Paso 3: Identificar path cells
        # Una celda (r,c) está en algún camino óptimo si:
        # dist_f[r][c] + dist_b[r][c] - cost(r,c) == optimal
        # (cost se descuenta porque aparece contado en ambos BFS)
        path_cells = set()
        for r in range(n):
            for c in range(n):
                if board.board[r][c] == opp:
                    continue  # Celda del oponente, no forma parte del camino
                if dist_f[r][c] >= INF or dist_b[r][c] >= INF:
                    continue  # Celda inalcanzable
                cost = 0 if board.board[r][c] == player_id else 1
                if dist_f[r][c] + dist_b[r][c] - cost == optimal:
                    path_cells.add((r, c))

        # Paso 4: Identificar landmark cells mediante propagación de restricciones
        # Una celda vacía es landmark si bloquearla (asignarla al oponente)
        # incrementa d* — viola la restricción de conexión óptima (Bloque 3)
        landmark_cells = set()
        for r, c in path_cells:
            if board.board[r][c] != 0:
                continue  # Solo verificar celdas vacías
            board.board[r][c] = opp           # Bloqueo temporal
            new_dist = self._bfs_distance(board, player_id)
            board.board[r][c] = 0             # Restaurar siempre
            if new_dist > optimal:
                landmark_cells.add((r, c))    # Bloquearla rompe el camino óptimo

        return path_cells, landmark_cells


    # CSP: DETECCIÓN DE JUGADAS FORZADAS (Bloque 3)
    def _immediate_move_csp(self, board: HexBoard):
        """
        Detecta jugadas forzadas usando razonamiento CSP antes de invocar
        Minimax o MCTS, evitando búsqueda completa cuando no es necesaria.

        Recibe:
            board (HexBoard): estado actual del tablero.

        Retorna:
            tuple|None: (fila, columna) si se detecta jugada forzada,
                        None si no hay jugada forzada evidente.
        """
        n = self._n

        # Contadores O(1) — evitan scan O(N²) del tablero
        my_count  = self._my_count
        opp_count = self._opp_count

        # Paso 1: Victoria inmediata — si tenemos >= N-2 piezas, buscamos conexión
        if my_count >= max(1, n - 2):
            for r in range(n):
                for c in range(n):
                    if board.board[r][c] != 0:
                        continue
                    clone = board.clone()
                    clone.place_piece(r, c, self.player_id)
                    if clone.check_connection(self.player_id):
                        return (r, c)  # Jugada ganadora encontrada

        # Paso 2: Bloqueo inmediato — si el oponente puede ganar, bloquearlo
        if opp_count >= max(1, n - 2):
            for r in range(n):
                for c in range(n):
                    if board.board[r][c] != 0:
                        continue
                    clone = board.clone()
                    clone.place_piece(r, c, self.profile.opponent)
                    if clone.check_connection(self.profile.opponent):
                        return (r, c)  # Bloquear jugada ganadora del oponente

        # Paso 3: CSP landmark singleton — solo en Minimax (N <= 11)
        # Si el oponente tiene exactamente 1 landmark en fase media, bloquearlo
        # es obligatorio: su dominio factible tiene cardinalidad 1 (arc-consistency)
        if self._n <= MINIMAX_MAX_N and self._time_remaining() > 0.8:
            mid_game_threshold = n // 2  # Activar solo en fase media
            if opp_count >= mid_game_threshold:
                _, opp_lm = self._get_path_and_landmarks(
                    board, self.profile.opponent)
                if len(opp_lm) == 1:
                    lm = next(iter(opp_lm))
                    if board.board[lm[0]][lm[1]] == 0:
                        return lm  # Único landmark: bloquear obligatoriamente

        return None


    # ORDENAMIENTO DE MOVIMIENTOS (Bloques 1, 2, 3 y 5)
    def _order_moves_landmark(self, board: HexBoard) -> list:
        """
        Ordenamiento jerárquico completo para la raíz de Minimax.
        Combina landmarks STRIPS, path cells, score greedy e history.

        Recibe:
            board (HexBoard): estado actual del tablero.

        Retorna:
            list: movimientos ordenados por prioridad decreciente.
                  Orden: [TT-move, killers, resto por tier+greedy+history].
        """
        n    = self._n
        grid = board.board
        mid  = n / 2.0

        # Paso 1: Calcular path cells y landmarks de ambos jugadores (Bloques 3 y 5)
        my_path,  my_lm  = self._get_path_and_landmarks(board, self.player_id)
        opp_path, opp_lm = self._get_path_and_landmarks(board, self.profile.opponent)

        # Paso 2: Recuperar TT-move para probarlo primero (Bloque 2)
        tt_move = None
        entry   = self._tt.get(self._current_hash)
        if entry is not None:
            tt_move = entry[3]  # best_move almacenado en la TT

        # Paso 3: Puntuar todas las celdas vacías combinando tier + greedy + history
        scored = {}
        for r in range(n):
            for c in range(n):
                if grid[r][c] != 0:
                    continue

                # Score greedy: vecinos propios y del oponente + bias centro (Bloque 1)
                own_nb = opp_nb = 0
                for nr, nc in get_neighbors(r, c, n):
                    if   grid[nr][nc] == self.player_id:        own_nb += 1
                    elif grid[nr][nc] == self.profile.opponent:  opp_nb += 1

                base  = own_nb * 3.0 + opp_nb * 2.0
                base += self.profile.center_bias * (n - abs(r - mid) - abs(c - mid))
                base += HISTORY_SCALE * self._history.get((r, c), 0)  # History heuristic

                # Bonus de tier según relevancia al plan STRIPS (Bloque 5)
                if   (r, c) in my_lm:    bonus = BONUS_MY_LM    # Tier 1: mi landmark
                elif (r, c) in opp_lm:   bonus = BONUS_OPP_LM   # Tier 2: landmark rival
                elif (r, c) in my_path:  bonus = BONUS_MY_PATH   # Tier 3: mi path cell
                elif (r, c) in opp_path: bonus = BONUS_OPP_PATH  # Tier 4: path rival
                else:                    bonus = 0                # Tier 5: resto

                scored[(r, c)] = bonus + base

        if not scored:
            return []

        # Paso 4: Ordenar por score decreciente
        ordered = sorted(scored, key=lambda m: -scored[m])

        # Paso 5: Construir lista final con TT-move y killers al frente
        result: list = []
        used:   set  = set()

        if tt_move is not None and tt_move in scored:
            result.append(tt_move)
            used.add(tt_move)

        for km in self._killers.get(0, []):  # Killers del nivel raíz
            if km in scored and km not in used:
                result.append(km)
                used.add(km)

        for m in ordered:
            if m not in used:
                result.append(m)

        return result

    def _order_moves_fast(self, board: HexBoard,
                          droot: int, tt_move=None) -> list:
        """
        Ordenamiento rápido para nodos internos de Alpha-Beta y raíz de MCTS.
        Para N > 11 usa frontera BFS O(k) en vez de scan O(N²).

        Recibe:
            board    (HexBoard):   estado del tablero a ordenar.
            droot    (int):        nivel de profundidad desde la raíz
                                   (para buscar killers del nivel correspondiente).
            tt_move  (tuple|None): jugada de la TT para poner al frente.

        Retorna:
            list: movimientos ordenados por score greedy + history,
                  con TT-move y killers al frente.
        """
        n    = self._n
        grid = board.board
        mid  = n / 2.0

        # Paso 1: Obtener candidatos — frontera O(k) para MCTS, scan completo para Minimax
        if n > MINIMAX_MAX_N:
            # Frontera BFS: solo celdas adyacentes a piezas ya colocadas
            candidates = self._get_frontier(board, self._known_cells)
        else:
            # Minimax N<=11: scan completo para no perder candidatos relevantes
            candidates = [(r, c) for r in range(n) for c in range(n)
                          if grid[r][c] == 0]

        # Paso 2: Puntuar cada candidato con score greedy + history
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

        # Paso 3: Ordenar por score decreciente
        ordered = sorted(scored, key=lambda m: -scored[m])

        # Paso 4: Construir lista con TT-move y killers al frente
        result: list = []
        used:   set  = set()

        if tt_move is not None and tt_move in scored:
            result.append(tt_move)
            used.add(tt_move)

        for km in self._killers.get(droot, []):  # Killers del nivel droot
            if km in scored and km not in used:
                result.append(km)
                used.add(km)

        for m in ordered:
            if m not in used:
                result.append(m)

        return result


    # MINIMAX + ALPHA-BETA + IDAB (Bloque 2) — N <= 11
    def _play_minimax(self, board: HexBoard) -> tuple:
        """
        Punto de entrada del régimen Minimax. Configura parámetros y lanza IDAB.

        Recibe:
            board (HexBoard): estado actual del tablero.

        Retorna:
            tuple: (fila, columna) de la mejor jugada encontrada.
        """
        # Paso 1: Ajustar profundidad y beam según N y fase de juego
        self._configure_search()

        # Paso 2: Ordenar candidatos raíz con landmarks STRIPS (Bloque 5)
        root_moves = self._order_moves_landmark(board)
        if not root_moves:
            return self._fallback(board)

        # Paso 3: Lanzar iterative deepening alpha-beta
        return self._iterative_deepening(board, root_moves)

    def _configure_search(self):
        """
        Ajusta _max_depth, _beam y _eval_mode según el tamaño del tablero
        y la fase de la partida (apertura vs. resto).
        No recibe ni retorna valores; modifica el estado interno del agente.
        """
        n = self._n
        early = self._move_count <= 4  # True si estamos en los primeros 4 turnos

        if   n <= 5:  self._eval_mode="RICH"; self._max_depth=7 if early else 9;  self._beam=18
        elif n <= 7:  self._eval_mode="RICH"; self._max_depth=6 if early else 8;  self._beam=16
        elif n <= 9:  self._eval_mode="RICH"; self._max_depth=5 if early else 7;  self._beam=14 if early else 16
        elif n <= 11: self._eval_mode="RICH"; self._max_depth=4 if early else 6;  self._beam=12 if early else 14
        else:         self._eval_mode="FAST"; self._max_depth=3;                   self._beam=10

    def _iterative_deepening(self, board: HexBoard,
                              root_moves: list) -> tuple:
        """
        Iterative Deepening Alpha-Beta (IDAB): itera profundidades 1..max_depth
        con control de tiempo. Guarda la mejor jugada de la iteración completada
        más reciente como respaldo.

        Recibe:
            board      (HexBoard): estado actual del tablero.
            root_moves (list):     candidatos raíz ordenados por landmarks.

        Retorna:
            tuple: (fila, columna) de la mejor jugada encontrada.
        """
        best_move = root_moves[0]  # Respaldo inicial: primer candidato por landmarks
        best_val  = float('-inf')

        for depth in range(1, self._max_depth + 1):
            # Verificar que haya tiempo suficiente para una iteración completa
            if self._time_remaining() < self._time_limit * EARLY_STOP_MARGIN:
                break

            alpha, beta    = float('-inf'), float('inf')
            depth_best_val = float('-inf')
            depth_best_move = best_move

            # Reiniciar killers: contexto nuevo para cada iteración de profundidad
            self._killers.clear()

            # Explorar candidatos raíz limitados por beam
            for move in root_moves[:self._beam]:
                if self._time_remaining() < TIME_MARGIN:
                    break  # Corte de emergencia: tiempo agotado

                # Paso 1: Crear clon y aplicar movimiento raíz
                clone   = board.clone()
                clone.place_piece(move[0], move[1], self.player_id)

                # Paso 2: Actualizar hash incremental O(1) con el movimiento
                h_child = self._update_hash(
                    self._current_hash, move[0], move[1], self.player_id)

                # Paso 3: Evaluar con alpha-beta desde profundidad depth-1
                val = self._alphabeta(
                    clone, h_child, depth - 1, alpha, beta, False, droot=1)

                if val > depth_best_val:
                    depth_best_val  = val
                    depth_best_move = move

                alpha = max(alpha, depth_best_val)

            # Actualizar mejor jugada global si esta iteración mejoró
            if depth_best_val > best_val or depth == 1:
                best_val  = depth_best_val
                best_move = depth_best_move

            # Reordenar raíz: mejor jugada de la iteración va primero en la siguiente
            if best_move in root_moves:
                root_moves.remove(best_move)
                root_moves.insert(0, best_move)

        return best_move

    def _alphabeta(self, clone: HexBoard, h: int, depth: int, alpha: float, beta: float, maximizing: bool, droot: int) -> float:
        """
        Alpha-Beta recursivo con tabla de transposición de cotas,
        killer moves y history heuristic.

        Recibe:
            clone      (HexBoard): clon del tablero en el estado a evaluar.
            h          (int):      hash Zobrist del estado actual.
            depth      (int):      profundidad restante de búsqueda.
            alpha      (float):    mejor valor garantizado para el maximizador.
            beta       (float):    mejor valor garantizado para el minimizador.
            maximizing (bool):     True si es turno del maximizador (jugador propio).
            droot      (int):      nivel de profundidad desde la raíz (para killers).

        Retorna:
            float: valor minimax del estado, acotado por [alpha, beta].
        """
        # Paso 1: Verificar estados terminales
        if clone.check_connection(self.player_id):
            return  1_000_000.0  # Victoria propia
        if clone.check_connection(self.profile.opponent):
            return -1_000_000.0  # Derrota propia
        if depth == 0 or self._time_remaining() < TIME_MARGIN:
            return self._evaluate(clone)  # Profundidad agotada o tiempo crítico

        # Guardar ventana original para determinar tipo de cota al final
        alpha0 = alpha
        beta0  = beta

        # Paso 2: Consultar tabla de transposición con cotas (Bloque 2)
        tt_move = None
        entry   = self._tt.get(h)
        if entry is not None and entry[1] >= depth:
            cached_score, _, flag, cached_best = entry
            tt_move = cached_best  # Jugada almacenada: probar primero

            if flag == EXACT:
                return cached_score  # Valor exacto: devolver directamente
            if flag == LOWER_BOUND:
                alpha = max(alpha, cached_score)  # Acotar alpha hacia arriba
            elif flag == UPPER_BOUND:
                beta  = min(beta, cached_score)   # Acotar beta hacia abajo
            if alpha >= beta:
                return cached_score  # Corte por ventana vacía

        # Paso 3: Generar y ordenar movimientos (TT-move + killers + greedy)
        moves = self._order_moves_fast(clone, droot, tt_move)
        if not moves:
            return self._evaluate(clone)

        best_local = moves[0]  # Mejor jugada local encontrada hasta ahora
        opp        = self.profile.opponent

        # Paso 4: Expandir árbol según turno (maximizador o minimizador)
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
                    # Corte beta: registrar killer e history para este movimiento
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
                    # Corte alpha: registrar killer e history para este movimiento
                    self._register_cutoff(move, droot, depth)
                    break

        # Paso 5: Determinar tipo de cota y guardar en la TT
        if   value <= alpha0: flag = UPPER_BOUND  # No mejoró alpha: all-node
        elif value >= beta0:  flag = LOWER_BOUND  # Cortó beta: cut-node
        else:                 flag = EXACT         # Dentro de ventana: exacto
        self._tt_put(h, value, depth, flag, best_local)

        return value

    def _register_cutoff(self, move: tuple, droot: int, depth: int):
        """
        Registra un corte alpha o beta actualizando killer moves e history heuristic.

        Recibe:
            move  (tuple): jugada (r,c) que causó el corte.
            droot (int):   nivel de profundidad desde la raíz.
            depth (int):   profundidad restante en el momento del corte.
        """
        if move is None:
            return
        # Añadir a killers del nivel actual
        self._add_killer(droot, move)
        # Incrementar history con depth²: más peso a cortes en nodos profundos
        self._history[move] = self._history.get(move, 0) + depth * depth

    def _add_killer(self, droot: int, move: tuple):
        """
        Añade una jugada a la lista de killers del nivel droot (máximo 2).
        Usa política FIFO: si ya hay 2 killers, el más antiguo se elimina.

        Recibe:
            droot (int):   nivel de profundidad al que pertenece el killer.
            move  (tuple): jugada (r,c) a registrar como killer.
        """
        lst = self._killers.setdefault(droot, [])
        if move in lst:
            return  # Ya registrado, no duplicar
        lst.insert(0, move)  # Insertar al frente (killer más reciente primero)
        if len(lst) > 2:
            lst.pop()  # Eliminar el killer más antiguo


    # MCTS + ROLLOUT BFS (Bloque 4) — N > 11
    def _play_mcts(self, board: HexBoard) -> tuple:
        """
        Monte Carlo Tree Search con rollout BFS y win-rate warmup reordering.

        Recibe:
            board (HexBoard): estado actual del tablero.

        Retorna:
            tuple: (fila, columna) del hijo de la raíz con más visitas.
        """
        # Paso 1: Generar candidatos iniciales con ordenamiento greedy + frontera
        moves = self._order_moves_fast(board, droot=0)
        if not moves:
            return self._fallback(board)
        if len(moves) == 1:
            return moves[0]

        # Paso 2: Inicializar acumuladores de win-rate para el warmup reordering
        mcts_wins   = {m: 0.0 for m in moves}  # Victorias por primera jugada
        mcts_visits = {m: 0   for m in moves}  # Visitas por primera jugada
        WARMUP      = max(20, len(moves))       # Iteraciones antes de reordenar

        # Paso 3: Crear nodo raíz del árbol MCTS
        root = MCTSNode(
            move             = None,
            parent           = None,
            untried_moves    = list(moves),       # Candidatos ordenados por greedy
            player_who_moved = self.profile.opponent  # El oponente "jugó" antes de la raíz
        )

        iters = 0
        while self._time_remaining() > 0.05:
            node       = root
            sim_board  = board.clone()  # Clon independiente para esta simulación
            sim_player = self.player_id
            first_move = None  # Primera jugada propia en este path (para warmup)

            # FASE 1: SELECCIÓN — descender por UCB1 hasta nodo no expandido
            while node.is_fully_expanded() and node.children:
                node = node.best_child(UCB_C)
                sim_board.place_piece(
                    node.move[0], node.move[1], node.player_who_moved)
                if first_move is None and node.player_who_moved == self.player_id:
                    first_move = node.move
                sim_player = 3 - node.player_who_moved
                if sim_board.check_connection(1) or sim_board.check_connection(2):
                    break  # Estado terminal encontrado durante la selección

            # FASE 2: EXPANSIÓN — crear nuevo nodo hijo con siguiente candidato
            if node.untried_moves and not (
                sim_board.check_connection(1) or sim_board.check_connection(2)
            ):
                move = node.untried_moves.pop(0)  # Primer candidato no explorado
                sim_board.place_piece(move[0], move[1], sim_player)

                # Nodos internos usan frontera BFS (sin _known_cells del estado raíz)
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

            # FASE 3: SIMULACIÓN — estimar resultado con rollout BFS
            result = self._rollout_bfs(sim_board)

            # FASE 4: BACKPROPAGACIÓN — propagar resultado hacia la raíz
            win_val = 1.0 if result == self.player_id else (0.5 if result == 0 else 0.0)
            while node is not None:
                node.visits += 1
                if result == self.player_id:
                    node.wins += 1.0
                elif result == 0:
                    node.wins += 0.5  # Empate técnico (posición equilibrada)
                node = node.parent

            # Acumular estadísticas de primera jugada para el warmup
            if first_move is not None and first_move in mcts_wins:
                mcts_wins[first_move]   += win_val
                mcts_visits[first_move] += 1

            iters += 1

            # WARMUP REORDERING: tras WARMUP iteraciones, reordenar untried_moves
            # de la raíz por win-rate provisional (Hill-Climbing sobre jugadas)
            if iters == WARMUP and root.untried_moves:
                root.untried_moves.sort(
                    key=lambda m: (
                        -(mcts_wins[m] / mcts_visits[m]) if mcts_visits[m] > 0
                        else 0.0
                    )
                )

        if not root.children:
            return moves[0]  # Fallback: sin hijos expandidos, primer candidato greedy

        # Paso 4: Seleccionar el hijo con más visitas (criterio más robusto que win-rate)
        return max(root.children, key=lambda nd: nd.visits).move

    def _rollout_bfs(self, board: HexBoard) -> int:
        """
        Estima el ganador probable del estado actual mediante distancias BFS,
        sin simular jugadas hasta el estado terminal.

        Recibe:
            board (HexBoard): estado del tablero tras la expansión MCTS.

        Retorna:
            int: player_id del ganador probable, o 0 si hay empate técnico.
        """
        # Paso 1: Verificar si el estado ya es terminal
        if board.check_connection(self.player_id):
            return self.player_id
        if board.check_connection(self.profile.opponent):
            return self.profile.opponent

        # Paso 2: Calcular distancias mínimas de ambos jugadores
        d_mine = self._bfs_distance(board, self.player_id)
        d_opp  = self._bfs_distance(board, self.profile.opponent)

        # Clampear infinito para comparación segura
        max_d  = float(self._n * self._n)
        if d_mine == float('inf'): d_mine = max_d
        if d_opp  == float('inf'): d_opp  = max_d

        # Paso 3: Determinar ganador por diferencia de distancias
        # Umbral adaptativo: en tableros grandes, 2 pasos de diferencia ya es decisivo
        threshold = 2 if self._n > 13 else 3
        diff = d_opp - d_mine  # Positivo = ventaja propia

        if   diff >=  threshold: return self.player_id       # Ventaja clara propia
        elif diff <= -threshold: return self.profile.opponent # Ventaja clara rival
        elif d_mine < d_opp:     return self.player_id        # Ventaja leve propia
        elif d_opp  < d_mine:    return self.profile.opponent # Ventaja leve rival
        else:                    return 0                      # Empate técnico

    
    # FUNCIÓN DE EVALUACIÓN HEURÍSTICA (Bloque 1) — compartida por Minimax
    def _evaluate(self, board: HexBoard) -> float:
        """
        Función de evaluación multicomponente para estados no terminales.
        Usada en nodos hoja del árbol Minimax.

        Recibe:
            board (HexBoard): estado del tablero a evaluar.

        Retorna:
            float: score de la posición (positivo = ventaja propia,
                   negativo = desventaja).
        """
        my_id  = self.player_id
        opp_id = self.profile.opponent
        max_d  = float(self._n * self._n + 5)  # Valor para distancia infinita

        # Paso 1: Componente principal — diferencia de distancias BFS
        raw_mine = self._bfs_distance(board, my_id)
        raw_opp  = self._bfs_distance(board, opp_id)
        dist_mine = raw_mine if raw_mine != float('inf') else max_d
        dist_opp  = raw_opp  if raw_opp  != float('inf') else max_d

        # dist_opp - dist_mine: positivo si el oponente está más lejos de ganar
        score = W_PATH * (dist_opp - dist_mine)

        # Paso 2: Componentes adicionales en modo RICH (solo N <= 11)
        if self._eval_mode == "RICH":
            # Componente 2: conexiones virtuales (pares con 2 celdas vacías compartidas)
            vc_mine = self._count_virtual_connections(board, my_id)
            vc_opp  = self._count_virtual_connections(board, opp_id)
            score  += W_VIRTUAL * (vc_mine - vc_opp)

            # Componente 3: mayor componente conexa de piezas propias
            cc_mine = self._largest_connected_component(board, my_id)
            cc_opp  = self._largest_connected_component(board, opp_id)
            score  += W_CONNECT * (cc_mine - cc_opp)

            # Componente 4: potencial de puente (pares de piezas a distancia 1 o 2)
            bp_mine = self._bridge_potential(board, my_id)
            bp_opp  = self._bridge_potential(board, opp_id)
            score  += W_BRIDGE * (bp_mine - bp_opp)

            # Componente 5: control del centro del tablero
            score  += (self._center_control(board, my_id)
                       - self._center_control(board, opp_id))

        # Paso 3: Escalar por agresividad adaptativa del jugador
        return score * (0.5 + self.profile.aggression)

    def _bfs_distance(self, board: HexBoard, player_id: int) -> float:
        """
        0-1 BFS que devuelve solo la distancia mínima total (sin la matriz completa).
        Variante optimizada para ser usada repetidamente en la evaluación.

        Recibe:
            board     (HexBoard): estado actual del tablero.
            player_id (int):      jugador para el que se calcula la distancia.

        Retorna:
            float: distancia mínima para conectar los dos bordes del jugador,
                   o float('inf') si no existe camino viable.
        """
        n   = self._n
        INF = float('inf')
        opp = 3 - player_id
        dist = [[INF] * n for _ in range(n)]
        dq   = deque()

        # Definir fuentes y función objetivo según el jugador
        if player_id == 1:
            # P1 conecta columna 0 -> columna N-1
            sources = [(r, 0) for r in range(n)]
            goal_fn = lambda r, c: c == n - 1
        else:
            # P2 conecta fila 0 -> fila N-1
            sources = [(0, c) for c in range(n)]
            goal_fn = lambda r, c: r == n - 1

        # Insertar fuentes con su coste inicial
        for r, c in sources:
            cell = board.board[r][c]
            if cell == opp:
                continue
            cost = 0 if cell == player_id else 1
            if cost < dist[r][c]:
                dist[r][c] = cost
                (dq.appendleft if cost == 0 else dq.append)((cost, r, c))

        # Expandir hasta encontrar el borde objetivo
        while dq:
            cost, r, c = dq.popleft()
            if cost > dist[r][c]:
                continue
            if goal_fn(r, c):
                return cost  # Primera llegada al objetivo = distancia mínima
            for nr, nc in get_neighbors(r, c, n):
                cell = board.board[nr][nc]
                if cell == opp:
                    continue
                step     = 0 if cell == player_id else 1
                new_cost = cost + step
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    (dq.appendleft if step == 0 else dq.append)((new_cost, nr, nc))

        return INF  # Sin camino posible

    def _count_virtual_connections(self, board: HexBoard, player_id: int) -> int:
        """
        Cuenta los pares de piezas propias con exactamente 2 celdas vacías
        compartidas como vecinas (conexión virtual garantizada).

        Recibe:
            board     (HexBoard): estado del tablero.
            player_id (int):      jugador cuyas conexiones virtuales se cuentan.

        Retorna:
            int: número de conexiones virtuales del jugador.
        """
        n = self._n; grid = board.board; count = 0; seen: set = set()

        # Obtener todas las celdas propias
        own = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == player_id]

        for r1, c1 in own:
            nb1 = set(get_neighbors(r1, c1, n))

            for m1r, m1c in nb1:
                if grid[m1r][m1c] != 0:
                    continue  # Celda mediadora debe estar vacía

                for r2, c2 in get_neighbors(m1r, m1c, n):
                    if (r2, c2) == (r1, c1) or grid[r2][c2] != player_id:
                        continue  # Debe ser otra pieza propia

                    br = frozenset(((r1, c1), (r2, c2)))
                    if br in seen:
                        continue  # Par ya contado

                    # Verificar que exista una segunda celda mediadora vacía compartida
                    nb2 = set(get_neighbors(r2, c2, n))
                    shared = nb1 & nb2  # Vecinos comunes de ambas piezas
                    for m2r, m2c in shared:
                        if (m2r, m2c) != (m1r, m1c) and grid[m2r][m2c] == 0:
                            seen.add(br)
                            count += 1
                            break

        return count

    def _largest_connected_component(self, board: HexBoard, player_id: int) -> int:
        """
        Calcula el tamaño de la mayor componente conexa de piezas del jugador
        mediante DFS iterativo.

        Recibe:
            board     (HexBoard): estado del tablero.
            player_id (int):      jugador cuya componente conexa se analiza.

        Retorna:
            int: número de piezas en la mayor componente conexa.
        """
        n = self._n; grid = board.board; visited: set = set(); largest = 0

        for r in range(n):
            for c in range(n):
                if grid[r][c] != player_id or (r, c) in visited:
                    continue

                # DFS iterativo desde la celda (r,c)
                size = 0; stack = [(r, c)]
                while stack:
                    cr, cc = stack.pop()
                    if (cr, cc) in visited:
                        continue
                    visited.add((cr, cc))
                    size += 1
                    for nr, nc in get_neighbors(cr, cc, n):
                        if grid[nr][nc] == player_id and (nr, nc) not in visited:
                            stack.append((nr, nc))
                largest = max(largest, size)

        return largest

    def _bridge_potential(self, board: HexBoard, player_id: int) -> float:
        """
        Calcula el potencial de puente: señal de conectividad potencial entre
        piezas propias a distancia 1 (vecinos directos) o 2 (vecinos de vecinos).

        Recibe:
            board     (HexBoard): estado del tablero.
            player_id (int):      jugador cuyo potencial se evalúa.

        Retorna:
            float: score acumulado (distancia 1 = +2.0, distancia 2 = +1.0
                   por cada par de piezas).
        """
        n = self._n; grid = board.board
        cells = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == player_id]
        score = 0.0

        for idx, (r1, c1) in enumerate(cells):
            s1 = set(get_neighbors(r1, c1, n))
            for r2, c2 in cells[idx + 1:]:  # Evitar pares duplicados
                if (r2, c2) in s1:
                    score += 2.0  # Par a distancia 1: vecinos directos
                else:
                    # Verificar si están a distancia 2 (vecinos de vecinos)
                    for nr, nc in s1:
                        if (r2, c2) in get_neighbors(nr, nc, n):
                            score += 1.0
                            break

        return score

    def _center_control(self, board: HexBoard, player_id: int) -> float:
        """
        Calcula el control del centro: suma de proximidades al centro de todas
        las piezas propias, escalada por W_CENTER.

        Recibe:
            board     (HexBoard): estado del tablero.
            player_id (int):      jugador cuyo control del centro se mide.

        Retorna:
            float: score de control del centro (mayor = piezas más centrales).
        """
        n = self._n; mid = n / 2.0; grid = board.board; total = 0.0
        for r in range(n):
            for c in range(n):
                if grid[r][c] == player_id:
                    # Proximidad = N - distancia Manhattan al centro
                    total += n - abs(r - mid) - abs(c - mid)
        return total * W_CENTER

    
    # TABLA DE TRANSPOSICIÓN ZOBRIST (Bloque 2)
    def _init_zobrist(self):
        """
        Inicializa la tabla Zobrist si el tamaño del tablero cambió.
        Genera números aleatorios de 64 bits para cada combinación
        (jugador, fila, columna) con semilla fija para reproducibilidad.
        No recibe ni retorna valores; modifica el estado interno del agente.
        """
        if self._zobrist_n == self._n and self._zobrist is not None:
            return  # Ya inicializada para este tamaño, no regenerar

        rng = random.Random(0xDEADBEEF)  # Semilla fija: misma tabla en todas las partidas
        n   = self._n

        # Tabla 3D: [jugador (0-2)][fila][columna] -> número de 64 bits
        self._zobrist = [
            [[rng.getrandbits(64) for _ in range(n)] for _ in range(n)]
            for _ in range(3)
        ]
        self._zobrist_n    = n
        self._tt.clear()       # Invalidar TT anterior (era para otro tamaño)
        self._current_hash = 0

    def _compute_hash(self, board: HexBoard) -> int:
        """
        Calcula el hash Zobrist completo del tablero en O(N²).
        Solo se llama una vez al inicio de cada partida; el resto de
        actualizaciones se realizan incrementalmente con _update_hash.

        Recibe:
            board (HexBoard): estado del tablero.

        Retorna:
            int: hash de 64 bits del estado completo del tablero.
        """
        h = 0; zob = self._zobrist
        for r, row in enumerate(board.board):
            for c, val in enumerate(row):
                if val:
                    h ^= zob[val][r][c]  # XOR acumulativo de todas las piezas
        return h

    def _update_hash(self, h: int, r: int, c: int, player_id: int) -> int:
        """
        Actualiza el hash Zobrist incrementalmente en O(1) al colocar una pieza.
        Propiedad XOR: aplicar dos veces la misma operación restaura el valor original.

        Recibe:
            h         (int): hash actual del tablero.
            r         (int): fila de la celda donde se coloca la pieza.
            c         (int): columna de la celda donde se coloca la pieza.
            player_id (int): jugador que coloca la pieza (1 o 2).

        Retorna:
            int: nuevo hash tras la colocación de la pieza.
        """
        return h ^ self._zobrist[player_id][r][c]

    def _tt_put(self, h: int, score: float, depth: int, flag: int, best_move):
        """
        Almacena una entrada en la tabla de transposición con política de evicción
        cuando la tabla está llena: primero entradas de profundidad <= 1, luego aleatorio.

        Recibe:
            h         (int):         hash Zobrist del estado.
            score     (float):       valor minimax calculado.
            depth     (int):         profundidad a la que fue calculado.
            flag      (int):         tipo de cota (EXACT, LOWER_BOUND, UPPER_BOUND).
            best_move (tuple|None):  mejor jugada encontrada en este nodo.
        """
        if len(self._tt) >= TT_MAX_SIZE:
            # Paso 1: Intentar evictar entradas superficiales (depth <= 1)
            shallow = [k for k, v in self._tt.items() if v[1] <= 1]
            victims = shallow[:TT_MAX_SIZE // 4]

            # Paso 2: Si no hay suficientes superficiales, completar con aleatorios
            if len(victims) < TT_MAX_SIZE // 4:
                extra = random.sample(
                    list(self._tt.keys()),
                    min(TT_MAX_SIZE // 4 - len(victims), len(self._tt)))
                for k in extra:
                    if k not in victims:
                        victims.append(k)

            # Paso 3: Eliminar las víctimas seleccionadas
            for k in victims:
                self._tt.pop(k, None)

        # Guardar nueva entrada: (score, depth, flag, best_move)
        self._tt[h] = (score, depth, flag, best_move)

    
    # UTILIDADES Y MÉTODOS AUXILIARES
    def _opening_move(self, board: HexBoard):
        """
        Retorna la celda central del tablero como jugada de apertura si está vacía.
        El centro tiene la mayor conectividad estructural en HEX.

        Recibe:
            board (HexBoard): estado actual del tablero.

        Retorna:
            tuple|None: (center, center) si disponible en los primeros 2 turnos,
                        None en caso contrario.
        """
        center = self._n // 2
        if self._move_count <= 2 and board.board[center][center] == 0:
            return (center, center)
        return None

    def _fallback(self, board: HexBoard):
        """
        Devuelve la primera celda vacía encontrada. Garantía de último recurso
        para que play() nunca retorne None.

        Recibe:
            board (HexBoard): estado actual del tablero.

        Retorna:
            tuple|None: (fila, columna) de la primera celda vacía, o None si
                        el tablero está completamente lleno.
        """
        for r in range(self._n):
            for c in range(self._n):
                if board.board[r][c] == 0:
                    return (r, c)
        return None

    def _find_new_piece(self, board: HexBoard):
        """
        Localiza la jugada nueva del oponente buscando primero en la frontera
        de celdas conocidas O(k·6), con fallback O(N²) para casos borde.

        Recibe:
            board (HexBoard): estado actual del tablero (ya incluye la jugada del oponente).

        Retorna:
            tuple|None: (fila, columna) de la nueva pieza del oponente,
                        None si no se encuentra (tablero sin cambios).
        """
        grid = board.board
        n    = self._n

        # Paso 1: Buscar en frontera O(k·6) — la nueva pieza casi siempre
        # es adyacente a una pieza ya colocada
        for r, c in self._known_cells:
            for nr, nc in get_neighbors(r, c, n):
                if (grid[nr][nc] == self.profile.opponent
                        and (nr, nc) not in self._known_cells):
                    return (nr, nc)

        # Paso 2: Fallback O(N²) — garantía de correctitud del hash Zobrist
        # Se activa solo si la frontera falla (primera jugada, jugada aislada)
        for r in range(n):
            for c in range(n):
                if (grid[r][c] == self.profile.opponent
                        and (r, c) not in self._known_cells):
                    return (r, c)

        return None

    def _get_empty_cells(self, board: HexBoard) -> list:
        """
        Devuelve todas las celdas vacías del tablero en O(N²).
        Usado como respaldo cuando la frontera no está disponible.

        Recibe:
            board (HexBoard): estado del tablero.

        Retorna:
            list: lista de tuplas (r, c) con todas las celdas vacías.
        """
        n = self._n
        return [(r, c) for r in range(n) for c in range(n)
                if board.board[r][c] == 0]

    def _get_frontier(self, board: HexBoard,
                       occupied: set = None) -> list:
        """
        Devuelve las celdas vacías adyacentes a piezas ya colocadas (frontera BFS).
        Reduce el espacio de candidatos de O(N²) a O(k·6) cuando se provee 'occupied'.

        Recibe:
            board    (HexBoard):  estado del tablero.
            occupied (set|None):  conjunto de celdas ocupadas conocidas.
                                  Si se proporciona, itera solo sobre ellas O(k·6).
                                  Si es None, hace scan completo O(N²) del tablero.

        Retorna:
            list: celdas vacías de la frontera. Si la frontera está vacía
                  (tablero vacío), retorna las celdas del anillo central.
        """
        n    = self._n
        grid = board.board
        seen: set = set()
        frontier  = []

        if occupied:
            # O(k·6): iterar solo sobre celdas ocupadas conocidas
            for r, c in occupied:
                for nr, nc in get_neighbors(r, c, n):
                    if grid[nr][nc] == 0 and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        frontier.append((nr, nc))
        else:
            # O(N²): scan completo para sim_board en MCTS (sin estado incremental)
            for r in range(n):
                for c in range(n):
                    if grid[r][c] == 0:
                        continue
                    for nr, nc in get_neighbors(r, c, n):
                        if grid[nr][nc] == 0 and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            frontier.append((nr, nc))

        # Si frontera vacía (tablero vacío), usar anillo central como candidatos
        if not frontier:
            mid = n // 2
            for r in range(max(0, mid-1), min(n, mid+2)):
                for c in range(max(0, mid-1), min(n, mid+2)):
                    if grid[r][c] == 0 and (r, c) not in seen:
                        seen.add((r, c))
                        frontier.append((r, c))
        return frontier

    def _time_remaining(self) -> float:
        """
        Calcula el tiempo restante del presupuesto del turno actual.

        Retorna:
            float: segundos restantes (negativo si se excedió el límite).
        """
        return self._time_limit - (time.time() - self._start_time)