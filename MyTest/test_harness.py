"""
test_harness.py — SmartPlayer vs SmartPlayer en tableros de distintos tamaños
"""
import time
import sys
from board import HexBoard
from solution import SmartPlayer

BOARD_SIZES = [5, 7, 9, 11, 13, 14, 17, 20]
TIME_WARN   = 4.5
TIME_LIMIT  = 5.0
MAX_TURNS   = 400

R="\033[91m"; Y="\033[93m"; G="\033[92m"; C="\033[96m"; B="\033[1m"; X="\033[0m"

def print_board(board):
    n = board.size
    sym = {0:"·", 1:"R", 2:"B"}
    for r in range(n):
        print("  " + " "*r + " ".join(sym[board.board[r][c]] for c in range(n)))
    print()

def play_game(n):
    board   = HexBoard(n)
    players = {1: SmartPlayer(1), 2: SmartPlayer(2)}
    for pid in (1,2):
        players[pid]._n = n
        players[pid]._configure_regime()

    stats = {"n":n,"turns":[],"winner":None,"disqualified":False,
             "disq_player":None,"invalid_moves":0,"total_time":0.0}

    print(f"\n{'─'*72}")
    print(f"{B}  N={n}  │  P1🔴(IZQ↔DER) vs P2🔵(ARR↔ABA)"
          f"  │  d_max: P1={players[1]._max_depth} P2={players[2]._max_depth}"
          f"  eval={players[1]._eval_mode}{X}")
    print(f"{'─'*72}")
    print(f"  {'Turno':<6}{'Jugador':<8}{'Mov':<10}{'Tiempo(s)':<13}{'Válido':<8}Estado")
    print(f"  {'─'*5} {'─'*7} {'─'*9} {'─'*12} {'─'*7} {'─'*20}")

    current = 1
    t_game  = time.time()

    for turn in range(1, MAX_TURNS+1):
        p   = players[current]
        t0  = time.time()
        mv  = p.play(board.clone())
        elapsed = time.time() - t0

        r, c      = mv
        in_bounds = 0 <= r < n and 0 <= c < n
        was_empty = in_bounds and board.board[r][c] == 0
        valid     = in_bounds and was_empty
        if not valid: stats["invalid_moves"] += 1

        if elapsed >= TIME_LIMIT:
            st = f"{R}❌ DESCALIFICADO ({elapsed:.2f}s){X}"
            stats["disqualified"] = True; stats["disq_player"] = current
        elif elapsed >= TIME_WARN:
            st = f"{Y}⚠️  LENTO ({elapsed:.2f}s){X}"
        elif not valid:
            st = f"{R}❌ MOV INVÁLIDO{X}"
        else:
            st = f"{G}✓{X}"

        icon = "🔴" if current==1 else "🔵"
        print(f"  {turn:<6}P{current}{icon:<6}({r},{c}){'':>4}{elapsed:<13.4f}{'✓' if valid else '✗':<8}{st}")

        stats["turns"].append({"turn":turn,"player":current,"move":mv,
                               "time":elapsed,"valid":valid,"depth":p._max_depth})

        if stats["disqualified"]: break

        if valid: board.place_piece(r, c, current)

        if board.check_connection(current):
            stats["winner"] = current
            break

        current = 2 if current==1 else 1

    stats["total_time"] = time.time() - t_game

    if n <= 11:
        print(f"\n  Tablero final:")
        print_board(board)
    return stats

def print_summary(s):
    turns  = s["turns"]
    times  = [t["time"] for t in turns]
    avg_t  = sum(times)/len(times) if times else 0
    max_t  = max(times) if times else 0
    max_d  = max(t["depth"] for t in turns) if turns else 0
    close  = [t for t in turns if TIME_WARN <= t["time"] < TIME_LIMIT]

    print(f"\n  {'─'*60}")
    print(f"  {B}RESUMEN N={s['n']}{X}")
    print(f"  Turnos: {len(turns)}  │  T.total: {s['total_time']:.2f}s"
          f"  │  T.prom: {avg_t:.4f}s  │  T.max: {max_t:.4f}s")
    print(f"  Prof.max configurada: {max_d}  │  Inválidas: {s['invalid_moves']}")

    if s["disqualified"]:
        print(f"  Veredicto: {R}{B}❌ DESCALIFICADO (P{s['disq_player']} > 5s){X}")
    elif s["winner"]:
        icon = "🔴" if s["winner"]==1 else "🔵"
        print(f"  Veredicto: {G}{B}✓ VÁLIDA — Ganador P{s['winner']} {icon}{X}")
    else:
        print(f"  Veredicto: {Y}⚠️  Sin ganador (límite turnos){X}")

    if close:
        print(f"  {Y}⚠️  {len(close)} turno(s) cerca del límite:{X}")
        for t in close:
            print(f"     Turno {t['turn']} P{t['player']} → {t['time']:.4f}s")

def print_global(all_stats):
    print(f"\n{'═'*72}")
    print(f"{B}{C}  RESUMEN GLOBAL{X}")
    print(f"{'═'*72}")
    print(f"  {'N':<6}{'Turnos':<8}{'T.prom':<12}{'T.max':<12}{'d.max':<8}{'Inváli.':<9}Veredicto")
    print(f"  {'─'*5} {'─'*7} {'─'*11} {'─'*11} {'─'*7} {'─'*8} {'─'*18}")

    for s in all_stats:
        turns = s["turns"]
        times = [t["time"] for t in turns]
        avg_t = sum(times)/len(times) if times else 0
        max_t = max(times) if times else 0
        max_d = max(t["depth"] for t in turns) if turns else 0

        if s["disqualified"]: vrd = f"{R}❌ DESCALIFICADO{X}"
        elif s["winner"]:     vrd = f"{G}✓ OK (P{s['winner']}){X}"
        else:                 vrd = f"{Y}⚠️  Sin ganador{X}"

        print(f"  {s['n']:<6}{len(turns):<8}{avg_t:<12.4f}{max_t:<12.4f}{max_d:<8}{s['invalid_moves']:<9}{vrd}")

    bad = [s for s in all_stats if s["disqualified"] or s["invalid_moves"]>0]
    print(f"\n  {'─'*60}")
    if not bad:
        print(f"  {G}{B}✅ Todos los tableros: sin descalificaciones ni movimientos inválidos.{X}")
    else:
        for s in bad:
            if s["disqualified"]: print(f"  {R}❌ N={s['n']}: descalificación{X}")
            if s["invalid_moves"]: print(f"  {R}❌ N={s['n']}: {s['invalid_moves']} movimiento(s) inválido(s){X}")
    print()

if __name__ == "__main__":
    print(f"{'═'*72}")
    print(f"{B}{C}  TEST HARNESS — SmartPlayer vs SmartPlayer{X}")
    print(f"{'═'*72}")
    print(f"  Tableros : {BOARD_SIZES}")
    print(f"  Límite   : {TIME_LIMIT}s/jugada  │  Alerta: {TIME_WARN}s  │  Corte automático: SÍ\n")

    all_stats = []
    for n in BOARD_SIZES:
        s = play_game(n)
        print_summary(s)
        all_stats.append(s)
        if s["disqualified"]:
            print(f"  {Y}→ Continuando con siguiente N...{X}\n")

    print_global(all_stats)
