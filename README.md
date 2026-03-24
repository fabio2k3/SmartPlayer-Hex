# Proyecto 1 — Jugador Autónomo para HEX
**Inteligencia Artificial — Facultad de Matemática y Computación**  
Fabio Victor Alonso Bañobre

---

## Descripción del Proyecto

HEX es un juego de estrategia para dos jugadores en un tablero de $N \times N$ celdas hexagonales. El Jugador 1 debe conectar los bordes izquierdo y derecho; el Jugador 2, los bordes superior e inferior. El primero en lograrlo gana.

El objetivo del proyecto es desarrollar un **agente autónomo** capaz de tomar decisiones óptimas dentro de un límite de 5 segundos por jugada, para cualquier tamaño de tablero.

---

## Solución Implementada

El agente adopta una **estrategia híbrida** que selecciona el algoritmo de búsqueda según el tamaño del tablero:

- **N ≤ 11 → Minimax con Poda Alpha-Beta e Iterative Deepening (IDAB):** búsqueda exhaustiva y determinista enriquecida con tabla de transposición Zobrist, killer moves, history heuristic y ordenamiento de acciones basado en landmarks STRIPS.

- **N > 11 → Monte Carlo Tree Search (MCTS):** búsqueda estocástica guiada por UCB1, con rollout informado por BFS, frontera de expansión O(k) y reordenamiento adaptativo por win-rate.

Ambos regímenes incorporan razonamiento CSP para detección de jugadas forzadas y planificación STRIPS para priorizar acciones relevantes al plan ganador actual. Para una explicación detallada de cada decisión de diseño se recomienda la lectura del informe técnico incluido en este repositorio.

---

## Estructura del Repositorio

```
Fabio_Alonso_Bañobre/
│
├── solution.py           # Implementación del agente SmartPlayer
│
├── informe.pdf           # Informe técnico del proyecto
│                         # Contiene la justificación completa de cada
│                         # decisión de diseño y los algoritmos implementados.
│                         # Se recomienda su lectura para comprender
│                         # en profundidad la solución desarrollada.
│
└── MyTest/               # Entorno de prueba local
    ├── boardTest.py      # Implementación del tablero HEX para pruebas
    ├── playerTest.py     # Clase base Player para pruebas
    ├── solution.py       # Copia del agente para ejecutar los tests
    └── test_hex.py       # Script de prueba: enfrenta dos agentes
                          # y muestra resultados por tamaño de tablero
```

---

## Ejecución del Test Local

```bash
cd MyTest
python test_hex.py
```

El script ejecuta partidas entre dos instancias del agente para distintos valores de N, mostrando el ganador, el número de turnos y los tiempos por jugada.

---

## Dependencias

No se requieren librerías externas. El agente utiliza únicamente módulos de la biblioteca estándar de Python.
