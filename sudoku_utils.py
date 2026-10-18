import math


def units(N):

    b = math.isqrt(N)
    rows = [[(r, c) for c in range(N)] for r in range(N)]
    cols = [[(r, c) for r in range(N)] for c in range(N)]
    boxes = [[(br * b + r, bc * b + c) for r in range(b) for c in range(b)]
             for br in range(b) for bc in range(b)]
    return rows + cols + boxes


def verify_sudoku(puzzle, solution):
    if not solution:
        return False
    N = len(puzzle)
    target = set(range(1, N + 1))
    clues_ok = all(puzzle[r][c] in (0, solution[r][c]) for r in range(N) for c in range(N))
    units_ok = all({solution[r][c] for r, c in unit} == target for unit in units(N))
    return clues_ok and units_ok