import json
import os
import random
import time

from pysat.solvers import Glucose4

from sudoku_sat import SATSudokuSolver

# Tỉ lệ ô gợi ý mục tiêu theo cỡ lưới (0 = xóa tới mức tối thiểu còn nghiệm duy nhất)
DEFAULT_TARGET = {9: 0.0, 16: 0.30, 25: 0.35, 36: 0.40, 49: 0.45, 64: 0.50}


def shuffled_solution(b, rng):
    N = b * b

    def pattern(r, c):
        return (b * (r % b) + r // b + c) % N

    def shuffled_axis():
        groups = rng.sample(range(b), b)
        return [g * b + x for g in groups for x in rng.sample(range(b), b)]

    rows, cols = shuffled_axis(), shuffled_axis()
    digits = rng.sample(range(1, N + 1), N)
    return [[digits[pattern(r, c)] for c in cols] for r in rows]


def make_puzzle(b, seed, target_ratio=None, time_limit=None):
    N = b * b
    rng = random.Random(seed)
    if target_ratio is None:
        target_ratio = DEFAULT_TARGET.get(N, 0.4)
    goal = target_ratio * N * N
    deadline = time.monotonic() + time_limit if time_limit else None

    full = shuffled_solution(b, rng)

    # SAT solver giữ ràng buộc Sudoku (không có gợi ý) và cấm đúng nghiệm gốc.
    # => đề là duy nhất  <=>  UNSAT khi giả định các ô gợi ý hiện có.
    enc = SATSudokuSolver(N, "seqcounter", "glucose")
    cnf, _ = enc.encode([[0] * N for _ in range(N)])
    solver = Glucose4(bootstrap_with=cnf.clauses)
    solver.add_clause([-enc.var(r, c, full[r][c] - 1) for r in range(N) for c in range(N)])

    grid = [row[:] for row in full]
    cells = [(r, c) for r in range(N) for c in range(N)]
    rng.shuffle(cells)
    n_clues = N * N
    try:
        for r, c in cells:
            if n_clues <= goal:
                break
            if deadline and time.monotonic() > deadline:
                break
            value, grid[r][c] = grid[r][c], 0
            clues = [enc.var(rr, cc, grid[rr][cc] - 1)
                     for rr in range(N) for cc in range(N) if grid[rr][cc]]
            if solver.solve(assumptions=clues):
                grid[r][c] = value
            else:
                n_clues -= 1
    finally:
        solver.delete()
    return grid, n_clues


def load_or_make(b, seed, target_ratio=None, time_limit=None, cache_dir=None):
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"sudoku_{b * b}_seed{seed}.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data["grid"], data["clues"]
    grid, clues = make_puzzle(b, seed, target_ratio, time_limit)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"grid": grid, "clues": clues}, f)
    return grid, clues