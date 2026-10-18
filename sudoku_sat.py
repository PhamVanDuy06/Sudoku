"""
Giải Sudoku bằng SAT (PySAT).

Mô hình: biến x[r][c][v] đúng <=> ô (r, c) mang giá trị v + 1. Ràng buộc "đúng một" (ALO + AMO)
cho mỗi ô, mỗi giá trị trên mỗi hàng, cột, khối; các ô gợi ý là mệnh đề đơn.
Kỹ thuật AMO được chọn qua `enc_type` (xem amo_encodings.py).
"""
import math
import multiprocessing as mp
import queue
import time

from pysat.formula import CNF
from pysat.solvers import Cadical195, Glucose4, MapleChrono, Minisat22

from amo_encodings import encode_amo
from sudoku_utils import units, verify_sudoku

SAT_SOLVER_CLASSES = {
    "cadical": Cadical195,
    "glucose": Glucose4,
    "minisat": Minisat22,
    "maple": MapleChrono,
}


class SATSudokuSolver:
    def __init__(self, N=9, enc_type="pairwise", solver_name="cadical"):
        self.N = N
        self.b = math.isqrt(N)
        self.enc_type = enc_type
        self.solver_name = solver_name
        self.top_id = N ** 3  # số hiệu biến lớn nhất đang dùng (biến phụ đánh số tiếp sau đó)

    # ---- đánh số biến ----
    def var(self, r, c, v):
        return r * self.N * self.N + c * self.N + v + 1

    def inv_var(self, var_id):
        var_id -= 1
        return var_id // (self.N * self.N), (var_id // self.N) % self.N, var_id % self.N

    # ---- sinh CNF ----
    def add_exactly_one(self, cnf, lits):
        cnf.append(lits)  # ALO: ít nhất một
        amo, self.top_id = encode_amo(lits, self.enc_type, self.top_id)  # AMO: nhiều nhất một
        for clause in amo:
            cnf.append(clause)

    def encode(self, grid):
        """Trả về (CNF, thời gian sinh CNF tính bằng ms)."""
        t0 = time.perf_counter()
        N = self.N
        cnf = CNF()
        self.top_id = N ** 3

        for r in range(N):
            for c in range(N):
                if grid[r][c]:
                    cnf.append([self.var(r, c, grid[r][c] - 1)])  # ô gợi ý

        for r in range(N):
            for c in range(N):
                self.add_exactly_one(cnf, [self.var(r, c, v) for v in range(N)])  # mỗi ô một giá trị

        for unit in units(N):  # mỗi giá trị xuất hiện đúng một lần trong mỗi hàng, cột, khối
            for v in range(N):
                self.add_exactly_one(cnf, [self.var(r, c, v) for r, c in unit])

        return cnf, (time.perf_counter() - t0) * 1000

    def extract(self, model):
        sol = [[0] * self.N for _ in range(self.N)]
        for lit in model:
            if 0 < lit <= self.N ** 3:
                r, c, v = self.inv_var(lit)
                sol[r][c] = v + 1
        return sol

    # ---- giải ----
    def build(self, grid):
        cnf, t_encode = self.encode(grid)
        solver_cls = SAT_SOLVER_CLASSES.get(self.solver_name.lower(), Cadical195)
        t0 = time.perf_counter()
        solver = solver_cls(bootstrap_with=cnf.clauses)
        t_load = (time.perf_counter() - t0) * 1000
        info = {"t_encode": t_encode, "t_load": t_load,
                "vars": self.top_id, "clauses": len(cnf.clauses)}
        return solver, info

    def _search(self, solver, grid):
        t0 = time.perf_counter()
        sat = solver.solve()
        t_solve = (time.perf_counter() - t0) * 1000
        if not sat:
            return "UNSAT", None, t_solve
        sol = self.extract(solver.get_model())
        return ("OK" if verify_sudoku(grid, sol) else "WRONG"), sol, t_solve

    def solve(self, grid):
        solver, info = self.build(grid)
        try:
            status, sol, t_solve = self._search(solver, grid)
        finally:
            solver.delete()
        return {"status": status, "t_build": info["t_encode"] + info["t_load"],
                "t_solve": t_solve, "solution": sol, **info}


# Giải có timeout: chạy trong tiến trình con để có thể dừng cả CaDiCaL (không ngắt được).

def _worker(q, grid, enc_type, solver_name):
    try:
        s = SATSudokuSolver(len(grid), enc_type, solver_name)
        solver, info = s.build(grid)
        q.put(("built", info))
        status, _, t_solve = s._search(solver, grid)
        q.put(("done", {"status": status, "t_solve": t_solve}))
    except Exception as e:  # noqa: BLE001
        q.put(("error", str(e)))


def _receive(q, timeout):
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return None


def solve_with_timeout(grid, enc_type, solver_name, timeout, build_limit=900):

    ctx = mp.get_context("spawn")  # bắt buộc trên Windows
    q = ctx.Queue()
    proc = ctx.Process(target=_worker, args=(q, grid, enc_type, solver_name))
    proc.start()
    result = {"status": "ERROR", "t_build": 0.0, "t_solve": 0.0}
    try:
        msg = _receive(q, build_limit)
        if msg is None:
            result["status"] = "ERROR: build quá hạn"
            return result
        kind, payload = msg
        if kind == "error":
            result["status"] = f"ERROR: {payload}"
            return result
        result.update(payload)
        result["t_build"] = payload["t_encode"] + payload["t_load"]

        msg = _receive(q, timeout)
        if msg is None:
            result.update(status="TIMEOUT", t_solve=timeout * 1000.0)
            return result
        kind, payload = msg
        if kind == "error":
            result["status"] = f"ERROR: {payload}"
        else:
            result.update(payload)
        return result
    finally:
        if proc.is_alive():
            proc.terminate()
        proc.join()