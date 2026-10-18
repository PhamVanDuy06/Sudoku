import time
import math
import os
from ortools.sat.python import cp_model
from sudoku_sat import verify_sudoku

# Nạp DLL CPLEX
CPLEX_DIR = r"C:\Users\THINKPAD\Desktop\Sudoku"
cplex_bin = os.path.join(CPLEX_DIR, "cplex", "bin", "x64_win64")
cpo_bin = os.path.join(CPLEX_DIR, "cpoptimizer", "bin", "x64_win64")
os.environ["PATH"] = cplex_bin + os.pathsep + cpo_bin + os.pathsep + os.environ.get("PATH", "")

from docplex.cp.config import context
from docplex.cp.model import CpoModel
from docplex.mp.model import Model as CplexMIPModel
import gurobipy as gp
from gurobipy import GRB

context.solver.local.execfile = os.path.join(cpo_bin, "cpoptimizer.exe")

class CPILPSolvers:
    @staticmethod
    def solve_cpsat(grid, timeout_sec=60):
        N = len(grid)
        b = int(math.isqrt(N))

        t0 = time.perf_counter()
        model = cp_model.CpModel()
        vars_grid = {}
        for r in range(N):
            for c in range(N):
                if grid[r][c] != 0:
                    vars_grid[(r, c)] = model.NewConstant(grid[r][c])
                else:
                    vars_grid[(r, c)] = model.NewIntVar(1, N, f"x_{r}_{c}")

        for r in range(N): model.AddAllDifferent([vars_grid[(r, c)] for c in range(N)])
        for c in range(N): model.AddAllDifferent([vars_grid[(r, c)] for r in range(N)])
        for br in range(b):
            for bc in range(b):
                model.AddAllDifferent([vars_grid[(br * b + r, bc * b + c)] for r in range(b) for c in range(b)])
        t_build = (time.perf_counter() - t0) * 1000

        solver = cp_model.CpSolver()
        # ÉP ĐƠN LUỒNG & TIMEOUT ĐỂ CÔNG BẰNG VỚI SAT
        solver.parameters.num_workers = 1
        solver.parameters.max_time_in_seconds = timeout_sec

        t1 = time.perf_counter()
        status = solver.Solve(model)
        t_solve = (time.perf_counter() - t1) * 1000

        ok = (status == cp_model.OPTIMAL or status == cp_model.FEASIBLE)
        sol = None
        if ok:
            sol = [[solver.Value(vars_grid[(r, c)]) for c in range(N)] for r in range(N)]
        return {"ok": verify_sudoku(grid, sol), "t_build": t_build, "t_solve": t_solve}

    @staticmethod
    def solve_cplex_cp(grid, timeout_sec=60):
        N = len(grid)
        b = int(math.isqrt(N))
        t0 = time.perf_counter()
        try:
            mdl = CpoModel(name='CPLEX_CP')
            x = [[mdl.integer_var(1, N, name=f"x_{r}_{c}") for c in range(N)] for r in range(N)]
            for r in range(N):
                for c in range(N):
                    if grid[r][c] != 0: mdl.add(x[r][c] == grid[r][c])
            for r in range(N): mdl.add(mdl.all_diff(x[r]))
            for c in range(N): mdl.add(mdl.all_diff([x[r][c] for r in range(N)]))
            for br in range(b):
                for bc in range(b):
                    mdl.add(mdl.all_diff([x[br * b + r][bc * b + c] for r in range(b) for c in range(b)]))
            t_build = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            # ÉP 1 THREAD VÀ TIMEOUT
            msol = mdl.solve(LogVerbosity='Quiet', TimeLimit=timeout_sec, Workers=1)
            t_solve = (time.perf_counter() - t1) * 1000

            ok = msol.is_solution()
            sol = None
            if ok:
                sol = [[msol[x[r][c]] for c in range(N)] for r in range(N)]
            return {"ok": verify_sudoku(grid, sol), "t_build": t_build, "t_solve": t_solve}
        except Exception as e:
            return {"ok": False, "t_build": 0, "t_solve": "Lỗi runtime/bộ nhớ"}

    @staticmethod
    def solve_gurobi(grid, timeout_sec=60):
        N = len(grid)
        b = int(math.isqrt(N))
        t0 = time.perf_counter()
        try:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("Threads", 1)  # Đơn luồng
            env.setParam("TimeLimit", timeout_sec)
            env.start()

            model = gp.Model("Gurobi_Sudoku", env=env)
            x = model.addVars(N, N, N, vtype=GRB.BINARY, name="x")

            for r in range(N):
                for c in range(N):
                    if grid[r][c] != 0: model.addConstr(x[r, c, grid[r][c] - 1] == 1)
            for r in range(N):
                for c in range(N): model.addConstr(x.sum(r, c, '*') == 1)
            for r in range(N):
                for v in range(N): model.addConstr(x.sum(r, '*', v) == 1)
            for c in range(N):
                for v in range(N): model.addConstr(x.sum('*', c, v) == 1)
            for br in range(b):
                for bc in range(b):
                    for v in range(N):
                        model.addConstr(gp.quicksum(x[br * b + r, bc * b + c, v] for r in range(b) for c in range(b)) == 1)
            t_build = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            model.optimize()
            t_solve = (time.perf_counter() - t1) * 1000

            ok = (model.status == GRB.OPTIMAL)
            sol = None
            if ok:
                sol = [[0]*N for _ in range(N)]
                for r in range(N):
                    for c in range(N):
                        for v in range(N):
                            if x[r, c, v].X > 0.5: sol[r][c] = v + 1
            return {"ok": verify_sudoku(grid, sol), "t_build": t_build, "t_solve": t_solve}
        except gp.GurobiError:
            return {"ok": False, "t_build": 0, "t_solve": "Vượt giới hạn Trial (2000 biến)"}

    @staticmethod
    def solve_cplex_mip(grid, timeout_sec=60):
        N = len(grid)
        b = int(math.isqrt(N))
        t0 = time.perf_counter()
        try:
            mdl = CplexMIPModel(name='CPLEX_MIP')
            mdl.set_log_output(False)
            mdl.parameters.threads = 1  # Đơn luồng
            mdl.parameters.timelimit = timeout_sec

            x = mdl.binary_var_cube(N, N, N, name='x')
            for r in range(N):
                for c in range(N):
                    if grid[r][c] != 0: mdl.add_constraint(x[r, c, grid[r][c] - 1] == 1)
            for r in range(N):
                for c in range(N): mdl.add_constraint(mdl.sum(x[r, c, v] for v in range(N)) == 1)
            for r in range(N):
                for v in range(N): mdl.add_constraint(mdl.sum(x[r, c, v] for c in range(N)) == 1)
            for c in range(N):
                for v in range(N): mdl.add_constraint(mdl.sum(x[r, c, v] for r in range(N)) == 1)
            for br in range(b):
                for bc in range(b):
                    for v in range(N):
                        mdl.add_constraint(mdl.sum(x[br * b + r, bc * b + c, v] for r in range(b) for c in range(b)) == 1)
            t_build = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            # DÙNG API NATIVE TRỰC TIẾP, KHÔNG XUẤT SUBPROCESS FILE LP
            sol = mdl.solve()
            t_solve = (time.perf_counter() - t1) * 1000

            ok = (sol is not None)
            res_grid = None
            if ok:
                res_grid = [[0]*N for _ in range(N)]
                for r in range(N):
                    for c in range(N):
                        for v in range(N):
                            if sol.get_value(x[r, c, v]) > 0.5: res_grid[r][c] = v + 1
            return {"ok": verify_sudoku(grid, res_grid), "t_build": t_build, "t_solve": t_solve}
        except Exception:
            return {"ok": False, "t_build": 0, "t_solve": "Vượt giới hạn Community (1000 biến)"}