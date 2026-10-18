"""
Các solver CP / ILP để so sánh với SAT: Google CP-SAT, CPLEX CP, Gurobi (MIP), CPLEX MIP.
Tất cả chạy 1 luồng, có timeout. Mỗi hàm solve_* trả về dict:
    {"status": OK | TIMEOUT | INFEASIBLE | WRONG | "ERROR: ...", "t_build": ms, "t_solve": ms, ...}
    t_build = thời gian dựng mô hình; t_solve = thời gian solver tìm nghiệm.

Cấu hình CPLEX: đặt biến môi trường CPLEX_DIR (thư mục chứa cplex\\ và cpoptimizer\\), hoặc để
project nằm cạnh thư mục đó (code tự dò). Có thể đặt CPO_EXEC trỏ tới cpoptimizer.exe.
"""
import os
import re
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

from sudoku_utils import units, verify_sudoku

# Cấu hình đường dẫn CPLEX

def _detect_cplex_dir():
    candidates = [os.environ.get("CPLEX_DIR", ""),
                  os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    for d in candidates:
        if d and os.path.isdir(os.path.join(d, "cplex")):
            return d
    return ""


CPLEX_DIR = _detect_cplex_dir()
CPO_EXEC = os.environ.get("CPO_EXEC", "")
if CPLEX_DIR:
    _cplex_bin = os.path.join(CPLEX_DIR, "cplex", "bin", "x64_win64")
    _cpo_bin = os.path.join(CPLEX_DIR, "cpoptimizer", "bin", "x64_win64")
    os.environ["PATH"] = os.pathsep.join([_cplex_bin, _cpo_bin, os.environ.get("PATH", "")])
    if not CPO_EXEC and os.path.exists(os.path.join(_cpo_bin, "cpoptimizer.exe")):
        CPO_EXEC = os.path.join(_cpo_bin, "cpoptimizer.exe")

# Nạp thư viện

HAS = {"cpsat": False, "cplex_cp": False, "cplex_mip": False, "gurobi": False}
_IMPORT_ERRORS = {}

try:
    from ortools.sat.python import cp_model
    HAS["cpsat"] = True
except Exception as e:  # noqa: BLE001
    _IMPORT_ERRORS["cpsat"] = str(e)

try:
    from docplex.cp.config import context
    from docplex.cp.model import CpoModel
    if CPO_EXEC:
        context.solver.local.execfile = CPO_EXEC
    HAS["cplex_cp"] = True
except Exception as e:  # noqa: BLE001
    _IMPORT_ERRORS["cplex_cp"] = str(e)

try:
    from docplex.mp.model import Model as CplexMIPModel
    HAS["cplex_mip"] = True
except Exception as e:  # noqa: BLE001
    _IMPORT_ERRORS["cplex_mip"] = str(e)

try:
    import gurobipy as gp
    from gurobipy import GRB
    HAS["gurobi"] = True
except Exception as e:  # noqa: BLE001
    _IMPORT_ERRORS["gurobi"] = str(e)

_gurobi_env = None


# Hàm phụ

def _ms(t0):
    return (time.perf_counter() - t0) * 1000


def _solved(grid, solution, t_build, t_solve, **extra):
    status = "OK" if verify_sudoku(grid, solution) else "WRONG"
    return {"status": status, "t_build": t_build, "t_solve": t_solve, **extra}


def _failed(status, t_build=0.0, t_solve=0.0, **extra):
    return {"status": status, "t_build": t_build, "t_solve": t_solve, **extra}


def _decode_binary(N, value):
    """Đổi biến nhị phân x[r, c, v] (value(r, c, v) > 0.5) thành lưới N x N."""
    sol = [[0] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            for v in range(N):
                if value(r, c, v) > 0.5:
                    sol[r][c] = v + 1
    return sol


class CPILPSolvers:
    @staticmethod
    def available():
        return dict(HAS)

    @staticmethod
    def import_errors():
        return dict(_IMPORT_ERRORS)

    # ----------------------------------------------------------------- CP-SAT
    @staticmethod
    def solve_cpsat(grid, timeout_sec=60):
        N = len(grid)
        try:
            t0 = time.perf_counter()
            model = cp_model.CpModel()
            x = {(r, c): (model.NewConstant(grid[r][c]) if grid[r][c]
                          else model.NewIntVar(1, N, f"x_{r}_{c}"))
                 for r in range(N) for c in range(N)}
            for unit in units(N):
                model.AddAllDifferent([x[cell] for cell in unit])
            t_build = _ms(t0)

            solver = cp_model.CpSolver()
            solver.parameters.num_workers = 1
            solver.parameters.max_time_in_seconds = timeout_sec
            t1 = time.perf_counter()
            status = solver.Solve(model)
            t_solve = _ms(t1)

            if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                sol = [[solver.Value(x[r, c]) for c in range(N)] for r in range(N)]
                return _solved(grid, sol, t_build, t_solve)
            if status == cp_model.INFEASIBLE:
                return _failed("INFEASIBLE", t_build, t_solve)
            return _failed("TIMEOUT", t_build, t_solve)
        except Exception as e:  # noqa: BLE001
            return _failed(f"ERROR: {e}")

    # --------------------------------------------------------------- CPLEX CP
    @staticmethod
    def solve_cplex_cp(grid, timeout_sec=60):
        N = len(grid)
        try:
            t0 = time.perf_counter()
            mdl = CpoModel(name="CPLEX_CP")
            x = {(r, c): mdl.integer_var(1, N, name=f"x_{r}_{c}")
                 for r in range(N) for c in range(N)}
            for (r, c), var in x.items():
                if grid[r][c]:
                    mdl.add(var == grid[r][c])
            for unit in units(N):
                mdl.add(mdl.all_diff([x[cell] for cell in unit]))
            t_build = _ms(t0)

            t1 = time.perf_counter()
            msol = mdl.solve(LogVerbosity="Quiet", TimeLimit=timeout_sec, Workers=1)
            t_solve = _ms(t1)  # gồm cả thời gian khởi động tiến trình cpoptimizer
            try:
                t_engine = msol.get_solve_time() * 1000  # thời gian do chính solver báo
            except Exception:  # noqa: BLE001
                t_engine = None

            if msol.is_solution():
                sol = [[msol[x[r, c]] for c in range(N)] for r in range(N)]
                return _solved(grid, sol, t_build, t_solve, t_engine=t_engine)
            if "infeasible" in str(msol.get_solve_status()).lower():
                return _failed("INFEASIBLE", t_build, t_solve, t_engine=t_engine)
            return _failed("TIMEOUT", t_build, t_solve, t_engine=t_engine)
        except Exception as e:  # noqa: BLE001
            return _failed(f"ERROR: {e}")

    # ----------------------------------------------------------------- Gurobi
    @staticmethod
    def _gurobi_environment():
        global _gurobi_env
        if _gurobi_env is None:
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.setParam("Threads", 1)
            env.start()
            _gurobi_env = env
        return _gurobi_env

    @staticmethod
    def solve_gurobi(grid, timeout_sec=60):
        N = len(grid)
        model = None
        try:
            env = CPILPSolvers._gurobi_environment()
            t0 = time.perf_counter()
            model = gp.Model("Gurobi_Sudoku", env=env)
            model.Params.TimeLimit = timeout_sec
            x = model.addVars(N, N, N, vtype=GRB.BINARY, name="x")
            for r in range(N):
                for c in range(N):
                    if grid[r][c]:
                        model.addConstr(x[r, c, grid[r][c] - 1] == 1)
                    model.addConstr(x.sum(r, c, "*") == 1)  # mỗi ô một giá trị
            for unit in units(N):
                for v in range(N):
                    model.addConstr(gp.quicksum(x[r, c, v] for r, c in unit) == 1)
            model.update()
            t_build = _ms(t0)

            t1 = time.perf_counter()
            model.optimize()
            t_solve = _ms(t1)

            if model.Status == GRB.OPTIMAL:
                sol = _decode_binary(N, lambda r, c, v: x[r, c, v].X)
                return _solved(grid, sol, t_build, t_solve)
            if model.Status == GRB.TIME_LIMIT:
                return _failed("TIMEOUT", t_build, t_solve)
            if model.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
                return _failed("INFEASIBLE", t_build, t_solve)
            return _failed(f"ERROR: Gurobi status {model.Status}", t_build, t_solve)
        except Exception as e:  # noqa: BLE001
            return _failed(f"ERROR: {e}")
        finally:
            if model is not None:
                model.dispose()

    # -------------------------------------------------------------- CPLEX MIP
    @staticmethod
    def _build_cplex_mip(grid):
        N = len(grid)
        mdl = CplexMIPModel(name="CPLEX_MIP")
        x = {(r, c, v): mdl.binary_var(name=f"x_{r}_{c}_{v}")
             for r in range(N) for c in range(N) for v in range(N)}
        for r in range(N):
            for c in range(N):
                if grid[r][c]:
                    mdl.add_constraint(x[r, c, grid[r][c] - 1] == 1)
                mdl.add_constraint(mdl.sum(x[r, c, v] for v in range(N)) == 1)
        for unit in units(N):
            for v in range(N):
                mdl.add_constraint(mdl.sum(x[r, c, v] for r, c in unit) == 1)
        return mdl, x

    @staticmethod
    def _find_cplex_exe():
        exe = shutil.which("cplex")
        if exe:
            return exe
        if CPLEX_DIR:
            candidate = os.path.join(CPLEX_DIR, "cplex", "bin", "x64_win64", "cplex.exe")
            if os.path.exists(candidate):
                return candidate
        return None

    @staticmethod
    def _parse_cplex_solution(path, N):
        sol = [[0] * N for _ in range(N)]
        for var in ET.parse(path).getroot().iter("variable"):
            parts = var.get("name", "").split("_")
            if len(parts) == 4 and parts[0] == "x" and float(var.get("value", "0")) > 0.5:
                r, c, v = map(int, parts[1:])
                sol[r][c] = v + 1
        return sol

    @staticmethod
    def _explain_cplex_output(out):
        low = out.lower()
        if "problem size limits" in low or "error 1016" in low:
            return "ERROR: CPLEX Community Edition - vượt giới hạn kích thước (1000 biến/ràng buộc)"
        if "time limit exceeded" in low:
            return "TIMEOUT"
        if "infeasible" in low:
            return "INFEASIBLE"
        hints = [ln.strip() for ln in out.splitlines()
                 if any(k in ln.lower() for k in ("error", "license", "exceeded"))]
        last = out.strip().splitlines()[-1] if out.strip() else "no output"
        return f"ERROR: cplex.exe: {(' | '.join(hints[:3]) or last)[:250]}"

    @staticmethod
    def _solve_cplex_mip_exe(grid, mdl, t_build, timeout_sec):

        N = len(grid)
        exe = CPILPSolvers._find_cplex_exe()
        if not exe:
            return _failed("ERROR: không tìm thấy cplex.exe (đặt CPLEX_DIR hoặc thêm vào PATH)", t_build)

        t0 = time.perf_counter()
        lp_text = mdl.export_as_lp_string()
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "model.lp"), "w", encoding="utf-8") as f:
                f.write(lp_text)
            t_build += _ms(t0)

            command = [exe, "-c", "read model.lp", "set threads 1",
                       f"set timelimit {timeout_sec}", "optimize", "write sol.sol"]
            t1 = time.perf_counter()
            try:
                proc = subprocess.run(command, cwd=tmp, capture_output=True, text=True,
                                      timeout=timeout_sec + 120)
            except subprocess.TimeoutExpired:
                return _failed("TIMEOUT", t_build, _ms(t1), engine="cplex.exe")
            t_wall = _ms(t1)

            out = (proc.stdout or "") + (proc.stderr or "")
            match = re.search(r"Solution time\s*=\s*([\d.]+)\s*sec", out)
            t_solve = float(match.group(1)) * 1000 if match else t_wall
            extra = {"engine": "cplex.exe", "t_wall": t_wall}

            sol_path = os.path.join(tmp, "sol.sol")
            if os.path.exists(sol_path):
                sol = CPILPSolvers._parse_cplex_solution(sol_path, N)
                if any(any(row) for row in sol):
                    return _solved(grid, sol, t_build, t_solve, **extra)
            return _failed(CPILPSolvers._explain_cplex_output(out), t_build, t_solve, **extra)

    @staticmethod
    def solve_cplex_mip(grid, timeout_sec=60):
        N = len(grid)
        try:
            t0 = time.perf_counter()
            mdl, x = CPILPSolvers._build_cplex_mip(grid)
            mdl.set_log_output(False)
            mdl.parameters.threads = 1
            mdl.parameters.timelimit = timeout_sec
            t_build = _ms(t0)

            t1 = time.perf_counter()
            try:
                sol = mdl.solve()
            except Exception as e:  # noqa: BLE001
                msg = str(e).lower()
                if any(k in msg for k in ("runtime", "community", "problem size", "size limit", "1016")):
                    return CPILPSolvers._solve_cplex_mip_exe(grid, mdl, t_build, timeout_sec)
                raise
            t_solve = _ms(t1)

            if sol is not None:
                grid_sol = _decode_binary(N, lambda r, c, v: sol.get_value(x[r, c, v]))
                return _solved(grid, grid_sol, t_build, t_solve, engine="python-api")
            status = str(mdl.solve_details.status).lower() if mdl.solve_details else ""
            if "limit" in status:
                return _failed("TIMEOUT", t_build, t_solve)
            return _failed(f"INFEASIBLE ({status})", t_build, t_solve)
        except Exception as e:  # noqa: BLE001
            return _failed(f"ERROR: {e}")