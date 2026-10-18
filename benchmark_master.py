import argparse
import csv
import os
import statistics
import time
from collections import defaultdict

from tabulate import tabulate

from amo_encodings import ENCODING_NAMES
from puzzle_gen import load_or_make
from sudoku_sat import SAT_SOLVER_CLASSES, SATSudokuSolver, solve_with_timeout

SAT_SOLVER_NAMES = list(SAT_SOLVER_CLASSES)
COMMUNITY_MIP_LIMIT = 1000  # số biến tối đa của CPLEX Community Edition


# Hàm phụ

def median(values):
    return statistics.median(values) if values else float("nan")


def stdev(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  -> đã lưu {path}")


def get_instances(b, n_instances, cache_dir, gen_limit):
    """Sinh (hoặc đọc lại từ cache) n_instances đề cỡ N = b^2. Trả về list (lưới, số gợi ý)."""
    N = b * b
    instances = []
    for i in range(n_instances):
        t0 = time.perf_counter()
        grid, clues = load_or_make(b, seed=1000 * N + i, time_limit=gen_limit, cache_dir=cache_dir)
        print(f"  [gen] {N}x{N} đề {i + 1}/{n_instances}: {clues} gợi ý "
              f"({time.perf_counter() - t0:.1f}s)", flush=True)
        instances.append((grid, clues))
    return instances


def run_sat(grid, enc, solver, timeout):
    """Giải bằng SAT; có timeout (tiến trình con) nếu timeout > 0, ngược lại chạy trực tiếp."""
    if timeout:
        return solve_with_timeout(grid, enc, solver, timeout)
    return SATSudokuSolver(len(grid), enc, solver).solve(grid)


# Thí nghiệm 1: kỹ thuật AMO x SAT solver

def experiment_amo_vs_solvers(sizes, n_instances, repeats, timeout, out_dir, gen_limit):
    raw = []
    for N in sizes:
        print(f"\n{'=' * 90}\n  THÍ NGHIỆM 1: {len(ENCODING_NAMES)} AMO x {len(SAT_SOLVER_NAMES)} "
              f"SAT solver, Sudoku {N}x{N}, {n_instances} đề duy nhất nghiệm\n{'=' * 90}")
        instances = get_instances(int(N ** 0.5), n_instances, os.path.join(out_dir, "instances"), gen_limit)
        print(f"  Số gợi ý trung bình: {statistics.mean(c for _, c in instances):.1f} / {N * N}")

        table = []
        for enc in ENCODING_NAMES:
            cells, info = [], None  # info = (số biến, số mệnh đề, t_build) của lần giải thành công gần nhất
            for solver in SAT_SOLVER_NAMES:
                times, failures = [], 0
                for idx, (grid, _) in enumerate(instances):
                    for rep in range(repeats):
                        res = run_sat(grid, enc, solver, timeout)
                        raw.append({"N": N, "instance": idx, "rep": rep, "encoding": enc,
                                    "solver": solver, "status": res["status"],
                                    "t_build": res.get("t_build"), "t_solve": res.get("t_solve"),
                                    "vars": res.get("vars"), "clauses": res.get("clauses")})
                        if res["status"] == "OK":
                            times.append(res["t_solve"])
                            info = (res["vars"], res["clauses"], res["t_build"])
                        else:
                            failures += 1
                cell = f"{median(times):.1f}" if times else "N/A"
                cells.append(cell + (f"  ({failures} lỗi/timeout)" if failures else ""))
            size_info = f"{info[0]} / {info[1]:,}" if info else "N/A"
            build_info = f"{info[2]:.0f}" if info else "N/A"
            table.append([enc.upper(), size_info, build_info] + cells)

        headers = ["AMO", "Biến / Clauses", "Build (ms)"] + [f"{s} (ms)" for s in SAT_SOLVER_NAMES]
        print(tabulate(table, headers=headers, tablefmt="grid"))
        print("  (ô = trung vị t_solve trên các đề giải được)")

    write_csv(os.path.join(out_dir, "exp1_raw.csv"), raw)



# Thí nghiệm 2: SAT vs CP vs ILP

def build_methods(sat_enc, sat_solver, timeout, skip=()):

    from solvers_cp_ilp import CPILPSolvers  # nạp muộn để tiến trình con SAT không phải nạp lại

    available = CPILPSolvers.available()
    methods = {f"SAT {sat_solver}+{sat_enc}":
                   lambda g: solve_with_timeout(g, sat_enc, sat_solver, timeout)}
    optional = [("cpsat", "CP-SAT (Google)", CPILPSolvers.solve_cpsat),
                ("cplex_cp", "CPLEX CP", CPILPSolvers.solve_cplex_cp),
                ("gurobi", "Gurobi (MIP)", CPILPSolvers.solve_gurobi),
                ("cplex_mip", "CPLEX MIP", CPILPSolvers.solve_cplex_mip)]
    for key, name, func in optional:
        if available[key]:
            methods[name] = lambda g, f=func: f(g, timeout)
        else:
            print(f"[!] Bỏ qua {key}: {CPILPSolvers.import_errors().get(key)}")

    methods = {name: fn for name, fn in methods.items()
               if not any(s.lower() in name.lower() for s in skip)}
    print("  Phương pháp sẽ chạy:", ", ".join(methods), flush=True)
    return methods


def warm_up(methods, cache_dir, gen_limit):

    grid, _ = load_or_make(3, seed=1, time_limit=gen_limit, cache_dir=cache_dir)
    for name, fn in methods.items():
        print(f"  warm-up {name}: {fn(grid)['status']}")


def summarize(raw, sizes, method_names):
    groups = defaultdict(list)
    for r in raw:
        groups[(r["N"], r["method"])].append(r)

    headers = ["Cỡ", "Phương pháp", "Giải được", "Timeout", "Lỗi",
               "Med solve (ms)", "Std solve", "Med build (ms)", "Med total (ms)"]
    table = []
    for N in sizes:
        for name in method_names:
            rows = groups[(N, name)]
            if rows and all(r["status"] == "SKIPPED" for r in rows):
                table.append([f"{N}x{N}", name, "N/A (license)"] + ["-"] * 6)
                continue
            ok = [r for r in rows if r["status"] == "OK"]
            timeouts = sum(r["status"] == "TIMEOUT" for r in rows)
            errors = sum(r["status"].startswith("ERROR") for r in rows)
            solve = [r["t_solve"] for r in ok]
            build = [r["t_build"] for r in ok]
            total = [r["t_build"] + r["t_solve"] for r in ok]
            table.append([f"{N}x{N}", name, f"{len(ok)}/{len(rows)}", timeouts, errors] +
                         ([f"{median(solve):.1f}", f"{stdev(solve):.1f}",
                           f"{median(build):.1f}", f"{median(total):.1f}"] if ok else ["-"] * 4))
    return headers, table


def experiment_solvers_multi_sizes(sizes, n_instances, timeout, sat_enc, sat_solver,
                                   out_dir, gen_limit, skip=()):
    print(f"\n{'=' * 90}\n  THÍ NGHIỆM 2: SAT vs CP vs ILP (1 luồng, timeout {timeout}s, "
          f"{n_instances} đề/cỡ)\n{'=' * 90}")
    cache_dir = os.path.join(out_dir, "instances")
    methods = build_methods(sat_enc, sat_solver, timeout, skip)
    warm_up(methods, cache_dir, gen_limit)

    raw = []
    for N in sizes:
        instances = get_instances(int(N ** 0.5), n_instances, cache_dir, gen_limit)
        avg_clues = statistics.mean(c for _, c in instances)
        print(f"\n[*] {N}x{N}: {n_instances} đề, gợi ý TB {avg_clues:.1f} ({avg_clues / N ** 2:.0%})")

        for idx, (grid, clues) in enumerate(instances):
            for name, fn in methods.items():
                record = {"N": N, "instance": idx, "clues": clues, "method": name}
                if name == "CPLEX MIP" and N ** 3 > COMMUNITY_MIP_LIMIT:
                    # CPLEX Community Edition giới hạn 1000 biến: không chạy được cỡ này
                    raw.append({**record, "status": "SKIPPED", "t_build": None, "t_solve": None})
                    continue
                res = fn(grid)
                print(f"    [{N}x{N}] đề {idx + 1}/{n_instances} {name}: {res['status']} "
                      f"(solve {res['t_solve']:.0f} ms)", flush=True)
                raw.append({**record, "status": res["status"],
                            "t_build": res["t_build"], "t_solve": res["t_solve"]})

    headers, table = summarize(raw, sizes, methods)
    print("\n" + tabulate(table, headers=headers, tablefmt="grid"))
    print("  (thống kê chỉ tính trên các đề giải đúng; timeout/lỗi được đếm riêng)")
    print("  N/A (license): CPLEX MIP bản Community giới hạn 1000 biến nên chỉ chạy ở 9x9")

    write_csv(os.path.join(out_dir, "exp2_raw.csv"), raw)
    with open(os.path.join(out_dir, "exp2_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Thực nghiệm Sudoku: SAT vs CP/ILP")
    ap.add_argument("--exp", nargs="+", type=int, default=[1, 2], help="thí nghiệm cần chạy (1 và/hoặc 2)")
    ap.add_argument("--sizes", nargs="+", type=int, default=[9, 16, 25],
                    help="cỡ lưới N của thí nghiệm 2 (số chính phương: 9 16 25 36 49 64)")
    ap.add_argument("--exp1-sizes", nargs="+", type=int, default=[9, 16],
                    help="cỡ lưới của thí nghiệm 1")
    ap.add_argument("--instances", type=int, default=10, help="số đề mỗi cỡ")
    ap.add_argument("--repeats", type=int, default=1, help="số lần lặp mỗi đề (thí nghiệm 1)")
    ap.add_argument("--timeout", type=int, default=60, help="giới hạn thời gian giải (giây)")
    ap.add_argument("--sat-enc", default="seqcounter", choices=ENCODING_NAMES,
                    help="kỹ thuật AMO của SAT ở thí nghiệm 2 (nên chọn theo kết quả thí nghiệm 1)")
    ap.add_argument("--sat-solver", default="cadical", choices=SAT_SOLVER_NAMES)
    ap.add_argument("--out", default="results", help="thư mục lưu kết quả")
    ap.add_argument("--skip", nargs="*", default=[],
                    help='bỏ qua phương pháp có tên chứa chuỗi này, ví dụ: --skip "CPLEX MIP"')
    ap.add_argument("--gen-limit", type=int, default=60,
                    help="số giây tối đa để sinh mỗi đề (quá hạn thì giữ số gợi ý hiện có)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    if 1 in a.exp:
        experiment_amo_vs_solvers(a.exp1_sizes, a.instances, a.repeats, a.timeout, a.out, a.gen_limit)
    if 2 in a.exp:
        experiment_solvers_multi_sizes(a.sizes, a.instances, a.timeout, a.sat_enc, a.sat_solver,
                                       a.out, a.gen_limit, a.skip)


if __name__ == "__main__":
    main()