
import io
import csv
import time
import tracemalloc
from datetime import datetime
from contextlib import redirect_stdout


def profile_call(fn, *args, capture_print=True, **kwargs):
    """
    Measure time, peak memory (KB via tracemalloc), and number of printed chars.
    Returns: (result, {"time_sec", "peak_mem_kb", "printed_chars"})
    """
    buf = io.StringIO()
    tracemalloc.start()
    t0 = time.perf_counter()
    if capture_print:
        with redirect_stdout(buf):
            result = fn(*args, **kwargs)
    else:
        result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {
        "time_sec": elapsed,
        "peak_mem_kb": peak / 1024.0,
        "printed_chars": len(buf.getvalue()) if capture_print else 0,
    }


def write_metrics_csv(rows, csv_path="metrics_summary.csv"):
    # union header
    header = set()
    for r in rows:
        header |= set(r.keys())
    header = ["timestamp"] + sorted(header - {"timestamp"})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        from datetime import datetime as _dt
        for r in rows:
            rr = dict(r)
            rr.setdefault("timestamp", _dt.now().isoformat(timespec="seconds"))
            w.writerow(rr)
    print(f"[CSV] Wrote metrics table to {csv_path}")

def bnb_with_cuts(amount_cents, coin_values, max_first=True):
    """Standalone BnB that doesn’t depend on Tp.py"""
    best = None
    best_cost = float("inf")
    nodes = calls = cuts = 0
    t0 = time.perf_counter()

    def dfs(i, r, prefix, partial_cost):
        nonlocal best, best_cost, nodes, calls, cuts
        calls += 1
        if r == 0:
            if partial_cost < best_cost:
                best = prefix + [(coin_values[j], 0) for j in range(i, len(coin_values))]
                best_cost = partial_cost
            return
        if i == len(coin_values):
            return
        d = coin_values[i]
        max_k = r // d
        k_range = range(max_k, -1, -1) if max_first else range(0, max_k + 1)
        for k in k_range:
            nodes += 1
            new_cost = partial_cost + k
            if new_cost >= best_cost:
                cuts += 1
                continue
            dfs(i + 1, r - k * d, prefix + [(d, k)], new_cost)

    dfs(0, amount_cents, [], 0)
    t1 = time.perf_counter()
    return {
        "best": best, "best_cost": best_cost,
        "elapsed_sec": t1 - t0, "nodes": nodes, "calls": calls, "cuts": cuts
    }


def make_run_greedy_and_print(greedy_change):
    def _run(amount_cents, denoms):
        sol, total_coins, r = greedy_change(amount_cents, denoms)
        parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in sol if k > 0) or "(no coins)"
        print("Greedy:", "{"+parts+f"}}  (coins: {total_coins}) | remainder: {r/100:.2f}")
        return {"solution": sol, "coins": total_coins, "remainder": r}
    return _run

def make_run_exhaustive_iter_and_print_all(all_solutions_iter):
    def _run(amount_cents, denoms):
        count = 0
        for sol in all_solutions_iter(amount_cents, denoms):
            count += 1
            parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in sol if k > 0) or "(no coins)"
            coins = sum(k for _, k in sol)
            print(f"{count:6d}. "+"{"+parts+f"}} (coins: {coins})")
        print(f"Total solutions: {count}")
        return {"count": count}
    return _run

def make_run_collect_all_and_keep_best(all_solutions_iter):
    def _run(amount_cents, denoms):
        best = None
        best_cost = float("inf")
        bucket = []
        for sol in all_solutions_iter(amount_cents, denoms):
            bucket.append(sol)
            cost = sum(k for _, k in sol)
            if cost < best_cost:
                best, best_cost = sol, cost
        print("Best (from stored set): coins =", best_cost)
        return {"best": best, "best_cost": best_cost, "count": len(bucket)}
    return _run

def make_run_best_stop_early_and_print(best_solution_stop_early):
    def _run(amount_cents, denoms):
        sol = best_solution_stop_early(amount_cents, denoms)
        if sol is None:
            print("No solution")
            return None
        parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in sol if k > 0) or "(no coins)"
        coins = sum(k for _, k in sol)
        print("Best (stop early):", "{"+parts+f"}} (coins: {coins})")
        return {"best": sol, "coins": coins}
    return _run

def make_run_bnb_and_print():
    def _run(amount_cents, denoms):
        res = bnb_with_cuts(amount_cents, denoms, max_first=True)
        parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in res["best"] if k > 0) or "(no coins)"
        print("BnB best:", "{"+parts+f"}} (coins: {res['best_cost']})")
        print(f"nodes={res['nodes']}  calls={res['calls']}  cuts={res['cuts']}")
        return res
    return _run
