

import time
import heapq

from metrics_calc import (
    profile_call,
    write_metrics_csv,
    make_run_greedy_and_print,
    make_run_exhaustive_iter_and_print_all,
    make_run_collect_all_and_keep_best,
    make_run_best_stop_early_and_print,
    make_run_bnb_and_print,
)




# this is the Greedy search
def greedy_change(amount_cents, coin_values):
    sol, r = [], amount_cents
    for d in coin_values:
        k = r // d
        sol.append((d, k))
        r -= k * d
        if r == 0:
            sol += [(dd, 0) for dd in coin_values[len(sol):]]
            break
    total_coins = sum(k for _, k in sol)
    return sol, total_coins, r


# in his funcion we get all he solution
def all_solutions_iter(amount_cents, coin_values):
    n = len(coin_values)
    stack = [(0, amount_cents, [])]  # (index, remainder, prefix)
    while stack:
        i, r, pref = stack.pop()
        if r == 0:
            yield pref + [(coin_values[j], 0) for j in range(i, n)]
            continue
        if i == n:
            continue
        d = coin_values[i]
        max_k = r // d
        for k in range(0, max_k + 1):
            stack.append((i + 1, r - k * d, pref + [(d, k)]))


def count_all_solutions_iter(amount_cents, coin_values):
    return sum(1 for _ in all_solutions_iter(amount_cents, coin_values))


# optimal coin countt we will use i in early stopping 
def optimal_coin_count(amount_cents, coin_values):
    INF = 10**9
    dp = [0] + [INF] * amount_cents
    for a in range(1, amount_cents + 1):
        best = INF
        for d in coin_values:
            if d <= a and dp[a - d] + 1 < best:
                best = dp[a - d] + 1
        dp[a] = best
    return dp[amount_cents]


def best_solution_stop_early(amount_cents, coin_values):
    target = optimal_coin_count(amount_cents, coin_values)
    if target >= 10**9:
        return None
    for sol in all_solutions_iter(amount_cents, coin_values):
        if sum(k for _, k in sol) == target:
            return sol
    return None


# use recursion
def all_solutions_recursive(amount_cents, coin_values, i=0, prefix=None, counter=None):
    if counter is not None:
        counter['calls'] = counter.get('calls', 0) + 1
    if prefix is None:
        prefix = []
    if amount_cents == 0:
        if counter is not None:
            counter['solutions'] = counter.get('solutions', 0) + 1
        yield prefix + [(d, 0) for d in coin_values[i:]]
        return
    if i == len(coin_values):
        return
    d = coin_values[i]
    for k in range(amount_cents // d, -1, -1):  # max..0 (original Prg3)
        if counter is not None:
            counter['nodes'] = counter.get('nodes', 0) + 1
        yield from all_solutions_recursive(
            amount_cents - k * d, coin_values, i + 1, prefix + [(d, k)], counter=counter
        )


# write he recursion to a file 
def write_all_solutions_recursive(amount_eur, denoms_eur, txt_path="RecursiveSolution.txt"):
    AMT = int(round(amount_eur * 100))
    DEN = sorted({int(round(v * 100)) for v in denoms_eur}, reverse=True)
    with open(txt_path, "w", encoding="utf-8") as f:
        for sol in all_solutions_recursive(AMT, DEN):
            parts = [f"{d/100:.2f}x{k}" for d, k in sol if k > 0] or ["(no coins)"]
            coins = sum(k for _, k in sol)
            f.write(" + ".join(parts) + f"  | coins={coins}\n")
    return txt_path


# iterative and wriing to a csv file 
def write_all_solutions(amount_eur, denoms_eur, txt_path="all_solutions.txt", csv_path="all_solutions.csv"):
    AMT = int(round(amount_eur * 100))
    DEN = sorted({int(round(v * 100)) for v in denoms_eur}, reverse=True)
    count = 0
    with open(txt_path, "w", encoding="utf-8") as ft, open(csv_path, "w", encoding="utf-8") as fc:
        header = [f"d_{d/100:.2f}" for d in DEN] + ["total_coins"]
        fc.write(",".join(header) + "\n")
        for sol in all_solutions_iter(AMT, DEN):
            parts = [f"{d/100:.2f}x{k}" for d, k in sol if k > 0] or ["(no coins)"]
            coins = sum(k for _, k in sol)
            ft.write(" + ".join(parts) + f"  | coins={coins}\n")
            row = [str(k) for _, k in sol] + [str(coins)]
            fc.write(",".join(row) + "\n")
            count += 1
    return count, txt_path, csv_path


# recursion in reverse order 
def all_solutions_recursive_order(amount_cents, coin_values, i=0, prefix=None, max_first=False, counter=None):
    if counter is not None:
        counter["calls"] = counter.get("calls", 0) + 1
    if prefix is None:
        prefix = []
    if amount_cents == 0:
        if counter is not None:
            counter["solutions"] = counter.get("solutions", 0) + 1
        yield prefix + [(d, 0) for d in coin_values[i:]]
        return
    if i == len(coin_values):
        return
    d = coin_values[i]
    max_k = amount_cents // d
    k_range = range(max_k, -1, -1) if max_first else range(0, max_k + 1)
    for k in k_range:
        if counter is not None:
            counter["nodes"] = counter.get("nodes", 0) + 1
        yield from all_solutions_recursive_order(
            amount_cents - k * d, coin_values, i + 1, prefix + [(d, k)], max_first=max_first, counter=counter
        )


def all_solutions_recursive_prg3bis(amount_cents, coin_values):
    return all_solutions_recursive_order(amount_cents, coin_values, max_first=False)


# compute all solution but we did not display 
def collect_all_solutions_recursive(amount_cents, coin_values):
    out = []
    for sol in all_solutions_recursive(amount_cents, coin_values):
        out.append(sol)
    return out


# compute he bes cos
def best_solution_trace(amount_cents, coin_values, trace_path="ImproveIncrementally.txt"):
    best = None
    best_cost = float("inf")
    with open(trace_path, "w", encoding="utf-8") as f:
        for sol in all_solutions_recursive(amount_cents, coin_values):
            cost = sum(k for _, k in sol)
            if cost < best_cost:
                best = sol
                best_cost = cost
                parts = [f"{d/100:.2f}x{k}" for d, k in sol if k > 0] or ["(no coins)"]
                f.write("IMPROVES ➜ " + " + ".join(parts) + f"  | coins={cost}\n")
    return best, best_cost, trace_path


# keep only the best solution with the report time 
def best_solution_keep_only(amount_cents, coin_values):
    t0 = time.perf_counter()
    best = None
    best_cost = float("inf")
    visited = 0
    for sol in all_solutions_recursive(amount_cents, coin_values):
        visited += 1
        cost = sum(k for _, k in sol)
        if cost < best_cost:
            best = sol
            best_cost = cost
    t1 = time.perf_counter()
    return {"best": best, "best_cost": best_cost, "solutions_enumerated": visited, "elapsed_sec": t1 - t0}


# adding a cut to the algo
def best_solution_branch_and_bound(amount_cents, coin_values, max_first=True):
    best = None
    best_cost = float("inf")
    nodes, calls = 0, 0
    t0 = time.perf_counter()

    def dfs(i, r, prefix, partial_cost):
        nonlocal best, best_cost, nodes, calls
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
                continue  # prune
            dfs(i + 1, r - k * d, prefix + [(d, k)], new_cost)

    dfs(0, amount_cents, [], 0)
    t1 = time.perf_counter()
    return {"best": best, "best_cost": best_cost, "elapsed_sec": t1 - t0, "nodes": nodes, "calls": calls}


# comparison
def compare_pruning_orders(amount_cents, coin_values):
    a = best_solution_branch_and_bound(amount_cents, coin_values, max_first=True)
    b = best_solution_branch_and_bound(amount_cents, coin_values, max_first=False)
    return {
        "order_max_first": {"best_cost": a["best_cost"], "nodes": a["nodes"], "time": a["elapsed_sec"]},
        "order_min_first": {"best_cost": b["best_cost"], "nodes": b["nodes"], "time": b["elapsed_sec"]},
    }



# the last program which is optional 
def k_best_solutions(amount_cents, coin_values, k=5):
    heap = []  # (-cost, idx, sol)
    idx = 0
    for sol in all_solutions_iter(amount_cents, coin_values):
        cost = sum(c for _, c in sol)
        item = (-cost, idx, sol)
        if len(heap) < k:
            heapq.heappush(heap, item)
        else:
            if -heap[0][0] > cost:
                heapq.heapreplace(heap, item)
        idx += 1
    top = sorted([(-c, i, s) for (c, i, s) in heap])
    return [s for _, _, s in top]


def best_and_two_less_worst(amount_cents, coin_values):
    best = None
    best_cost = float("inf")
    all_solutions = []
    idx = 0
    for sol in all_solutions_iter(amount_cents, coin_values):
        cost = sum(k for _, k in sol)
        all_solutions.append((cost, idx, sol))
        if cost < best_cost:
            best, best_cost = sol, cost
        idx += 1
    two_worst = sorted(all_solutions, key=lambda x: (-x[0], x[1]))[:2]
    return best, [w[2] for w in two_worst]


#the main function 
if __name__ == "__main__":
    L = [5, 2, 1, 0.5, 0.2, 0.1, 0.05]
    m = 12.35

    # som conversiob before computations 
    DEN = sorted({int(round(v * 100)) for v in L}, reverse=True)
    AMT = int(round(m * 100))

    # Program 1
    g_sol, g_coins, g_rem = greedy_change(AMT, DEN)
    g_parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in g_sol if k > 0) or "(no coins)"
    print("Greedy: {" + g_parts + f"}}  (coins: {g_coins}) | remainder: {g_rem/100:.2f}")

    # Program 2
    print("\nFirst 5 solutions (iterative, exhaustive):")
    for idx, sol in zip(range(5), all_solutions_iter(AMT, DEN)):
        parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in sol if k > 0) or "(no coins)"
        coins = sum(k for _, k in sol)
        print(f"{idx+1:2d}. "+"{"+parts+f"}} (coins: {coins})")

    total = count_all_solutions_iter(AMT, DEN)
    print("\nTotal number of solutions (iterative):", total)

    # Use early stopping 
    best = best_solution_stop_early(AMT, DEN)
    if best is not None:
        parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in best if k > 0) or "(no coins)"
        coins = sum(k for _, k in best)
        print("\nBest solution (stop early): "+"{"+parts+f"}} (coins: {coins})")
    else:
        print("\nBest solution (stop early): None")

    # Program 3
    print("\nFirst 3 solutions (recursive):")
    for idx, sol in zip(range(3), all_solutions_recursive(AMT, DEN)):
        parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in sol if k > 0) or "(no coins)"
        coins = sum(k for _, k in sol)
        print(f"{idx+1:2d}. "+"{"+parts+f"}} (coins: {coins})")

    stats = {}
    r_total = 0
    for _ in all_solutions_recursive(AMT, DEN, counter=stats):
        r_total += 1
    print("\n[Recursive totals]")
    print("Total solutions (recursive):", r_total)
    print("Recursive calls:", stats.get('calls', 0))
    print("Branch choices tried (nodes):", stats.get('nodes', 0))

    # writing to a file 
    path_rec = write_all_solutions_recursive(m, L) 
    print(f"\n Wrote recursive list to {path_rec}")
    print("After validation, compare with:  diff RecursiveSoluttion.txt RecursiveSoluttion.txt")

    # keep the iterative writer too 
    total_written, txt_file, csv_file = write_all_solutions(m, L)
    print(f"\n[Iterative writer] Wrote {total_written} solutions to:\n - {txt_file}\n - {csv_file}")

    # Program 3 bis (reverse order)
    print("\n[Prg3bis] First 3 solutions with reversed order:")
    for idx, sol in zip(range(3), all_solutions_recursive_prg3bis(AMT, DEN)):
        parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in sol if k > 0) or "(no coins)"
        print(f"{idx+1:2d}. "+"{"+parts+f"}} (coins: {sum(k for _, k in sol)})")

    # Program 4 
    coll = collect_all_solutions_recursive(AMT, DEN)
    print(f"\n[Prg4] Collected {len(coll)} solutions (no display during search). Now printing 3:")
    for idx, sol in enumerate(coll[:3], 1):
        parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in sol if k > 0) or "(no coins)"
        print(f"{idx:2d}. "+"{"+parts+f"}} (coins: {sum(k for _, k in sol)})")

    # Program 5
    best5, cost5, trace_file = best_solution_trace(AMT, DEN)
    print(f"\n[Prg5] Best coins = {cost5}. Trace ➜ {trace_file}")

    # Program 6
    res6 = best_solution_keep_only(AMT, DEN)
    print(f"\n[Prg6] best_cost={res6['best_cost']}  solutions_enumerated={res6['solutions_enumerated']}  time={res6['elapsed_sec']:.6f}s")

    # Program 7
    bb = best_solution_branch_and_bound(AMT, DEN, max_first=True)
    print(f"\n[Prg7] BnB best_cost={bb['best_cost']}  nodes={bb['nodes']}  calls={bb['calls']}  time={bb['elapsed_sec']:.6f}s")

    # Program 7 bis
    cmp_orders = compare_pruning_orders(AMT, DEN)
    print("[Prg7bis] order comparison:")
    for name, row in cmp_orders.items():
        print(f"  {name}: best_cost={row['best_cost']} nodes={row['nodes']} time={row['time']:.6f}s")

    # The optional part
    top5 = k_best_solutions(AMT, DEN, k=5)
    print("\n[Prog 8] Top-5 (fewest coins):")
    for i, sol in enumerate(top5, 1):
        parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in sol if k > 0) or "(no coins)"
        print(f"  {i}. "+"{"+parts+f"}}  (coins: {sum(k for _, k in sol)})")

    
    best_sol, two_worst = best_and_two_less_worst(AMT, DEN)
    bp = " + ".join(f"{d/100:.2f}x{k}" for d, k in best_sol if k > 0) or "(no coins)"
    print("\n[Prog 8′] Best solution:", "{"+bp+f"}} (coins: {sum(k for _, k in best_sol)})")
    print("Two less-worst solutions:")
    for i, sol in enumerate(two_worst, 1):
        parts = " + ".join(f"{d/100:.2f}x{k}" for d, k in sol if k > 0) or "(no coins)"
        print(f"  {i}. "+"{"+parts+f"}} (coins: {sum(k for _, k in sol)})")


run_greedy_and_print = make_run_greedy_and_print(greedy_change)
run_exhaustive_iter_and_print_all = make_run_exhaustive_iter_and_print_all(all_solutions_iter)
run_collect_all_and_keep_best = make_run_collect_all_and_keep_best(all_solutions_iter)
run_best_stop_early_and_print = make_run_best_stop_early_and_print(best_solution_stop_early)
run_bnb_and_print = make_run_bnb_and_print()

metrics_rows = []

    # 1) Greedy
g_out, g_m = profile_call(run_greedy_and_print, AMT, DEN)
print(f"[Greedy] time={g_m['time_sec']:.6f}s  peak_mem={g_m['peak_mem_kb']:.1f}KB  printed={g_m['printed_chars']} chars")
metrics_rows.append({
        "algorithm": "greedy",
        "time_sec": g_m["time_sec"],
        "peak_mem_kb": g_m["peak_mem_kb"],
        "printed_chars": g_m["printed_chars"],
        "coins": g_out["coins"],
        "remainder": g_out["remainder"]/100.0
    })

    # 2) Exhaustive (iterative, prints everything)
enum_out, enum_m = profile_call(run_exhaustive_iter_and_print_all, AMT, DEN)
print(f"[Exhaustive/Iter] time={enum_m['time_sec']:.6f}s  peak_mem={enum_m['peak_mem_kb']:.1f}KB  printed={enum_m['printed_chars']} chars")
metrics_rows.append({
        "algorithm": "exhaustive_iter_print_all",
        "time_sec": enum_m["time_sec"],
        "peak_mem_kb": enum_m["peak_mem_kb"],
        "printed_chars": enum_m["printed_chars"],
        "solutions": enum_out["count"]
    })

    # 3) Exhaustive storing all solutions
stored_out, stored_m = profile_call(run_collect_all_and_keep_best, AMT, DEN)
print(f"[Exhaustive store] time={stored_m['time_sec']:.6f}s  peak_mem={stored_m['peak_mem_kb']:.1f}KB  printed={stored_m['printed_chars']} chars  solutions={stored_out['count']}")
metrics_rows.append({
        "algorithm": "exhaustive_store_all",
        "time_sec": stored_m["time_sec"],
        "peak_mem_kb": stored_m["peak_mem_kb"],
        "printed_chars": stored_m["printed_chars"],
        "solutions": stored_out["count"],
        "best_cost": stored_out["best_cost"]
    })

    # 4) Early-stop
early_out, early_m = profile_call(run_best_stop_early_and_print, AMT, DEN)
print(f"[Early-Stop] time={early_m['time_sec']:.6f}s  peak_mem={early_m['peak_mem_kb']:.1f}KB  printed={early_m['printed_chars']} chars")
if early_out:
        metrics_rows.append({
            "algorithm": "early_stop",
            "time_sec": early_m["time_sec"],
            "peak_mem_kb": early_m["peak_mem_kb"],
            "printed_chars": early_m["printed_chars"],
            "best_cost": early_out["coins"]
        })

    # 5) Branch & Bound with cuts
bnb_out, bnb_m = profile_call(run_bnb_and_print, AMT, DEN)
print(f"[BnB] time={bnb_m['time_sec']:.6f}s  peak_mem={bnb_m['peak_mem_kb']:.1f}KB  printed={bnb_m['printed_chars']} chars")
metrics_rows.append({
        "algorithm": "branch_and_bound",
        "time_sec": bnb_m["time_sec"],
        "peak_mem_kb": bnb_m["peak_mem_kb"],
        "printed_chars": bnb_m["printed_chars"],
        "best_cost": bnb_out["best_cost"],
        "nodes": bnb_out["nodes"],
        "calls": bnb_out["calls"],
        "cuts": bnb_out["cuts"]
    })

    # 6) Save CSV
write_metrics_csv(metrics_rows, csv_path="metrics_summary.csv")