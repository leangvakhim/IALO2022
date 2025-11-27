from wsn import wsn
import numpy as np
from ialo import ialo
import matplotlib.pyplot as plt
from benchmark import (
    sphere,
    schwefel_1_2,
    schwefel_2_22,
    rosenbrock,
    quartic,
    rastrigin,
    ackley,
    griewank,
    penalized

)

print("1. Benchmark")
print("2. Coverage")
val = int(input("Enter a value: "))

if val == 1:
    benchmarks = [
        {"name": "F1: Sphere",        "func": sphere,        "lb": -100,  "ub": 100,  "dim": 30},
        {"name": "F2: Schwefel 2.22", "func": schwefel_2_22, "lb": -10,   "ub": 10,   "dim": 30},
        {"name": "F3: Schwefel 1.2",  "func": schwefel_1_2,  "lb": -100,  "ub": 100,  "dim": 30},
        {"name": "F4: Rosenbrock",    "func": rosenbrock,    "lb": -30,   "ub": 30,   "dim": 30},
        {"name": "F5: Quartic",       "func": quartic,       "lb": -1.28, "ub": 1.28, "dim": 30},
        {"name": "F6: Rastrigin",     "func": rastrigin,     "lb": -5.12, "ub": 5.12, "dim": 30},
        {"name": "F7: Ackley",        "func": ackley,        "lb": -32,   "ub": 32,   "dim": 30},
        {"name": "F8: Griewank",      "func": griewank,      "lb": -600,  "ub": 600,  "dim": 30},
        {"name": "F9: Penalized",     "func": penalized,     "lb": -50,   "ub": 50,   "dim": 30},
    ]

    pop_size = 30
    max_iter = 500
    list_val = {bm['name']: [] for bm in benchmarks}
    results = {bm['name']: [] for bm in benchmarks}
    times = 30

    for bm in benchmarks:
        print(f"\nProcessing {bm['name']}...")
        for _ in range(times):
            optimizer = ialo(
                bm['func'],
                bm['dim'],
                bm['lb'],
                bm['ub'],
                pop_size,
                max_iter
            )

            best_pos, best_score, curve = optimizer.optimize()
            results[bm['name']].append(curve)
            list_val[bm['name']].append(best_score)

    for name, scores in list_val.items():
        print(f"name is: {name}")
        mean_val = np.mean(scores)
        std_val = np.std(scores)
        min_val = np.max(scores)
        max_val = np.min(scores)

        print(f"Mean values: {mean_val:.4e}")
        print(f"Std values: {std_val:.4e}")
        print(f"Min values: {min_val:.4e}")
        print(f"Max values: {max_val:.4e}")


    # plt.figure(figsize=(12, 8))
    # for name, curve in results.items():
    #     safe_curve = np.array(curve)
    #     if np.min(safe_curve) <= 0:
    #          safe_curve = safe_curve + 1e-300

    #     plt.plot(safe_curve, label=name)

    # plt.xlabel('Iterations')
    # plt.ylabel('Fitness (Best Score)')
    # plt.yscale('log')
    # plt.title('IALO Convergence on All 9 Benchmark Functions')
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.grid(True, which="both", ls="-", alpha=0.5)
    # plt.tight_layout()
    # plt.show()

elif val == 2:
    l = 50.0
    n = 50
    rs = 5.0
    re = rs / 2.0

    params = {
        'alpha1': 1, 'alpha2': 0,
        'beta1': 1, 'beta2': 2
    }

    wsn_cov = wsn(l, n, rs, re, params)

    dim = 2 * n
    lb = 0.0
    ub = l
    pop_size = 30
    max_iter = 100

    optimizer = ialo(
        wsn_cov.objective_function,
        dim,
        lb,
        ub,
        pop_size,
        max_iter
    )

    best_pos, best_score, curve = optimizer.optimize()
    final_coverage = -best_score

    best_sensors = best_pos.reshape((n, 2))

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    for i in range(n):
        circle = plt.Circle((best_sensors[i, 0], best_sensors[i, 1]), rs, color='blue', alpha=0.2, linewidth=0)
        ax.add_artist(circle)
        plt.plot(best_sensors[i, 0], best_sensors[i, 1], 'r.', markersize=5)

    plt.xlim(0, l)
    plt.ylim(0, l)
    plt.title(f"IALO Coverages: {final_coverage*100:.2f}%")
    plt.xlabel("L (m)")
    plt.ylabel("L (m)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

