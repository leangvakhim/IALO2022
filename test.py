from wsn import wsn
from ialo import ialo
import matplotlib.pyplot as plt
from benchmark import (
    sphere,

)

print("1. Benchmark")
print("2. Coverage")
val = int(input("Enter a value: "))

if val == 1:
    abs = 1
elif val == 2:
    l = 50.0
    n = 30
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
    max_iter = 1000

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

