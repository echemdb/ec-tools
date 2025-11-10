r"""
---------------------------
 Benchmark with Python code & transonic (python backend)
---------------------------
"""

import sys
import time

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm

from ec_tools import semi_integration as si

# Define file name of export csv
if len(sys.argv) == 2:
    FILENAME = sys.argv[1]
else:
    FILENAME = "benchmark_python"

FILENAME = "".join(["results/", FILENAME])

# Define list of Element sizes
N_list = [1000, 2500, 5000, 7500, 10000, 12500, 15000, 20000]

print("--------------------------")
print("Benchmark with Python code")
print("--------------------------")

# Define if csv export is desired
# export_csv = True

# --------------------------------------
# Fast Riemann-Liouville transformation
# --------------------------------------

# Initialize
t_FRLT = np.zeros(np.size(N_list))
e_FRLT_max = np.zeros(np.size(N_list))
e_rel_FRLT_max = np.zeros(np.size(N_list))

# Print header
print("\n FRLT Algorithm\n")
print(" N      |  t (s) | max abs err | max rel err")

for i in range(0, np.size(N_list)):
    # No. of Elems
    N = N_list[i]

    # generate test values
    x = np.linspace(0, 8, N + 1)
    y = norm.pdf(x, 4, 1)

    # run first calculation twice
    if i == 0:
        si.semi_integration(
            si.semi_integration(y, x, alg="frlt", transonic_backend="python"),
            x,
            alg="frlt",
            transonic_backend="python",
        )

    # calculation of alg
    t_i = time.time()
    d_FRLT = si.semi_integration(
        si.semi_integration(y, x, alg="frlt", transonic_backend="python"),
        x,
        alg="frlt",
        transonic_backend="python",
    )
    t_FRLT[i] = time.time() - t_i

    d_FRLT = d_FRLT[1:]  # first value is zero

    # Reference values
    d_ref = cumulative_trapezoid(y, x)

    # absolute error
    e_FRLT = np.abs(d_FRLT - d_ref)
    e_FRLT_max[i] = np.max(e_FRLT)

    # relative error
    e_rel_FRLT = e_FRLT / (np.abs(d_ref))
    e_rel_FRLT_max[i] = np.max(e_rel_FRLT)
    # Print results
    print(
        f"{N:.1e}",
        "|",
        f"{t_FRLT[i]:6.3}",
        "|   ",
        f"{e_FRLT_max[i]:.2e}",
        "|",
        f"{e_rel_FRLT_max[i]:.2e}",
    )


# --------------------------------------
# Gruenwald
# --------------------------------------

# Initialize
t_G1 = np.zeros(np.size(N_list))
e_G1_max = np.zeros(np.size(N_list))
e_rel_G1_max = np.zeros(np.size(N_list))

# Print header
print("\n Gruenwald Algorithm\n")
print(" N      |  t (s) | max abs err | max rel err")

for i in range(0, np.size(N_list)):
    # No. of Elems
    N = N_list[i]

    # generate test values
    x = np.linspace(0, 8, N + 1)
    y = norm.pdf(x, 4, 1)

    # run first calculation twice
    if i == 0:
        si.semi_integration(
            si.semi_integration(y, x, alg="g1", transonic_backend="python"),
            x,
            alg="g1",
            transonic_backend="python",
        )

    # calculation of alg
    t_i = time.time()
    d_G1 = si.semi_integration(
        si.semi_integration(y, x, alg="g1", transonic_backend="python"),
        x,
        alg="g1",
        transonic_backend="python",
    )
    t_G1[i] = time.time() - t_i

    d_G1 = d_G1[:-1]

    # Reference values
    d_ref = cumulative_trapezoid(y, x)

    # absolute error
    e_G1 = np.abs(d_G1 - d_ref)
    e_G1_max[i] = np.max(e_G1)

    # relative error
    e_rel_G1 = e_G1 / (np.abs(d_ref))
    e_rel_G1_max[i] = np.max(e_rel_G1)

    # Results
    print(
        f"{N:.1e}",
        "|",
        f"{t_G1[i]:6.3}",
        "|   ",
        f"{e_G1_max[i]:.2e}",
        "|",
        f"{e_rel_G1_max[i]:.2e}",
    )

# --------------------------------------
# Riemann-Liouville
# --------------------------------------

# Initialize
t_R1 = np.zeros(np.size(N_list))
e_R1_max = np.zeros(np.size(N_list))
e_rel_R1_max = np.zeros(np.size(N_list))

# Print header
print("\nRiemann-Liouville Algorithm\n")
print(" N      |  t (s) | max abs err | max rel err")

for i in range(0, np.size(N_list)):
    # No. of Elems
    N = N_list[i]

    # generate test values
    x = np.linspace(0, 8, N + 1)
    y = norm.pdf(x, 4, 1)

    # run first calculation twice
    if i == 0:
        si.semi_integration(
            si.semi_integration(y, x, alg="r1", transonic_backend="python"),
            x,
            alg="r1",
            transonic_backend="python",
        )
    # calculation of alg
    t_i = time.time()
    d_R1 = si.semi_integration(
        si.semi_integration(y, x, alg="r1", transonic_backend="python"),
        x,
        alg="r1",
        transonic_backend="python",
    )
    t_R1[i] = time.time() - t_i

    d_R1 = d_R1[:-1]

    # Reference values
    d_ref = cumulative_trapezoid(y, x)

    # absolute error
    e_R1 = np.abs(d_R1 - d_ref)
    e_R1_max[i] = np.max(e_R1)

    # relative error
    e_rel_R1 = e_R1 / (np.abs(d_ref))
    e_rel_R1_max[i] = np.max(e_rel_R1)

    # Results
    print(
        f"{N:.1e}",
        "|",
        f"{t_R1[i]:6.3}",
        "|   ",
        f"{e_R1_max[i]:.2e}",
        "|",
        f"{e_rel_R1_max[i]:.2e}",
    )

# if export_csv is True:
# export results (by pandas)

results = {
    "N": N_list,
    "t_FRLT": t_FRLT,
    "e_FRLT_max": e_FRLT_max,
    "e_rel_FRLT_max": e_rel_FRLT_max,
    "t_G1": t_G1,
    "e_G1_max": e_G1_max,
    "e_rel_G1_max": e_rel_G1_max,
    "t_R1": t_R1,
    "e_R1_max": e_R1_max,
    "e_rel_R1_max": e_rel_R1_max,
}
df_res = pd.DataFrame(results)
df_res.to_csv("".join([FILENAME, ".csv"]), index=False)
