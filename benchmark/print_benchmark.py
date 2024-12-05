r"""
Load computed benchmark results and generates performance test plots
for each algorithm with different speed up.
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd

if len(sys.argv) == 2:
    FIGNAME = sys.argv[1]
else:
    FIGNAME = "benchmark"

# for draft mode set False
SAVE_PLOTS = True

# plot size
plt.rcParams["figure.figsize"] = [10, 8]

## Define Filenames
# with python backend
PYTHON_FILE = "results/benchmark_python.csv"
# with numba
NUMBA_FILE = "results/benchmark_numba.csv"
# with pythran
PYTHRAN_FILE = "results/benchmark_pythran.csv"
# with native rust code
# RUST_FILE = "results/benchmark_rust.csv"

# import python result values
df_python = pd.read_csv(PYTHON_FILE)
# import numba result values
df_numba = pd.read_csv(NUMBA_FILE)
# import transonic result values
df_pythran = pd.read_csv(PYTHRAN_FILE)
# import rust result values
# df_ru = pd.read_csv(RUST_FILE)

# Column entries
col_lst = [
    "t_FRLT",
    "e_FRLT_max",
    "e_rel_FRLT_max",
    "t_G1",
    "e_G1_max",
    "e_rel_G1_max",
    "t_R1",
    "e_R1_max",
    "e_rel_R1_max",
]
# Methods (short)
mtd_lst = ["python", "numba", "pythran"]
# Methods (long)
method_lst = ["Pyhton", "Numba", "Pythran"]

df = pd.DataFrame()
df["N"] = df_python["N"]
# for i in range(len(col_lst)):
for i in enumerate(col_lst):
    ## import from
    # python
    df = df.join(df_python[i[1]])
    df = df.rename(columns={i[1]: "".join([i[1], "_", mtd_lst[0]])})
    # numba
    df = df.join(df_numba[i[1]])
    df = df.rename(columns={i[1]: "".join([i[1], "_", mtd_lst[1]])})
    # pythran
    df = df.join(df_pythran[i[1]])
    df = df.rename(columns={i[1]: "".join([i[1], "_", mtd_lst[2]])})

## create lists for times and errors
# Times
t_lst = []  # (short)
t_method_lst = []  # (long)
for i in range(0, len(col_lst), 3):
    for j in enumerate(mtd_lst):
        t_lst.append("".join([col_lst[i], "_", j[1]]))
        t_method_lst.append("".join([col_lst[i], " ", j[1]]))

# (absolute) errors
e_lst = []  # (short)
e_method_lst = []  # (long)
for i in range(1, len(col_lst), 3):
    for j in enumerate(mtd_lst):
        e_lst.append("".join([col_lst[i], "_", j[1]]))
        e_method_lst.append("".join([col_lst[i], " ", j[1]]))

# relative errors
e_rel_lst = []  # (short)
e_rel_method_lst = []  # (long)
for i in range(2, len(col_lst), 3):
    for j in enumerate(mtd_lst):
        e_rel_lst.append("".join([col_lst[i], "_", j[1]]))
        e_rel_method_lst.append("".join([col_lst[i], " ", j[1]]))

## Define colors for differen algorithms
#   FRTLT , G1 , R1
col = ["g", "b", "r"]
# Define symbols for different approachees
#       py , nu , tr_py
sym = [".:", "x:", "+:"]

# create list of all variations
sym_lst = []
for i in enumerate(col):
    for j in enumerate(sym):
        sym_lst.append("".join([i[1], j[1]]))

## Plot performance test for each alg and each approach
fig = plt.figure()
ax = plt.subplot(111)

for i in enumerate(t_lst):
    ax.semilogy(
        df_python["N"], df[i[1]], sym_lst[i[0]], label=t_method_lst[i[0]], linewidth=2.0
    )

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

plt.title(
    "Time performance test \nfor different alg by different approach", fontsize=20
)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11)

plt.xlabel("No. of Elements", fontsize=18)
plt.ylabel(r"$t \,/\,\mathrm{s} $", fontsize=18)
ax.tick_params(right=True, top=True, direction="in")

if SAVE_PLOTS:
    print("save png for time performance")
    plt.savefig("".join(["images/", FIGNAME, "_time.png"]), dpi=600)

## Plot performance test for each alg and each approach
plt.rcParams["figure.figsize"] = [10, 8]
fig = plt.figure()
ax = plt.subplot(111)

for i in range(len(t_lst)):
    ax.semilogy(
        df_python["N"], df[e_lst[i]], sym_lst[i], label=e_method_lst[i], linewidth=2.0
    )

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

plt.title(
    "Max absolute error of performance test \nfor different alg by different approach",
    fontsize=20,
)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11)

plt.xlabel("No. of Elements", fontsize=18)
plt.ylabel("absolute error", fontsize=18)
ax.tick_params(right=True, top=True, direction="in")

if SAVE_PLOTS:
    print("save png for absolute error of performance test")
    plt.savefig("".join(["images/", FIGNAME, "_abs_err.png"]), dpi=600)

## Plot performance test for each alg and each approach
plt.rcParams["figure.figsize"] = [10, 8]
fig = plt.figure()
ax = plt.subplot(111)

for i in range(len(t_lst)):
    ax.semilogy(
        df_python["N"],
        df[e_rel_lst[i]],
        sym_lst[i],
        label=e_rel_method_lst[i],
        linewidth=2.0,
    )

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])

plt.title(
    "Max relative error of performance test \nfor different alg by different approach",
    fontsize=20,
)
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=11)

plt.xlabel("No. of Elements", fontsize=18)
plt.ylabel("relative error", fontsize=18)
ax.tick_params(right=True, top=True, direction="in")

if SAVE_PLOTS:
    print("save png for relative error of performance test")
    plt.savefig("".join(["images/", FIGNAME, "_rel_err.png"]), dpi=600)
