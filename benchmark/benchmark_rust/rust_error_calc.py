r"""
--------------------------------------
 Error Calculation of rust results
--------------------------------------
"""

import sys

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm

if len(sys.argv) == 2:
    FILENAME = sys.argv[1]
else:
    FILENAME = "results/rs_benchmark"

# Load Dataframe with time and element infos
df = pd.read_csv(("rust_implemention/" + FILENAME + "_time.csv"))

# save temp data
e_FRLT_max = np.zeros(len(df["N"]))
e_G1_max = np.zeros(len(df["N"]))
e_R1_max = np.zeros(len(df["N"]))
e_rel_FRLT_max = np.zeros(len(df["N"]))
e_rel_G1_max = np.zeros(len(df["N"]))
e_rel_R1_max = np.zeros(len(df["N"]))

# import values and calculate errors
for i in range(0, len(df["N"])):
    # No. of Elems
    N = df["N"][i]
    # generate test values
    t = np.linspace(0, 8, N + 1)
    I = norm.pdf(t, 4, 1) # noqa: E741

    # with default values
    # Reference values
    d_ref = cumulative_trapezoid(I, t)
    # Import results
    df_ru = pd.read_csv(
        ("rust_implemention/" + FILENAME + "_" + str(df["N"][i]) + ".csv")
    )

    # FRLT alg
    # -------------------------------
    d_FRLT = df_ru["res_FRLT"][1:].values  # [1:]  # first value is zero
    # (max) absolute error
    e_FRLT = np.abs(d_FRLT - d_ref)
    e_FRLT_max[i] = np.max(e_FRLT)
    # (max) relative error
    e_rel_FRLT = e_FRLT / (np.abs(d_ref))
    e_rel_FRLT_max[i] = np.max(e_rel_FRLT)

    # G1 alg
    # -------------------------------
    d_G1 = df_ru["res_G1"][:-1].values
    # absolute error
    e_G1 = np.abs(d_G1 - d_ref)
    e_G1_max[i] = np.max(e_G1)
    # relative error
    e_rel_G1 = e_G1 / (np.abs(d_ref))
    e_rel_G1_max[i] = np.max(e_rel_G1)

    # R1 alg
    # -------------------------------
    d_R1 = df_ru["res_R1"][:-1].values
    # absolute error
    e_R1 = np.abs(d_R1 - d_ref)
    e_R1_max[i] = np.max(e_R1)
    # relative error
    e_rel_R1 = e_R1 / (np.abs(d_ref))
    e_rel_R1_max[i] = np.max(e_rel_R1)

# save error values to Dataframe
df["e_FRLT_max"] = e_FRLT_max
df["e_G1_max"] = e_G1_max
df["e_R1_max"] = e_R1_max
df["e_rel_FRLT_max"] = e_rel_FRLT_max
df["e_rel_G1_max"] = e_rel_G1_max
df["e_rel_R1_max"] = e_rel_R1_max

# export all results
df.to_csv("../results/benchmark_rust.csv", index=False)
