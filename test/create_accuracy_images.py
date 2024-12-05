r"""
Create images to visualize the accuracy of semi integration algorithms
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import zeta
from scipy.stats import norm

from ec_tools import semi_integration as si

# define which set of images should be ploted and saved
PRINT_ACCURACY_SEMI = False
PRINT_ACCURACY_FULL = False
PRINT_ACCURACY_REL = False

if PRINT_ACCURACY_SEMI:
    N = 1000
    x = np.linspace(0, 10, N)
    V = -0.5

    # f = const
    C = 1
    y1 = np.ones(N) * C
    ref1 = 2 * C * np.sqrt(x / np.pi)
    G1_lim1 = np.abs((V * (V + 1)) / (2 * np.ones(N) * N))
    delta = x[1] - x[0]

    # f = x
    y2 = x
    ref2 = (4 * x ** (3 / 2)) / (3 * np.sqrt(np.pi))
    G1_lim2 = np.abs((V * (V - 1)) / (2 * np.ones(N) * N))
    R1_lim2 = np.ones(N) * (1 - V) / N * (zeta(V) / N ** (-V) - V / (12 * N))

    # f=x^2
    y3 = x**2
    ref3 = (16 * x ** (5 / 2)) / (15 * np.sqrt(np.pi))
    G1_lim3 = np.abs(V * (V - 2) / (2 * np.ones(N) * N))
    R1_lim3 = (
        np.ones(N)
        * ((2 - V) * (1 - V))
        / (N**2)
        * ((zeta(V)) - (zeta(V - 1)) + 1 / 6)
    )

    # Apply algorithms
    G1 = si.gruenwald(y1, delta, V)
    R1 = si.riemann(y1, delta, V)
    fR1 = si.fast_riemann(y1, delta, V)

    G2 = si.gruenwald(y2, delta, V)
    R2 = si.riemann(y2, delta, V)
    fR2 = si.fast_riemann(y2, delta, V)

    G3 = si.gruenwald(y3, delta, V)
    R3 = si.riemann(y3, delta, V)
    fR3 = si.fast_riemann(y3, delta, V)

    # calculate rel error
    errG1 = np.abs(G1 - ref1) / ref1
    errR1 = np.abs(R1 - ref1) / ref1
    errFR1 = np.abs(fR1 - ref1) / ref1

    errG2 = np.abs(G2 - ref2) / ref2
    errR2 = np.abs(R2 - ref2) / ref2
    errFR2 = np.abs(fR2 - ref2) / ref2

    errG3 = np.abs(G3 - ref3) / ref3
    errR3 = np.abs(R3 - ref3) / ref3
    errFR3 = np.abs(fR3 - ref3) / ref3

    # plot f=const
    plt.semilogy(x, errG1, "g", linewidth=2.0, label="G1")
    plt.semilogy(x, errR1, "b", linewidth=2.0, label="R1")
    plt.plot(x, errFR1, "m", linewidth=2.0, label="FR")
    plt.plot(x, G1_lim1, "r", linewidth=2.0, label="limit G1")
    plt.title("Accuracy Test with $y=1$", fontsize=20)
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel("relative error", fontsize=18)
    plt.tick_params(right=True, top=True, direction="in")
    plt.legend(fontsize=18)
    plt.savefig("".join(["data/images/Accuracy_C.png"]), dpi=300)

    # plot f=x
    fig = plt.figure()
    plt.semilogy(x, errG2, "g", linewidth=2.0, label="G1")
    plt.semilogy(x, errR2, "b", linewidth=2.0, label="R1")
    plt.plot(x, errFR2, "m", linewidth=2.0, label="FR")
    plt.plot(x, G1_lim2, "r", linewidth=2.0, label="limit G1")
    plt.title("Accuracy Test with $y=x$", fontsize=20)
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel("relative error", fontsize=18)
    plt.tick_params(right=True, top=True, direction="in")
    plt.legend(fontsize=18)
    plt.savefig("".join(["data/images/Accuracy_x.png"]), dpi=300)

    # plot f=x^2
    fig = plt.figure()
    plt.semilogy(x, errG3, "g", linewidth=2.0, label="G1")
    plt.semilogy(x, errR3, "b", linewidth=2.0, label="R1")
    plt.plot(x, errFR3, "m", linewidth=2.0, label="FR")
    plt.plot(x, G1_lim3, "r", linewidth=2.0, label="limit G1")
    plt.plot(x, np.abs(R1_lim3), "r:", linewidth=2.0, label="limit R1")
    plt.title("Accuracy Test with $y=x^2$", fontsize=20)
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel("relative error", fontsize=18)
    plt.tick_params(right=True, top=True, direction="in")
    plt.legend(fontsize=18)
    plt.savefig("".join(["data/images/Accuracy_x2.png"]), dpi=300)

if PRINT_ACCURACY_FULL:
    N = 1000
    x = np.linspace(0, 10, N)
    V = -0.5

    # f = const
    C = 1
    y1 = np.ones(N) * C
    ref1 = x[1:]
    delta = x[1] - x[0]

    # f = x
    y2 = x
    ref2 = 0.5 * x**2
    ref2 = ref2[1:]

    # f=x^2
    y3 = x**2
    ref3 = (1 / 3) * x**3
    ref3 = ref3[1:]

    # numerical full integration
    T1 = cumulative_trapezoid(y1, x)
    T2 = cumulative_trapezoid(y2, x)
    T3 = cumulative_trapezoid(y3, x)

    # Apply algorithms
    G1 = si.gruenwald(si.gruenwald(y1, delta, V), delta, V)[:-1]
    R1 = si.riemann(si.riemann(y1, delta, V), delta, V)[:-1]
    fR1 = si.fast_riemann(si.fast_riemann(y1, delta, V), delta, V)[1:]

    G2 = si.gruenwald(si.gruenwald(y2, delta, V), delta, V)[:-1]
    R2 = si.riemann(si.riemann(y2, delta, V), delta, V)[:-1]
    fR2 = si.fast_riemann(si.fast_riemann(y2, delta, V), delta, V)[1:]

    G3 = si.gruenwald(si.gruenwald(y3, delta, V), delta, V)[:-1]
    R3 = si.riemann(si.riemann(y3, delta, V), delta, V)[:-1]
    fR3 = si.fast_riemann(si.fast_riemann(y3, delta, V), delta, V)[1:]

    # calculate rel error
    errT1 = np.abs(T1 - ref1) / ref1
    errG1 = np.abs(G1 - ref1) / ref1
    errR1 = np.abs(R1 - ref1) / ref1
    errFR1 = np.abs(fR1 - ref1) / ref1

    errT2 = np.abs(T2 - ref2) / ref2
    errG2 = np.abs(G2 - ref2) / ref2
    errR2 = np.abs(R2 - ref2) / ref2
    errFR2 = np.abs(fR2 - ref2) / ref2

    errT3 = np.abs(T3 - ref3) / ref3
    errG3 = np.abs(G3 - ref3) / ref3
    errR3 = np.abs(R3 - ref3) / ref3
    errFR3 = np.abs(fR3 - ref3) / ref3

    # plot f=const
    plt.semilogy(x[:-1], errG1, "g", linewidth=2.0, label="G1")
    plt.semilogy(x[:-1], errR1, "b", linewidth=2.0, label="R1")
    plt.plot(x[:-1], errFR1, "m", linewidth=2.0, label="FR")
    plt.plot(x[:-1], errT1, "r", linewidth=2.0, label="Trap")
    plt.title("Accuracy Test (full Integration) with $y=1$", fontsize=20)
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel("relative error", fontsize=18)
    plt.tick_params(right=True, top=True, direction="in")
    plt.legend(fontsize=18)
    plt.savefig("".join(["data/images/Accuracy_full_C.png"]), dpi=300)

    # plot f=x
    fig = plt.figure()
    plt.semilogy(x[:-1], errG2, "g", linewidth=2.0, label="G1")
    plt.semilogy(x[:-1], errR2, "b", linewidth=2.0, label="R1")
    plt.plot(x[:-1], errFR2, "m", linewidth=2.0, label="FR")
    plt.plot(x[:-1], errT2, "r", linewidth=2.0, label="Trap")
    plt.title("Accuracy Test (full Integration) with $y=x$", fontsize=20)
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel("relative error", fontsize=18)
    plt.tick_params(right=True, top=True, direction="in")
    plt.legend(fontsize=18)
    plt.savefig("".join(["data/images/Accuracy_full_x.png"]), dpi=300)

    # # plot f=x^2
    fig = plt.figure()
    plt.semilogy(x[:-1], errG3, "g", linewidth=2.0, label="G1")
    plt.semilogy(x[:-1], errR3, "b", linewidth=2.0, label="R1")
    plt.plot(x[:-1], errFR3, "m", linewidth=2.0, label="FR")
    plt.plot(x[:-1], errT3, "r", linewidth=2.0, label="Trap")
    plt.title("Accuracy Test (full Integration) with $y=x^2$", fontsize=20)
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel("relative error", fontsize=18)
    plt.tick_params(right=True, top=True, direction="in")
    plt.legend(fontsize=18)
    plt.savefig("".join(["data/images/Accuracy_full_x2.png"]), dpi=300)

if PRINT_ACCURACY_REL:
    # Accuracy tests
    n_max = [1000, 10000]

    for N in n_max:
        # Test data
        x = np.linspace(0, 10, N + 1)
        y = norm.pdf(x, 5, 1)

        delta_x = x[1] - x[0]

        # apply semi integration methods
        d_ref = cumulative_trapezoid(y, x)
        d1 = si.fast_riemann(si.fast_riemann(y, delta_x), delta_x)[1:]
        d2 = si.riemann(si.riemann(y, delta_x), delta_x)
        d3 = si.gruenwald(si.gruenwald(y, delta_x), delta_x)

        # calc absolute and relative errors

        err_1 = np.abs(d1 - d_ref)
        err_2 = np.abs(d2[:-1] - d_ref)
        err_3 = np.abs(d3[:-1] - d_ref)

        relerr_1 = np.abs(d1 - d_ref) / (np.abs(d_ref))
        relerr_2 = np.abs(d2[:-1] - d_ref) / (np.abs(d_ref))
        relerr_3 = np.abs(d3[:-1] - d_ref) / (np.abs(d_ref))

        # Input plot
        if N == n_max[0]:
            fig, ax = plt.subplots(1)
            plt.plot(x, y, "b", linewidth=5.0)
            plt.xlabel(r"$x$", fontsize=18)
            plt.ylabel(r"$y$", fontsize=18)
            plt.title("Test data", fontsize=18)
            ax.tick_params(right=True, top=True, direction="in")
            plt.savefig("".join(["data/images/TestData.png"]), dpi=300)

            # Integration plot
            fig, ax = plt.subplots(1)
            plt.plot(x[:-1], d_ref, "b", linewidth=2.0)
            plt.plot(x[:-1], d1, "r", linewidth=2.0)
            plt.plot(x, d2, "c", linewidth=2.0)
            plt.plot(x, d3, "m", linewidth=2.0)
            plt.xlabel(r"$x$", fontsize=18)
            plt.ylabel(r"$\int y\mathrm{d}x$", fontsize=18)
            plt.legend(["scipy", "fast Riemann", "Riemann", "Gruenwald"], fontsize=14)
            plt.title("Full integration", fontsize=18)
            ax.tick_params(right=True, top=True, direction="in")
            plt.savefig("".join(["data/images/full_int.png"]), dpi=300)

        # abs err plot
        fig, ax = plt.subplots(1)
        plt.semilogy(x[:-1], err_1, "r", linewidth=2.0)
        plt.semilogy(x[:-1], err_2, "c", linewidth=2.0)
        plt.semilogy(x[:-1], err_3, "m", linewidth=2.0)
        plt.xlabel(r"$x$", fontsize=18)
        plt.ylabel("absolute error", fontsize=18)
        plt.legend(["fast Riemann", "Riemann", "Gruenwald"], fontsize=14)
        plt.title("Test Case: Absolute Error", fontsize=18)
        ax.tick_params(right=True, top=True, direction="in")
        plt.savefig("".join(["data/images/abserr_", str(N), ".png"]), dpi=300)

        # rel err plot
        fig, ax = plt.subplots(1)
        plt.semilogy(x[:-1], relerr_1, "r", linewidth=2.0)
        plt.semilogy(x[:-1], relerr_2, "c", linewidth=2.0)
        plt.semilogy(x[:-1], relerr_3, "m", linewidth=2.0)
        plt.xlabel(r"$x$", fontsize=18)
        plt.ylabel("relative error", fontsize=18)
        plt.legend(["fast Riemann", "Riemann", "Gruenwald"], fontsize=14)
        plt.title("Test Case: Relative Error", fontsize=18)
        ax.tick_params(right=True, top=True, direction="in")
        plt.savefig("".join(["data/images/relerr_", str(N), ".png"]), dpi=300)
