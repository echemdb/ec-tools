r"""
Create images to visualize the semi integration
"""
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import norm

from ec_tools import semi_integration as si

# define which set of images should be ploted and saved
PRINT_PARAMETRIZATION = True
PRINT_DIFFERENTIATION = True
PRINT_FR_PARAMETRIZATION = True
PRINT_FUNCTIONALITY_DOUBLE = True
PRINT_FUNCTIONALITY_MULTIPLE_G = True
PRINT_FUNCTIONALITY_MULTIPLE_FR = True


if PRINT_PARAMETRIZATION:
    # test values
    x = np.linspace(0, 10, 2001)
    y = norm.pdf(x, 5, 1)

    delta_x = x[1] - x[0]

    # reference
    d_ref = cumulative_trapezoid(y, x, initial=0)

    # order of semi integrals
    v = np.round(np.linspace(-0.9, -0.1, 9), 1)

    # Initialize matrix for all computed gruenwald semi integrals
    G = np.zeros((len(v), len(x)))

    for i in enumerate(v):
        G[i[0]] = si.gruenwald(y, delta_x, i[1])

        # Initialize matrix for all computed fast riemann semi integrals
    FR = np.zeros((len(v), len(x)))

    for i in enumerate(v):
        FR[i[0]] = si.fast_riemann(y, delta_x, i[1])

    # linestyle
    ls = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]
    # create 4 subplots
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Varying semi integrations", fontsize=20)

    for i in enumerate(v):
        ax[0, 0].plot(
            x,
            G[i[0]],
            ls=ls[i[0]],
            linewidth=2.0,
            color=(0, 1 + i[1], 0),
            label=f"G: v= {i[1]}",
        )
        ax[1, 0].semilogy(
            x,
            G[i[0]],
            ls=ls[i[0]],
            linewidth=2.0,
            color=(0, 1 + i[1], 0),
            label=f"G: v= {i[1]}",
        )

        ax[0, 1].plot(
            x,
            FR[i[0]],
            ls=ls[i[0]],
            linewidth=2.0,
            color=(0, 0, 1 + i[1]),
            label=f"FR: v= {i[1]}",
        )
        ax[1, 1].semilogy(
            x,
            FR[i[0]],
            ls=ls[i[0]],
            linewidth=2.0,
            color=(0, 0, 1 + i[1]),
            label=f"FR: v= {i[1]}",
        )

    for i in range(0, len(ax)):
        ax[0, i].plot(x, d_ref, "r", linewidth=2.0, label="num int")
        ax[1, i].plot(x, d_ref, "r", linewidth=2.0, label="num int")

    for i in range(len(ax)):
        ax[0, i].set_xlim([0, 10])
        ax[0, i].set_xlabel(r"$x$", fontsize=18)
        ax[0, i].set_ylabel(r"$(\mathrm{d}y/\mathrm{d}x)^v$", fontsize=18)
        ax[0, i].tick_params(right=True, top=True, direction="in")

        ax[1, i].set_xlim([0, 10])
        ax[1, i].set_xlabel(r"$x$", fontsize=18)
        ax[1, i].set_ylabel(r"$\log (\mathrm{d}y/\mathrm{d}x)^v$", fontsize=18)
        ax[1, i].tick_params(right=True, top=True, direction="in")

    ax[0, 0].legend()
    ax[0, 1].legend()
    ax[0, 0].set_title("Gruenwald", fontsize=18)
    ax[0, 1].set_title("Fast Riemann", fontsize=18)

    plt.savefig("data/images/varying_semiint.png", dpi=600)

if PRINT_DIFFERENTIATION:
    # semidifferentiation
    # test values
    x = np.linspace(0, 10, 2001)
    y1 = norm.pdf(x, 5, 1)

    # reference
    y = cumulative_trapezoid(y1, x, initial=0)
    # simple differentiation
    dy = np.diff(y) / np.diff(x)

    delta_x = x[1] - x[0]

    dg = si.gruenwald(si.gruenwald(y, delta_x, 0.5), delta_x, 0.5)
    dr = si.riemann(si.riemann(y, delta_x, v=0.5), delta_x, v=0.5)

    # absoulte error
    err_r = np.abs(y1 - dr)
    err_g = np.abs(y1 - dg)
    # relative error
    rel_err_r = np.abs(y1 - dr) / np.abs(y1)
    rel_err_g = np.abs(y1 - dg) / np.abs(y1)

    # plot
    fig, ax = plt.subplots(3, 1, figsize=(12, 12))
    ax[0].plot(x, y1, "r", linewidth=2.0)
    ax[0].plot(x, dr, "b--", linewidth=2.0)
    ax[0].plot(x, dg, "c:", linewidth=2.0)
    ax[0].set_xlabel(r"$x$", fontsize=18)
    ax[0].set_ylabel(r"$y$", fontsize=18)
    ax[0].legend(["initial", "riemann", "gruenwald"], fontsize=18)
    ax[0].tick_params(right=True, top=True, direction="in")
    ax[0].set_title("Full differentiation", fontsize=26)
    ax[0].set_xlim([0, 10])

    ax[1].semilogy(x, err_r, "b--", linewidth=2.0)
    ax[1].semilogy(x, err_g, "c:", linewidth=2.0)
    ax[1].set_xlabel(r"$x$", fontsize=18)
    ax[1].set_ylabel("abs error", fontsize=18)
    ax[1].tick_params(right=True, top=True, direction="in")
    ax[1].legend(["riemann", "gruenwald"], fontsize=18)
    ax[0].set_xlim([0, 10])

    ax[2].semilogy(x, rel_err_r, "b--", linewidth=2.0)
    ax[2].semilogy(x, rel_err_g, "c:", linewidth=2.0)
    ax[2].set_xlabel(r"$x$", fontsize=18)
    ax[2].set_ylabel("rel error", fontsize=18)
    ax[2].tick_params(right=True, top=True, direction="in")
    ax[2].legend(["riemann", "gruenwald"], fontsize=18)
    ax[2].set_xlim([0, 10])

    plt.savefig("data/images/full_diff.png", dpi=600)

if PRINT_FR_PARAMETRIZATION:
    # Testing of c1 & c2 parameters for Fast-Riemann

    # number of elements
    N = 1000
    x = np.linspace(0, 10, N)
    # semi-integration
    V = -0.5

    # Define set of C parameters:
    C1 = [1, 3, 10, 30, 100]
    C2 = C1

    # case 1 (y=constant)
    C0 = 1
    y1 = np.ones(N) * C0
    ref1 = 2 * C0 * np.sqrt(x / np.pi)

    # case 2 (y=x)
    y2 = x
    ref2 = (4 * x ** (3 / 2)) / (3 * np.sqrt(np.pi))

    # case 3 (y=x^2)
    y3 = x**2
    ref3 = (16 * x ** (5 / 2)) / (15 * np.sqrt(np.pi))

    # time matrix
    t = np.zeros((np.size(C1), np.size(C2)))

    # plot case 1
    fig = plt.figure()
    fig, ax = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
    )

    ax[0].tick_params(right=True, top=True, direction="in")
    ax[1].tick_params(right=True, top=True, direction="in")

    # plot case function
    box = ax[0].get_position()
    ax[0].plot(x, y1, linewidth=5.0)
    ax[0].set_ylabel("$y$", fontsize=18)
    ax[0].set_xlabel("$x$", fontsize=18)
    ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[0].xaxis.set_tick_params(which="both", labelbottom=True)

    ax[0].set_title("Test Function $y=1$ \n", fontsize=20)

    for i in enumerate(C1):
        for j in enumerate(C2):
            tmp = time.time()
            fR = si.fast_riemann(y1, x[2] - x[1], V, c1=i[1], c2=j[1])
            t[i[0]][j[0]] = time.time() - tmp
            errFR = (fR - ref1) / ref1
            ax[1].semilogy(
                x,
                np.abs(errFR),
                color=(0, (i[0] * 2 + 1) / (np.size(C1) * 2 + 1), (j[0] + 1) / 10),
                label=f"C1:{i[1]}, C2:{j[1]}",
                linewidth=2.0,
            )

    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    # plt.xlim([0,0.5])
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel("relative error", fontsize=18)
    # fig.suptitle(
    ax[1].set_title(
        "Parametrization Test for Fast Riemann with $n=1000$\n", fontsize=20
    )
    plt.savefig("data/images/FR_Para_C_1000.png", dpi=600)

    # time plot
    fig = plt.figure(figsize=(12, 7))
    for i in enumerate(C1):
        plt.loglog(C1, t[i[0]], "x-", label="".join(["$C_2$: ", str(i[1])]))
    plt.legend(fontsize=18)
    plt.xlabel("$C_1$", fontsize=18)
    plt.ylabel(r"$t\;/\;s$", fontsize=18)
    plt.title("Time Performance of fast Riemann with different Parameters", fontsize=20)
    plt.tick_params(right=True, top=True, direction="in")
    plt.savefig("data/images/FR_Para_C_1000_time.png", dpi=600)

    # plot case 2
    fig = plt.figure()
    fig, ax = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
    )

    ax[0].tick_params(right=True, top=True, direction="in")
    ax[1].tick_params(right=True, top=True, direction="in")

    # plot case function
    box = ax[0].get_position()
    ax[0].plot(x, y2, linewidth=5.0)
    ax[0].set_ylabel("$y$", fontsize=18)
    ax[0].set_xlabel("$x$", fontsize=18)
    ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[0].xaxis.set_tick_params(which="both", labelbottom=True)

    ax[0].set_title("Test Function $y=x$ \n", fontsize=20)

    for i in enumerate(C1):
        for j in enumerate(C2):
            tmp = time.time()
            fR = si.fast_riemann(y2, x[2] - x[1], V, c1=i[1], c2=j[1])
            t[i[0]][j[0]] = time.time() - tmp
            errFR = (fR - ref2) / ref2
            ax[1].semilogy(
                x,
                np.abs(errFR),
                color=(0, (i[0] * 2 + 1) / (np.size(C1) * 2 + 1), (j[0] + 1) / 10),
                label=f"C1:{i[1]}, C2:{j[1]}",
                linewidth=2.0,
            )

    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    # plt.xlim([0,0.5])
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel("relative error", fontsize=18)
    # fig.suptitle(
    ax[1].set_title(
        "Parametrization Test for Fast Riemann with $n=1000$\n", fontsize=20
    )
    plt.savefig("data/images/FR_Para_x_1000.png", dpi=600)

    # time plot
    fig = plt.figure(figsize=(12, 7))
    for i in enumerate(C1):
        plt.loglog(C1, t[i[0]], "x-", label="".join(["$C_2$: ", str(i[1])]))
    plt.legend(fontsize=18)
    plt.xlabel("$C_1$", fontsize=18)
    plt.ylabel(r"$t\;/\;s$", fontsize=18)
    plt.title("Time Performance of fast Riemann with different Parameters", fontsize=20)
    plt.tick_params(right=True, top=True, direction="in")
    plt.savefig("data/images/FR_Para_x_1000_time.png", dpi=600)

    # plot case 3
    fig = plt.figure()
    fig, ax = plt.subplots(
        2, 1, figsize=(10, 10), sharex=True, gridspec_kw={"height_ratios": [1, 3]}
    )

    ax[0].tick_params(right=True, top=True, direction="in")
    ax[1].tick_params(right=True, top=True, direction="in")

    # plot case function
    box = ax[0].get_position()
    ax[0].plot(x, y3, linewidth=5.0)
    ax[0].set_ylabel("$y$", fontsize=18)
    ax[0].set_xlabel("$x$", fontsize=18)
    ax[0].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax[0].xaxis.set_tick_params(which="both", labelbottom=True)

    ax[0].set_title("Test Function $y=x^2$ \n", fontsize=20)

    for i in enumerate(C1):
        for j in enumerate(C2):
            tmp = time.time()
            fR = si.fast_riemann(y3, x[2] - x[1], V, c1=i[1], c2=j[1])
            t[i[0]][j[0]] = time.time() - tmp
            errFR = (fR - ref3) / ref3
            ax[1].semilogy(
                x,
                np.abs(errFR),
                color=(0, (i[0] * 2 + 1) / (np.size(C1) * 2 + 1), (j[0] + 1) / 10),
                label=f"C1:{i[1]}, C2:{j[1]}",
                linewidth=2.0,
            )

    box = ax[1].get_position()
    ax[1].set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    ax[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=7)
    # plt.xlim([0,0.5])
    plt.xlabel(r"$x$", fontsize=18)
    plt.ylabel("relative error", fontsize=18)
    # fig.suptitle(
    ax[1].set_title(
        "Parametrization Test for Fast Riemann with $n=1000$\n", fontsize=20
    )
    plt.savefig("data/images/FR_Para_x2_1000.png", dpi=600)

    # time plot
    fig = plt.figure(figsize=(12, 7))
    for i in enumerate(C1):
        plt.loglog(C1, t[i[0]], "x-", label="".join(["$C_2$: ", str(i[1])]))
    plt.legend(fontsize=18)
    plt.xlabel("$C_1$", fontsize=18)
    plt.ylabel(r"$t\;/\;s$", fontsize=18)
    plt.title("Time Performance of fast Riemann with different Parameters", fontsize=20)
    plt.tick_params(right=True, top=True, direction="in")
    plt.savefig("data/images/FR_Para_x2_1000_time.png", dpi=600)

if PRINT_FUNCTIONALITY_DOUBLE:
    N = 1000
    x = np.linspace(0, 10, N + 1)

    y = np.ones(N + 1)  # constant
    delta_x = x[1] - x[0]

    d_ref = cumulative_trapezoid(y, x)

    M = 1000
    v = np.linspace(-10, 0, M + 1) / 10
    v2 = v[::-1]

    # Gruenwald
    err_G = np.zeros((np.size(v)))

    for i in enumerate(v):
        d_G = si.gruenwald(si.gruenwald(y, delta_x, i[1]), delta_x, v2[i[0]])[:-1]

        err_G[i[0]] = np.max(np.abs((d_G - d_ref) / d_ref))

    plt.rcParams["figure.figsize"] = [8, 8]
    plt.figure()
    plt.semilogy(v, err_G, "rx:", linewidth=2.0)
    plt.title("Composition Test: Gruenwald with variing $v$ values", fontsize=20)
    plt.xlabel("$v1$ or\n $v2-1$", fontsize=18)
    plt.ylabel("max relative error", fontsize=18)
    plt.tick_params(right=True, top=True, direction="in")
    plt.savefig("data/images/double_varying_semiint_G.png", dpi=600)

    # Fast Riemann
    err_FR = np.zeros((np.size(v)))

    for i in enumerate(v):
        d_FR = si.fast_riemann(si.fast_riemann(y, delta_x, i[1]), delta_x, v2[i[0]])[1:]

        err_FR[i[0]] = np.max(np.abs((d_FR - d_ref) / d_ref))

    plt.figure()
    plt.semilogy(v, err_FR, "mx:", linewidth=2.0)
    plt.title("Composition Test: Fast Riemann with variing v values", fontsize=20)
    plt.xlabel("$v1$ or\n $v2-1$", fontsize=18)
    plt.ylabel("max relative error", fontsize=18)
    plt.tick_params(right=True, top=True, direction="in")
    plt.savefig("data/images/double_varying_semiint_FR.png", dpi=600)


if PRINT_FUNCTIONALITY_MULTIPLE_G:
    N = 1000
    x = np.linspace(0, 10, N + 1)
    y = np.ones(N + 1)  # constant

    delta_x = x[1] - x[0]

    d_ref = cumulative_trapezoid(y, x)

    v4 = [-1 / 3, -1 / 4, -1 / 5, -1 / 6, -1 / 8, -1 / 10]

    # Plott
    plt.rcParams["figure.figsize"] = [9, 9]
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2, hspace=0, wspace=0.4)
    ax = gs.subplots(sharex="col")

    fig.suptitle("Composition Test for Gruenwald\n y=1 with n=1000", fontsize=20)

    for i in enumerate(v4):
        COUNT = 0
        d_FR = y
        while COUNT > -1:
            COUNT += i[1]
            # prevent rounding error
            if i[0] != (0):
                COUNT = np.round(COUNT, 15)

            d_FR = si.gruenwald(d_FR, delta_x, i[1])

        d_FR = d_FR[:-1]
        err_FR = np.abs((d_FR - d_ref) / d_ref)
        if i[0] <= 2:
            ax[i[0]][0].plot(
                x[:-1], err_FR, color=(0, (1 + i[0]) / 6, 0), linewidth=4.0
            )
            ax[i[0]][0].tick_params(right=True, top=True, direction="in")
            ax[i[0]][0].legend([f"v: {np.round(i[1],4)}"])
        else:
            ax[i[0] - 3][1].plot(
                x[:-1], err_FR, color=(0, (1 + i[0]) / 6, 0), linewidth=4.0
            )
            ax[i[0] - 3][1].tick_params(right=True, top=True, direction="in")
            ax[i[0] - 3][1].legend([f"v: {np.round(i[1],4)}"])

    ax[1][0].set_ylabel("relative error", fontsize=18)
    ax[1][1].set_ylabel("relative error", fontsize=18)
    ax[2][0].set_xlabel("$x$", fontsize=18)
    ax[2][1].set_xlabel("$x$", fontsize=18)
    plt.savefig("data/images/triple_varying_semiint_G.png", dpi=600)

if PRINT_FUNCTIONALITY_MULTIPLE_FR:
    N = 1000
    x = np.linspace(0, 10, N + 1)
    y = np.ones(N + 1)  # constant

    delta_x = x[1] - x[0]

    d_ref = cumulative_trapezoid(y, x)

    v4 = [-1 / 3, -1 / 4, -1 / 5, -1 / 6, -1 / 8, -1 / 10]

    # Plott 1
    plt.rcParams["figure.figsize"] = [9, 9]
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2, hspace=0, wspace=0.4)
    ax = gs.subplots(sharex="col")

    fig.suptitle("Composition Test for Fast Riemann\n $y=1$ with $n=1000$", fontsize=20)

    for i in enumerate(v4):
        COUNT = 0
        d_FR = y
        while COUNT > -1:
            COUNT += i[1]
            if i[0] != (0):
                COUNT = np.round(COUNT, 15)

            d_FR = si.fast_riemann(d_FR, delta_x, i[1])

        d_FR = d_FR[1:]
        err_FR = np.abs((d_FR - d_ref) / d_ref)

        if i[0] <= 2:
            J1 = i[0]
            J2 = 0
        else:
            J1 = i[0] - 3
            J2 = 1

        ax[J1][J2].semilogy(x[:-1], err_FR, color=(0, 0, (1 + i[0]) / 6), linewidth=4.0)
        ax[J1][J2].tick_params(right=True, top=True, direction="in")
        ax[J1][J2].legend([f"v: {np.round(i[1],4)}\nmax err: {np.max(err_FR):.1e}"])

    ax[1][0].set_ylabel("relative error", fontsize=18)
    ax[1][1].set_ylabel("relative error", fontsize=18)
    ax[2][0].set_xlabel("$x$", fontsize=18)
    ax[2][1].set_xlabel("$x$", fontsize=18)
    plt.savefig("data/images/triple_varying_semiint_FR_init.png", dpi=600)

    # Plott 2
    plt.rcParams["figure.figsize"] = [9, 9]
    fig = plt.figure()
    gs = fig.add_gridspec(3, 2, hspace=0, wspace=0.4)
    ax = gs.subplots(sharex="col")

    fig.suptitle(
        "Composition Test for Fast Riemann\n $y=1$ with $n=1000$ and $C1=C2=10$",
        fontsize=20,
    )

    for i in enumerate(v4):
        COUNT = 0
        d_FR = y
        while COUNT > -1:
            COUNT += i[1]
            if i[0] != (0):
                COUNT = np.round(COUNT, 15)

            d_FR = si.fast_riemann(d_FR, delta_x, i[1], 10, 10)

        d_FR = d_FR[1:]
        err_FR = np.abs((d_FR - d_ref) / d_ref)

        if i[0] <= 2:
            J1 = i[0]
            J2 = 0
        else:
            J1 = i[0] - 3
            J2 = 1

        ax[J1][J2].semilogy(x[:-1], err_FR, color=(0, 0, (1 + i[0]) / 6), linewidth=4.0)
        ax[J1][J2].tick_params(right=True, top=True, direction="in")
        ax[J1][J2].legend([f"v: {np.round(i[1],4)}\nmax err: {np.max(err_FR):.1e}"])

    ax[1][0].set_ylabel("relative error", fontsize=18)
    ax[1][1].set_ylabel("relative error", fontsize=18)
    ax[2][0].set_xlabel("$x$", fontsize=18)
    ax[2][1].set_xlabel("$x$", fontsize=18)
    plt.savefig("data/images/triple_varying_semiint_FR_opt.png", dpi=600)
