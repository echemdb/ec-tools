r"""
Loose collection of helper functions used in several places in the package.
"""

from collections import namedtuple

import numpy as np

x0_indices = namedtuple("x0_indices", ["pos", "neg"])
extrema_indices = namedtuple("extrema_indices", ["pos", "neg"])


def find_x0_indices(arr):
    """Takes x and y as numpy arrays and returns the ordered zero points by assuming linear behavior between
    the points.
    The mode determines if all or either positive or negative zero points should be returned.

    TODO:: see #5
    - add non-linear interpolation

    TESTS:

    >>> x = np.array([1, 1, -1, -1, 1])
    >>> find_x0_indices(x)
    x0_indices(pos=array([3]), neg=array([1]))

    >>> x = np.array([1, 1, -1, -1, 0.9])
    >>> find_x0_indices(x)
    x0_indices(pos=array([4]), neg=array([1]))

    Special case where y is exactly zero:
    >>> x = np.array([1, 1, 0, -1, 1])
    >>> find_x0_indices(x)
    x0_indices(pos=array([3]), neg=array([2]))

    """
    signs = np.diff(np.sign(arr))

    # use only one expression at first to reduce the array size for subsequent steps
    crossings = np.where(signs != 0)[0]

    exact_crossings = crossings[np.where((signs[crossings] == 1) | (signs[crossings] == -1))[0]]
    exact_crossings_pos = exact_crossings[np.where(signs[exact_crossings] > 0)[0]]
    exact_crossings_neg = exact_crossings[np.where(signs[exact_crossings] < 0)[0]]

    non_exact_crossings = crossings[np.where((signs[crossings] > 1) | (signs[crossings] < -1))[0]]
    # determine index of which value is closer to zero
    non_exact_crossings_optimized = np.where(
        np.abs(arr[non_exact_crossings]) <= np.abs(arr[non_exact_crossings + 1]),
        non_exact_crossings,
        non_exact_crossings + 1,
    )
    non_exact_crossings_pos = non_exact_crossings_optimized[np.where(signs[non_exact_crossings] > 0)[0]]
    non_exact_crossings_neg = non_exact_crossings_optimized[np.where(signs[non_exact_crossings] < 0)[0]]

    return x0_indices(
        np.sort(np.concatenate([exact_crossings_pos[1::2], non_exact_crossings_pos])),
        np.sort(np.concatenate([exact_crossings_neg[1::2], non_exact_crossings_neg])),
    )


def find_x0_values(x, y, mode="all"):
    """Takes x and y as numpy arrays and returns the ordered zero points by assuming linear behavior between
    the points.
    The mode determines if all or either positive or negative zero points should be returned.

    TODO:: see #5
    - add non-linear interpolation

    TESTS:

    >>> x = np.array([10, 10.5, 11, 11.5, 12])
    >>> y = np.array([1, 1, -1, -1, 1])
    >>> find_x0_values(x, y)
    array([10.75, 11.75])

    >>> x = np.array([10, 10.5, 11, 11.5, 12])
    >>> y = np.array([1, 1, -1, -1, 1])
    >>> find_x0_values(x, y, mode='pos')
    array([11.75])

    >>> x = np.array([10, 10.5, 11, 11.5, 12])
    >>> y = np.array([1, 1, -1, -1, 1])
    >>> find_x0_values(x, y, mode='neg')
    array([10.75])

    Special case where y is exactly zero:
    >>> x = np.array([10, 10.5, 11, 11.5, 12])
    >>> y = np.array([1, 1, 0, -1, 1])
    >>> find_x0_values(x, y)
    array([11.  , 11.75])

    """
    signs = np.diff(np.sign(y))

    if mode == "all":
        exact_crossings = np.where((signs == 1) | (signs == -1))[0]
        non_exact_crossings = np.where((signs > 1) | (signs < -1))[0]
    elif mode == "pos":
        exact_crossings = np.where(signs == 1)[0]
        non_exact_crossings = np.where(signs > 1)[0]
    elif mode == "neg":
        exact_crossings = np.where(signs == -1)[0]
        non_exact_crossings = np.where(signs < -1)[0]

    m = (y[non_exact_crossings] - y[non_exact_crossings + 1]) / (
        x[non_exact_crossings] - x[non_exact_crossings + 1]
    )
    delta_x = -y[non_exact_crossings] / m

    return np.sort(np.concatenate([x[exact_crossings[1::2]], delta_x + x[non_exact_crossings]]))


def find_extrema_indices(y, mode="all"):
    """Return the indices of the extrema of an array which holds values of a periodical linearly changing signal.
    The `mode` determines if the positive `pos`, negative `neg`, or `all` extrema are returned.
    One example for such a signal is the potential in cyclic voltammetry, a electrochemical method.

    TODO:
      - always return both types of vertices grouped because regularly both are needed

    TESTS:

    >>> E = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.10, 0.11, 0.12,
    ... 0.13, 0.14])
    >>> find_extrema_indices(E)
    extrema_indices(pos=array([5]), neg=array([11]))

    >>> find_extrema_indices(E, mode="pos")
    extrema_indices(pos=array([5]), neg=None)
    """
    signs = np.diff(np.sign(np.diff(y)))

    if mode == "pos":
        extrema_pos = np.where(signs == -2)[0]
        return extrema_indices(extrema_pos + 1, None)  # raise indices by one to accomodate for differentiation
    elif mode == "neg":
        extrema_neg = np.where(signs == 2)[0]
        return extrema_indices(None, extrema_neg + 1)
    else:
        extrema = np.where(signs != 0)[0]
        extrema_pos = extrema[np.where(signs[extrema] < 0)[0]]
        extrema_neg = extrema[np.where(signs[extrema] > 0)[0]]
        return extrema_indices(extrema_pos + 1, extrema_neg + 1)


def discrete_scan_rates(t, x):
    """Return array of discrete scan rates of given t and x arrays.

    TESTS:

    >>> t = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    >>> E = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.10, 0.11, 0.12,
    ... 0.13, 0.14])
    >>> discrete_scan_rates(t, E)
    array([0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
               0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005])
    """

    return np.abs(np.diff(x) / np.diff(t))


def determine_scan_rate(t, x):
    """Return scan rate of given t and x arrays.

    TESTS:

    >>> t = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    >>> E = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.10, 0.11, 0.12,
    ... 0.13, 0.14])
    >>> float(determine_scan_rate(t, E))
    0.005
    """

    return discrete_scan_rates(t, x).mean()


def detect_voltammetric_measurement(t, E, threshold=0.05):
    """Probe if the potential is composed by linear potential sweeps which is the case for cyclic voltammetry.
    Default threshold is 5 percent of scan rate.

    TODO::
     - not working for data with multiple parts having different scan rates

    TESTS:

    >>> t = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    >>> E = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.10, 0.11, 0.12,
    ... 0.13, 0.14])
    >>> detect_voltammetric_measurement(t, E)
    np.True_

    # Potential step measurement
    >>> t = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    >>> E = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26, 0.26])
    >>> detect_voltammetric_measurement(t, E)
    np.False_
    """

    discrete_rates = discrete_scan_rates(t, E)
    mean_scan_rate = discrete_rates.mean()
    return (np.abs(discrete_rates / mean_scan_rate) - 1 < threshold).all()


def cycles_indices(E, mode):
    """Return a list of index tuples defining the beginnung and the end of the cycles from spliting in `mode`.
    `mode` allows `inital` and `vertex` which splits either at initial potential crossings or at the vertices.
    If the intial potential lays outside the vertices i.e. negative of the negative vertices or positive of the
    positive vertices `mode=initial` will fail.

    .. plot::

        import matplotlib.pyplot as plt
        import numpy as np
        from ec_tools.helper import cycles_indices
        from scipy import signal

        t = np.linspace(0, np.pi+1, 1000)
        cycles = signal.sawtooth(np.pi * t + 1, width=0.5)
        indices = cycles_indices(cycles, mode="initial")

        fig, axs = plt.subplots(2)

        axs[0].hlines(y=cycles[0], xmin=0, xmax=np.pi+1, color="red")
        axs[0].plot(t[0:indices[0]-1], cycles[0:indices[0]-1])
        axs[0].plot(t[indices[0]:indices[1]-1], cycles[indices[0]:indices[1]-1])

        axs[1].hlines(y=cycles[0], xmin=0, xmax=np.pi+1, color="red")
        axs[1].plot(t[0:indices[0]], cycles[0:indices[0]], marker = 'o')
        axs[1].plot(t[indices[0]:indices[1]], cycles[indices[0]:indices[1]], marker = 'o')
        axs[1].set_xlim((1.9, 2.1))
        axs[1].set_ylim((-0.45, -0.25))

    TESTS:

    >>> E = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.10, 0.11, 0.12,
    ... 0.13, 0.14])
    >>> cycles_indices(E, mode="initial")
    array([11])

    >>> E = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.0999, 0.09, 0.10, 0.11, 0.12,
    ... 0.13, 0.14])
    >>> cycles_indices(E, mode="initial")
    array([11])

    >>> E = np.array([0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.10,
    ... 0.11, 0.12, 0.13, 0.14])
    >>> cycles_indices(E, mode="initial")
    Traceback (most recent call last):
    ValueError: The initial potential is not between the negative and positive vertices.

    >>> E = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.10, 0.11, 0.12,
    ... 0.13, 0.14])
    >>> cycles_indices(E, mode="vertex")
    extrema_indices(pos=array([5]), neg=array([11]))

    >>> E = np.array([0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.10,
    ... 0.11, 0.12, 0.13, 0.14])
    >>> cycles_indices(E, mode="vertex")
    extrema_indices(pos=array([7]), neg=array([13]))
    """
    vertex_indices = find_extrema_indices(E)
    if mode == "initial":
        initial_potential = E[0]

        if len(
            np.where((E[vertex_indices.pos] < initial_potential) | (E[vertex_indices.neg] > initial_potential))[0]
        ):
            raise ValueError("The initial potential is not between the negative and positive vertices.")
        initial_crossings = find_x0_indices(E - initial_potential)
        if E[1] - E[0] > 0:
            return initial_crossings.pos
        else:
            return initial_crossings.neg
    elif mode == "vertex":
        return vertex_indices


def detect_step(t, x):
    """Returns the index of the step in given t and x arrays.
    Index is the where the changed value of t located.

    TESTS:

    >>> t = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
    ... 1.6, 1.7, 1.8, 1.9, 2.0])
    >>> E = np.array([-0.205383, -0.204468, -0.204773, -0.205078, 0.500183, 0.500488, 0.501099,
    ... 0.500183, 0.500488, 0.500488, 0.500183, 0.500488, 0.500488, 0.500488, 0.500183, 0.499878,
    ... 0.499878, 0.500183, 0.500183, 0.499878, 0.500488])
    >>> int(detect_step(t, E))
    4
    """
    return np.abs(np.diff(x) / np.diff(t)).argmax() + 1
