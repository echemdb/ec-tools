r"""
Loose collection of helper functions used in several places in the package.
"""

import numpy as np


def find_x0_values(x, y, mode="all"):
    """Takes x and y as numpy arrays and returns the ordered zero points by assuming linear behavior between
    the points.
    The mode determines if all or either positive or negative zero points should be returned.

    TODO:: see #5
    - add non-linear interpolation

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

    return np.sort(
        np.concatenate([x[exact_crossings[1::2]], delta_x + x[non_exact_crossings]])
    )


def determine_scan_rate(t, x):
    """Return scan rate of given t and x arrays.

    TEST:
    >>> t = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32])
    >>> E = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14])
    >>> determine_scan_rate(t, E)
    0.005
    """

    return np.abs(np.diff(x) / np.diff(t)).mean()

def find_extrema_indeces(y, mode="all"):
    """Return the indexes of limits of cyclic voltammetry.

    TEST:
    >>> E = np.array([0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.14, 0.13, 0.12, 0.11, 0.10, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14])
    >>> find_extrema_indeces(E)
    array([ 5, 11])

    >>> find_extrema_indeces(E, mode="pos")
    array([5])
    """
    signs = np.diff(np.sign(np.diff(y)))

    if mode == "all":
        extrema = np.where((signs == 2) | (signs == -2))[0]
    elif mode == "pos":
        extrema = np.where(signs == -2)[0]
    elif mode == "neg":
        extrema = np.where(signs == 2)[0]

    return extrema + 1

def detect_step(t, x):
    """Returns the index of the step in given t and x arrays.
    Index is the where the changed value of t located.
    TEST:
    >>> t = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5,
    ... 1.6, 1.7, 1.8, 1.9, 2.0])
    >>> E = np.array([-0.205383, -0.204468, -0.204773, -0.205078, 0.500183, 0.500488, 0.501099,
    ... 0.500183, 0.500488, 0.500488, 0.500183, 0.500488, 0.500488, 0.500488, 0.500183, 0.499878,
    ... 0.499878, 0.500183, 0.500183, 0.499878, 0.500488])
    >>> detect_step(t, E)
    4
    """
    return np.abs(np.diff(x) / np.diff(t)).argmax() + 1
