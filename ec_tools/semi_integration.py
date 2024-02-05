r"""
The differentiation and the integration are common mathematical operations. 
The differentiation of an arbitrary function is often expressed by :math:`\frac{\mathrm{d}}{\mathrm{d}x} f(x)`.
Interestingly, the integration can also be defined in a similar way:

.. math:: \frac{\mathrm{d}^{-1}}{\mathrm{d}x^{-1}} f(x)= \int_0^x f(\tau) \mathrm{d}\tau

Here it must be considered that a lower limit must be defined so that the integral is completely defined.
These operations can be then displayed in a more general way by:

.. math:: \frac{\mathrm{d}^{v}}{\mathrm{d}x^{v}} f(x) = \int_0^x f(\tau) \mathrm{d}\tau

With:

* :math:`v= 1`: Differentiation

* :math:`v=-1`: Integration

The so-called semi-operators are :math:`v=\frac{1}{2}` for semi differentiation and :math:`v=-\frac{1}{2}` for semi integration. 
The following figure visualizes the idea of semi integration and semi differentiation.

.. image:: ../doc/files/images/semidif.png
  :width: 600
  :alt: Image 

A semi integration of a peak function (bottom left) results in a hybrid function (top) and 
by applying another semi integration it is transformed into a wave-like function (bottom right), which is equal to 
performing a regular integration of the peak function. The operations in opposite direction are analog, 
except that semi differentiations and a regular differentiation are performed, respectively.

Semi Integration Methods
------------------------
Several methods exist for applying semi integration and semi differentiation, respectively. 
These computations generally need discrete values, i.e. the function graph (see above) has to be seperated into discrete finite values:

.. math:: f(0), f(\delta), ..., f((N-1)\delta), f(N\delta)

It is assumed that the step size :math:`\delta` is equidistant, meaning for a fixed set of x-Values N:

.. math:: \delta = \frac{x_N}{N}

The following algorithms (Gruenwald and Riemann & Liouville) are taken from Oldham in :cite:p:`oldham_fractional_2006` and the fast Riemann from Pajkossy et. al. in :cite:p:`pajkossy_fast_1984_65`.


Gruenwald Algorithms
^^^^^^^^^^^^^^^^^^^^
One method of semi integration was introduced by Gruenwald :cite:p:`grunwald_uber_1867_441` and
Oldham shows in his web ressource 1244 from :cite:p:`oldham_electrochemical_2013` how this type of semi integration can be applied as an algorithm, called G1.

It can be generally expressed by the sum of the discrete function values multiplied with weights :math:`w_n` 
divided by the stepsize:

.. math:: \frac{\mathrm{d}^{\pm 0.5}}{\mathrm{d}t^{\pm 0.5}} f(t) =\frac{1}{\delta^{\pm 0.5}} \sum_{n=0}^{N-1} w_n f(n\delta)

The G1 algorithm is ideal for (cyclic) voltammograms, where the early signals are small. 
**Note**, that these algorithms are less suitable for step and pulse techniques, in which the initial currents are large :cite:p:`oldham_electrochemical_2013`.

The weights can be expressed in different ways. The single weights :math:`w_n` often depend on their predecessor :math:`w_{n-1}`. 
As the factorial expression could lead to an computive intensive overflow, this algorithm needs to be simplified.

The **Gruenwald G1 semi integration algorithm** is defined as follows:

.. math:: \frac{\mathrm{d}^{- 0.5}}{\mathrm{d}t^{- 0.5}} f(t) \approx \sqrt{\delta} \sum_{n=1}^{N} w_{N-n} f(n\delta)

Which can be also displayed in reverse summation to allow a more efficient
implementation:

.. math:: \frac{\mathrm{d}^{- 0.5}}{\mathrm{d}t^{- 0.5}} f(t) \approx \sqrt{\delta} \sum_{n=N}^{1} w_{N-n} f(n\delta)

With:

* :math:`w_0 = 1`

* :math:`w_n = \frac{(n-0.5)w_{n-1}}{n} = (1-\frac{0.5}{n})w_{n-1}`

The previous definition can also be applied as  **Gruenwald G1 semi differentiation algorithm** with some changes:

.. math:: \frac{\mathrm{d}^{0.5}}{\mathrm{d}t^{0.5}} f(t) \approx \frac{1}{\sqrt{\delta}} \sum_{n=0}^{N-1} w_{N-n} f(n\delta)

With:

* :math:`w_0 = 1`

* :math:`w_n = \frac{(n-1.5)w_{n-1}}{n}`


Riemann and Liouville Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Another method to determine the semi integral was introduced by Riemann and Liouville :cite:p:`riemann_versuch_2013_331` and 
described by Oldham in :cite:p:`oldham_electrochemical_2013` as R1 algorithm in his web ressource 1244. 
These sort of algorithms are mainly straightforward general-purpose algorithms.

The algorithm for the **R1 semi integration** is defined by:

.. math:: 

   \frac{\mathrm{d}^{-\frac{1}{2}}}{\mathrm{d}t^{-\frac{1}{2}}}f(t) = \frac{4}{3} \sqrt{\frac{\delta}{\pi}} \left[ f(N\delta) + \left\{ \frac{3}{2}\sqrt{N} - N^\frac{3}{2} + (N-1)^\frac{3}{2} \right\}f(0) \right.\\

   \left. + \sum_{n=1}^{N-1} \left\{ (N-n+1)^\frac{3}{2} - 2 (N-n)^\frac{3}{2} + (N-n-1)^\frac{3}{2} \right\}f(n\delta) \right]

Similar to the G1 algorithm is the R1 for semi integration and semi differentiation. But it is not usable for application to currents that arise from potential steps or leaps :cite:p:`oldham_electrochemical_2013`, as:

#. The large current at :math:`t=0`, immediately following the step is impossible to measure accurately and even if it would be possible,
   it is likely to be largely composed of a chemically uninteresting non-faradaic component. But the algorithm still requires a value of :math:`f(0)`.

#. The algorithm is based on the assumption that :math:`f(t)` can be treated as an assemblage of linear segments, 
   whereas faradaic currents arising from a potential step and are non-linear with time. 

The general definition for the **R1 semi differentiation** is:

.. math::

   \frac{\mathrm{d}^{\frac{1}{2}}}{\mathrm{d}t^{\frac{1}{2}}}f(t) = \frac{2}{\sqrt{\pi\delta}} \left[ f(N\delta) + \left\{ \frac{1}{2\sqrt{N}} - \sqrt{N} + \sqrt{N-1}\right\}f(0) \right.\\

   \left. + \sum_{n=1}^{N-1} \left\{ \sqrt{N-n+1} - 2 \sqrt{N-n} + \sqrt{N-n-1} \right\}f(n\delta) \right]


Fast Riemann
^^^^^^^^^^^^

The following algorithm was introduced by Pajkossy et al. in 1984 :cite:p:`pajkossy_fast_1984_65` and is based on the Riemann-Liouville transformation (RLT). 
Its big advantage is, that the computation time increases only linearly with the number of points (:math:`N`). 
Here it is necessary to define some input variables (besides :math:`t` and :math:`I(t)` data), where :math:`q` is equal to :math:`v`, describing a semi integration or semi differentiation. 
:math:`\delta_t` defines the constant time intervall (i.e. :math:`t_2 - t_1`) and :math:`c_1, c_2` are constant values,
which is set to :math:`c_1=8, c_2=2` by default, as recommended by Pajkossy. With these variables, the  algorithm can be described as pseudo code:

.. image:: ../test/data/images/alg_fast_riemann.png
  :width: 500
  :alt: Image

Here, :math:`R` represents the calculated semi integral, i.e. :math:`R \approx \frac{\mathrm{d}^{v}}{\mathrm{d}t^{v}} I(t)`.

"""

import numpy as np
from transonic import jit


def semi_integration(y, x, v=-0.5, alg="frlt", transonic_backend="pythran", d_tol=1e-5):
    r"""
    A generalized call is implemented, in which the user can define with ``v`` the operation i.e. semi integration
    (``v`` :math:`=-0.5`) or semi differentiation (``v`` :math:`=0.5`) for a predefined dataset (``y`` and ``t``).
    Furthermore he can optionally set a flag (``alg``) to select a specific algorithms and with ``transonic_backend``
    he apply a speed up by transonic (with numba or pythran backend).

    Available algorithms (``alg`` ):

    ``frlt``: Fast Riemann-Liouville transformation (default)

    ``g1``: Gruenwald

    ``r1``: Riemann and Liouville

    Available backends (``transonic_backend`` ):

    ``python``: Transonic package with python backend (default)

    ``numba``: Transonic package with numba backend

    ``pythran``: Transonic package with pythran backend

    ``d_tol`` (by default: :math:`1 \cdot 10^{-5}`) defines the maximum relational difference between individual step
    and the average step size. It can be modified, if the time steps are not equally spaced.
    """

    # Calc average step size
    deltas = np.diff(x)
    delta_x = deltas.mean()

    # warning if avg. delta_x differs to much from single ones
    if np.max(np.abs(deltas / delta_x - 1)) >= d_tol:
        print("Warning: step size tolerance reached!")

    if alg not in ["g1", "r1", "frlt"] or transonic_backend not in [
        "python",
        "numba",
        "pythran",
    ]:
        raise ValueError("\nNo matching setting for alg or transonic_backend")

    if alg == "frlt":
        if transonic_backend == "python":

            @jit(backend="python")
            def fast_riemann_python(y, delta_x, v):
                return fast_riemann(y, delta_x, v)

            return fast_riemann_python(y, delta_x, v)

        if transonic_backend == "numba":

            @jit(backend="numba")
            def fast_riemann_numba(y, delta_x, v):
                return fast_riemann(y, delta_x, v)

            return fast_riemann_numba(y, delta_x, v)

        if transonic_backend == "pythran":

            @jit(backend="pythran")
            def fast_riemann_pythran(y, delta_x, v):
                return fast_riemann(y, delta_x, v)

            return fast_riemann_pythran(y, delta_x, v)

    elif alg == "g1":
        if transonic_backend == "python":

            @jit(backend="python")
            def gruenwald_python(y, delta_x, v):
                return gruenwald(y, delta_x, v)

            return gruenwald_python(y, delta_x, v)

        if transonic_backend == "numba":

            @jit(backend="numba")
            def gruenwald_numba(y, delta_x, v):
                return gruenwald(y, delta_x, v)

            return gruenwald_numba(y, delta_x, v)

        if transonic_backend == "pythran":

            @jit(backend="pythran")
            def gruenwald_pythran(y, delta_x, v):
                return gruenwald(y, delta_x, v)

            return gruenwald_pythran(y, delta_x, v)

    elif alg == "r1":
        if transonic_backend == "python":

            @jit(backend="python")
            def riemann_python(y, delta_x, v):
                return riemann(y, delta_x, v)

            return riemann_python(y, delta_x, v)

        if transonic_backend == "numba":

            @jit(backend="numba")
            def riemann_numba(y, delta_x, v):
                return riemann(y, delta_x, v)

            return riemann_numba(y, delta_x, v)

        if transonic_backend == "pythran":

            @jit(backend="pythran")
            def riemann_pythran(y, delta_x, v):
                return riemann(y, delta_x, v)

            return riemann_pythran(y, delta_x, v)


def gruenwald(y, delta_x, v=-0.5):
    """
    Gruenwald Algorithm
    ^^^^^^^^^^^^^^^^^^^

    Implementation of the Gruenwald algorithm for
    semi-integration (``v`` :math:`=-0.5`) and semi-differentiation (``v``:math:` =0.5`)
    based on Oldham :cite:p:`oldham_electrochemical_2013`.

    Input:

    ``y``: y-values

    ``delta_x``: step size (i.e. x2-x1)

    ``v``: :math:`-0.5` (default) or in range :math:`-1 < v < 1`

    EXAMPLES:

    Simple examples to verify the algorithm by applying semi-integration twice (i.e. resulting in a normal integration)
    and comparing the result with a numerical integration from scipy. First, the input data is a function with constant :math:`y`:

    >>> from scipy.integrate import cumulative_trapezoid
    >>> x = np.linspace(0,1000, 1001)
    >>> delta_x = x[1] - x[0]
    >>> y = np.array([1]*1001)
    >>> np.allclose(gruenwald(gruenwald(y,delta_x),delta_x)[:-1], cumulative_trapezoid(y,x), rtol=1e-15)
    True

    Second test with more application-related values from a gaussian distribution function (from scipy):

    >>> from scipy.stats import norm
    >>> from scipy.integrate import cumulative_trapezoid
    >>> x = np.linspace(0, 8, 1001)
    >>> delta_x = x[1] - x[0]
    >>> y = norm.pdf(x,4,1)
    >>> np.allclose(gruenwald(gruenwald(y,delta_x),delta_x)[:-1], cumulative_trapezoid(y,x), rtol=1e-1)
    True

    """
    if v != (-0.5 or 0.5):
        print("\nWarning: algorithm is only tested for v=0.5 and v=-0.5.")
        print("         Other values for v are not verified!\n")

    # No. of steps
    n_max = y.size
    # initialize with zeros
    g_1 = np.zeros(n_max)
    for N in range(1, n_max + 1):
        # value for n = N with w0 = 1
        g_i = y[0]
        #      go from N to 0
        for i in range(N - 1, 0, -1):
            g_i = g_i * (1 - (v + 1) / i) + y[N - i]
        g_1[N - 1] = g_i * np.sqrt(delta_x)
    return g_1


def riemann(y, delta_x, v=-0.5):
    """

    Riemann Algorithm
    ^^^^^^^^^^^^^^^^^

    Implementation of the Riemann algorithm for
    semi-integration (``v`` :math:`=-0.5`) and semi-differentiation (``v``:math:` =0.5`)
    based on Oldham :cite:p:`oldham_electrochemical_2013`.

    Input:

    ``y``: y-values

    ``delta_x``: step size (i.e. x2-x1)

    ``v``: -0.5 (default) or 0.5

    EXAMPLES:

    Simple examples to verify the algorithm by applying semi-integration twice (i.e. resulting in a normal integration)
    and comparing the result with a numerical integration from scipy. First, the input data is a function with constant :math:`y`:

    >>> from scipy.integrate import cumulative_trapezoid
    >>> x = np.linspace(0,1000, 1001)
    >>> delta_x = x[1] - x[0]
    >>> y = np.array([1]*1001)
    >>> np.allclose(riemann(riemann(y,delta_x),delta_x)[:-1], cumulative_trapezoid(y,x), rtol=1e-0)
    True

    Second, test with more application-related values from a gaussian distribution function (from scipy):

    >>> from scipy.stats import norm
    >>> from scipy.integrate import cumulative_trapezoid
    >>> x = np.linspace(0, 8, 1001)
    >>> delta_x = x[1] - x[0]
    >>> y = norm.pdf(x,4,1)
    >>> np.allclose(riemann(riemann(y,delta_x),delta_x)[:-1], cumulative_trapezoid(y,x), rtol=1e-0)
    True

    """

    if v not in (0.5, -0.5):
        raise ValueError("\nError: This algorithm accepts only v=0.5 and v=-0.5.")

    # No. of steps
    n_max = y.size
    # initialize with zeros
    r_1 = np.zeros(n_max)

    if v == -0.5:
        sqrt_d_pi = (4 / 3) * np.sqrt(delta_x / np.pi)
    elif v == 0.5:
        sqrt_d_pi = 2 / np.sqrt(delta_x * np.pi)

    for N in range(1, n_max + 1):
        r_i = 0
        for i in range(1, N):
            r_i += y[i - 1] * (
                (N - i + 1) ** (1 - v) - 2 * (N - i) ** (1 - v) + (N - i - 1) ** (1 - v)
            )

        r_1[N - 1] = sqrt_d_pi * (
            y[N - 1]
            + y[0] * ((1 - v) * N ** (-v) - N ** (1 - v) + (N - 1) ** (1 - v))
            + r_i
        )

    return r_1


def fast_riemann(y, delta_x=1, q=-0.5, c1=8, c2=2):
    """

    Fast Riemann Algorithm
    ^^^^^^^^^^^^^^^^^^^^^^

    Implementation of the fast Riemann algorithm for semi-integration.
    based on Pajkossy et al :cite:p:`pajkossy_fast_1984_65`. Return the semiintegral R of order :math:`q` for :math:`y`
    with the :math:`x` interval :math:`\delta_x` and the filter constants :math:`c_1` and :math:`c_2`.

    Input:

    ``y``: y-values

    ``delta_x``: step size (i.e. x2-x1), by default 1

    ``v``: -0.5 (default) or 1 < q < 0

    ``c1, c2``: filter constants (default c1: 8, c2: 2)

    EXAMPLES:
    Simple examples to verfiy the algorithm by applying semi-integration twice (i.e. resulting in a normal integration)
    and comparing the result with a numerical integration from scipy. First, the input data is a function with constant :math:`y`:

    >>> from scipy.integrate import cumulative_trapezoid
    >>> x = np.linspace(0,1000, 1001)
    >>> delta_x = x[1] - x[0]
    >>> y = np.array([1]*1001)
    >>> np.allclose(fast_riemann(fast_riemann(y, delta_x=delta_x), delta_x=delta_x), cumulative_trapezoid(y,x,initial=0), rtol=2.5e-03)
    True

    Second, test with more application-related values from a gaussian distribution function (from scipy):

    >>> from scipy.stats import norm
    >>> from scipy.integrate import cumulative_trapezoid
    >>> x = np.linspace(0, 1000, 1001)
    >>> delta_x = x[1] - x[0]
    >>> y = norm.pdf(x,500,1)
    >>> np.allclose(fast_riemann(fast_riemann(y, delta_x=delta_x), delta_x=delta_x), cumulative_trapezoid(y,x,initial=0), rtol=1e-0)
    True

    Increase the number of samples:

    >>> x = np.linspace(0,1000, 10001)
    >>> delta_x = x[1] - x[0]
    >>> y = np.array([1]*10001)
    >>> np.allclose(fast_riemann(fast_riemann(y, delta_x=delta_x), delta_x=delta_x), cumulative_trapezoid(y,x,initial=0), rtol=5e-03)
    True

    >>> from scipy.stats import norm
    >>> from scipy.integrate import cumulative_trapezoid
    >>> x = np.linspace(0,1000, 10001)
    >>> delta_x = x[1] - x[0]
    >>> y = norm.pdf(x,500,1)
    >>> np.allclose(fast_riemann(fast_riemann(y, delta_x=delta_x), delta_x=delta_x), cumulative_trapezoid(y,x,initial=0), rtol=1e-0)
    True
    """

    if q > 0:
        raise ValueError(
            "This algorithm works right now only for semi integration, i.e. (q<0)"
        )

    if q != -0.5:
        print("\nWarning: algorithm is only tested for v=0.5 and v=-0.5.")
        print("         Other values for v are not verified!\n")

    def prepare_kernel(q, delta_x, N, c1, c2):
        r"""
        Setup the integration kernel with the order q, the x interval delat_x, the length of the array N,
        and the filter constants c1 and c2.
        """
        tau0 = delta_x * N**0.5
        a0 = np.sin(np.pi * q) / (np.pi * q * tau0**q)
        # total number of filters
        n_filters = 2 * c1 * c2 + 1
        # dimension according to the number of filters
        # filter weights
        w1 = np.zeros(n_filters)
        w2 = np.zeros(n_filters)
        # auxiliary array
        s = np.zeros(n_filters)

        for i in range(2 * c1 * c2):
            j = i - c1 * c2
            a_j = (a0 / c2) * np.exp(j / c2)
            t_j = tau0 * np.exp(-j / (q * c2))
            w1[i] = t_j / (delta_x + t_j)
            w2[i] = a_j * (1 - w1[i])
        return s, w1, w2

    N = y.size
    R = np.zeros(N)
    s, w1, w2 = prepare_kernel(q, delta_x, N, c1, c2)
    for k in range(1, N):
        for i in range(s.size):
            s[i] = s[i] * w1[i] + y[k] * w2[i]
            R[k] = R[k] + s[i]
    return R
