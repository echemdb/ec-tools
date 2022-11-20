import numpy as np

# G1 Gruenwald Semi-Integration and Semi-Differentiation
# on base from Oldham: Electrochemical Science and Technology, 2012


def G1_Semiint(I,t):
    '''
    Doctest procedure
    >>> import pandas as pd
    >>> from scipy.integrate import cumulative_trapezoid
    >>> df1 = pd.read_csv('TestFiles/Testfile_1.csv')
    >>> np.allclose(G1_Semiint(G1_Semiint(df1['I'],df1['t']), df1['t'])[:-1], cumulative_trapezoid(df1['I'],df1['t']), rtol=1e-12)
    True
    >>> df2 = pd.read_csv('TestFiles/Testfile_2.csv')
    >>> np.allclose(G1_Semiint(G1_Semiint(df2['I'],df2['t']), df2['t'])[:-1], cumulative_trapezoid(df2['I'],df2['t']), rtol=1e-2)
    True
    '''
    # (equidistant) time step
    delta = t[1]-t[0]
    # No. of steps
    N_max = I.size
    # initialize with zeros
    G1 = np.zeros(N_max)
    for N in range(0,N_max):
        # value for n = N with w0 = 1
        G1_i = I[0]; 
        #      go from N to 0
        for n in range(N,0,-1):
            #print(n-1)
            G1_i = G1_i*((n)-0.5)/(n) + I[N-n+1]
        G1[N] = G1_i*np.sqrt(delta)
    return(G1)

def G1_Semidiff(I,t):
    # (equidistant) time step
    delta = t[1]-t[0]
    # No. of steps
    N_max = I.size
    # initialize with zeros
    G1 = np.zeros(N_max)
    for N in range(0,N_max):
        # value for n = N with w0 = 1
        G1_i = I[0]; 
        #      go from N to 0
        for n in range(N,0,-1):
            #print(n-1)
            G1_i = G1_i*((n)-1.5)/(n) + I[N-n+1]
        G1[N] = G1_i/np.sqrt(delta)
    return(G1)

# R1 Riemann and Liouville Semi-Integration and Semi-Differentiation
# on base from Oldham: Electrochemical Science and Technology, 2012
def R1_Semiint1(I,t):
    '''
    Doctest procedure
    >>> import pandas as pd
    >>> from scipy.integrate import cumulative_trapezoid
    >>> df1 = pd.read_csv('TestFiles/Testfile_1.csv')
    >>> np.allclose(R1_Semiint1(R1_Semiint1(df1['I'],df1['t']), df1['t'])[:-1], cumulative_trapezoid(df1['I'],df1['t']), rtol=1e-0)
    True
    >>> df2 = pd.read_csv('TestFiles/Testfile_2.csv')
    >>> np.allclose(R1_Semiint1(R1_Semiint1(df2['I'],df2['t']), df2['t'])[:-1], cumulative_trapezoid(df2['I'],df2['t']), rtol=1e-0)
    True
    '''
    # (equidistant) time step
    delta = t[1] - t[0]
    # No. of steps
    N_max = I.size
    # initialize with zeros
    R1 = np.zeros(N_max)

    for N in range(1,N_max+1):
        R1_i = 0
        for n in range(1,N):
            R1_i += I[n-1]*((N-n+1)**(3/2) - 2*(N-n)**(3/2) + (N-n-1)**(3/2))
            
        R1[N-1] = (4/3)*np.sqrt(delta/np.pi)*(I[N-1] + I[0]*(1.5*np.sqrt(N)-N**(3/2) + (N-1)**(3/2)) + R1_i)
    
    return(R1)

def R1_Semidiff1(I,t):
    delta = t[1] - t[0]
    # No. of steps
    N_max = I.size
    # initialize with zeros
    R1 = np.zeros(N_max)

    for N in range(1,N_max+1):
        R1_i = 0
        for n in range(1,N):
            R1_i += I[n-1]*(np.sqrt(N-n+1) - 2*np.sqrt(N-n) + np.sqrt(N-n-1))
            
        R1[N-1] = (2/np.sqrt(np.pi*delta))*(I[N-1] + I[0]*(1/(2*np.sqrt(N)) - np.sqrt(N) + np.sqrt(N-1)) + R1_i)
    
    return(R1)  


#Implementation of a algorithm for semi-integration.
#Fast Riemann-Liouville transformation (differintergration) - FRLT
#based on
#Pajkossy, T., Nyikos, L., 1984. Fast algorithm for differintegration. Journal of Electroanalytical Chemistry and Interfacial Electrochemistry 179, 65–69. https://doi.org/10.1016/S0022-0728(84)80275-2
"""
 TODO:
- evaluate if it is possible have varying Δx i.e. passing x array
"""

def prepare_kernel(q, delta_x, N, c1, c2):
    """
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


def semi_integration(y, q=-0.5, delta_x=1, c1=8, c2=2):
    """
    Return the semiintegral R of order q for y with the x interval delta_x and the filter constants
    c1 and c2.

    Semi-integrating two times with order q = -0.5 should give the same result as integrating once.
    The relative error should not exceed 0.25 percent for 1000 and 0.5 percent per 10000 integration steps.
    TODO:: see #6
    - add test for non constant y values

    TEST:
    >>> from scipy.integrate import cumulative_trapezoid
    >>> x = np.linspace(0,1000, 1001)
    >>> delta_x = x[1] - x[0]
    >>> y = np.array([1]*1001)
    >>> np.allclose(semi_integration(semi_integration(y, delta_x=delta_x), delta_x=delta_x), cumulative_trapezoid(y,x,initial=0), rtol=2.5e-03)
    True
    >>> x = np.linspace(0,1000, 10001)
    >>> delta_x = x[1] - x[0]
    >>> y = np.array([1]*10001)
    >>> np.allclose(semi_integration(semi_integration(y, delta_x=delta_x), delta_x=delta_x), cumulative_trapezoid(y,x,initial=0), rtol=5e-03)
    True
    >>> import pandas as pd
    >>> df2 = pd.read_csv('TestFiles/Testfile_2.csv')
    >>> delta_x = df2['t'][1] - df2['t'][0]
    >>> np.allclose(semi_integration(semi_integration(df2['I'],delta_x=delta_x),delta_x=delta_x), cumulative_trapezoid(df2['I'],delta_x=delta_x), rtol=1e-0)
    True
    """

    N = y.size
    R = np.zeros(N)
    s, w1, w2 = prepare_kernel(q, delta_x, N, c1, c2)
    for k in range(1, N):
        for i in range(s.size):
            s[i] = s[i] * w1[i] + y[k] * w2[i]
            R[k] = R[k] + s[i]
    return R
