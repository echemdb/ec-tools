
# check if  doctest can load csv files

import numpy as np
import pandas as pd
import doctest

# import numerical integration
from scipy.integrate import cumulative_trapezoid



def G1_Semiint1(I,t):
    # Gruenwald-G1 semiintegration algorithm
    '''
    Doctest procedure
    >>> df = pd.read_csv('Testfile_1.csv')
    >>> np.allclose(G1_Semiint1(G1_Semiint1(df['I'],df['t']), df['t'])[:-1], cumulative_trapezoid(df['I'],df['t']), rtol=1e-12)
    True
    >>> df = pd.read_csv('Testfile_2.csv')
    >>> np.allclose(G1_Semiint1(G1_Semiint1(df['I'],df['t']), df['t'])[:-1], cumulative_trapezoid(df['I'],df['t']), rtol=1e-12)
    True
    '''

    # (equidistant) time step
    delta = t[1]-t[0]
    # No. of steps
    N_max = I.size
    # initialize with zeros
    G1 = np.zeros(N_max)

    sqr_d = np.sqrt(delta)
    
    for N in range(1,N_max+1):
        # value for n = N with w0 = 1
        G1_i = I[0]; 
        #      go from N to 0
        for n in range(N-1,0,-1):
            G1_i = G1_i*(1-(0.5)/n) + I[N-n]
            
        G1[N-1] = G1_i*sqr_d
    return(G1)

