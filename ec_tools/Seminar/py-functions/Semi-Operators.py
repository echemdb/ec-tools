import numpy as np

# G1 Gruenwald Semi-Integration and Semi-Differentiation
# on base from Oldham: Electrochemical Science and Technology, 2012


def G1_Semiint(I,t):
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