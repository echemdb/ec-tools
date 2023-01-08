# Based on Oldham: Electrochemical Science and Technology, 2012 (web resources: Web#1244, Web#1245) 
# Calculates nernstian steady-state voltammetry at a hemispherical microelectrode (eq. 12:19)

import numpy as np

def current_hemispherical1(T=298.15, D_R=1e-9, D_O=1e-9, c_bR=1, r_hemi=5e-6, E_oi=0, E_0=-0.25, E_final=0.25, u=0.025, N_max=1000, v=0.5):
    # Universal gas constant
    R = 8.31446261815324; 
    # Faraday constant
    F = 96485.33212331; 
    
    # (optional) calculate Numbers
    N = np.array(range(0,N_max))

    # create time segments
    delta_t = (E_final -E_0)/(u*N_max)
    t = N*delta_t

    # create time-dependent Potential
    E = E_0 + u*t

    # calculate Current (in nA)
    I = 1e9*(2*np.pi*F*D_R*D_O*c_bR*r_hemi)/(D_O+D_R*np.exp(-F*(E-E_oi)/(R*T)))
    
    return(t,I)