#--------------------------------------
# Benchmark with Python code & Numba jit
#--------------------------------------

import os
import sys
import time
import csv
import numpy as np
from numba import jit
from scipy.integrate import cumulative_trapezoid

# get location of the relevant functions (in parent path)
# and import them
script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..' )
sys.path.append( mymodule_dir )
import hemispherical_electrode as he
import Semi_Operators_numba as SO

# Define list of Element sizes
#N_list = [1000,2500,5000,7500,10000,12500,15000,20000]
N_list = [1000,5000,10000,15000,20000,25000,30000]
# Define file name of export csv
FILENAME = "results/numba_Benchmark_Results" 

# Define if csv export is desired
csv_export = True

#--------------------------------------
# Fast Riemann-Liouville transformation
#--------------------------------------

# Initialize
t_FRLT = np.zeros(np.size(N_list))
e_FRLT_max = np.zeros(np.size(N_list))
e_rel_FRLT_max = np.zeros(np.size(N_list))

# Print header
print('\n FRLT Algorithm\n')
print(' N      |  t (s) | max abs err | max rel err')

for i in range(0, np.size(N_list)):
    
    # No. of Elems
    N = N_list[i]

    # generate test potential
    [t,I] = he.current_hemispherical1(N_max =N); # with default values

    delta_x = t[1] - t[0]
    
    # calculation of alg
    t_i = time.time()
    d_FRLT = SO.semi_integration(SO.semi_integration(I,  delta_x=delta_x), delta_x=delta_x)
    t_FRLT[i] = time.time() - t_i
    
    d_FRLT = d_FRLT[1:] # first value is zero

    # Reference values
    d_ref = cumulative_trapezoid(I,t)

    # absolute error
    e_FRLT = np.abs(d_FRLT-d_ref)
    e_FRLT_max[i] = np.max(e_FRLT)

    # relative error
    e_rel_FRLT = e_FRLT/(np.abs(d_ref))
    e_rel_FRLT_max[i] = np.max(e_rel_FRLT)
    # Print results
    print( '{:.1e}'.format(N),'| {:6.3}'.format(t_FRLT[i]), '|    {:.2e}'.format(e_FRLT_max[i]),'| {:.2e}'.format(e_rel_FRLT_max[i]))

# export current state


#df_res.to_csv(''.join([FILENAME,'.csv']), index=False)

#--------------------------------------
# Gruenwald
#--------------------------------------

# Initialize
t_G1 = np.zeros(np.size(N_list))
e_G1_max = np.zeros(np.size(N_list))
e_rel_G1_max = np.zeros(np.size(N_list))

# Print header
print('\n Gruenwald Algorithm\n')
print(' N      |  t (s) | max abs err | max rel err')

for i in range(0, np.size(N_list)):

    # No. of Elems
    N = N_list[i]

    # generate test potential
    [t,I] = he.current_hemispherical1(N_max =N); # with default values
    
    # calculation of alg
    t_i = time.time()
    d_G1 = SO.G1_Semiint(SO.G1_Semiint(I,t), t)
    t_G1[i] = time.time() - t_i
    
    d_G1 = d_G1[:-1]

    # Reference values
    d_ref = cumulative_trapezoid(I,t)

    # absolute error
    e_G1 = np.abs(d_G1-d_ref)
    e_G1_max[i] = np.max(e_G1)
    
    # relative error
    e_rel_G1 = e_G1/(np.abs(d_ref))
    e_rel_G1_max[i] = np.max(e_rel_G1)

    # Results
    print( '{:.1e}'.format(N),'| {:6.3}'.format(t_G1[i]), '|    {:.2e}'.format(e_G1_max[i]),'| {:.2e}'.format(e_rel_G1_max[i]))

# export current state
#df_res.to_csv(''.join([FILENAME,'.csv']), index=False)

#--------------------------------------
# Riemann-Liouville
#--------------------------------------

# Initialize
t_R1 = np.zeros(np.size(N_list))
e_R1_max = np.zeros(np.size(N_list))
e_rel_R1_max = np.zeros(np.size(N_list))

# Print header
print('\nRiemann-Liouville Algorithm\n')
print(' N      |  t (s) | max abs err | max rel err')

for i in range(0, np.size(N_list)):

    # No. of Elems
    N = N_list[i]

    # generate test potential
    [t,I] = he.current_hemispherical1(N_max =N); # with default values

    # calculation of alg
    t_i = time.time()
    d_R1 = SO.R1_Semiint(SO.R1_Semiint(I,t), t)
    t_R1[i] = time.time() - t_i

    d_R1 = d_R1[:-1]

        # Reference values
    d_ref = cumulative_trapezoid(I,t)

    # absolute error
    e_R1 = np.abs(d_R1-d_ref)
    e_R1_max[i] = np.max(e_R1)

    # relative error
    e_rel_R1 = e_R1/(np.abs(d_ref))
    e_rel_R1_max[i] = np.max(e_rel_R1)

    # Results
    print( '{:.1e}'.format(N),'| {:6.3}'.format(t_R1[i]), '|    {:.2e}'.format(e_R1_max[i]),'| {:.2e}'.format(e_rel_R1_max[i]))


# export results (by pandas)
import pandas as pd

results = {'N':N_list, 't_FRLT':t_FRLT,'e_FRLT_max':e_FRLT_max, 'e_rel_FRLT_max':e_rel_FRLT_max,
                  't_G1':t_G1,'e_G1_max':e_G1_max, 'e_rel_G1_max':e_rel_G1_max,
                  't_R1':t_R1,'e_R1_max':e_R1_max, 'e_rel_R1_max':e_rel_R1_max}
df_res = pd.DataFrame(results)

df_res.to_csv(''.join([FILENAME,'.csv']), index=False)

