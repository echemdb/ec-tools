#--------------------------------------
# Visualize the performed benchmark
#--------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

myprint = True

# Define csv file name
FILENAME = "Py_Benchmark_Results_1.csv" 

# Define image name
FIGNAME = "Test"

# import result values
df = pd.read_csv(FILENAME)
print(df)


if myprint == True:
    # Print them
    plt.plot(df['N'],df['e_FRLT_max'])
    # Settings
    plt.xlabel('No of Elems')
    plt.ylabel('Time [s]')
    
    # save figure
    path = ['images/',FIGNAME]
    plt.savefig(''.join(path))