#import essentials
import numpy as np 
import matplotlib.pyplot as plt 
import scipy 
import scipy.stats as stats
import random 

#Use custom style and colours
plt.style.use('shendrukGroupStyle')
import shendrukGroupFormat as ed


#need to work through this properly 
import argparse
parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='Code for Checkpoint 1 of MVP',
    epilog='Not sure what to write in here yet.'
)

#passing arguments
parser.add_argument('bounds', nargs='+', type=float,
                   help='Integration bounds as: x1 x2 y1 y2 [z1 z2 ...]')
parser.add_argument('-v', '--verbose', action='store_true',
                   help='Enable verbose output')
parser.add_argument('-n', '--npoints', type=int, default=1000,
                   help='Number of Monte Carlo points (default: 1000)')

args = parser.parse_args()

# Correct way to access the bounds argument
print(f"Testing arg is {args.bounds}")


#need to initialise lattice
#need to generate random numbers 
#need to initialise energies
#functions for all of the above

#function for monte carlo integration 
def monteCarloIntegrator(function, bounds, nPoints=1000):
    """
    Monte Carlo Integration over specified bounds
    
    Parameters:
    -----
    function : callable 
        Function to integrate via monte carlo
    bounds : list of arrays which define the bounds, can be 2D or 3D
        [(x_min, x_max), (y_min, y_max), ...]
    n : integer
        Number of samples 
    """


    #initialise I=0
    I=0

    for i in range(nPoints):
        #need to generate a random point based on bounds passed in
        #spent a lot of time working out argsparse
        pass


#function
def function(x, y, z=0):
    return x**2+y**2



#------ MAIN ------

monteCarloIntegrator(function, args.bounds, nPoints=10)





