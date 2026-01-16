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
# parser.add_argument('bounds', nargs='+', type=float,
#                    help='Integration bounds as: x1 x2 y1 y2 [z1 z2 ...]')
parser.add_argument('-v', '--verbose', action='store_true',
                   help='Enable verbose output')
parser.add_argument('-n', '--n', type=int, default=1000,
                   help='Number of Monte Carlo points (default: 1000)')
parser.add_argument('-e', '--thermalEnergy', type=float, default=1, 
                    help='Thermal energy of the system')
parser.add_argument('-d', '--dynamics', type=str, default='Glauber',
                    help='The type of dynamics you want to run the Ising simulation with')

args = parser.parse_args()


#function to calculate energy from nearest neighbours
#this function should handle the periodic boundary conditions
def energy(coord, J=1):
    pass


#initialise grid
grid=np.random.choice([-1, 1], size=(args.n, args.n))

#choosing dynamics 
if args.dynamics.lower()=='kawasaki':
    #do kawasaki dynamics
    print('kawasaki')
elif args.dynamics.lower()=='glauber':
    print('Using Glauber Dynamics')

    #pick a random site
    randCoord=np.random.choice(np.linspace(0, 49, 50), size=2)
    
    #access value on the grid
    print(grid[int(randCoord[0])][int(randCoord[1])], 'grid randcoord')

#show grid
# plt.imshow(grid)
# plt.grid(False)
# plt.show()