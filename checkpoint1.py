#import essentials
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use('TkAgg')
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
def energy(grid, coord, J=1):
    """
    Function which calculates energy change in flipping a spin at a given coordinate in Ising model 
    
    param grid: The current state of the Ising grid
    param coord: The coordinate of the spin we are going to flip
    param J: Coupling constant 
    """

    #get grid dimensions
    Lx=len(grid)
    Ly=len(grid[0])

    #extract coordinate
    xcoord=int(coord[0])
    ycoord=int(coord[1])

    #get spin at site
    spin=grid[xcoord][ycoord]

    #Get neighboring spins with periodic BC
    #Wrap indices using modulo operator
    n1 = grid[(xcoord - 1) % Lx][ycoord]
    n2 = grid[(xcoord + 1) % Lx][ycoord]
    n3 = grid[xcoord][(ycoord + 1) % Ly]
    n4 = grid[xcoord][(ycoord - 1) % Ly]
    
    #calculate current energy
    eChange = 2 * J * spin * (n1 + n2 + n3 + n4)
    
    return eChange
    


#initialise grid, number of timesteps to simulate for and how often to update plot
grid=np.random.choice([-1, 1], size=(args.n, args.n))
updateInterval = 200  # Update plot every 20 Monte Carlo step
totalSteps = 100000000
    

#choosing dynamics 
if args.dynamics.lower()=='kawasaki':
    #do kawasaki dynamics
    print('kawasaki')


elif args.dynamics.lower() == 'glauber':
    print('Using Glauber Dynamics')
    
    plt.ion()
    
    # Create initial figure with performance optimizations
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Use faster rendering options
    im = ax.imshow(grid, vmin=-1, vmax=1, interpolation='none')  # 'none' is fastest for pixel data
    plt.colorbar(im, ax=ax)
    ax.set_title('Glauber Dynamics')
    plt.grid(False)
    #turn off ticks
    plt.xticks([])
    plt.yticks([])

    
    # Disable autoscaling for better performance
    ax.set_autoscale_on(False)
    
    # Initial draw
    fig.canvas.draw()
   
    for i in range(totalSteps):
        #Pick random site
        x, y = random.randint(0, 49), random.randint(0, 49)
        
        #Calculate energy change
        eChange = energy(grid, [x, y])
        
        #Glauber dynamics
        if eChange < 0:
            grid[x][y] *= -1
        elif random.random() < np.exp(-eChange / args.thermalEnergy):
            grid[x][y] *= -1
        
        #Update plot periodically (not every step)
        if i % updateInterval == 0 or i == totalSteps - 1:
            #Update the data
            im.set_data(grid)  #Use set_data instead of set_array (slightly faster)
            
            #Use more efficient drawing methods
            fig.canvas.draw_idle()  #More efficient than plt.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)  
    
    #Final update
    fig.canvas.draw()
    
    plt.ioff()
    plt.show()
        
       

    
    
    #grid[int(randCoord[0])][int(randCoord[1])]=10 #for viewing and debugging

#show grid
# plt.imshow(grid)
# plt.grid(False)
# plt.colorbar()
# plt.show()