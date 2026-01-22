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
parser.add_argument('-n', '--n', type=int, default=50,
                   help='Number of Monte Carlo points (default: 1000)')
parser.add_argument('-e', '--thermalEnergy', type=float, default=1, 
                    help='Thermal energy of the system')
parser.add_argument('-d', '--dynamics', type=str, default='Glauber',
                    help='The type of dynamics you want to run the Ising simulation with')
parser.add_argument('-c', '--couplingConstant', type=float, default='1',
                    help='The value of the coupling constant')

args = parser.parse_args()


#function to calculate energy from nearest neighbours
#this function should handle the periodic boundary conditions
def energyChange(grid, coord, J=1):
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

#function to get the sum of the nearest neighbour spins
def nnSum(x, y):
    #get grid coordinates
    Lx=len(grid)
    Ly=len(grid[0])

    return grid[(x - 1) % Lx][y] + grid[(x + 1) % Lx][y] + grid[x][(y + 1) % Ly] + grid[x][(y - 1) % Ly]
    


#initialise grid, number of timesteps to simulate for and how often to update plot
grid=np.random.choice([-1, 1], size=(args.n, args.n))
updateInterval = 1000  # Update plot every 20 Monte Carlo step
totalSteps = 10000000000
    

#choosing dynamics 
#this is not correct 
if args.dynamics.lower() == 'kawasaki':
    print('Using Kawasaki Dynamics')

    plt.ion()
    
    # Create initial figure with performance optimizations
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Use faster rendering options
    im = ax.imshow(grid, vmin=-1, vmax=1, interpolation='none')  # 'none' is fastest for pixel data
    plt.colorbar(im, ax=ax)
    ax.set_title('Kawasaki Dynamics')
    plt.grid(False)
    #turn off ticks
    plt.xticks([])
    plt.yticks([])

    # Disable autoscaling for better performance
    ax.set_autoscale_on(False)
    
    # Initial draw
    fig.canvas.draw()
    
    # Simple implementation (may have small error for neighboring spins)
    for i in range(totalSteps):
        # Pick two different random sites
        x1, y1 = random.randint(0, args.n-1), random.randint(0, args.n-1)
        x2, y2 = random.randint(0, args.n-1), random.randint(0, args.n-1)

        si=grid[x1, y1]
        sj=grid[x2, y2]

        #if the spins are the same then we can pass
        if si==sj:
            deltaE=0

        else: #if spins are different

            #get sum of nearest neighbours for energy calcualations
            Si = nnSum(x1, y1)
            Sj = nnSum(x2, y2)

            #Check for nearest neighbours with periodic boundaries
            are_neighbours = (
                (x1 == x2 and (abs(y1 - y2) == 1 or abs(y1 - y2) == args.n-1)) or
                (y1 == y2 and (abs(x1 - x2) == 1 or abs(x1 - x2) == args.n-1))
            )

            #if nearest neighbours, need to correct spin sum 
            if are_neighbours:
                Si -= sj
                Sj -= si

            #calculate energy before and after swapping the spins, like -J*( (spin at i*Sum of spins at i) + (spin at j*Sum of spins at j)) and then swapping the spins
            E_before = -args.couplingConstant * si * Si - sj * Sj
            E_after  = -args.couplingConstant * sj * Si - si * Sj

            #calculate energy change 
            deltaE = E_after - E_before

            #if negative or within boltzmann probability then swap spins
            if deltaE <= 0 or random.random() < np.exp(-deltaE / args.thermalEnergy):
                grid[x1, y1] = sj
                grid[x2, y2] = si
        
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
        x, y = random.randint(0, args.n-1), random.randint(0, args.n-1)
        
        #Calculate energy change
        eChange = energyChange(grid, [x, y])
        
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