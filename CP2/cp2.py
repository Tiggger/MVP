import numpy as np
import matplotlib.pyplot as plt
import scipy 
from numba import njit
import argparse
import csv


plt.style.use('shendrukGroupStyle')
import shendrukGroupFormat as ed

#################
#Args
#################

parser = argparse.ArgumentParser(
    prog='ProgramName',
    description='Code for Checkpoint 1 of MVP',
    epilog='Written by John Whitfield'
)

parser.add_argument('-n', '--n', type=int, default=50,
                   help='Length of simulation box in number of sites, default is 50')

parser.add_argument('-r', '--random', type=int, default='1',
                    help='Whether you want a random initial condition or not, by default it is turned on')

parser.add_argument('-s', '--startingCondition', type=str, default=None,
                    help='The starting condition that you want to try options include: {Need to add options once implemented}')

parser.add_argument('-p', '--pAlive', type=float, default=0.5,
                    help='The probability that a cell is alive upon intialisation')

parser.add_argument('-g', '--graphics', type=int, default='1',
                    help='Graphics mode (visualisation), 0 for off, 1 for on')

parser.add_argument('-d', '--debug', type=int, default='0',
                    help='Debug mode (prints a lot of things to help), 0 for off, 1 for on. Default is off')

args = parser.parse_args()


#################
# Functions
#################

#Conway procedure to update grid
def conwayProcedure(grid):

    #ways to roll
    rolls = [
        (1, 0),   #down
        (-1, 0),  #up
        (0, 1),   #right
        (0, -1),  #left
        (1, 1),   #down-right
        (1, -1),  #down-left
        (-1, 1),  #up-right
        (-1, -1)  #up-left
    ]

    #need to implement conway procedure using np.roll
    tempGrid=np.zeros_like(grid, dtype=int)
    newGrid=np.zeros_like(grid, dtype=int)

    #go through rolling conditions to sum neighbours
    for dx, dy in rolls:
        tempGrid+=np.roll(grid, shift=(dx, dy), axis=(0,1))
    
    #update
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            cellState=grid[i,j]
            sumNeighbours=tempGrid[i,j]

            #conways rules
            if cellState==1 and sumNeighbours < 2:
                newGrid[i,j]=0
            elif cellState==1 and (sumNeighbours == 3 or sumNeighbours ==2):
                newGrid[i,j]=1
            elif cellState==1 and sumNeighbours > 3:
                newGrid[i,j]=0
            elif cellState==0 and sumNeighbours==3:
                newGrid[i,j]=1
    
    #print(newGrid, 'newGrid')
    return newGrid

#################
#Plotting Functions
#################

#Initialse video plotter
def init_plot(grid, title='Ising Dynamics'):
    """Initialize interactive plot."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(grid, vmin=-1, vmax=1, interpolation='none')
    ax.set_title(title)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_autoscale_on(False)
    fig.canvas.draw()
    return fig, ax, im

#function to update the plot with current state
def update_plot(im, fig, grid):
    """Update the figure with the current grid."""
    im.set_data(grid)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


#################
#Measurement functions
#################


#function to measure number of active sites
def measureActiveSites(grid):
    unique, counts = np.unique(grid, return_counts=True)

    return counts[1]

#################
# Error Function - Bootstrapping
#################

def bootstrap(data, function, *args, n_bootstrap=1000):
    """Bootstrap error estimation"""
    bootstrap_values = []
    
    for i in range(n_bootstrap):
        # Resample with replacement
        samples = np.random.choice(data, size=len(data), replace=True)
        # Calculate statistic on resampled data
        value = function(samples, *args)
        bootstrap_values.append(value)
    
    # The standard deviation of bootstrap distribution is the error estimate
    return np.std(bootstrap_values)


#################
# Simulation Parameters
#################

#put in variables like this as numba doesn't like when you pass args.X in
L = args.n
sweep = L * L
pAlive=args.pAlive

#for measuring absorption times
nMeas = 1000
absorptionTimes=[]
simulationAbsorped=False

#lists to store data

#some arbitrary number of steps that is large, such that when visualising, it plays for a long time
visualiserSteps = 10_000_000_000_000
updateCounter=0

# ------------------------------
# Main loop
# ------------------------------



#graphics mode
if args.graphics==1:

    #initialise grid
    if args.random==1:
        #initialising grid
        grid = np.random.choice(np.array([1, 0], dtype=np.int8), size=(L, L), p=[pAlive, 1-pAlive])
    #glider
    elif args.random==0 and args.startingCondition.lower()=='glider':
        grid = np.zeros((L, L), dtype=np.int8)
        
        glider = np.array([[0, 1, 0],
                       [0, 0, 1],
                       [1, 1, 1]], dtype=np.int8)
        
        start_y = L // 2 - 1
        start_x = L // 2 - 1
        grid[start_y:start_y+3, start_x:start_x+3] = glider

    #initialise figure for plotting
    fig, ax, im = init_plot(grid, "Conway's Game of Life")

    for step in range(visualiserSteps):
        #do conway procedure to update grid
        grid = conwayProcedure(grid)

        #visually update the grid
        update_plot(im, fig, grid)

#non grpahics mode
elif args.graphics==0:

    #repeat simulation for the number of measurements that we want
    for sim in range(nMeas):
        print(sim)

        #initialise random grid
        grid = np.random.choice(np.array([1, 0], dtype=np.int8), size=(L, L), p=[pAlive, 1-pAlive])
        dAcounter=0

        #simulate for a long amount of time
        for step in range(visualiserSteps):

            #get number of active sites before we update
            activeSitesBefore=measureActiveSites(grid)

            #do conway for the grid
            grid = conwayProcedure(grid)

            #get the number of active sites again after we update
            activeSitesAfter=measureActiveSites(grid)

            dA=activeSitesAfter-activeSitesBefore

            if dA==0:
                dAcounter+=1

            #debug condition
            if args.debug:
                #print(grid, 'grid')
                print('step: ', step, 'change in active sites', dA)

            #if we have detected that the simulation has been absorped, then reset the grid
            if dAcounter>=5:
                absorptionTimes.append(step)
                break

    with open('absorptionTimes.csv', 'w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['Data in rows as: absorptionTimes'])
        writer.writerow(absorptionTimes)
    
    #linear scale
    plt.xlabel(r'$\tau_\mathrm{Absorption}$')
    plt.ylabel(r'$p(\tau_\mathrm{Absorption})$')
    plt.hist(absorptionTimes, density=True, bins=20, log=False)
    plt.show()

    #log scale
    plt.xlabel(r'$\tau_\mathrm{Absorption}$')
    plt.ylabel(r'$p(\tau_\mathrm{Absorption})$')
    plt.hist(absorptionTimes, density=True, bins=20, log=True)
    plt.show()