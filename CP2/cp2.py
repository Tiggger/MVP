import numpy as np
import matplotlib.pyplot as plt
import scipy 
from numba import njit
import argparse
import csv


plt.style.use('shendrukGroupStyle')
import shendrukGroupFormat as ed


#Args
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

parser.add_argument('-p', '--pAlive', type=float, default=None,
                    help='The probability that a cell is alive upon intialisation')

parser.add_argument('-g', '--graphics', type=int, default='1',
                    help='Graphics mode (visualisation), 0 for off, 1 for on')

args = parser.parse_args()

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

#Plotting Functions
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

# Error Function - Bootstrapping
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

# Simulation Parameters

#put in variables like this as numba doesn't like when you pass args.X in
L = args.n
sweep = L * L
pAlive=args.pAlive

#carried over from checkpoint 1
n_warm = 25000
n_meas = 15000

#lists to store data

#some arbitrary number of steps that is large, such that when visualising, it plays for a long time
visualiserSteps = 10_000_000_000_000
sweepCounter=0

# ------------------------------
# Main loop
# ------------------------------

#initialising grid
grid = np.random.choice(np.array([1, 0], dtype=np.int8), size=(L, L), p=[pAlive, 1-pAlive])

#graphics mode
if args.graphics==1:

    #initialise figure for plotting
    fig, ax, im = init_plot(grid, "Conway's Game of Life")

    for step in range(visualiserSteps):
        #do conway procedure to update grid
        grid = conwayProcedure(grid)

        #visually update the grid
        update_plot(im, fig, grid)

#non grpahics mode
elif args.graphics==0:

    for step in range(visualiserSteps):
        grid = conwayProcedure(grid)
        print(grid, 'grid')
