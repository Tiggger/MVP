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

parser.add_argument('-g', '--graphics', type=int, default='1',
                    help='Graphics mode (visualisation), 0 for off, 1 for on')

args = parser.parse_args()


@njit
def calculateEnergy(grid, J):
    """Count each bond only once."""
    L = grid.shape[0]
    energy = 0.0

    for i in range(L):
        for j in range(L):
            # Only count right and down bonds to avoid double counting
            right_j = (j + 1) % L
            down_i = (i + 1) % L
            energy += grid[i, j] * (grid[i, right_j] + grid[down_i, j])
    
    return -J * energy  



# Numba Monte Carlo Kernel
#function to execute a singular glauber procedure
@njit
def conwayProcedure(grid):
    #need to implement conway procedure using np.roll
    pass

#function to execute the glauber procedure, including some warm up
@njit
def runConway(grid, n_warm, n_meas, sweep):
    """
    Run game of life
    Doesn't return anything yet, but when clear will be implemented
    """

    L = grid.shape[0]
    #could create zero lists to store if needed
    # M_vals = np.zeros(n_meas)
    # E_vals = np.zeros(n_meas)

    # -------- warm-up --------
    for step in range(n_warm * sweep):
        conwayProcedure(grid, L)

    # -------- measurements --------
    for m in range(n_meas):
        for step in range(sweep):
            conwayProcedure(grid, L)

        #calculate Ms and Es
        # M_vals[m] = np.sum(grid)
        # E_vals[m] = calculateEnergy(grid, J)

    # return M_vals, E_vals
    return True

# Plotting Functions
#Initialse video plotter
def init_plot(grid, title='Ising Dynamics'):
    """Initialize interactive plot."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(7,6))
    im = ax.imshow(grid, vmin=-1, vmax=1, interpolation='none')
    plt.colorbar(im, ax=ax)
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



# Error Functions
#shift down 
#np.roll(list, 1, axis=0)
#shift left
#np.roll(list, -1, axis=1)

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

if args.graphics==1:

    grid = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    print(grid, 'grid')

    fig, ax, im = init_plot(grid, "Conway's Game of Life")

    for step in range(visualiserSteps):
        runConway(grid, n_meas, n_warm, sweep)

    update_plot(im, fig, grid)
