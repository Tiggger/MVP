import numpy as np
import matplotlib.pyplot as plt
import scipy 
from numba import njit
import argparse
import csv
from matplotlib.colors import ListedColormap, BoundaryNorm

plt.style.use('shendrukGroupStyle')
import shendrukGroupFormat as ed

# plt.rcParams['figure.dpi'] = 300
# plt.rcParams['savefig.dpi'] = 300

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

parser.add_argument('-psi', '--psi', type=float, default=0.5,
                    help='Probability of a susceptible being made into infected. Default is 0.5')

parser.add_argument('-pir', '--pir', type=float, default=0.5,
                    help='Probability of an infected getting recovered. Defauly is 0.5')

parser.add_argument('-prs', '--prs', type=float, default=0.5,
                    help='Probability of a recovered becoming susceptible again. Default is 0.5')


args = parser.parse_args()


#Plotting functions
def init_plot(grid, title='Ising Dynamics'):
    """Initialize interactive plot."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(7,6))
    cmap = ListedColormap(["#4daf4a", "#e41a1c", "#377eb8"])  # S, I, R
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    im = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='none')
    ax.set_title(title)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_autoscale_on(False)
    cbar = fig.colorbar(im, ax=ax, ticks=[-1, 0, 1], fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(["S", "I", "R"])
    cbar.set_label("State")
    fig.canvas.draw()
    return fig, ax, im

def update_plot(im, fig, grid):
    """Update the figure with the current grid."""
    im.set_data(grid)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)

#sirs update function
@njit
def doSirsUpdate(L, grid, psi, pir, prs):
    #pick a site
    x = np.random.randint(0, L)
    y = np.random.randint(0, L)

    #get value and list of neighbours
    candidate = grid[x, y]
    neighbourList = [grid[(x-1) % L, y], grid[(x+1) % L, y], grid[x, (y-1) % L], grid[x, (y+1) % L]]

    #if candidate if susceptible
    if candidate==-1:
        #if next to an infected
        if 0 in neighbourList:
            #probability change condition
            if np.random.rand()<psi:
                #make infected
                grid[x,y]=0

    #if candidate is infected
    elif candidate==0:
        if np.random.rand()<pir:
            grid[x,y]=1

    #if candidate is recovered
    elif candidate==1:
        if np.random.rand()<prs:
            grid[x,y]=-1

#Arguments
L=args.n
sweep = L * L
visualiserSteps=1_000_000_000

psi=args.psi
pir=args.pir
prs=args.prs







#graphics mode
if args.graphics==1:

    #initialise grid
    if args.random==1:
        #initialising grid - conventional choice that S=-1, I=0, R=1
        grid = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(L, L))

        #initialise figure for plotting
        fig, ax, im = init_plot(grid, "SIRS")

        for step in range(visualiserSteps):

            #do sirs update 
            doSirsUpdate(L, grid, psi, pir, prs)

            #only update grid after a sweep
            if step % sweep == 0 or step==visualiserSteps-1:

                #visually update the grid
                update_plot(im, fig, grid)

elif args.graphics==2:
    #initialise grid
    if args.random==1:
        #initialising grid - conventional choice that S=-1, I=0, R=1
        grid = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(L, L))

        #initialise figure for plotting
        fig, ax, im = init_plot(grid, "SIRS")

        for step in range(visualiserSteps):

            #do sirs update 
            doSirsUpdate(L, grid, psi, pir, prs)

            #only update grid after a sweep
            if step % sweep == 0 or step==visualiserSteps-1:

                #visually update the grid
                update_plot(im, fig, grid)
