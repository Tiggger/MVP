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

parser.add_argument('-d', '--debug', type=bool, default=False,
                    help='Debug mode (prints a lot of things to help), 0 for off, 1 for on. Default is off')

parser.add_argument('-psi', '--psi', type=float, default=0.5,
                    help='Probability of a susceptible being made into infected. Default is 0.5')

parser.add_argument('-pir', '--pir', type=float, default=0.5,
                    help='Probability of an infected getting recovered. Defauly is 0.5')

parser.add_argument('-prs', '--prs', type=float, default=0.5,
                    help='Probability of a recovered becoming susceptible again. Default is 0.5')


args = parser.parse_args()


#Plotting functions
def init_plot(grid, title='Ising Dynamics', show_populations=False):
    """Initialize interactive plot."""
    plt.ion()
    if show_populations:
        fig, (ax, ax_pop) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, ax = plt.subplots(figsize=(7,6))
        ax_pop = None

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

    lines = None
    histories = None
    if ax_pop is not None:
        s_count = np.count_nonzero(grid == -1)
        i_count = np.count_nonzero(grid == 0)
        r_count = np.count_nonzero(grid == 1)
        histories = {
            "sweeps": [0],
            "S": [s_count],
            "I": [i_count],
            "R": [r_count],
        }
        line_s, = ax_pop.plot(histories["sweeps"], histories["S"], color="#4daf4a", lw=2, label="S")
        line_i, = ax_pop.plot(histories["sweeps"], histories["I"], color="#e41a1c", lw=2, label="I")
        line_r, = ax_pop.plot(histories["sweeps"], histories["R"], color="#377eb8", lw=2, label="R")
        lines = {"S": line_s, "I": line_i, "R": line_r}

        ax_pop.set_title("Population vs Sweep")
        ax_pop.set_xlabel("Sweep")
        ax_pop.set_ylabel("Count")
        ax_pop.set_xlim(0, 10)
        ax_pop.set_ylim(0, grid.size)
        ax_pop.legend(frameon=False)

    fig.canvas.draw()
    return fig, ax, im, ax_pop, lines, histories

def update_plot(im, fig, grid, sweep_count=None, ax_pop=None, lines=None, histories=None):
    """Update the figure with the current grid."""
    im.set_data(grid)

    if ax_pop is not None and lines is not None and histories is not None and sweep_count is not None:
        s_count = np.count_nonzero(grid == -1)
        i_count = np.count_nonzero(grid == 0)
        r_count = np.count_nonzero(grid == 1)

        histories["sweeps"].append(sweep_count)
        histories["S"].append(s_count)
        histories["I"].append(i_count)
        histories["R"].append(r_count)

        lines["S"].set_data(histories["sweeps"], histories["S"])
        lines["I"].set_data(histories["sweeps"], histories["I"])
        lines["R"].set_data(histories["sweeps"], histories["R"])

        right_edge = max(10, sweep_count)
        ax_pop.set_xlim(0, right_edge)

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

#average infected calculation function


#Arguments
L=args.n
sweep = L * L
visualiserSteps=1_000_000_000

psi=args.psi
pir=args.pir
prs=args.prs

debug=args.debug

#for phase space diagram maker
res=25
numMeas=1000 #low for testing
avgInfectedResults=np.zeros([res, res])



#graphics mode
if args.graphics==1:

    #initialise grid
    if args.random==1:
        #initialising grid - conventional choice that S=-1, I=0, R=1
        grid = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(L, L))

        #initialise figure for plotting
        fig, ax, im, _, _, _ = init_plot(grid, "SIRS", show_populations=False)

        for step in range(visualiserSteps):

            #do sirs update 
            doSirsUpdate(L, grid, psi, pir, prs)

            #only update grid after a sweep
            if (step + 1) % sweep == 0 or step==visualiserSteps-1:

                #visually update the grid
                update_plot(im, fig, grid)

#graphics mode to plot 
elif args.graphics==2:
    #initialise grid
    if args.random==1:
        #initialising grid - conventional choice that S=-1, I=0, R=1
        grid = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(L, L))

        #initialise figure for plotting
        fig, ax, im, ax_pop, lines, histories = init_plot(grid, "SIRS", show_populations=True)

        for step in range(visualiserSteps):

            #do sirs update 
            doSirsUpdate(L, grid, psi, pir, prs)

            #only update grid after a sweep
            if (step + 1) % sweep == 0 or step==visualiserSteps-1:
                sweep_count = (step + 1) // sweep

                #visually update the grid
                update_plot(im, fig, grid, sweep_count=sweep_count, ax_pop=ax_pop, lines=lines, histories=histories)


#measurement mode



elif args.graphics==0:
    if args.random==1:

        #set linspaces for psi and prs
        psiVals=np.linspace(0, 1, res)
        prsVals=np.linspace(0, 1, res)

        #looping through parameter set
        for psi in range(res):
            for prs in range(res):
                
                #do 1 simulation, but measure 1000 times

                if debug:
                    print(f'Doing sim with psi={psiVals[psi]} and prs={prsVals[prs]}')
                
                #initialise random grid
                grid = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(L, L))

                #initialise list to save results for this parameter set
                infectedResults=[]

                #warm up for 100 sweeps
                for _ in range(100*sweep):
                    doSirsUpdate(L, grid, psiVals[psi], 0.5, prsVals[prs]) #hard code pir as 0.5
                
                #measurements for 1000 sweeps
                for _ in range(1000*sweep):
                    #do update
                    doSirsUpdate(L, grid, psiVals[psi], 0.5, prsVals[prs])

                    #measure infected
                    infectedResults.append(np.count_nonzero(grid==0))
                
                #take average of infected results
                avgInfected=np.average(infectedResults)

                #normalise by grid size
                avgInfected/=(L*L)

                # print([psi, prs])
                avgInfectedResults[psi, prs]=avgInfected

    
    plt.imshow(avgInfectedResults)
    plt.grid(False)
    plt.colorbar()
    plt.show()
