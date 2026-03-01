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

parser.add_argument('-m', '--measurementMode', type=int, default=0,
                    help='Defines what type of measurement you want to make. 0 - heatmap for varying psi and prs, pir=0.5. 1 - contour plot of variance in infected for prs=0.5, psi=0.2-0.5. Default is 0.')

parser.add_argument('-if', '--immunityFraction', type=float, default=0,
                    help='Defines the fraction of agents that are immune. Please note that immunity is permanent. Default is 0')



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
def doSirsUpdate(L, grid, psi, pir, prs, immunityMatrix=None):
    #pick a site
    x = np.random.randint(0, L)
    y = np.random.randint(0, L)

    if immunityMatrix is not None:
        immunityStatus=immunityMatrix[x, y]

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
        if immunityMatrix is not None:
            if immunityStatus==0:
                if np.random.rand()<prs:
                    grid[x,y]=-1
            # else:
            #     print('Theyre immune!')
        else:
            if np.random.rand()<prs:
                grid[x,y]=-1


#bootstrap error function
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


#function to calculate variance
def calcVar(data, N):
    return (np.average(data**2)-np.average(data)**2)/N


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
numMeas=1000
avgInfectedResults=np.zeros([res, res])

infectedVar=np.zeros(res)
infectedVarError=np.zeros(res)

avgInfectedFracImmuneResults=np.zeros(res)



#graphics mode
if args.graphics==1:

    #initialise grid
    if args.random==1:

        if args.immunityFraction == 0:
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
        
        elif args.immunityFraction != 0:
            print('Immunity mode running')
            grid = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(L, L))

            immunityMatrix = np.random.choice(np.array([True, False], dtype=np.int8), size=(L, L), p=[args.immunityFraction, 1-args.immunityFraction])

            #initialise figure for plotting
            fig, ax, im, _, _, _ = init_plot(grid, "SIRS", show_populations=False)

            for step in range(visualiserSteps):

                #do sirs update 
                doSirsUpdate(L, grid, psi, pir, prs, immunityMatrix=immunityMatrix)

                #only update grid after a sweep
                if (step + 1) % sweep == 0 or step==visualiserSteps-1:

                    #visually update the grid
                    update_plot(im, fig, grid)

#graphics mode to plot 
elif args.graphics==2:
    #initialise grid
    if args.random==1:

        if args.immunityFraction == 0:
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

        if args.immunityFraction != 0:
            #initialising grid - conventional choice that S=-1, I=0, R=1
            grid = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(L, L))

            immunityMatrix = np.random.choice(np.array([True, False], dtype=np.int8), size=(L, L), p=[args.immunityFraction, 1-args.immunityFraction])

            #initialise figure for plotting
            fig, ax, im, ax_pop, lines, histories = init_plot(grid, "SIRS", show_populations=True)

            for step in range(visualiserSteps):

                #do sirs update 
                doSirsUpdate(L, grid, psi, pir, prs, immunityMatrix=immunityMatrix)

                #only update grid after a sweep
                if (step + 1) % sweep == 0 or step==visualiserSteps-1:
                    sweep_count = (step + 1) // sweep

                    #visually update the grid
                    update_plot(im, fig, grid, sweep_count=sweep_count, ax_pop=ax_pop, lines=lines, histories=histories)



#measurement mode
elif args.graphics==0:
    if args.random==1:

        if args.measurementMode==0:

            print('Measuring phase space')

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

            #save results
            with open('phaseSpace.csv', 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['psi_index', 'prs_index', 'avgInfected'])  # header
                for psi in range(res):
                    for prs in range(res):
                        writer.writerow([psi, prs, avgInfectedResults[psi, prs]])

            #plot results
            plt.imshow(avgInfectedResults)
            plt.grid(False)
            plt.colorbar()
            plt.xlabel(r'$P_{S \rightarrow I}$')
            plt.ylabel(r"$P_{R \rightarrow S}$")
            plt.title(r"$\langle I \rangle / N$")
            plt.show()
    
        elif args.measurementMode==1:
            print('Measuring variance')

            #set linspaces for psi and prs
            psiVals=np.linspace(0.2, 0.5, res)
            prsVals=0.5

            #looping through parameter set
            for psi in range(res):
                
                if debug:
                    print(f'Doing sim with psi={psiVals[psi]}')
                
                #initialise random grid
                grid = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(L, L))

                #initialise list to save results for this parameter set
                infectedResults=[]

                #warm up for 100 sweeps
                for _ in range(100*sweep):
                    doSirsUpdate(L, grid, psiVals[psi], 0.5, 0.5) #hard code pir as 0.5
                
                #measurements for 10000 sweeps
                for _ in range(100*sweep):
                    #do update
                    doSirsUpdate(L, grid, psiVals[psi], 0.5, 0.5)

                    #measure infected
                    infectedResults.append(np.count_nonzero(grid==0))
                    
                var=calcVar(np.array(infectedResults), L*L)

                error=bootstrap(infectedResults, calcVar, L*L, n_bootstrap=100)

                # print([psi, prs])
                infectedVar[psi]=var
                infectedVarError[psi]=error
            
            #need to handle error

            with open('varResults.csv', 'w', newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['Data in rows as: Variance, error on variance'])
                writer.writerow(infectedVar)
                writer.writerow(infectedVarError)
            
            plt.plot(psiVals, infectedVar)
            plt.fill_between(psiVals, infectedVar-infectedVarError, infectedVar+infectedVarError, label='Bootstrap')
            plt.xlabel(r'$P_{S\rightarrow I}$')
            plt.ylabel(r'$\frac{\langle I^2\rangle-\langle I\rangle^2}{N}$')
            plt.show()

        #looking at the fraction of immune
        elif args.measurementMode == 2:
            print('Measuring Infections as function of fraction of immune')

            fracsImmune=np.linspace(0, 1, res)

            for frac in range(res):

                if debug:
                    print(f'Simulating with immunity fraction {fracsImmune[frac]}')

                grid = np.random.choice(np.array([-1, 0, 1], dtype=np.int8), size=(L, L))

                immunityMatrix = np.random.choice(np.array([True, False], dtype=np.int8), size=(L, L), p=[fracsImmune[frac], 1-fracsImmune[frac]])

                infectedResults=[]


                #warm up for 100 sweeps - not sure if we want to do warm up with immunity or not
                for _ in range(100*sweep):
                    doSirsUpdate(L, grid, 0.5, 0.5, 0.5)

                for _ in range(1000*sweep):
                    doSirsUpdate(L, grid, 0.5, 0.5, 0.5, immunityMatrix=immunityMatrix)

                    infectedResults.append(np.count_nonzero(grid==0))
                
                #take average of infected results
                avgInfected=np.average(infectedResults)

                #normalise by grid size
                avgInfected/=(L*L)

                # print([psi, prs])
                avgInfectedFracImmuneResults[frac]=avgInfected
            
            with open('fracImmuneResults.csv', 'w', newline='') as f:
                writer=csv.writer(f)
                writer.writerow(['Data in rows as: frac immune, average infected'])
                writer.writerow(fracsImmune)
                writer.writerow(avgInfectedFracImmuneResults)
            
            plt.plot(fracsImmune, avgInfectedFracImmuneResults)
            plt.xlabel(r'$f_\mathrm{Im}$')
            plt.ylabel(r'$\langle I\rangle/N$')
            plt.show()


