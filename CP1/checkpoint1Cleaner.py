import numpy as np
import random
import matplotlib.pyplot as plt
from numba import njit

# ------------------------------
# Helper functions
# ------------------------------

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
    epilog='Written by John Whitfield'
)

#passing arguments
# parser.add_argument('bounds', nargs='+', type=float,
#                    help='Integration bounds as: x1 x2 y1 y2 [z1 z2 ...]')
parser.add_argument('-v', '--verbose', action='store_true',
                   help='Enable verbose output')
parser.add_argument('-n', '--n', type=int, default=50,
                   help='Number of Monte Carlo points (default: 1000)')
parser.add_argument('-t', '--thermalEnergy', type=float, default=1, 
                    help='Thermal energy of the system')
parser.add_argument('-d', '--dynamics', type=str, default='Glauber',
                    help='The type of dynamics you want to run the Ising simulation with')
parser.add_argument('-c', '--couplingConstant', type=float, default='1',
                    help='The value of the coupling constant')
parser.add_argument('-g', '--graphics', type=float, default='0',
                    help='Graphics modee, 0 for off, 1 for on')

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

def nnSum(x, y, grid):
    """Sum of nearest neighbours with periodic boundaries."""
    Lx, Ly = grid.shape
    return (grid[(x - 1) % Lx, y] +
            grid[(x + 1) % Lx, y] +
            grid[x, (y + 1) % Ly] +
            grid[x, (y - 1) % Ly])

def nnSum_corrected(x1, y1, x2=None, y2=None, grid=None):
    """Nearest-neighbour sum, corrected for mutual bond if sites are neighbours."""
    S = nnSum(x1, y1, grid)
    if x2 is not None and y2 is not None:
        Lx, Ly = grid.shape
        are_neighbours = (
            (x1 == x2 and (abs(y1 - y2) == 1 or abs(y1 - y2) == Ly-1)) or
            (y1 == y2 and (abs(x1 - x2) == 1 or abs(x1 - x2) == Lx-1))
        )
        if are_neighbours:
            S -= grid[x2, y2]  # remove mutual bond
    return S

def metropolis_accept(deltaE, T):
    """Metropolis criterion: accept move if ΔE ≤ 0 or with probability e^(-ΔE/T)."""
    return deltaE <= 0 or random.random() < np.exp(-deltaE / T)

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

def update_plot(im, fig, grid):
    """Update the figure with the current grid."""
    im.set_data(grid)
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)

# ------------------------------
# Simulation parameters
# ------------------------------

grid = np.random.choice([-1, 1], size=(args.n, args.n))

sweep = args.n**2 #1 sweep is N (no. spins on the grid)
sweepCounter=0
warmUpSweeps=100
warmUpComplete=False

measurementCounter=0
numMeasurements=1000 #1000

totalSteps = 10_000_000_000
J = args.couplingConstant
T = args.thermalEnergy

susceptabilities=[]

# ------------------------------
# Kawasaki Dynamics
# ------------------------------

if args.dynamics.lower() == 'kawasaki':
    print("Using Kawasaki Dynamics")
    fig, ax, im = init_plot(grid, "Kawasaki Dynamics")
    
    for step in range(totalSteps):
        # Pick two random distinct sites
        x1, y1 = random.randint(0, args.n-1), random.randint(0, args.n-1)
        x2, y2 = random.randint(0, args.n-1), random.randint(0, args.n-1)
        
        si, sj = grid[x1, y1], grid[x2, y2]
        
        if si != sj:
            Si = nnSum_corrected(x1, y1, x2, y2, grid)
            Sj = nnSum_corrected(x2, y2, x1, y1, grid)
            
            # Compute ΔE for exchange
            E_before = -J * (si * Si + sj * Sj)
            E_after  = -J * (sj * Si + si * Sj)
            deltaE = E_after - E_before
            
            if metropolis_accept(deltaE, T):
                grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]
        
        #if we have swept once, we can show this on the screen
        if step % sweep == 0 or step == totalSteps - 1:
            sweepCounter+=1

            if sweepCounter==warmUpSweeps:
                print('Warm up complete')

            update_plot(im, fig, grid)
            print(f"sweepCounter: {sweepCounter}")
        
        

# ------------------------------
# Glauber Dynamics
# ------------------------------

elif args.dynamics.lower() == 'glauber':
    print("Using Glauber Dynamics")

    if args.graphics == 1:
        fig, ax, im = init_plot(grid, "Glauber Dynamics")

    temps=np.linspace(1, 3, 20) #need to change this to 20
    chis=[]

    for t in temps:
        print(f'Simulating T={t}')
        grid = np.random.choice([-1, 1], size=(args.n, args.n))

        tempM=[]
        warmUpComplete=False
        sweepCounter=0
                
        for step in range(totalSteps):
            # Pick a random site
            x, y = random.randint(0, args.n-1), random.randint(0, args.n-1)
            s = grid[x, y]
            S = nnSum(x, y, grid)
            
            # ΔE if spin flips
            deltaE = 2 * J * s * S
            
            if metropolis_accept(deltaE, T):
                grid[x, y] *= -1
            
            #if we have swept then
            if step % sweep == 0 or step == totalSteps - 1:
                #update visuals
                if args.graphics==1:
                    update_plot(im, fig, grid)

                #keep track of the sweeps
                sweepCounter+=1

                #if warm up complete
                if sweepCounter == warmUpSweeps:
                    print('Warm up complete')
                    warmUpComplete=True

                #if warm up complete and sweep is divisible by 10 and we are in the measurement game mode
                if warmUpComplete and sweepCounter % 10 == 0:
                    print('Taking measurement')
                    #measure M
                    M=np.sum(grid)
                    tempM.append(M)

                    if len(tempM) >= numMeasurements:
                        break
                
                #for testing
                print(f"Sweep: {sweepCounter}")

        M_arr = np.array(tempM)

        chi = (1 / (args.n**2 * T)) * (np.mean(M_arr**2) - np.mean(M_arr)**2)
        chis.append(chi)
            
        
        
print(temps, chis)


            
        

# ------------------------------
# Finalise
# ------------------------------

if args.graphics==1:
    plt.ioff()
    plt.close('all')   
plt.figure()
plt.plot(temps, chis, 'o-')
plt.xlabel(r"$k_B T$")
plt.ylabel(r"$\chi$")
plt.show()
