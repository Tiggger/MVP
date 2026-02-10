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


parser.add_argument('-d', '--dynamics', type=str, default='Glauber',
                    help='The type of dynamics you want to run the visual Ising simulation with, default is Glauber')

parser.add_argument('-c', '--couplingConstant', type=float, default='1',
                    help='The value of the coupling constant, default value is 1')

parser.add_argument('-g', '--graphics', type=float, default='1',
                    help='Graphics mode (visualisation), 0 for off, 1 for on')

args = parser.parse_args()

if args.thermalEnergy == 0:
    print('Thermal Energy is 0, resetting to 1')
    args.thermalEnergy=1
    
# Energy Calculator
# @njit
# def calculateEnergy(grid, J):
#     """Numba JIT version - manual loops."""
#     L = grid.shape[0]
#     energy = 0.0

#     #this replicates np.roll, which numba doesn't like
#     for i in range(L):
#         for j in range(L):
#             right_j = (j + 1) % L
#             up_i = (i - 1) % L
#             energy += grid[i, j] * (grid[i, right_j] + grid[up_i, j])
#     return (-J * energy)/2 #to avoid double counting

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
def glauberProcedure(L, J, grid, beta):
    #pick a site
    x = np.random.randint(0, L)
    y = np.random.randint(0, L)

    #get value and neighbours
    s = grid[x, y]
    S = (grid[(x-1) % L, y] + grid[(x+1) % L, y] + grid[x, (y-1) % L] + grid[x, (y+1) % L])

    #Energy change of flipping
    dE = 2.0 * J * s * S

    #Monte Carlo Integrator
    if dE <= 0.0 or np.random.rand() < np.exp(-beta * dE):
        grid[x, y] = -s

#function to execute the glauber procedure, including some warm up
@njit
def glauber_run(grid, beta, J, n_warm, n_meas, sweep):
    """
    Run Glauber dynamics at fixed temperature.
    Returns magnetisation after each measurement sweep.
    """

    L = grid.shape[0]
    M_vals = np.zeros(n_meas)
    E_vals = np.zeros(n_meas)

    # -------- warm-up --------
    for step in range(n_warm * sweep):
        glauberProcedure(L, J, grid, beta)

    # -------- measurements --------
    for m in range(n_meas):
        for step in range(sweep):
            glauberProcedure(L, J, grid, beta)

        #calculate Ms and Es
        M_vals[m] = np.sum(grid)
        E_vals[m] = calculateEnergy(grid, J)

    return M_vals, E_vals

#function for kawasaki procedure
@njit
def kawasakiProcedure(L, J, grid, beta):
    #Pick two random sites
    x1 = np.random.randint(0, L)
    y1 = np.random.randint(0, L)
    
    x2 = np.random.randint(0, L)
    y2 = np.random.randint(0, L)

    
    #get spins
    si, sj = grid[x1, y1], grid[x2, y2]
    
    #if not the same coord picked (if the same, nothing changes)
    if si != sj:
        #get sum of neighbours
        Si = (grid[(x1-1) % L, y1] + grid[(x1+1) % L, y1] + grid[x1, (y1-1) % L] + grid[x1, (y1+1) % L])
        Sj = (grid[(x2-1) % L, y2] + grid[(x2+1) % L, y2] + grid[x2, (y2-1) % L] + grid[x2, (y2+1) % L])

        #check if neighbours
        are_neighbours = ((x1 == x2 and (abs(y1 - y2) == 1 or abs(y1 - y2) == L-1)) or (y1 == y2 and (abs(x1 - x2) == 1 or abs(x1 - x2) == L-1)))

        #if they are neighbours, need to make a correction
        if are_neighbours:
            Si-=grid[x2, y2]
            Sj-=grid[x1, y1]
        
        #Compute ΔE for exchange
        EBefore = -J * (si * Si + sj * Sj)
        EAfter  = -J * (sj * Si + si * Sj)
        dE = EAfter - EBefore
        
        #check metropolis conditions and update
        if dE < 0 or np.random.rand() < np.exp(-beta * dE):
            grid[x1, y1], grid[x2, y2] = grid[x2, y2], grid[x1, y1]

#function to execute the kawasaki procedure
@njit
def kawasakiRun(grid, beta, J, n_warm, n_meas, sweep):

    L = grid.shape[0]
    M_vals = np.zeros(n_meas)
    E_vals = np.zeros(n_meas)

    #warm up
    for step in range(n_warm * sweep):
        kawasakiProcedure(L, J, grid, beta)
        

    for m in range(n_meas):
        for step in range(sweep):
            kawasakiProcedure(L, J, grid, beta)

        M_vals[m] = np.sum(grid)
        E_vals[m] = calculateEnergy(grid, J)
    
    return M_vals, E_vals

# Function to calculate specific heat capacity from energy equation

def calculateSpecificHeat(energies, T, N):
    return (np.mean(energies**2)-np.mean(energies)**2) / (T**2 * N)



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
J = args.couplingConstant

temps = np.linspace(1.0, 3.0, 20)

n_warm = 25000
n_meas = 15000

#storing data to lists for easy plotting and saving
chis_glaub = []

Ms_glaub = []
Ms_Errors_Low_glaub = []
Ms_Errors_High_glaub = []

Es_glaub = []

cs_glaub = []
csErrors_glaub=[]


chis_kawa = []

Ms_kawa = []
Es_kawa = []

cs_kawa = []
csErrors_kawa=[]

#some arbitrary number of steps that is large, such that when visualising, it plays for a long time
visualiserSteps = 10_000_000_000_000
sweepCounter=0


# ------------------------------
# Temperature Loop
# ------------------------------

if args.graphics==0:
    for T in temps:
        print(f"Running T = {T:.2f}")
        beta = 1.0 / T

        #------------------------------
        #Glauber
        #------------------------------

        #initialise grid
        grid_glaub = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))

        #run glauber
        M_vals_glaub, E_vals_glaub = glauber_run(grid_glaub, beta, J, n_warm, n_meas, sweep)

        #magnetisation and susceptibility
        M_mean_glaub = np.mean(np.abs(M_vals_glaub)) / (L * L)
        chi_glaub = (np.mean(M_vals_glaub**2) - np.mean(M_vals_glaub)**2) / (L * L * T)

        #energy and specific heat per spin
        E_mean_glaub = np.mean(E_vals_glaub) / (L*L)
        c_glaub = calculateSpecificHeat(E_vals_glaub, T, L*L)

        #error calculations
        cErrors_glaub=bootstrap(E_vals_glaub, calculateSpecificHeat, T, L*L)
        #print(cErrors_glaub, 'cerrosrglaub')

        #save
        csErrors_glaub.append(cErrors_glaub)

        #saving data
        Ms_glaub.append(M_mean_glaub)
        chis_glaub.append(chi_glaub)

        Es_glaub.append(E_mean_glaub)
        cs_glaub.append(c_glaub)

        #-------------------------------
        #Kawasaki
        #-------------------------------

        #initialise grid, this causes for magnetisation to change slightly between temperatures
        grid_kawa = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))

        M_vals_kawa, E_vals_kawa = kawasakiRun(grid_kawa, beta, J, n_warm, n_meas, sweep)

        #Magnetisation and Magnetic Susceptibility
        M_mean_kawa = np.mean(np.abs(M_vals_kawa)) / (L * L)
        chi_kawa = (np.mean(M_vals_kawa**2) - np.mean(M_vals_kawa)**2) / (L * L * T)

        #energy and specific heat per spin
        E_mean_kawa = np.mean(E_vals_kawa) / (L*L)
        #c_kawa = (np.mean(E_vals_kawa**2)-np.mean(E_vals_kawa)**2)/ (L*L*T**2)
        c_kawa = calculateSpecificHeat(E_vals_kawa, T, L*L)

        #want to calculate errors
        cErrors_kawa=bootstrap(E_vals_kawa, calculateSpecificHeat, T, L*L)
        print(cErrors_kawa, 'cerrosrkawa')
        csErrors_kawa.append(cErrors_kawa)

        #saving data
        Ms_kawa.append(M_mean_kawa)
        chis_kawa.append(chi_kawa)

        Es_kawa.append(E_mean_kawa)
        cs_kawa.append(c_kawa)

    #--------------------------
    #Saving data to CSV
    #------------------------------

    with open('GlauberData.csv', 'w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['Data in rows as: temps, Magnetisations, Susceptibilities, Energies, Specific Heats'])
        writer.writerow(temps)
        writer.writerow(Ms_glaub)
        writer.writerow(chis_glaub)
        writer.writerow(Es_glaub)
        writer.writerow(cs_glaub)
        writer.writerow(csErrors_glaub)
    
    with open('KawasakiData.csv', 'w', newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['Data in rows as: temps, Magnetisations, Susceptibilities, Energies, Specific Heats'])
        writer.writerow(temps)
        writer.writerow(Ms_kawa)
        writer.writerow(chis_kawa)
        writer.writerow(Es_kawa)
        writer.writerow(cs_kawa)
        writer.writerow(csErrors_kawa)

    #convert to numpy list for plotting
    cs_glaub=np.array(cs_glaub)
    csErrors_glaub=np.array(csErrors_glaub)

    cs_kawa=np.array(cs_kawa)
    csErrors_kawa=np.array(csErrors_kawa)


    #------------------------------
    #Plot Results
    #------------------------------

    plt.figure()
    plt.plot(temps, Ms_glaub, "o-", label='Glauber', color='blue', marker=',')
    plt.plot(temps, Ms_kawa, "-o", label='Kawasaki', color='red', marker=',')
    plt.legend(loc='best')
    plt.xlabel(r"$k_B T$")
    plt.ylabel(r"$\overline{M}/N$")
    plt.title("Magnetisation vs Temperature")
    plt.show()

    plt.figure()
    plt.plot(temps, Es_glaub, "o-", label='Glauber', color='blue', marker=',')
    plt.plot(temps, Es_kawa, "o-", label='Kawasaki', color='red', marker=',')
    plt.legend(loc='best')
    plt.xlabel(r"$k_B T$")
    plt.ylabel(r"$\chi$")
    plt.title("Energy vs Temperature")
    plt.show()

    plt.figure()
    plt.plot(temps, chis_glaub, "o-", label='Glauber', color='blue', marker=',')
    plt.plot(temps, chis_kawa, "o-", label='Kawasaki', color='red', marker=',')
    plt.legend(loc='best')
    plt.xlabel(r"$k_B T$")
    plt.ylabel(r"$\chi$")
    plt.title("Susceptibility vs Temperature")
    plt.show()


    #Specific heats
    plt.figure()
    plt.plot(temps, cs_glaub, "o-", label='Glauber', color='blue', marker=',')
    plt.fill_between(temps, cs_glaub-csErrors_glaub, cs_glaub+csErrors_glaub, color='blue', alpha=0.3)

    plt.plot(temps, cs_kawa, "o-", label='Kawasaki', color='red', marker=',')
    plt.fill_between(temps, cs_kawa-csErrors_kawa, cs_kawa+csErrors_kawa, color='red', alpha=0.3)
    
    plt.legend(loc='best')
    plt.xlabel(r"$k_B T$")
    plt.ylabel(r"$C$")
    plt.title("Specific Heat Per Spin vs Temperature")
    plt.show()


#hanlding visualisation mode
elif args.graphics==1:

    grid = np.random.choice(np.array([-1, 1], dtype=np.int8), size=(L, L))
    beta = 1.0 / args.thermalEnergy

    #kawasaki visualiser
    if args.dynamics.lower() == 'kawasaki':
        #initialise
        fig, ax, im = init_plot(grid, "Kawasaki Dynamics")

        #simulate
        for step in range(visualiserSteps):
    
            kawasakiProcedure(L, J, grid, beta)

            if step % sweep == 0 or step == visualiserSteps - 1:
                #sweepCounter+=1

                update_plot(im, fig, grid)
    
    #glauber visualiser
    if args.dynamics.lower() == 'glauber':
        fig, ax, im = init_plot(grid, "Glauber Dynamics")

        for step in range(visualiserSteps):

            glauberProcedure(L, J, grid, beta)

            if step % sweep == 0 or step == visualiserSteps - 1:
                #sweepCounter+=1

                update_plot(im, fig, grid)
