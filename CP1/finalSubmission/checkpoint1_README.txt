This is a read me file for checkpoint 1.

Author: John Whitfield, 06.02.25
Description:
This code implements a 2D Ising model on a square lattice with periodic boundary conditions, using both Glauber and Kawasaki Monte Carlo dynamics. It supports either full temperature sweeps for measuring thermodynamic observables or real-time visualisation of the spin dynamics.
The code is optimised using Numba for performance and can compute magnetisation, energy, susceptibility, and specific heat, including bootstrap error estimates.

Features
2D square lattice Ising model with periodic boundary conditions
Glauber (non-conserved magnetisation) dynamics
Kawasaki (conserved magnetisation) dynamics
Fast Monte Carlo updates via numba.njit
Temperature sweeps and thermodynamic measurements
Bootstrap error estimation for specific heat
Optional real-time visualisation of spin configurations
CSV output for post-processing and plotting

Requirements
This code requires the following Python packages:
numpy
matplotlib
scipy
numba
argparse (standard library)
csv (standard library)
Additionally, it assumes access to:
shendrukGroupStyle (Matplotlib style)
shendrukGroupFormat (custom plotting utilities)
If these are not available, remove or comment out:
plt.style.use('shendrukGroupStyle')
import shendrukGroupFormat as ed

Running the Code
The script is designed to be run from the command line.

Basic Usage
python3 numbaSimulation.py
This will:
Run a Glauber Dynamics simulation for t=1, j=1 and visualise it.


Command Line Arguments
Argument	Flag	Description	Default
Lattice size	-n, --n	Length of lattice (L × L)	50
Temperature	-t, --thermalEnergy	Temperature for visualisation mode	1.0
Dynamics	-d, --dynamics	Glauber or Kawasaki	Glauber
Coupling	-c, --couplingConstant	Coupling constant J	1
Graphics mode	-g, --graphics	0 = measurements, 1 = visualisation	1

Simulation Modes
1. Visualisation mode (--graphics 1)
Runs the simulation using the dynamics, size, and other paramters that you wish.
Doesn't run any calculations.
Runs for a long number of times steps (effectively infinite).
Spins are shown as ±1 values
Periodic updates every Monte Carlo sweep
Useful for observing domain growth and phase ordering

Examples:
python ising.py -g 1 -d Glauber -t 2.0
python ising.py -g 1 -d Kawasaki -t 1.5


2. Measurement Mode (--graphics 0)
Runs temperature sweeps and computes:
Magnetisation per spin
Energy per spin
Magnetic susceptibility
Specific heat per spin
For each temperature:
System is thermalised
Measurements are taken over many Monte Carlo sweeps
Bootstrap resampling is used to estimate errors in the specific heat

Output files:
GlauberData.csv
KawasakiData.csv
Each CSV contains rows that are consisted of the average values of:
Temperatures
Magnetisation
Susceptibility
Energy
Specific heat
Specific heat error

Produces Graphs of the above too.



Periodic boundary conditions
One Monte Carlo sweep = L**2 attempted updates

General features of different dynamics.
Glauber Dynamics
Single spin flips
Magnetisation is not conserved

Kawasaki Dynamics
Spin exchanges
Magnetisation is conserved

Performance Notes
Critical routines are JIT-compiled with numba.njit
First run will be slower due to compilation
Larger lattices benefit significantly from JIT acceleration

Known Limitations / Notes
Error estimates are currently only implemented for specific heat
Bootstrap assumes uncorrelated samples (no binning applied)
Custom plotting styles are optional but assumed present