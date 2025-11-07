# %%
import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from simulation.pde_model.guzzo.simulation_1d import Simulation1D
from simulation.pde_model.guzzo.parameters import Parameters


signal_threshold = 0.5
# The initial density correspond to the density of u + v
initial_density = 0.45
fluctuation_level = 0.001
par = Parameters()
# You can choose either 'local' or 'directional' signaling
# signal_type = 'local'
signal_type = 'directional'

sim = Simulation1D(signal_type, initial_density, signal_threshold, fluctuation_level)
print('u =', np.mean(np.sum(sim.un[0, :, :] * par.dr, axis=0)))
print('v =', np.mean(np.sum(sim.un[1, :, :] * par.dr, axis=0)))
print('u + v =', np.mean(np.sum(sim.un[:, :, :] * par.dr, axis=(0,1))))

# Plot the reversal functions
signal = np.linspace(0, 2*initial_density, par.nx)
sim.refractory_period_function(signal)
sim.reversal_rate_function(signal)
plt.plot(signal, sim.f_rp[0])
plt.plot(signal, sim.f_rr[0])
plt.show()

# Launch the simulation
sim.start(T=10, alpha=2)

# %%
# Launch the kymograph plot
figsize = (15,15)
fontsize = 40
cmap = plt.get_cmap('hot')
sim.plot_kymograph(figsize, fontsize, cmap)

