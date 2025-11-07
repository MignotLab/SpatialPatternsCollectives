# %%
import sys, os
import cupy as cp
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from simulation.pde_model.guzzo.simulation_1d_cupy import Simulation1D
from simulation.pde_model.guzzo.parameters_cupy import Parameters


signal_threshold = 0.5
# The initial density correspond to the density of u + v
initial_density = 0.45
fluctuation_level = 0.001
par = Parameters()
# You can choose either 'local' or 'directional' signaling
# signal_type = 'local'
signal_type = 'directional'

sim = Simulation1D(signal_type, initial_density, signal_threshold, fluctuation_level)
print('signal type:', signal_type)
print('u =', cp.mean(cp.sum(sim.un[0, :, :] * par.dr, axis=0)))
print('v =', cp.mean(cp.sum(sim.un[1, :, :] * par.dr, axis=0)))
print('u + v =', cp.mean(cp.sum(sim.un[:, :, :] * par.dr, axis=(0,1))))
print('u0 =', cp.mean(cp.sum(sim.u0[:, :, :] * par.dr, axis=(0,1))))
plt.figure()
plt.plot(cp.asnumpy(sim.par.r), cp.asnumpy(sim.u0[0, :, 0]))
plt.xlabel('r')
plt.ylabel('density')

# Plot the reversal functions
signal = cp.linspace(0, 2*initial_density, par.nx)
sim.refractory_period_function(signal)
sim.reversal_rate_function(signal)
plt.plot(cp.asnumpy(signal), cp.asnumpy(sim.f_rp)[0])
plt.plot(cp.asnumpy(signal), cp.asnumpy(sim.f_rr)[0])
plt.show()

# Launch the simulation
sim.start(T=50, alpha=2)

# %%
# Kymograph plot
cmap = plt.get_cmap('hot')
# vmin, vmax = 0, 3
vmin, vmax = None, None
sim.plot_kymograph(cmap, vmin, vmax)

