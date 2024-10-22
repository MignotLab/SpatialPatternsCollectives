# %%
import numpy as np
import cupy as cp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import gc
import os
os.environ['CUPY_TF32'] = '1'


from parameters_igoshin import Parameters
from tools import Tools

class Simulation1D:


    def __init__(self, initial_density, signal_threshold, fluctuation_level):

        self.par = Parameters()
        self.tool = Tools()
        self.signal_threshold = signal_threshold
        self.initial_density = initial_density
        self.fluctuation_level = fluctuation_level

        # The Heaviside function
        self.sigmoid = 1 / (1 + cp.exp(-self.par.alpha_sigmoid * (self.par.rrep - self.par.delta_phi_r)))

        # The initial density with fluctuation un
        self.U = 0.5 * (self.initial_density * self.par.w_0 * (self.par.w_0 + self.w_1(self.initial_density))
             / (self.w_1(self.initial_density) * self.par.delta_phi_r + cp.pi * self.par.w_0))
        self.u0 = self.U / self.w(self.initial_density)
        self.u0 *= 1 + cp.random.normal(loc=0, scale=self.fluctuation_level, size=(2*self.par.np, self.par.nx), dtype=self.par.dtype)
        # self.u0 *= 1 + cp.random.normal(loc=0, scale=self.fluctuation_level, size=(2*self.par.np, self.par.nx))
        self.un = self.u0.copy()

        # Initialize the storage for the kymograph plot
        self.data_kymo = []

        # Matrices for derivative
        self.M_right, self.M_left = cp.zeros(self.un.shape, dtype=self.par.dtype), cp.zeros(self.un.shape, dtype=self.par.dtype)
        self.index_0 = 0
        self.index_delta_phi_r = int(self.par.delta_phi_r / self.par.dp)
        self.index_pi = int(self.par.phi_mid / self.par.dp) + 1
        self.index_pi_plus_delta_phi_r = self.index_pi + self.index_delta_phi_r

        self.M_right, self.M_left = cp.zeros(self.un.shape, dtype=float), cp.zeros(self.un.shape, dtype=float)
        self.M_right[:self.index_pi, :] = 1
        self.M_left[self.index_pi:, :] = 1

        # Save
        self.path_folder_sample = 'results/sample_igoshin__q=' + str(round(self.par.q, 1)) +'__w_1_sur_w_0=' + str(round(self.w_1(self.initial_density) / self.par.w_0, 2)) + '__initial_density=' + str(round(self.initial_density, 2)) + '__signal_threshold=' + str(round(self.signal_threshold, 2)) + '/'
        self.tool.initialize_directory_or_file(self.path_folder_sample)
        self.path_file_kymograph = self.path_folder_sample+'kymograph/data_kymo__time_init='+str(self.par.start_time_save_kymo)+'save_frequency_kymo='+str(self.par.save_frequency_kymo)+'_minutes'+'.csv'
        self.tool.initialize_directory_or_file(self.path_file_kymograph)
        self.path_folder_sample_plots_simu = self.path_folder_sample+'plots_simu/'
        self.tool.initialize_directory_or_file(self.path_folder_sample_plots_simu)

    def w_1(self, rho):
        """
        Function \omega_1
        
        """
        return self.par.w_n * rho**self.par.q / (rho**self.par.q + self.signal_threshold**self.par.q)
    

    def w(self, rho):
        """
        Function \omega^\pm
        
        """
        return self.par.w_0 + self.w_1(rho) * self.sigmoid
    

    def store_density(self, t):
        """
        Save un at specific time

        Parameters
        ----------
        t : int or float
            The time value at which to save the data.

        Returns
        -------
        None
        """
        if t > self.par.start_time_save_kymo:
            if t % int(1 / self.par.dt * self.par.save_frequency_kymo) == 0:
                self.data_kymo.append(cp.sum(self.un[:, :] * self.par.dp, axis=0))


    def save_kymo(self):
        """
        Save the storage of the density in a csv file for the kymographe plot
        
        """
        with open(self.path_file_kymograph, 'a') as f:
            np.savetxt(f, cp.array(self.data_kymo), delimiter=',')


    def plot_kymograph(self, cmap, vmin=None, vmax=None):
        """
        Kymograph plot
        
        """
        self.kymo = pd.read_csv(self.path_file_kymograph, header=None).values
        fig, ax = plt.subplots(figsize=self.par.figsize)
        if vmin and vmax:
            im = plt.imshow(self.kymo, cmap=cmap, extent=[0, self.par.lx, len(self.kymo)*self.par.save_frequency_kymo, self.par.start_time_save_kymo], aspect='auto', vmin=vmin, vmax=vmax)  # Set vmin and vmax to 0 and 7
        else:
            im = plt.imshow(self.kymo, cmap=cmap, extent=[0, self.par.lx, len(self.kymo)*self.par.save_frequency_kymo, self.par.start_time_save_kymo], aspect='auto')  # Set vmin and vmax to 0 and 7
        ax.set_ylabel('Time (min)', fontsize=self.par.fontsize)
        ax.set_xlabel(r'Space ($\mu$m)', fontsize=self.par.fontsize)
        ax.tick_params(labelsize=self.par.fontsize_ticks)
        # Division de l'axe pour ajouter la colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        # Ajout de la colorbar
        cbar = plt.colorbar(im, cax=cax)
        cbar.set_label('Density of bacteria', fontsize=self.par.fontsize)
        if vmin and vmax:
            cbar.set_ticks(np.arange(vmin, vmax+1, 1))  # Set the tick positions for the colorbar
        cbar.ax.tick_params(labelsize=self.par.fontsize_ticks)
        fig.savefig(self.path_folder_sample+'kymograph/kymo_igoshin_main.png', bbox_inches='tight', dpi=self.par.dpi)
        fig.savefig(self.path_folder_sample+'kymograph/kymo_igoshin_main.svg', dpi=self.par.dpi)
    

    def plot(self, t, save_frequency):
        """
        Plot and save
        
        """
        if t % int(1 / self.par.dt * save_frequency) == 0:
            fig, ax = plt.subplots(figsize=self.par.figsize)

            x = self.par.x.get()
            rho_0 = cp.sum(self.u0[:, :].copy() * self.par.dp, axis=0).get()
            u = cp.sum(self.un[:self.index_pi, :] * self.par.dp, axis=0).get()
            v = cp.sum(self.un[self.index_pi:, :] * self.par.dp, axis=0).get()

            ax.plot(x, 0.5*rho_0, label=r'$\bar\rho$', color='k')
            ax.plot(x, u, label=r'$\rho^+$', color='royalblue', linewidth=3)
            ax.plot(x, v, label=r'$\rho^-$', color='limegreen', linewidth=3)
            ax.set_xlim(0, self.par.lx)
            ax.set_ylim(0, 4.5)

            ax.set_xlabel(r'Space ($\mu$m)', fontsize=self.par.fontsize)
            ax.set_ylabel('Density of bacteria', fontsize=self.par.fontsize)
            ax.tick_params(labelsize=self.par.fontsize_ticks)

            ax.legend(loc="upper right", fontsize=self.par.fontsize_ticks)

            fig.savefig(self.path_folder_sample_plots_simu+str(int(t * self.par.dt / save_frequency))+'.png', bbox_inches='tight', dpi=self.par.dpi)
            plt.close()
            gc.collect()
    

    def start(self, T):
        """
        Start the simulation when the signal is on the directional density
        
        """
        t = int(T / self.par.dt)

        for i in tqdm(range(t)):

            self.plot(i, self.par.save_frequency)
            self.store_density(i)

            rho_opp_u = 2 * cp.sum(self.un[:self.index_pi, :] * self.par.dp, axis=0)
            rho_opp_v = 2 * cp.sum(self.un[self.index_pi:, :] * self.par.dp, axis=0)
            rho_opp = cp.concatenate((cp.tile(rho_opp_v, (self.par.np, 1)), 
                                      cp.tile(rho_opp_u, (self.par.np, 1))
                                    ))

            # Scheme
            # u_i^{n+1} = u_i^n 
            #             - v_0 * dt/dx * (u_i^n - u_{i \pm 1}^n)
            #             - dt/dp * (w_i * u_i^n - w_{i-1} * u_{i-1}^n)

            # roll_w = cp.roll(self.w(rho), shift=1, axis=0)
            # Boundary conditions
            # roll_w[self.index_0, :] = self.par.w_0
            # roll_w[self.index_delta_phi_r, :] = self.w(rho)[self.index_delta_phi_r, :]
            # roll_w[self.index_pi, :] = self.par.w_0
            # roll_w[self.index_pi_plus_delta_phi_r, :] = self.w(rho)[self.index_pi_plus_delta_phi_r, :]

            self.un[:, :] = (self.un[:, :]
                                
                            - self.par.v0 * self.par.dt / self.par.dx 
                            * (self.M_right[:, :] * self.un[:, :] - cp.roll(self.M_right[:, :] * self.un[:, :], shift=1, axis=1) 
                            +  self.M_left[:, :] * self.un[:, :] - cp.roll(self.M_left[:, :] * self.un[:, :], shift=-1, axis=1))

                            - self.par.dt / self.par.dp
                            * (self.w(rho_opp) * self.un[:, :] - cp.roll(self.w(rho_opp) * self.un[:, :], shift=1, axis=0))
                            )
        
        self.save_kymo()
        print('DENSITY END SIMULATION')
        print('u =', cp.mean(cp.sum(self.un[:self.index_pi, :] * self.par.dp, axis=0)))
        print('v =', cp.mean(cp.sum(self.un[self.index_pi:, :] * self.par.dp, axis=0)))
        print('u + v =', cp.mean(cp.sum(self.un[:, :] * self.par.dp, axis=0)))
        print('u_0 =', cp.mean(cp.sum(self.u0[:, :] * self.par.dp, axis=0)))



signal_threshold = 0.5
# The initial density correspond to the density of u + v
initial_density = 0.5
fluctuation_level = 0.01
par = Parameters()
sim = Simulation1D(initial_density, signal_threshold, fluctuation_level)
print('u0=', cp.mean(cp.sum(sim.u0 * sim.par.dp, axis=0)))
print('u =', cp.mean(cp.sum(sim.un[:sim.index_pi, :] * sim.par.dp, axis=0)))
print('v =', cp.mean(cp.sum(sim.un[sim.index_pi:, :] * sim.par.dp, axis=0)))
print('u_rp =', cp.mean(cp.sum(sim.un[:sim.index_delta_phi_r, :] * sim.par.dp, axis=0)))
print('v_rp =', cp.mean(cp.sum(sim.un[sim.index_pi:sim.index_pi_plus_delta_phi_r, :] * sim.par.dp, axis=0)))
print('u_rr =', cp.mean(cp.sum(sim.un[sim.index_delta_phi_r:sim.index_pi, :] * sim.par.dp, axis=0)))
print('v_rr =', cp.mean(cp.sum(sim.un[sim.index_pi_plus_delta_phi_r:, :] * sim.par.dp, axis=0)))
print('u + v =', cp.mean(cp.sum(sim.un[:, :] * sim.par.dp, axis=0)))
print('w_1/w_0=', sim.w_1(rho=initial_density) / sim.par.w_0)

# %%
# Launch the simulation
sim.start(T=50)

fig, ax = plt.subplots()
x = sim.par.x.get()
rho_0 = cp.sum(sim.u0[:, :] * sim.par.dp, axis=0).get()
u = cp.sum(sim.un[:sim.index_pi, :] * sim.par.dp, axis=0).get()
v = cp.sum(sim.un[sim.index_pi:, :] * sim.par.dp, axis=0).get()
ax.plot(x, 0.5*rho_0, label=r'$\rho_0$', color='k')
ax.plot(x, u, label='$u^+$', color='royalblue', linewidth=3)
ax.plot(x, v, label='$u^-$', color='limegreen', linewidth=3)
ax.legend()

# %%
# Launch the kymograph plot
cmap = plt.get_cmap('hot')
# vmin, vmax = 0, 3
vmin, vmax = None, None
sim.plot_kymograph(cmap, vmin, vmax)




# %%
x = np.linspace(0, 2*np.pi, 1000)
print(sim.sigmoid[:, 0].dtype)
x = np.linspace(0, 2*np.pi, sim.sigmoid[:,0].shape[0])
y = sim.sigmoid[:, 1].get()
plt.plot(x, y)

# %%
from parameters_igoshin import Parameters
par = Parameters()
def w_1(rho, q):
    """
    Function \omega_1
    
    """
    return par.w_n * rho**q / (rho**q + signal_threshold**q)

q = 8
rho = np.linspace(0, 2, 1000)
plt.plot(rho, w_1(rho, q))