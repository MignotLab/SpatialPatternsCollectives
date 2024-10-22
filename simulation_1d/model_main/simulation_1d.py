# %%
import numpy as np
import pandas as pd
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import gc

import simulation_1d.model_main.parameters as parameters
import tools


class SignalTypeError(Exception):
    pass


class Simulation1D:


    def __init__(self, signal_type, initial_density, signal_threshold, fluctuation_level):
        
        self.par = parameters.Parameters()
        self.tool = tools.Tools()
        self.signal_type = signal_type
        self.signal_threshold = signal_threshold
        self.initial_density = initial_density
        self.fluctuation_level = fluctuation_level

        self.a_rr = self.par.rr_max / self.signal_threshold

        self.f_rp = cp.zeros((2, self.par.nx), dtype=self.par.dtype)
        self.f_rr = cp.zeros((2, self.par.nx), dtype=self.par.dtype)

        self.un = cp.zeros((2, self.par.nr, self.par.nx), dtype=self.par.dtype)

        # The following will depend of the chosen signal
        if self.signal_type == 'directional':
            C, S = self.compute_simulation_parameters(self.initial_density/2)
            self.sample_name = 'result_simu/sample_directional_C=' + str(round(C, 4)) + '_S=' + str(round(S, 4)) + '_init_dens=' + str(round(self.initial_density, 2)) + '_thresh=' + str(round(self.signal_threshold, 2)) + '_rp=' + str(self.par.rp_max) + '_rr=' + str(cp.round(self.par.rr_max,1)) + '/'
            self.update_signal = self.signal_directional

            signal = cp.ones(self.f_rp.shape) * self.initial_density / 2
            self.refractory_period_function(signal)
            self.reversal_rate_function(signal)
            self.R = self.f_rp.copy()
            self.F = self.f_rr.copy()

        elif self.signal_type == 'local':
            C, S = self.compute_simulation_parameters(self.initial_density)
            self.sample_name = 'result_simu/sample_local_C=' + str(round(C, 4)) + '_S=' + str(round(S, 4)) + '_init_dens=' + str(round(self.initial_density, 2)) + '_thresh=' + str(round(self.signal_threshold, 2)) + '_rp=' + str(self.par.rp_max) + '_rr=' + str(cp.round(self.par.rr_max,1)) + '/'
            self.update_signal = self.signal_local

            signal = cp.ones(self.f_rp.shape) * self.initial_density
            self.refractory_period_function(signal)
            self.reversal_rate_function(signal)
            self.R = self.f_rp.copy()
            self.F = self.f_rr.copy()
            
        else:
            raise SignalTypeError('signal_type should be "local" or "directional"')

        U = 0.5 * self.initial_density / (self.R[:, cp.newaxis, :] + 1 / self.F[:, cp.newaxis, :])
        self.u0 = U * cp.exp(-self.F[:, cp.newaxis, :] * (self.par.r[cp.newaxis, :, cp.newaxis] - self.R[:, cp.newaxis, :]) * cp.maximum(0, cp.sign(self.par.r[cp.newaxis, :, cp.newaxis] - self.R[:, cp.newaxis, :])))
        # add fluctuation
        self.u0 *= 1 + cp.random.normal(loc=0, scale=self.fluctuation_level, size=(2, self.par.nr, self.par.nx))
        # self.rhotemp = cp.sum(self.u0, axis=(0, 1, 2)) * self.par.dr * self.par.dx * 1/2
        self.un = self.u0.copy()
        self.signal = cp.zeros((2, self.par.nx), dtype=float)

        # Initialize the storage for the kymograph plot
        self.data_kymo = []

        # Construction des matrices de transformation spatiales M_U et M_V
        self.M_r = cp.ones((2, self.par.nr, self.par.nx))
        self.M_r[:, -1, :] = 0
        self.M_right, self.M_left = cp.zeros(self.un.shape, dtype=float), cp.zeros(self.un.shape, dtype=float)
        self.M_right[0, :, :] = 1
        self.M_left[1, :, :] = 1
        
        # Paths
        self.path_file_kymograph = self.sample_name+'kymograph/data_kymo_'+self.signal_type+'_'+self.high_or_low_signal+'_time_init='+str(self.par.start_time_save_kymo)+'_save_frequency='+str(self.par.save_frequency)+'_minutes'+'.csv'
        self.tool.initialize_directory_or_file(self.path_file_kymograph)
        self.path_folder_sample_plots_simu = self.sample_name+'plots_simu/'
        self.tool.initialize_directory_or_file(self.path_folder_sample_plots_simu)
        

    def refractory_period_function(self, signal):
        """
        Refractory period function.
        If signal < signal_threshold the function is constant and equal to rp_max.
        else the function is decrease as 1 / signal
        
        """
        self.f_rp[:, :] = cp.minimum(self.par.rp_max, self.par.rp_max * self.signal_threshold / (signal + 10e-8))
    

    def reversal_rate_function(self, signal):
        """
        Reversal rate function.
        If signal < signal_threshold the function is linear.
        else the function is constant and equal to r_max
        
        """
        self.f_rr[:, :] = cp.minimum(self.par.rr_max, self.a_rr * signal)


    def compute_simulation_parameters(self, signal):
        """
        Compute the C function
        
        """
        self.refractory_period_function(cp.ones(self.f_rp.shape) * signal)
        self.reversal_rate_function(cp.ones(self.f_rr.shape) * signal)
        F = np.mean(cp.asnumpy(self.f_rr))
        R = np.mean(cp.asnumpy(self.f_rp))
        S = F * R
        print('initial density =', self.initial_density)
        if signal < self.signal_threshold:
            print('UNDER THRESHOLD')
            self.high_or_low_signal = "low_signalling"
            print('signal =', signal)
            print('signal_threshold =', self.signal_threshold)
            if self.signal_type == 'local':
                C = 0.5 / (1 + S)
            elif self.signal_type == 'directional':
                C = 1 / (1 + S)
            else:
                raise SignalTypeError('signal_type should be "local" or "directional"')
        else:
            print('UPPER THRESHOLD')
            self.high_or_low_signal = "high_signalling"
            print('signal =', signal)
            print('signal_threshold =', self.signal_threshold)
            if self.signal_type == 'local':
                C = 0.5 * S / (1 + S)
            elif self.signal_type == 'directional':
                C = S / (1 + S)
            else:
                raise SignalTypeError('signal_type should be "local" or "directional"')

        print('rp(rho_bar) =', R,
              '\n'+'rr(rho_bar) =', F,
              '\n'+'C =', C,
              '\n'+'S =' , S)
        
        return C, S


    def signal_directional(self):
        """
        Compute the signal through directional signal
        
        """
        self.signal = cp.sum(cp.roll(self.un[:, :, :], shift=+1, axis=0) * self.par.dr, axis=1)


    def signal_local(self):
        """
        Compute the signal through directional signal
        
        """
        self.signal = cp.sum(self.un[:, :, :] * self.par.dr, axis=(0,1))


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
                self.data_kymo.append(cp.sum(self.un[:, :, :] * self.par.dr, axis=(0, 1)))


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
        # Utiliser ScalarFormatter pour un formatage scientifique si nécessaire
        cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # cbar.ax.yaxis.get_offset_text().set_fontsize(self.par.fontsize_ticks)

        fig.savefig(self.sample_name+'kymograph/kymo_'+self.signal_type+'.png', bbox_inches='tight', dpi=self.par.dpi)
        fig.savefig(self.sample_name+'kymograph/kymo_'+self.signal_type+'.svg', dpi=self.par.dpi)


    def plot_density(self, t):
        """
        Plot of the density un at a specific time
        
        """
        if t % int(1 / self.par.dt * self.par.save_frequency) == 0:
            fig, ax = plt.subplots(figsize=self.par.figsize)

            x = self.par.x.get()
            rho_0 = cp.sum(self.u0[:, :, :].copy() * self.par.dr, axis=(0,1)).get()
            u = cp.sum(self.un[0] * self.par.dr, axis=0).get()
            v = cp.sum(self.un[1] * self.par.dr, axis=0).get()

            ax.plot(x, 0.5*rho_0, label=r'$\bar\rho$', color='k')
            ax.plot(x, u, label=r'$\rho^+$', color='royalblue', linewidth=3)
            ax.plot(x, v, label=r'$\rho^-$', color='limegreen', linewidth=3)
            ax.set_xlim(0, self.par.lx)
            if self.signal_type == 'directional':
                ax.set_ylim(0, 4.5)
            elif self.signal_type == 'local':
                ax.set_ylim(0, 2)

            ax.set_xlabel(r'Space ($\mu$m)', fontsize=self.par.fontsize)
            ax.set_ylabel('Density of bacteria', fontsize=self.par.fontsize)
            ax.tick_params(labelsize=self.par.fontsize_ticks)

            ax.legend(loc="upper right", fontsize=self.par.fontsize_ticks)

            # plt.gca().set_facecolor('black')
            fig.savefig(self.path_folder_sample_plots_simu+str(int(t * self.par.dt / self.par.save_frequency))+'.png', bbox_inches='tight', dpi=self.par.dpi)
            plt.close()
            gc.collect()


    def start(self, T, alpha):
        """
        Start the simulation when the signal is on the directional density
        
        """
        eps = 1e-8
        t = int(T / self.par.dt)

        for i in tqdm(range(t)):
            # save plot
            self.store_density(i)
            self.plot_density(i)
            self.update_signal()
            self.refractory_period_function(self.signal)
            new_f_rp = cp.swapaxes(cp.tile(self.f_rp, (self.par.nr, 1, 1)), axis1=1, axis2=0)
            rbool = 0.5 * (1 + cp.tanh((self.par.rrep - new_f_rp) / (alpha * self.par.dr))) # Sigmoïde
            self.reversal_rate_function(self.signal)
            
            self.un[:, :, :] = ((1 - self.par.dt * self.f_rr[:, cp.newaxis, :] * rbool[:, :, :]) * self.un[:, :, :]
                                + self.par.vr * self.par.dt / self.par.dr 
                                * (-self.M_r[:, :, :] * self.un[:, :, :] + cp.roll(self.M_r[:, :, :] * self.un[:, :, :], shift=1, axis=1))
                                + self.par.v0 * self.par.dt / self.par.dx 
                                * (self.M_right[:, :, :] * (-self.un[:, :, :] + cp.roll(self.un[:, :, :], shift=1, axis=2)) 
                                   + self.M_left[:, :, :] * (-self.un[:, :, :] + cp.roll(self.un[:, :, :], shift=-1, axis=2)))
                                )
                
            # reversals
            reversals = self.par.dt * self.f_rr[:, cp.newaxis, :] * rbool[:, :, :] * self.un[:, :, :]
            self.un[:, 0, :] += cp.roll(cp.sum(reversals[:, :, :], axis=1), shift=1, axis=0)

        self.save_kymo()
        print('DENSITY END SIMULATION')
        print('u =', cp.mean(cp.sum(self.un[0, :, :] * self.par.dr, axis=0)))
        print('v =', cp.mean(cp.sum(self.un[1, :, :] * self.par.dr, axis=0)))
        print('u + v =', cp.mean(cp.sum(self.un[:, :, :] * self.par.dr, axis=(0,1))))
    

signal_threshold = 0.5
# The initial density correspond to the density of u + v
initial_density = 0.45
fluctuation_level = 0.001
par = parameters.Parameters()
signal_type = 'local'
# signal_type = 'directional'
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

# %%
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
# Launch the kymograph plot
cmap = plt.get_cmap('hot')
# vmin, vmax = 0, 3
vmin, vmax = None, None
sim.plot_kymograph(cmap, vmin, vmax)


# %%
