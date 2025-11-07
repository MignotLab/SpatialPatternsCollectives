# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import csv
import gc
import os

import parameters
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

        self.f_rp = np.zeros((2, self.par.nx), dtype=float)
        self.f_rr = np.zeros((2, self.par.nx), dtype=float)

        self.un = np.zeros((2, self.par.nr, self.par.nx), dtype=float)

        self.refractory_period_function(np.ones(self.f_rp.shape) * self.initial_density)
        self.R = self.f_rp.copy()
        self.reversal_rate_function(np.ones(self.f_rr.shape) * self.initial_density)
        self.F = self.f_rr.copy()
        U = 0.5 * self.initial_density / (self.R[:, np.newaxis, :] + 1 / self.F[:, np.newaxis, :])
        self.u0 = U * np.exp(-self.F[:, np.newaxis, :] * (self.par.r[np.newaxis, :, np.newaxis] - self.R[:, np.newaxis, :]) * np.maximum(0, np.sign(self.par.r[np.newaxis, :, np.newaxis] - self.R[:, np.newaxis, :])))
        # add fluctuation
        self.u0 *= 1 + np.random.normal(loc=0, scale=self.fluctuation_level, size=(2, self.par.nr, self.par.nx))
        # self.rhotemp = np.sum(self.u0, axis=(0, 1, 2)) * self.par.dr * self.par.dx * 1/2
        self.un = self.u0.copy()
        self.signal = np.zeros((2, self.par.nx), dtype=float)

        # Initialize the storage for the kymograph plot
        self.data_kymo = []

        #construction des matrices de transformation spatiales M_U et M_V
        self.M_r = np.ones((2, self.par.nr, self.par.nx))
        self.M_r[:, -1, :] = 0
        self.M_right, self.M_left = np.zeros(self.un.shape, dtype=float), np.zeros(self.un.shape, dtype=float)
        self.M_right[0, :, :] = 1
        self.M_left[1, :, :] = 1

        # Create the folder save
        if self.signal_type == 'directional':
            C, S = self.compute_simulation_parameters(signal=self.initial_density/2)
            self.sample_name = 'result_simu/sample_directional_C=' + str(round(C, 4)) + '_S=' + str(round(S, 4)) + '_init_dens=' + str(round(self.initial_density, 2)) + '_thresh=' + str(round(self.signal_threshold, 2)) + '_rp=' + str(self.par.rp_max) + '_rr=' + str(np.round(self.par.rr_max,1)) + '/'
            self.update_signal = self.signal_directional

        elif self.signal_type == 'local':
            C, S = self.compute_simulation_parameters(signal=self.initial_density)
            self.sample_name = 'result_simu/sample_local_C=' + str(round(C, 4)) + '_S=' + str(round(S, 4)) + '_init_dens=' + str(round(self.initial_density, 2)) + '_thresh=' + str(round(self.signal_threshold, 2)) + '_rp=' + str(self.par.rp_max) + '_rr=' + str(np.round(self.par.rr_max,1)) + '/'
            self.update_signal = self.signal_local
        else:
            raise SignalTypeError('signal_type should be "local" or "directional"')
        
        # Paths
        self.path_folder_sample = self.sample_name+'sample/'
        self.path_file_kymograph = self.sample_name+'kymograph/data_kymo.csv'
        

    def refractory_period_function(self, signal):
        """
        Refractory period function.
        If signal < signal_threshold the function is constant and equal to rp_max.
        else the function is decrease as 1 / signal
        
        """
        self.f_rp[:, :] = np.minimum(self.par.rp_max, self.par.rp_max * self.signal_threshold / (signal + 10e-8))
    

    def reversal_rate_function(self, signal):
        """
        Reversal rate function.
        If signal < signal_threshold the function is linear.
        else the function is constant and equal to r_max
        
        """
        self.f_rr[:, :] = np.minimum(self.par.rr_max, self.a_rr * signal)


    def compute_simulation_parameters(self, signal):
        """
        Compute the C function
        
        """
        self.refractory_period_function(np.ones(self.f_rp.shape) * signal)
        self.reversal_rate_function(np.ones(self.f_rr.shape) * signal)
        S = self.f_rp[0, 0] * self.f_rr[0, 0]
        if signal < self.signal_threshold:
            print('UNDER THRESHOLD')
            print('signal =', signal)
            print('signal_threshold =', self.signal_threshold)
            C = 0.5 / (1 + S)
        else:
            print('UPPER THRESHOLD')
            print('signal =', signal)
            print('signal_threshold =', self.signal_threshold)
            C = 0.5 * S / (1 + S)

        print('rp =', self.f_rp[0, 0],
              '\n'+'rr =', self.f_rr[0, 0],
              '\n'+'C =', C,
              '\n'+'S =' , S)
        
        return C, S


    def signal_directional(self):
        """
        Compute the signal through directional signal
        
        """
        self.signal = np.sum(np.roll(self.un[:, :, :] * self.par.dr, shift=+1, axis=0), axis=1)


    def signal_local(self):
        """
        Compute the signal through directional signal
        
        """
        self.signal = np.sum(self.un[:, :, :] * par.dr, axis=(0,1))


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
                self.data_kymo.append(np.sum(self.un[:, :, :] * self.par.dr, axis=(0, 1)))


    def save_kymo(self):
        """
        Save the storage of the density in a csv file for the kymographe plot
        
        """
        with open(self.path_file_kymograph, 'a') as f:
            np.savetxt(f, self.data_kymo, delimiter=',')


    def plot_kymograph(self, figsize, fontsize, cmap):
        """
        Kymograph plot
        
        """
        kymo = pd.read_csv(self.path_file_kymograph, header=None).values
        fig, ax = plt.subplots(figsize=figsize)
        im = plt.imshow(kymo, cmap=cmap, extent=[0, self.par.lx, len(kymo)*self.par.save_frequency,0], aspect='auto')
        ax.set_xlabel('x ($\mu$m)', fontsize=fontsize)
        ax.set_ylabel('t (min)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=fontsize)
        fig.savefig(self.sample_name+'kymograph/kymo.png', bbox_inches='tight')


    def plot_density(self, t, fontsize):
        """
        Plot of the density un at a specific time
        
        """
        if t % int(1 / self.par.dt * self.par.save_frequency) == 0:
            fig, ax = plt.subplots(figsize=(12,8))
            ax.plot(self.par.x, np.sum(self.un[0] * self.par.dr, axis=0), zorder=1, linewidth=5, color='#B0E0E6')
            # ax.plot(self.par.x, np.sum((self.un[0] + self.un[1]) * self.par.dr, axis=0), zorder=2, linewidth=3, color='#B0E0E6')
            ax.plot(self.par.x, np.sum(self.un[1] * self.par.dr, axis=0), zorder=2, linewidth=5, color='k', linestyle='dotted')
            # ax.plot(self.par.x, np.sum((self.u0[0] + self.u0[1]) * self.par.dr, axis=0), zorder=1, linewidth=0.5, color='white')
            # ax.set_ylim(0, 3*self.initial_density)
            
            # ax.set_ylim(self.initial_density-fluctuation_level/2, self.initial_density+fluctuation_level/2)
            ax.set_ylim(0, self.initial_density*2.5)
            ax.set_xlabel('x (µm)', fontsize=fontsize)
            ax.set_ylabel('density', fontsize=fontsize)
            ax.tick_params(labelsize=fontsize)
            # plt.gca().set_facecolor('black')
            fig.savefig(self.path_folder_sample+str(int(t * self.par.dt / self.par.save_frequency))+'.png', bbox_inches='tight')
            plt.close()
            gc.collect()



    def start(self, T, alpha):
        """
        Start the simulation when the signal is on the directional density
        
        """
        # Initialize folders and files
        self.tool.initialize_directory_or_file(self.path_folder_sample)
        self.tool.initialize_directory_or_file(self.path_file_kymograph)
        eps = 1e-8
        t = int(T / self.par.dt)

        for i in tqdm(range(t)):
            # save plot
            self.store_density(i)
            self.plot_density(i, fontsize=60)
            self.update_signal()
            self.refractory_period_function(self.signal)
            new_f_rp = np.swapaxes(np.tile(self.f_rp, (self.par.nr, 1, 1)), axis1=1, axis2=0)
            rbool = 0.5 * (1 + np.tanh((self.par.rrep - new_f_rp) / (alpha * self.par.dr))) # Sigmoïde
            self.reversal_rate_function(self.signal)
            
            self.un[:, :, :] = ((1 - self.par.dt * self.f_rr[:, np.newaxis, :] * rbool[:, :, :]) * self.un[:, :, :]
                                + self.par.vr * self.par.dt / self.par.dr * (-self.M_r[:, :, :] * self.un[:, :, :] + np.roll(self.M_r[:, :, :] * self.un[:, :, :], shift=1, axis=1))
                                + self.par.v0 * self.par.dt / self.par.dx 
                                * (self.M_right[:, :, :] * (-self.un[:, :, :] + np.roll(self.un[:, :, :], shift=1, axis=2)) 
                                   + self.M_left[:, :, :] * (-self.un[:, :, :] + np.roll(self.un[:, :, :], shift=-1, axis=2)))
                                )
                
            # reversals
            reversals = self.par.dt * self.f_rr[:, np.newaxis, :] * rbool[:, :, :] * self.un[:, :, :]
            self.un[:, 0, :] += np.roll(np.sum(reversals[:, :, :], axis=1), shift=1, axis=0)

        self.save_kymo()
        print('DENSITY END SIMULATION')
        print('u =', np.mean(np.sum(self.un[0, :, :] * self.par.dr, axis=0)))
        print('v =', np.mean(np.sum(self.un[1, :, :] * self.par.dr, axis=0)))
        print('u + v =', np.mean(np.sum(self.un[:, :, :] * self.par.dr, axis=(0,1))))
    
if __name__ == '__main__':
    signal_threshold = 0.2
    # The initial density correspond to the density of u + v
    initial_density = 0.5
    fluctuation_level = 0.05
    par = parameters.Parameters()
    signal_type = 'directional'
    # signal_type = 'local'
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

    # Launch the kymograph plot
    figsize = (15,15)
    fontsize = 40
    cmap = plt.get_cmap('hot')
    sim.plot_kymograph(figsize, fontsize, cmap)
