import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

## .py files import
from .alignment import Alignment
from .attraction import Attraction
from .bacteria_generation import GenerateBacteria
from .bacteria_movement import Move
from .boundary_condition import Boundaries
from .directions import Direction
from .extracellular_matrix import Ecm
from .kymograph import Kymograph
from .neighbourhood import Neighbours
from .nodes_attachment import BacteriaBody
from .phantom_data import Phantom
from .plot import Plot
from .repulsion import Repulsion
from .reversal_signal import ReversalSignal
from .reversals import Reversal
from .rigidity import Rigidity
from .velocity_measurement import Velocity
from .viscosity import Viscosity
import utils as tl


class Main():


    def __init__(self,inst_par,sample,T):

        ## Parameters
        self.par = inst_par
        self.sample = sample
        self.T = T
        self.velocity_save = []
        self.cond_rev = np.zeros(self.par.n_bact, dtype=int)

        ## Init the data array
        self.gen = GenerateBacteria(inst_par=self.par)
        self.gen.generate_bacteria()
        ## Data moving in space without the boundary condition
        self.pha = Phantom(inst_gen=self.gen)
        ## Neighbourhood
        self.nei = Neighbours(inst_par=self.par, inst_gen=self.gen)
        ## Boundaries
        self.bound = Boundaries(inst_par=self.par, inst_gen=self.gen)
        ## Directions
        self.dir = Direction(inst_par=self.par, inst_gen=self.gen, inst_pha=self.pha, inst_nei=self.nei)
        ## Initialize the attachment of the nodes
        self.spg = BacteriaBody(inst_par=self.par, inst_gen=self.gen, inst_pha=self.pha, inst_dir=self.dir)
        # Rigidity
        self.rig = Rigidity(inst_par=self.par, inst_gen=self.gen, inst_pha=self.pha)
        ## Velocity measurement
        self.vel = Velocity(inst_par=self.par, inst_pha=self.pha)
        ## Viscosity
        self.visc = Viscosity(inst_par=self.par, inst_gen=self.gen, inst_pha=self.pha)
        ## Kymograph of trajectories
        if self.par.kymograph_plot:
            self.kym = Kymograph(inst_par=self.par, inst_gen=self.gen,T=self.T)
        else:
            self.kym = None

        ### Class needed lots of others classes
        ## Repulsion
        self.rep = Repulsion(inst_par=self.par,
                             inst_gen=self.gen,
                             inst_pha=self.pha,
                             inst_dir=self.dir,
                             inst_nei=self.nei)
        ## Attraction
        self.att = Attraction(inst_par=self.par,
                              inst_gen=self.gen,
                              inst_pha=self.pha,
                              inst_dir=self.dir,
                              inst_nei=self.nei)
        ## Alignment
        self.ali = Alignment(inst_par=self.par,
                             inst_gen=self.gen,
                             inst_pha=self.pha,
                             inst_dir=self.dir,
                             inst_nei=self.nei,
                             inst_uti=tl)
        ## EPS
        self.ecm = Ecm(inst_par=self.par,
                       inst_gen=self.gen,
                       inst_pha=self.pha,
                       inst_dir=self.dir,
                       inst_nei=self.nei,
                       inst_ali=self.ali,
                       inst_uti=tl)
        ## Reversal signal
        self.sig = ReversalSignal(inst_par=self.par,
                                  inst_gen=self.gen,
                                  inst_vel=self.vel,
                                  inst_dir=self.dir,
                                  inst_nei=self.nei)
        ## Reversals
        self.rev = Reversal(inst_par=self.par,
                            inst_gen=self.gen,
                            inst_pha=self.pha,
                            inst_sig=self.sig)
        ## Movement
        self.move = Move(inst_par=self.par,
                         inst_gen=self.gen,
                         inst_pha=self.pha,
                         inst_nei=self.nei,
                         inst_dir=self.dir,
                         inst_uti=tl)
        ## Plot
        self.plo = Plot(inst_par=self.par,
                        inst_gen=self.gen,
                        inst_dir=self.dir,
                        inst_rev=self.rev,
                        inst_ecm=self.ecm,
                        inst_move=self.move,
                        inst_kym=self.kym,
                        inst_nei=self.nei,
                        inst_sig=self.sig,
                        sample=self.sample)

    def start(self):

        # REVERSAL FUNCTIONS PLOT
        if self.par.rev_function_plot:
            signal = np.linspace(0, 2*self.par.s1, self.par.n_bact)
            self.sig.signal = signal
            self.rev.function_reversal_type()
            self.plo.reversal_functions(signal=signal, refractory_period=self.rev.P, reversal_rate=self.rev.R)
        # TRANSITION STOP PARAMETER
        self.transition_stop = False

        t = int(self.T/self.par.dt) # Number of iterations

        # File to save the coordinates of the simultion
        tl.initialize_directory_or_file(self.sample+'/')
        filename = self.sample+'/coords__'+str(self.par.n_bact)+'_bacts__tbf='+str(int(self.par.save_frequency_csv*60))+'_secondes__space_size='+str(self.par.space_size)+'.csv'
        x_columns, y_columns = tl.gen_coord_str(n=self.par.n_nodes, xy=False)
        column_names = ['frame', 'id'] + x_columns + y_columns + ['reversals', 'clock_tbr', self.par.signal_type, 'reversing']
        if self.par.save_other_reversal_signals:
            column_names += ['cumul_frustration_extra', 'directional_density_extra', 'local_density_extra']
        tl.initialize_directory_or_file(path=filename, columns=column_names)

        # Plot
        path = self.sample+'/'+str(int(0*self.par.dt/self.par.save_frequency_image))+".png"
        if self.par.plot_movie:
            self.plo.plotty(path=path)
        if self.par.plot_rippling_swarming_color:
            path_folder_rippling_swarming = self.sample+'/rippling_swarming_colored_plot/'
            tl.initialize_directory_or_file(path_folder_rippling_swarming)

        if self.par.plot_reversing_and_non_reversing:
            path_folder_reversing_and_non_reversing = self.sample+'/reversing_and_non_reversing_colored_plot/'
            tl.initialize_directory_or_file(path_folder_reversing_and_non_reversing)

        if self.par.plot_colored_nb_neighbors:
            path_folder_colored_nb_neighbors = self.sample+'/colored_nb_neighbors_plot/'
            tl.initialize_directory_or_file(path_folder_colored_nb_neighbors)

        if self.par.n_bact_prey > 0:
            fig, ax = plt.subplots(figsize=(32, 32))
            # plt.imshow(self.ecm.prey_grid[::-1, :])
            plt.imshow(self.ecm.prey_grid)
            plt.colorbar()
            fig.savefig(self.sample+'/ecm_pre_map.png')
            plt.close()

        print('SIMULATION '+self.sample+' IS RUNNING\n')
        ## Iteration over time
        for i in tqdm(range(t)):
            # BOUNDARIES
            self.bound.periodic()
            # CONDITIONAL SPACE
            self.gen.update_conditional_space()
            # NODES DIRECTIONS
            self.dir.set_nodes_direction()
            # SAVE NODES POSITION IN 
            self.vel.head_position_in()
            # NEIGHBOURS DETECTION
            self.nei.set_kn_nearest_neighbours_torus()
            self.nei.set_bacteria_index()
            self.nei.find_prey_neighbours() # Need to be after set_kn_nearest_neighbours_torus and set_bacteria_index
            self.dir.set_nodes_to_neighbours_direction_torus()
            self.dir.set_neighbours_direction()

            ## BODY MOVEMENT USING THE PREVIOUS POSITION AND NEIGHBOURHOOD
            # MOTILITY
            self.move.function_movement_type()
            # REPULSION
            self.rep.function_repulsion_type()
            # ATTRACTION
            self.att.function_attraction_type()
            # RIGIDITY
            self.rig.function_rigidity_type()
            # SPRINGS ACTION
            self.spg.nodes_spring() #  The node directions are recompute here before applying the node springs

            ## HEAD ROTATION
            # ALIGNMENT
            time_now = i * self.par.dt
            if self.par.stop_alignment is None or self.par.stop_alignment > time_now:
                self.ali.function_alignment_type()
            else:
                self.transition_stop = True
            # EPS
            if self.par.stop_eps_follower is None or self.par.stop_eps_follower > time_now:
                self.ecm.function_ecm_follower_type()
            else:
                self.transition_stop = True

            # SAVE NODES POSITION OUT AND COMPUTE VELOCITY AND DISPLACEMENT
            self.vel.head_position_out()
            self.vel.displacement_in_out()
            self.vel.velocity_in_out()

            # REVERSALS
            if self.par.save_other_reversal_signals: # Save the reversal signals before calling the main signal function
                if self.par.signal_type == 'set_frustration_memory_exp_decrease':
                    self.sig.compute_frustration_memory_exp_decrease() # Compute the frustration without modifying frustration_memory
                else:
                    self.sig.set_frustration_memory_exp_decrease()
                cumul_frustration_signal = self.sig.signal.copy()

                self.sig.set_directional_density()
                directional_density_signal = self.sig.signal.copy()

                self.sig.set_local_density()
                local_density_signal = self.sig.signal.copy()
                
            # Apply the parameter selected signal function
            self.sig.function_signal_type()
            self.rev.function_reversal_type()
            # PLOTS
            if i % int(1/self.par.dt*self.par.save_frequency_image) == 0:
                path = self.sample+'/'+str(int(i*self.par.dt/self.par.save_frequency_image))+".png"
                if self.par.plot_movie:
                    self.plo.plotty(path=path, transition_stop=self.transition_stop)

                if self.par.plot_rippling_swarming_color:
                    path_rippling_swarming = path_folder_rippling_swarming+str(int(i*self.par.dt/self.par.save_frequency_image))+".png"
                    self.plo.plotty_rippling_swarming(path=path_rippling_swarming, t=i)

                if self.par.plot_reversing_and_non_reversing:
                    path_reversing_and_non_reversing = path_folder_reversing_and_non_reversing+str(int(i*self.par.dt/self.par.save_frequency_image))+".png"
                    self.plo.plotty_reversing_and_non_reversing(path=path_reversing_and_non_reversing)

                if self.par.plot_colored_nb_neighbors:
                    path_colored_nb_neighbors = path_folder_colored_nb_neighbors+str(int(i*self.par.dt/self.par.save_frequency_image))+".png"
                    self.plo.plotty_colored_nb_neighbors(path=path_colored_nb_neighbors)

            self.cond_rev[:] = self.cond_rev | self.rev.cond_rev
            if (i % int(1/self.par.dt*self.par.save_frequency_csv) == 0) and (i >= int(1/self.par.dt*self.par.save_in_csv_from_time)):
                time = np.ones(self.par.n_bact) * i / int(1/self.par.dt*self.par.save_frequency_csv)
                ids = np.arange(self.par.n_bact)
                if self.par.save_other_reversal_signals:
                    tl.append_to_csv(filename=filename, 
                                    data=np.concatenate((time[np.newaxis, :],
                                                        ids[np.newaxis, :],
                                                        self.gen.data[0, :, :],
                                                        self.gen.data[1, :, :],
                                                        self.cond_rev[np.newaxis, :],
                                                        self.rev.clock_tbr[np.newaxis, :],
                                                        self.sig.signal[np.newaxis, :],
                                                        self.rev.cond_reversing[np.newaxis, :],
                                                        cumul_frustration_signal[np.newaxis, :],
                                                        directional_density_signal[np.newaxis, :],
                                                        local_density_signal[np.newaxis, :],
                                                        ), axis=0).T
                                    )
                else:
                    tl.append_to_csv(filename=filename, 
                                    data=np.concatenate((time[np.newaxis, :],
                                                        ids[np.newaxis, :],
                                                        self.gen.data[0, :, :],
                                                        self.gen.data[1, :, :],
                                                        self.cond_rev[np.newaxis, :],
                                                        self.rev.clock_tbr[np.newaxis, :],
                                                        self.sig.signal[np.newaxis, :],
                                                        self.rev.cond_reversing[np.newaxis, :],
                                                        ), axis=0).T
                                    )
                self.cond_rev[:] = 0

            if self.par.velocity_plot & (i % int(1/self.par.dt*self.par.save_freq_velo) == 0):
                self.velocity_save.append(self.vel.velocity_norm[self.par.node_velocity_measurement])

            # KYMOGRAPH SAVE
            if self.par.kymograph_plot:
                self.kym.build_kymograph_density(index=i, save_kymo=self.par.kymograph_plot)

        # VELOCITY PLOT
        if self.par.velocity_plot:
            self.plo.velocity(velocity_list=self.velocity_save, velocity_max=self.par.v0*2.5, width_bin=0.2)

        # TBR PLOT
        if self.par.tbr_plot and self.par.reversal_type != 'off':
            self.plo.tbr(sample=self.sample, T=self.T)

        if self.par.tbr_cond_space_plot and self.par.reversal_type != 'off':
            self.plo.tbr_cond_space(sample=self.sample, T=self.T)

        # KYMOGRAPH PLOT
        if self.par.kymograph_plot:
            self.plo.kymograph_density(T=self.T, start=0)

        print('SIMULATION '+self.sample+' IS DONE\n')
