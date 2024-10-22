import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

## .py files import
import alignment
import attraction
import bacteria_generation
import bacteria_movement
import boundaries
import directions
import exopolysaccaride
import kymographe
import neighbourhood
import nodes_attachment
import phantom_data
import plot
import repulsion
import reversal_signal
import reversals
import rigidity
import tools
import velocity_measurement
import viscosity


class Main():


    def __init__(self,inst_par,sample,T):

        ## Parameters
        self.par = inst_par
        self.sample = sample
        self.T = T
        self.velocity_save = []
        self.cond_rev = np.zeros(self.par.n_bact, dtype=int)

        ## Init the data array
        self.gen = bacteria_generation.GenerateBacteria(inst_par=self.par)
        self.gen.function_generation_type()
        ## Data moving in space without the boundary condition
        self.pha = phantom_data.Phantom(inst_gen=self.gen)
        ## Neighbourhood
        self.nei = neighbourhood.Neighbours(inst_par=self.par, inst_gen=self.gen)
        ## Boundaries
        self.bound = boundaries.Boundaries(inst_par=self.par, inst_gen=self.gen)
        ## Directions
        self.dir = directions.Direction(inst_par=self.par, inst_gen=self.gen, inst_pha=self.pha, inst_nei=self.nei)
        ## Initialize the attachment of the nodes
        self.spg = nodes_attachment.BacteriaBody(inst_par=self.par, inst_gen=self.gen, inst_pha=self.pha, inst_dir=self.dir)
        # Rigidity
        self.rig = rigidity.Rigidity(inst_par=self.par, inst_gen=self.gen, inst_pha=self.pha)
        ## Velocity measurement
        self.vel = velocity_measurement.Velocity(inst_par=self.par, inst_pha=self.pha)
        ## Viscosity
        self.visc = viscosity.Viscosity(inst_par=self.par, inst_gen=self.gen, inst_pha=self.pha)
        ## Kymograph of trajectories
        if self.par.kymograph_plot:
            self.kym = kymographe.Kymograph(inst_par=self.par, inst_gen=self.gen,T=self.T)
        else:
            self.kym = None
        # Tools
        self.tool = tools.Tools()

        ### Class needed lots of others classes
        ## Repulsion
        self.rep = repulsion.Repulsion(inst_par=self.par,
                                       inst_gen=self.gen,
                                       inst_pha=self.pha,
                                       inst_dir=self.dir,
                                       inst_nei=self.nei)
        ## Attraction
        self.att = attraction.Attraction(inst_par=self.par,
                                         inst_gen=self.gen,
                                         inst_pha=self.pha,
                                         inst_dir=self.dir,
                                         inst_nei=self.nei)
        ## Alignment
        self.ali = alignment.Alignment(inst_par=self.par,
                                       inst_gen=self.gen,
                                       inst_pha=self.pha,
                                       inst_dir=self.dir,
                                       inst_nei=self.nei,
                                       inst_tool=self.tool)
        ## EPS
        self.eps = exopolysaccaride.Eps(inst_par=self.par,
                                        inst_gen=self.gen,
                                        inst_pha=self.pha,
                                        inst_dir=self.dir,
                                        inst_nei=self.nei,
                                        inst_ali=self.ali,
                                        inst_tool=self.tool)
        ## Reversal signal
        self.sig = reversal_signal.ReversalSignal(inst_par=self.par,
                                                  inst_gen=self.gen,
                                                  inst_vel=self.vel,
                                                  inst_dir=self.dir,
                                                  inst_nei=self.nei)
        ## Reversals
        self.rev = reversals.Reversal(inst_par=self.par,
                                      inst_gen=self.gen,
                                      inst_pha=self.pha,
                                      inst_sig=self.sig)
        ## Movement
        self.move = bacteria_movement.Move(inst_par=self.par,
                                           inst_gen=self.gen,
                                           inst_pha=self.pha,
                                           inst_dir=self.dir,
                                           inst_nei=self.nei,
                                           inst_eps=self.eps,
                                           inst_att=self.att,
                                           inst_vel = self.vel,
                                           inst_tool=self.tool)
        ## Plot
        self.plo = plot.Plot(inst_par=self.par,
                             inst_gen=self.gen,
                             inst_dir=self.dir,
                             inst_rev=self.rev,
                             inst_eps=self.eps,
                             inst_move = self.move,
                             inst_kym = self.kym,
                             sample=self.sample)

        # self.plo.compute_tbr()

    def start(self):

        # REVERSAL FUNCTIONS PLOT
        if self.par.rev_function_plot:
            signal = np.linspace(0, 2*self.par.s1, self.par.n_bact)
            self.sig.signal = signal
            self.rev.function_reversal_type()
            self.plo.reversal_functions(sample=self.sample, signal=signal, refractory_period=self.rev.P, reversal_rate=self.rev.R)
        
        # Create the array of the position of the cells
        # self.gen.function_generation_type()

        ## Iteration over time
        t = int(self.T/self.par.dt)
        velocity_middle = []
        velocity_head = []

        # File to save the coordinates of the simultion
        isExist = os.path.exists(self.sample+'/')
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.sample+'/')
            print("The new directory is created!")
        filename = self.sample+'/coords__'+str(self.par.n_bact)+'_bacts__tbf='+str(int(self.par.save_frequency_csv*60))+'_secondes__space_size='+str(self.par.space_size)+'.csv'
        column_names = ['frame','id']+self.tool.gen_coord_str(n=self.par.n_nodes)+['reversals', 'clock_tbr','cumul_frustration', 'reversing']
        self.tool.initialize_csv(filename=filename, column_names=column_names)

        # Plot
        path = self.sample+'/'+str(int(0*self.par.dt/self.par.save_frequency))+".png"
        if self.par.plot_movie:
            self.plo.plotty(path=path)
        if self.par.plot_rippling_swarming_color:
            path_folder_rippling_swarming = 'rippling_swarming_'+self.sample+'/'
            isExist = os.path.exists(path_folder_rippling_swarming)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(path_folder_rippling_swarming)
                print("The new directory is created!")

        if self.par.plot_reversing_and_non_reversing:
            path_folder_reversing_and_non_reversing = 'reversing_and_non_reversing_'+self.sample+'/'
            isExist = os.path.exists(path_folder_reversing_and_non_reversing)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(path_folder_reversing_and_non_reversing)
                print("The new directory is created!")

        print('SIMULATION '+self.sample+' IS RUNNING\n')
        for i in tqdm(range(t)):

            # BOUNDARIES
            self.bound.periodic()
            # NEIGHBOURS DETECTION
            self.nei.set_kn_nearest_neighbours_torus()
            self.nei.set_bacteria_index()
            # CONDITIONAL SPACE
            self.gen.function_conditional_space()
            # NODES DIRECTIONs
            self.dir.set_nodes_direction()
            # SAVE NODES POSITION IN 
            self.vel.head_position_in()
            # dist, ind = nei.neighbours_scipy_kn_euclidian()
            # x_dir_nodes_to_nei, y_dir_nodes_to_nei, nodes_to_neighbours_distance = dir.nodes_to_neighbours_euclidian_direction(x_nodes=nei.x, y_nodes=nei.y, ind=ind)
            self.dir.set_nodes_to_neighbours_direction_torus()
            self.dir.set_neighbours_direction()
            # MOTILITY
            self.move.function_movement_type()
            # REPULSION
            self.rep.function_repulsion_type()
            # ATTRACTION
            self.att.function_attraction_type()
            # RIGIDITY
            self.rig.function_rigidity_type()
            # SPRINGS ACTION
            self.spg.nodes_spring()
            # ALIGNMENT
            self.ali.function_alignment_type()
            # EPS
            self.eps.function_eps_follower_type()
            # SAVE NODES POSITION OUT AND COMPUTE VELOCITY AND DISPLACEMENT
            self.vel.head_position_out()
            self.vel.displacement_in_out()
            self.vel.velocity_in_out()
            # REVERSALS
            self.sig.function_signal_type()
            self.rev.function_reversal_type()
            # PLOTS
            if i % int(1/self.par.dt*self.par.save_frequency) == 0:
                path = self.sample+'/'+str(int(i*self.par.dt/self.par.save_frequency))+".png"
                # with open(self.sample+'/coord.csv', 'a') as record_append:
                #     # Finish it
                #     np.savetxt(record_append, np.concatenate((self.gen.data[0],self.gen.data[1]),axis=0).T, fmt='%.2f', delimiter=',', header=','.join(self.tool.gen_coord_str(n=self.par.n_nodes)), comments='')
                # self.plo.plotty(data=self.gen.data, path=path, point_size=100, nodes_direction_head=self.dir.nodes_direction[:,0,:],position_focal_adhesion_point=None,eps_grid=None)
                if self.par.plot_movie:
                    self.plo.plotty(path=path)

                if self.par.plot_rippling_swarming_color:
                    path_rippling_swarming = path_folder_rippling_swarming+str(int(i*self.par.dt/self.par.save_frequency))+".png"
                    self.plo.plotty_rippling_swarming(path=path_rippling_swarming, t=i)

                if self.par.plot_reversing_and_non_reversing:
                    path_reversing_and_non_reversing = path_folder_reversing_and_non_reversing+str(int(i*self.par.dt/self.par.save_frequency))+".png"
                    self.plo.plotty_reversing_and_non_reversing(path=path_reversing_and_non_reversing)

            self.cond_rev[:] = self.cond_rev | self.rev.cond_rev
            if i % int(1/self.par.dt*self.par.save_frequency_csv) == 0:
                time = np.ones(self.par.n_bact) * i / int(1/self.par.dt*self.par.save_frequency_csv)
                ids = np.arange(self.par.n_bact)
                self.tool.append_to_csv(filename=filename, 
                                        data=np.concatenate((time[np.newaxis, :],
                                                             ids[np.newaxis, :],
                                                             self.gen.data[0, :, :],
                                                             self.gen.data[1, :, :],
                                                             self.cond_rev[np.newaxis, :],
                                                             self.rev.clock_tbr[np.newaxis, :],
                                                             self.sig.signal[np.newaxis, :],
                                                             self.rev.cond_reversing[np.newaxis, :],
                                                             ), axis=0).T)
                self.cond_rev[:] = 0

            if self.par.velocity_plot & (i % int(1/self.par.dt*self.par.save_freq_velo) == 0):
                self.velocity_save.append(self.vel.velocity_norm[self.par.node_velocity_measurement])
                # path_rev = "sample_rev/"+str(int(i*dt/save_frequency))+".png"
                # plo.plotty_rev(x=rev_p.rev_to_plot_x, y=rev_p.rev_to_plot_y, path=path_rev, point_size=50, fig=fig_rev,ax=ax_rev)
                # rev_p.initialize_save_reversals()

                # path_eps = "sample_eps/"+str(int(i*dt/self.par.save_frequency))+".png"
                # plo.plotty_eps(path=path_eps)

            # KYMOGRAPH SAVE
            if self.par.kymograph_plot:
                self.kym.build_kymograph_density(index=i, save_kymo=self.par.kymograph_plot)

        # VELOCITY PLOT
        if self.par.velocity_plot:
            self.plo.velocity(velocity_list=self.velocity_save, velocity_max=self.par.v0*2.5, width_bin=0.2)

        # TBR PLOT
        if self.par.tbr_plot:
            self.plo.tbr(sample=self.sample, T=self.T)

        if self.par.tbr_cond_space_plot:
            self.plo.tbr_cond_space(sample=self.sample, T=self.T)

        # KYMOGRAPH PLOT
        if self.par.kymograph_plot:
            self.plo.kymograph_density(T=self.T, start=0)

        print('SIMULATION '+self.sample+' IS DONE\n')
