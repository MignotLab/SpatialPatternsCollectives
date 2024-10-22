import numpy as np
import pandas as pd

class SignalTypeError(Exception):
    
    pass

class SaveSignalFrustrationTypeError(Exception):
    
    pass

class ReversalSignal:
    """
    Compute the value of the reversal signal.
    signal_type could be: "set_local_frustration", "set_frustration_memory", "mean_local_polarity", "mean_local_polarity_memory", "mean_local_polarity_head", "sum_local_polarity", "sum_local_neg_polarity", "set_local_density", "set_local_nodes_density"; default is "set_frustration_memory".
    
    """

    def __init__(self,inst_par,inst_gen,inst_vel,inst_dir,inst_nei):
        
        # Instance objects
        self.par = inst_par
        self.gen = inst_gen
        self.vel = inst_vel
        self.dir = inst_dir
        self.nei = inst_nei

        if self.par.reversal_type == 'no_reversal':
            self.chosen_signal_fonction = self.function_doing_nothing
        elif self.par.signal_type == 'set_local_frustration':
            self.chosen_signal_fonction = self.set_local_frustration
        elif self.par.signal_type == 'set_frustration_memory':
            self.chosen_signal_fonction = self.set_frustration_memory
        elif self.par.signal_type == 'mean_local_polarity':
            self.chosen_signal_fonction = self.mean_local_polarity
        elif self.par.signal_type == 'mean_local_polarity_memory':
            self.chosen_signal_fonction = self.mean_local_polarity_memory
        elif self.par.signal_type == 'mean_local_polarity_head':
            self.chosen_signal_fonction = self.mean_local_polarity_head
        elif self.par.signal_type == 'sum_local_polarity':
            self.chosen_signal_fonction = self.sum_local_polarity
        elif self.par.signal_type == 'sum_local_neg_polarity':
            self.chosen_signal_fonction = self.sum_local_neg_polarity
        elif self.par.signal_type == 'set_local_density':
            self.chosen_signal_fonction = self.set_local_density
        elif self.par.signal_type == 'set_local_nodes_density':
            self.chosen_signal_fonction = self.set_local_nodes_density
        else:
            # Print the different possible movement when the class is call
            print('signal_type could be: "set_local_frustration", "set_frustration_memory", "mean_local_polarity", "mean_local_polarity_memory", "mean_local_polarity_head", "sum_local_polarity", "sum_local_neg_polarity", "set_local_density", "set_local_nodes_density"; default is "set_frustration_memory"\n')
            raise SignalTypeError()
        
        if self.par.save_frustration == True:
            self.chosen_save_fonction = self.set_save_frustration
        elif self.par.save_frustration == False:
            self.chosen_save_fonction = self.function_doing_nothing
        else:
            print('save_frustration could be: True or False, default is False\n')
            raise SaveSignalFrustrationTypeError()

        # Signal object
        self.signal = np.zeros(self.par.n_bact, dtype=self.par.float_type)

        ### Class object for density signal
        self.array_k = np.ones(self.par.n_bact, dtype=self.par.float_type) * self.par.kn
        self.nb_neighbours = np.zeros(self.par.n_bact, dtype=self.par.float_type)
        ### Class objects for polarity signal
        self.local_polarity = np.zeros(self.par.n_bact, dtype=self.par.float_type)
        self.polarity_memory = np.zeros((self.par.n_bact, int(self.par.time_memory / self.par.dt)), dtype=self.par.float_type)
        ### Class object for frustration
        self.local_frustration = np.zeros(self.par.n_bact, dtype=self.par.float_type)
        self.frustration_memory = np.zeros((self.par.n_bact, int(self.par.time_memory / self.par.dt)), dtype=self.par.float_type)
        # self.save_local_frustration = np.array([])
        self.save_frustration = np.array([])
        # Correction do to the rate to keep a signal between 0 and 1
        tmp = np.zeros(int(self.par.time_memory / self.par.dt), dtype=self.par.float_type)
        # Construct correction array
        for i in range(len(tmp)):
            
            tmp = np.roll(tmp, shift=1)
            tmp[0] = self.par.max_frustration
            tmp *= np.exp(-self.par.rate * self.par.dt)

        self.correction = np.sum(tmp)

    def function_signal_type(self):
        """
        Function doing the movement type and the random movement depending on the choice of these movement in the parameter class
        
        """
        self.chosen_signal_fonction()
        self.chosen_save_fonction()

    def function_doing_nothing(self):
        """
        No reversals

        """
        pass

###################################################################################
################################### FRUSTRATION ###################################
###################################################################################

    def set_local_frustration(self):
        """
        Compute a signal between 0 and 1 due to the frustration measurement
        
        """
        self.signal = np.sum(self.dir.nodes_direction[:,0,:] * self.vel.velocity[:,0,:] / (self.par.v0 * self.par.dt), axis=0)
        self.signal[self.signal > 1] = 1
        self.signal[self.signal < -1] = -1
        self.signal = (self.signal + 1) / 2
        self.signal = np.abs(self.signal - 1)

    def set_frustration_memory(self):
        """
        Compute a signal between 0 and 1 due to the frustration measurement
        
        """
        # Compute the frustration at frame t
        # self.local_frustration = np.sum(self.dir.nodes_direction[:,0,:] * self.vel.velocity[:,0,:] / self.par.v0, axis=0) / (self.par.time_memory/self.par.dt)
        norm_square_vt = np.sum(self.par.v0 * self.dir.nodes_direction[:,0,:] * self.par.v0 * self.dir.nodes_direction[:,0,:], axis=0)
        norm_square_vr = np.sum(self.vel.velocity[:,0,:] * self.vel.velocity[:,0,:], axis=0)
        self.local_frustration = (1 - np.sum(self.par.v0 * self.dir.nodes_direction[:,0,:] * self.vel.velocity[:,0,:], axis=0) / np.maximum(norm_square_vt, norm_square_vr))
        # Rool the frustration memory array to remove the older frustration
        self.frustration_memory = np.roll(self.frustration_memory, shift=1, axis=1)
        # Add the new frustration as the first element
        self.frustration_memory[:,0] = self.local_frustration
        # Decrease exponentially the old frustrations
        self.frustration_memory[:,:] *= np.exp(-self.par.rate * self.par.dt)
        # Compute the new frustration at time t
        self.signal = np.sum(self.frustration_memory, axis=1) / self.correction

    def set_save_frustration(self):
        """
        Save instantaneous frustration and the cumulation of the frustration
        
        """
        # self.save_local_frustration = np.concatenate((self.save_local_frustration,self.local_frustration))
        self.save_frustration = np.concatenate((self.save_frustration,self.signal_frustration))

###################################################################################
#################################### POLARITY #####################################
###################################################################################

    def mean_local_polarity(self):
        """
        
        """
        # Find the index of the bacteria among the id of nodes
        cond_dist =  self.nei.dist > self.par.width
        bacteria_nodes_direction = np.reshape(np.repeat(self.dir.neighbours_direction[:,:,0], self.par.kn), (2,int(self.par.n_bact*self.par.n_nodes),self.par.kn))
        bacteria_nodes_angle = np.arctan2(bacteria_nodes_direction[1,:,:], bacteria_nodes_direction[0,:,:])
        neighbours_nodes_angle = np.arctan2(self.dir.neighbours_direction[1,:,:], self.dir.neighbours_direction[0,:,:])
        polarity = np.cos(bacteria_nodes_angle - neighbours_nodes_angle)
        polarity[self.nei.cond_same_bact | cond_dist] = np.nan
        # Reshape to mean polarity of a same bact
        polarity = np.nanmean(polarity.reshape((self.par.n_bact,int(self.par.n_nodes*self.par.kn)), order='F'), axis=1)
        # Bacteria without neighbour have a nan value, here it is convert to 1
        polarity[np.isnan(polarity)] = 1
        # Transform it between a signal between 0 and 1
        self.signal = 1 - (polarity + 1) / 2

    def mean_local_polarity_memory(self):
        """
        
        """
        # Find the index of the bacteria among the id of nodes
        cond_dist =  self.nei.dist > self.par.width
        bacteria_nodes_direction = np.reshape(np.repeat(self.dir.neighbours_direction[:,:,0], self.par.kn), (2,int(self.par.n_bact*self.par.n_nodes),self.par.kn))
        bacteria_nodes_angle = np.arctan2(bacteria_nodes_direction[1,:,:], bacteria_nodes_direction[0,:,:])
        neighbours_nodes_angle = np.arctan2(self.dir.neighbours_direction[1,:,:], self.dir.neighbours_direction[0,:,:])
        polarity = np.cos(bacteria_nodes_angle - neighbours_nodes_angle)
        polarity[self.nei.cond_same_bact | cond_dist] = np.nan
        # Reshape to mean polarity of a same bact
        polarity = np.nanmean(polarity.reshape((self.par.n_bact,int(self.par.n_nodes*self.par.kn)), order='F'), axis=1)
        # Bacteria without neighbour have a nan value, here it is convert to 1
        polarity[np.isnan(polarity)] = 1
        # Transform it between a signal between 0 and 1
        self.local_polarity = 1 - (polarity + 1) / 2

        # Rool the frustration memory array to remove the older frustration
        self.polarity_memory = np.roll(self.polarity_memory, shift=1, axis=1)
        # Add the new frustration as the first element
        self.polarity_memory[:,0] = self.local_polarity
        # Decrease exponentially the old frustrations
        self.polarity_memory[:,:] *= np.exp(-self.par.rate * self.par.dt)
        # # Compute the new frustration at time t
        # self.signal = np.sum(self.polarity_memory, axis=1) / self.correction
        # Compute the new frustration at time t
        self.signal = np.max(self.polarity_memory, axis=1)

    def mean_local_polarity_head(self):
        """
        
        """
        # Find the index of the bacteria among the id of nodes
        cond_dist =  self.nei.dist > self.par.width
        bacteria_nodes_direction = np.reshape(np.repeat(self.dir.neighbours_direction[:,:,0], self.par.kn), (2,int(self.par.n_bact*self.par.n_nodes),self.par.kn))
        bacteria_nodes_angle = np.arctan2(bacteria_nodes_direction[1,:,:], bacteria_nodes_direction[0,:,:])
        neighbours_nodes_angle = np.arctan2(self.dir.neighbours_direction[1,:,:], self.dir.neighbours_direction[0,:,:])
        polarity = np.cos(bacteria_nodes_angle - neighbours_nodes_angle)
        polarity[self.nei.cond_same_bact | cond_dist] = np.nan
        polarity_head = polarity[:self.par.n_bact,:]
        # Reshape to mean polarity of a same bact
        polarity_head = np.nanmean(polarity_head, axis=1)
        # Bacteria without neighbour have a nan value, here it is convert to 1
        polarity_head[np.isnan(polarity_head)] = 1
        # Transform it between a signal between 0 and 1
        self.signal = 1 - (polarity_head + 1) / 2

    def sum_local_polarity(self):
        """
        
        """
        # Find bacteria that are too 
        cond_dist =  self.nei.dist > self.par.width
        # Create condition to take only one time the same neighbour for one node
        sorted_id_bact = np.sort(self.nei.id_bact, axis=1)
        invert_sorted = np.argsort(np.argsort(self.nei.id_bact, axis=1), axis=1)
        cond_duplicate_neighbour = sorted_id_bact - np.roll(sorted_id_bact, shift=1, axis=1) == 0
        cond_duplicate_neighbour = np.take_along_axis(cond_duplicate_neighbour, invert_sorted, axis=1)
        bacteria_nodes_direction = np.reshape(np.repeat(self.dir.neighbours_direction[:,:,0], self.par.kn), (2,int(self.par.n_bact*self.par.n_nodes),self.par.kn))
        bacteria_nodes_angle = np.arctan2(bacteria_nodes_direction[1,:,:], bacteria_nodes_direction[0,:,:])
        neighbours_nodes_angle = np.arctan2(self.dir.neighbours_direction[1,:,:], self.dir.neighbours_direction[0,:,:])
        polarity = np.cos(bacteria_nodes_angle - neighbours_nodes_angle)
        polarity[self.nei.cond_same_bact | cond_dist | cond_duplicate_neighbour] = np.nan
        # Reshape to mean polarity of a same bact
        self.signal = -np.nansum(polarity.reshape((self.par.n_bact,self.par.n_nodes*self.par.kn), order='F'), axis=1) / self.par.n_nodes

    def sum_local_neg_polarity(self):
        """
        
        """
        # Find bacteria that are too 
        cond_dist =  self.nei.dist > self.par.width
        # Create condition to take only one time the same neighbour for one node
        sorted_id_bact = np.sort(self.nei.id_bact, axis=1)
        invert_sorted = np.argsort(np.argsort(self.nei.id_bact, axis=1), axis=1)
        cond_duplicate_neighbour = sorted_id_bact - np.roll(sorted_id_bact, shift=1, axis=1) == 0
        cond_duplicate_neighbour = np.take_along_axis(cond_duplicate_neighbour, invert_sorted, axis=1)
        bacteria_nodes_direction = np.reshape(np.repeat(self.dir.neighbours_direction[:,:,0], self.par.kn), (2,int(self.par.n_bact*self.par.n_nodes),self.par.kn))
        bacteria_nodes_angle = np.arctan2(bacteria_nodes_direction[1,:,:], bacteria_nodes_direction[0,:,:])
        neighbours_nodes_angle = np.arctan2(self.dir.neighbours_direction[1,:,:], self.dir.neighbours_direction[0,:,:])
        polarity = np.cos(bacteria_nodes_angle - neighbours_nodes_angle)
        polarity[self.nei.cond_same_bact | cond_dist | cond_duplicate_neighbour] = np.nan
        polarity[polarity > 0] = np.nan
        # Reshape to mean polarity of a same bact
        self.signal = -np.nansum(polarity.reshape((self.par.n_bact,self.par.n_nodes*self.par.kn), order='F'), axis=1) / (self.par.n_nodes * self.par.max_signal_sum_neg_polarity)
        # Transform it between a signal between 0 and 1
        self.signal[self.signal > 1] = 1


###################################################################################
##################################### DENSITY #####################################
###################################################################################

    # def set_local_density(self):
    #     """
    #     Measure the local density and transform it into a signal between 0 and 1
        
    #     """
        
    #     # Copy the input
    #     count = self.nei.id_bact.astype(dtype=float)

    #     # Condition for too far bacteria
    #     cond_dist = self.nei.dist > self.par.max_dist_signal_density
    #     count[cond_dist | self.nei.cond_same_bact] = np.nan

    #     # Reshape as (n_bact,n_nodes*k)
    #     count = count.reshape((self.par.n_bact,int(self.par.n_nodes*self.par.kn)),order='F')

    #     count = pd.DataFrame(count)
    #     count = count.apply(lambda x: pd.Series(x.unique()), axis=1)
    #     count = count.values

    #     self.nb_neighbours = np.sum(~np.isnan(count),axis=1)
    #     self.signal = self.nb_neighbours / self.par.max_neighbours_signal_density
    #     self.signal[self.signal > 1] = 1

    def set_local_density(self):
        """
        Measure the local density and transform it into a signal between 0 and 1
        
        """
        count_n = self.nei.id_bact.reshape((self.par.n_bact, int(self.par.n_nodes*self.par.kn)), order='F')
        cond_dist = self.nei.dist < self.par.width * 1.1
        cond_dist = cond_dist.reshape((self.par.n_bact, int(self.par.n_nodes*self.par.kn)), order='F')
        cond_same_bact = self.nei.cond_same_bact.reshape((self.par.n_bact, int(self.par.n_nodes*self.par.kn)), order='F')

        cond_dist[cond_same_bact] = False
        cond_dist = np.sum(cond_dist, axis=1).astype('bool')

        # Compute the difference of bact index
        cond_same_bact = np.take_along_axis(cond_same_bact, count_n.argsort(axis=1), axis=1)
        count_n = np.sort(count_n, axis=1)
        count_n = count_n[:, 1:] - count_n[:, :-1]
        count_n[count_n!=0] = 1
        # Put to zero the difference equal to one for the same bact id
        count_n[cond_same_bact[:, :-1]] = 0
        count_n = np.sum(count_n, axis=1)
        # Add one neighbour for the bacteria with minimum one neighbour
        count_n[cond_dist] += 1
        
        # # Copy the input
        # count = self.nei.id_bact.astype(dtype=float)
        # count = count.reshape((self.par.n_bact,int(self.par.n_nodes*self.par.kn)),order='F')
        # count = np.sort(count, axis=1)
        # count = count[:,1:] - count[:,:-1]
        # count[count!=0] = 1
        self.nb_neighbours = count_n
        self.signal = self.nb_neighbours / self.par.max_neighbours_signal_density
        self.signal[self.signal > 1] = 1

    def set_local_nodes_density(self):
        """
        Measure the local density and transform it into a signal between 0 and 1
        
        """
        # Find bacteria that are too 
        cond_dist =  self.nei.dist < self.par.width
        cond_dist[self.nei.cond_same_bact] = False
        # Reshape correctly the array with order F (Fortran)
        cond_dist = cond_dist.reshape((self.par.n_bact,int(self.par.n_nodes*self.par.kn)),order='F')
        self.nb_neighbours = np.sum(cond_dist, axis=1)
        self.signal = self.nb_neighbours / self.par.max_neighbours_signal_density
        self.signal[self.signal > 1] = 1
