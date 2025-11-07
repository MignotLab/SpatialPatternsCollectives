"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-06
"""
import numpy as np


class SignalTypeError(Exception):
    """
    Custom exception raised when an invalid signal type is provided.
    """
    pass


class SaveSignalFrustrationTypeError(Exception):
    """
    Custom exception raised when an invalid value for save_frustration is provided.
    """
    pass


class ReversalSignal:
    """
    Class to compute the value of the reversal signal for bacteria, based on various signal types.
    
    The `signal_type` parameter can be one of the following options:
    - "set_local_frustration": Local frustration value.
    - "set_frustration_memory": Memory-based frustration value (default).
    - "mean_local_polarity": Mean polarity of local neighborhood.
    - "mean_local_polarity_memory": Mean polarity with memory.
    - "mean_local_polarity_head": Mean polarity near the head of the bacterium.
    - "sum_local_polarity": Sum of local polarity.
    - "sum_local_neg_polarity": Sum of negative local polarity.
    - "set_local_density": Local density around the bacterium.
    - "set_local_nodes_density": Local node density.
    - "set_local_nodes_density_memory": Local node density with memory.
    - "set_local_nodes_density_head": Local node density near the head of the bacterium.
    
    The class also handles saving the computed frustration signal if the `save_frustration` parameter is set to True.
    
    Attributes:
    -----------
    par : object
        Instance of the Parameters class containing simulation settings.
    gen : object
        Instance of the class managing general data and behaviors in the simulation.
    vel : object
        Instance of the class managing velocity-related data.
    dir : object
        Instance of the class managing direction-related data.
    nei : object
        Instance of the class managing neighbor-related data.
    chosen_signal_function : method
        Selected function for calculating the reversal signal, depending on the signal_type parameter.
    chosen_save_function : method
        Selected function for saving the frustration signal.
    signal : numpy.ndarray
        Array holding the computed signal values for each bacterium.
    array_k : numpy.ndarray
        Array for storing local density information for each bacterium.
    nb_neighbors : numpy.ndarray
        Array for storing the number of neighbors around each bacterium.
    local_polarity : numpy.ndarray
        Array for storing the polarity of each bacterium.
    polarity_memory : numpy.ndarray
        Array for storing memory of the polarity of each bacterium.
    local_frustration : numpy.ndarray
        Array for storing local frustration values.
    frustration_memory : numpy.ndarray
        Array for storing memory of frustration values.
    save_frustration : numpy.ndarray
        Array for storing the frustration values that are saved during the simulation.
    correction : float
        Correction factor used to maintain the signal between 0 and 1, calculated based on the frustration memory.
    """
    def __init__(self, inst_par, inst_gen, inst_vel, inst_dir, inst_nei):
        # Instance objects: Assign the passed-in instances to class attributes.
        self.par = inst_par
        self.gen = inst_gen
        self.vel = inst_vel
        self.dir = inst_dir
        self.nei = inst_nei

        # Choose the appropriate signal calculation function based on the signal_type parameter.
        if self.par.reversal_type == 'off':
            self.chosen_signal_function = self.function_doing_nothing
        elif self.par.signal_type == 'set_local_frustration':
            self.chosen_signal_function = self.set_local_frustration
        elif self.par.signal_type == 'set_frustration_memory':
            self.chosen_signal_function = self.set_frustration_memory
        elif self.par.signal_type == 'set_frustration_memory_exp_decrease':
            self.chosen_signal_function = self.set_frustration_memory_exp_decrease
        elif self.par.signal_type == 'mean_local_polarity':
            self.chosen_signal_function = self.mean_local_polarity
        elif self.par.signal_type == 'mean_local_polarity_memory':
            self.chosen_signal_function = self.mean_local_polarity_memory
        elif self.par.signal_type == 'mean_local_polarity_head':
            self.chosen_signal_function = self.mean_local_polarity_head
        elif self.par.signal_type == 'sum_local_polarity':
            self.chosen_signal_function = self.sum_local_polarity
        elif self.par.signal_type == 'sum_local_neg_polarity':
            self.chosen_signal_function = self.sum_local_neg_polarity
        elif self.par.signal_type == 'set_local_density':
            self.chosen_signal_function = self.set_local_density
        elif self.par.signal_type == 'set_local_nodes_density':
            self.chosen_signal_function = self.set_local_nodes_density
        elif self.par.signal_type == 'set_directional_density':
            self.chosen_signal_function = self.set_directional_density
        elif self.par.signal_type == 'set_directional_nodes_density':
            self.chosen_signal_function = self.set_directional_nodes_density
        
        else:
            # Raise an error if an invalid signal_type is provided.
            print("""signal_type could be:
                - set_local_frustration
                - set_frustration_memory
                - set_local_density
                - set_local_nodes_density
                - mean_local_polarity
                - mean_local_polarity_memory
                - mean_local_polarity_head
                - sum_local_polarity
                - sum_local_neg_polarity
                - set_directional_density
                - set_directional_nodes_density"""
                  )
            raise SignalTypeError()
        
        # Choose the appropriate save function for storing the frustration signal.
        if self.par.save_frustration == True:
            self.chosen_save_function = self.set_save_frustration
        elif self.par.save_frustration == False:
            self.chosen_save_function = self.function_doing_nothing
        else:
            # Raise an error if an invalid value for save_frustration is provided.
            print('save_frustration could be: True or False, default is False\n')
            raise SaveSignalFrustrationTypeError()

        # Initialize signal array and other attributes.
        self.signal = np.zeros(self.par.n_bact, dtype=self.par.float_type)
        self.array_k = np.ones(self.par.n_bact, dtype=self.par.float_type) * self.par.kn
        self.nb_neighbors = np.zeros(self.par.n_bact, dtype=self.par.float_type)
        self.local_polarity = np.zeros(self.par.n_bact, dtype=self.par.float_type)
        self.polarity_memory = np.zeros((self.par.n_bact, int(self.par.time_memory / self.par.dt)), dtype=self.par.float_type)
        self.local_frustration = np.zeros(self.par.n_bact, dtype=self.par.float_type)
        self.frustration_memory = np.zeros((self.par.n_bact, int(self.par.time_memory / self.par.dt)), dtype=self.par.float_type)
        self.frustration_memory_exp_decay = np.zeros((self.par.n_bact, int(self.par.time_memory / self.par.dt)), dtype=self.par.float_type)
        self.save_frustration = np.array([])

        # Correction factor to ensure signal stays between 0 and 1.
        tmp = np.zeros(int(self.par.time_memory / self.par.dt), dtype=self.par.float_type)
        for i in range(len(tmp)):
            tmp = np.roll(tmp, shift=1)
            tmp[0] = self.par.max_frustration
            tmp *= np.exp(-self.par.rate * self.par.dt)

        self.correction = np.sum(tmp)
        self.nb_neighbors = np.zeros(self.par.n_bact, dtype=self.par.float_type)


    def function_signal_type(self):
        """
        Executes the selected movement and signal functions based on the signal type and save settings.
        
        This function calls:
        - chosen_signal_function: The function that computes the reversal signal (e.g., local frustration, polarity).
        - chosen_save_function: The function that handles saving the frustration signal.
        """
        self.chosen_signal_function()  # Call the function to compute the signal (e.g., frustration, polarity).
        self.chosen_save_function()    # Call the function to save the computed frustration signal (if enabled).


    def function_doing_nothing(self):
        """
        A placeholder function that does nothing, used when no reversal signal is applied.
        """
        pass


###################################################################################
################################### FRUSTRATION ###################################
###################################################################################


    def set_local_frustration(self):
        """
        Compute a signal between 0 and 1 based on the frustration measurement.

        The signal is calculated as the normalized frustration between bacteria based on their velocity and direction.
        - If the calculated signal is greater than 1, it is capped at 1.
        - If the calculated signal is less than -1, it is capped at -1.
        - The signal is then rescaled to be between 0 and 1, and the absolute value of (signal - 1) is taken.

        The result represents the frustration level of each bacterium.
        (experimental)
        """
        self.signal = np.sum(self.dir.nodes_direction[:, 0, :] * self.vel.velocity[:, 0, :] / (self.par.v0 * self.par.dt), axis=0)
        self.signal[self.signal > 1] = 1  # Cap signal to 1
        self.signal[self.signal < -1] = -1  # Cap signal to -1
        self.signal = (self.signal + 1) / 2  # Rescale to range [0, 1]
        self.signal = np.abs(self.signal - 1)  # Take the absolute difference from 1.


    def set_frustration_memory_exp_decrease(self):
        """
        Compute a signal between 0 and 1 based on the frustration measurement with an exponential memory decay.

        The function:
        - Calculates the current frustration using the dot product of the velocity and direction vectors.
        - Computes the frustration memory over time with an exponential decay, where the older frustrations are reduced by the rate `par.rate`.
        - The signal is normalized based on the total accumulated frustration memory and a correction factor.
        """
        norm_square_vt = np.sum(self.par.v0 * self.dir.nodes_direction[:, 0, :] * self.par.v0 * self.dir.nodes_direction[:, 0, :], axis=0)
        norm_square_vr = np.sum(self.vel.velocity[:, 0, :] * self.vel.velocity[:, 0, :], axis=0)
        self.local_frustration = (1 - np.sum(self.par.v0 * self.dir.nodes_direction[:, 0, :] * self.vel.velocity[:, 0, :], axis=0) / np.maximum(norm_square_vt, norm_square_vr))
        
        # Roll the frustration memory array to remove the oldest frustration values
        self.frustration_memory_exp_decay = np.roll(self.frustration_memory_exp_decay, shift=1, axis=1)
        
        # Add the current frustration as the first element in the memory
        self.frustration_memory_exp_decay[:, 0] = self.local_frustration
        
        # Apply exponential decay to the memory values
        self.frustration_memory_exp_decay[:, :] *= np.exp(-self.par.rate * self.par.dt)
        
        # Compute the final signal as the sum of the memory values, normalized by the correction factor
        self.signal = np.sum(self.frustration_memory_exp_decay, axis=1) / self.correction


    def compute_frustration_memory_exp_decrease(self):
        """
        Compute the frustration signal with exponential memory decay without altering the internal memory state.
        Used for diagnostic or preview purposes.
        """
        # Compute current local frustration
        norm_square_vt = np.sum(self.par.v0 * self.dir.nodes_direction[:, 0, :] * self.par.v0 * self.dir.nodes_direction[:, 0, :], axis=0)
        norm_square_vr = np.sum(self.vel.velocity[:, 0, :] * self.vel.velocity[:, 0, :], axis=0)
        local_frustration = (1 - np.sum(self.par.v0 * self.dir.nodes_direction[:, 0, :] * self.vel.velocity[:, 0, :], axis=0)
                            / np.maximum(norm_square_vt, norm_square_vr))

        # Copy memory and simulate update
        memory_copy = np.roll(self.frustration_memory_exp_decay.copy(), shift=1, axis=1)
        memory_copy[:, 0] = local_frustration
        memory_copy *= np.exp(-self.par.rate * self.par.dt)

        return np.sum(memory_copy, axis=1) / self.correction


    def set_frustration_memory(self):
        """
        Compute a signal between 0 and 1 based on the frustration measurement with memory.

        The function:
        - Calculates the current frustration using the dot product of the velocity and direction vectors.
        - Adds the current frustration to the memory array, which is rolled to remove the oldest values.
        - The signal is then computed as the mean of the frustration values in memory.
        """
        norm_square_vt = np.sum(self.par.v0 * self.dir.nodes_direction[:, 0, :] * self.par.v0 * self.dir.nodes_direction[:, 0, :], axis=0)
        norm_square_vr = np.sum(self.vel.velocity[:, 0, :] * self.vel.velocity[:, 0, :], axis=0)
        self.local_frustration = (1 - np.sum(self.par.v0 * self.dir.nodes_direction[:, 0, :] * self.vel.velocity[:, 0, :], axis=0) / np.maximum(norm_square_vt, norm_square_vr))
        
        # Roll the frustration memory array to remove the oldest frustration values
        self.frustration_memory = np.roll(self.frustration_memory, shift=1, axis=1)
        
        # Add the current frustration as the first element in the memory
        self.frustration_memory[:, 0] = self.local_frustration
        
        # Compute the final signal as the mean of the frustration memory
        self.signal = np.mean(self.frustration_memory, axis=1)


    def set_save_frustration(self):
        """
        Save the instantaneous frustration signal and its cumulative values.

        The function:
        - Adds the current frustration signal to the save_frustration array, which keeps track of the frustration signals over time.
        """
        # Concatenate the current frustration signal to the saved array
        self.save_frustration = np.concatenate((self.save_frustration, self.signal_frustration))


###################################################################################
################################### POLARITY ######################################
###################################################################################


    def mean_local_polarity(self):
        """
        Compute the mean local polarity of each bacterium based on its neighbours.
        """
        # Condition to check if the distance between neighbours is greater than the threshold width
        cond_dist = self.nei.dist > self.par.width * self.par.neighbor_distance_factor
        
        # Reshape the neighbour direction vectors for bacteria and calculate their angles
        bacteria_nodes_direction = np.reshape(np.repeat(self.dir.neighbours_direction[:,:,0], self.par.kn), (2, int(self.par.n_bact * self.par.n_nodes), self.par.kn))
        bacteria_nodes_angle = np.arctan2(bacteria_nodes_direction[1, :, :], bacteria_nodes_direction[0, :, :])
        neighbours_nodes_angle = np.arctan2(self.dir.neighbours_direction[1, :, :], self.dir.neighbours_direction[0, :, :])
        
        # Calculate the polarity as the cosine of the angle difference between bacteria and their neighbours
        polarity = np.cos(bacteria_nodes_angle - neighbours_nodes_angle)
        
        # Set polarity to NaN where bacteria are the same or the distance is too large
        polarity[self.nei.cond_same_bact | cond_dist] = np.nan
        
        # Reshape and compute the mean polarity for each bacterium
        polarity = np.nanmean(polarity.reshape((self.par.n_bact, int(self.par.n_nodes * self.par.kn)), order='F'), axis=1)
        
        # Bacteria without neighbours have a NaN value, here it is converted to 1
        polarity[np.isnan(polarity)] = 1
        
        # Transform the polarity into a signal between 0 and 1
        self.signal = 1 - (polarity + 1) / 2


    def mean_local_polarity_memory(self):
        """
        Compute the mean local polarity of each bacterium, accounting for memory effects over time.
        """
        # Condition to check if the distance between neighbours is greater than the threshold width
        cond_dist = self.nei.dist > self.par.width * self.par.neighbor_distance_factor
        
        # Reshape the neighbour direction vectors for bacteria and calculate their angles
        bacteria_nodes_direction = np.reshape(np.repeat(self.dir.neighbours_direction[:,:,0], self.par.kn), (2, int(self.par.n_bact * self.par.n_nodes), self.par.kn))
        bacteria_nodes_angle = np.arctan2(bacteria_nodes_direction[1, :, :], bacteria_nodes_direction[0, :, :])
        neighbours_nodes_angle = np.arctan2(self.dir.neighbours_direction[1, :, :], self.dir.neighbours_direction[0, :, :])
        
        # Calculate the polarity as the cosine of the angle difference between bacteria and their neighbours
        polarity = np.cos(bacteria_nodes_angle - neighbours_nodes_angle)
        
        # Set polarity to NaN where bacteria are the same or the distance is too large
        polarity[self.nei.cond_same_bact | cond_dist] = np.nan
        
        # Reshape and compute the mean polarity for each bacterium
        polarity = np.nanmean(polarity.reshape((self.par.n_bact, int(self.par.n_nodes * self.par.kn)), order='F'), axis=1)
        
        # Bacteria without neighbours have a NaN value, here it is converted to 1
        polarity[np.isnan(polarity)] = 1
        
        # Transform the polarity into a signal between 0 and 1
        self.local_polarity = 1 - (polarity + 1) / 2

        # Shift the polarity memory array to remove the oldest polarity values
        self.polarity_memory = np.roll(self.polarity_memory, shift=1, axis=1)
        
        # Insert the new polarity as the first element of the memory
        self.polarity_memory[:, 0] = self.local_polarity
        
        # Apply exponential decay to old polarity values in memory
        self.polarity_memory[:, :] *= np.exp(-self.par.rate * self.par.dt)
        
        # Compute the new signal as the maximum polarity value from the memory
        self.signal = np.max(self.polarity_memory, axis=1)


    def mean_local_polarity_head(self):
        """
        Compute the mean local polarity for the head of each bacterium.
        """
        # Condition to check if the distance between neighbours is greater than the threshold width
        cond_dist = self.nei.dist > self.par.width * self.par.neighbor_distance_factor
        
        # Reshape the neighbour direction vectors for bacteria and calculate their angles
        bacteria_nodes_direction = np.reshape(np.repeat(self.dir.neighbours_direction[:,:,0], self.par.kn), (2, int(self.par.n_bact * self.par.n_nodes), self.par.kn))
        bacteria_nodes_angle = np.arctan2(bacteria_nodes_direction[1, :, :], bacteria_nodes_direction[0, :, :])
        neighbours_nodes_angle = np.arctan2(self.dir.neighbours_direction[1, :, :], self.dir.neighbours_direction[0, :, :])
        
        # Calculate the polarity as the cosine of the angle difference between bacteria and their neighbours
        polarity = np.cos(bacteria_nodes_angle - neighbours_nodes_angle)
        
        # Set polarity to NaN where bacteria are the same or the distance is too large
        polarity[self.nei.cond_same_bact | cond_dist] = np.nan
        
        # Extract polarity values for the head of the bacteria (first n_bact entries)
        polarity_head = polarity[:self.par.n_bact, :]
        
        # Reshape and compute the mean polarity for each bacterium at the head
        polarity_head = np.nanmean(polarity_head, axis=1)
        
        # Bacteria without neighbours have a NaN value, here it is converted to 1
        polarity_head[np.isnan(polarity_head)] = 1
        
        # Transform the polarity into a signal between 0 and 1
        self.signal = 1 - (polarity_head + 1) / 2


    def sum_local_polarity(self):
        """
        Compute the sum of the local polarity for each bacterium.
        """
        # Condition to check if the distance between neighbours is greater than the threshold width
        cond_dist = self.nei.dist > self.par.width * self.par.neighbor_distance_factor
        
        # Create a condition to ensure that each neighbour is considered only once for a node
        sorted_id_bact = np.sort(self.nei.id_bact, axis=1)
        invert_sorted = np.argsort(np.argsort(self.nei.id_bact, axis=1), axis=1)
        cond_duplicate_neighbour = sorted_id_bact - np.roll(sorted_id_bact, shift=1, axis=1) == 0
        cond_duplicate_neighbour = np.take_along_axis(cond_duplicate_neighbour, invert_sorted, axis=1)
        
        # Reshape the neighbour direction vectors for bacteria and calculate their angles
        bacteria_nodes_direction = np.reshape(np.repeat(self.dir.neighbours_direction[:,:,0], self.par.kn), (2, int(self.par.n_bact * self.par.n_nodes), self.par.kn))
        bacteria_nodes_angle = np.arctan2(bacteria_nodes_direction[1, :, :], bacteria_nodes_direction[0, :, :])
        neighbours_nodes_angle = np.arctan2(self.dir.neighbours_direction[1, :, :], self.dir.neighbours_direction[0, :, :])
        
        # Calculate the polarity as the cosine of the angle difference between bacteria and their neighbours
        polarity = np.cos(bacteria_nodes_angle - neighbours_nodes_angle)
        
        # Set polarity to NaN where bacteria are the same, the distance is too large, or there are duplicate neighbours
        polarity[self.nei.cond_same_bact | cond_dist | cond_duplicate_neighbour] = np.nan
        
        # Reshape and compute the sum of the polarity for each bacterium
        self.signal = -np.nansum(polarity.reshape((self.par.n_bact, self.par.n_nodes * self.par.kn), order='F'), axis=1) / self.par.n_nodes


    def sum_local_neg_polarity(self):
        """
        Compute the sum of the local negative polarity for each bacterium, considering the effect of its neighbours.
        Only the negative polarity (polarity < 0) is summed.
        """
        # Condition to check if the distance between neighbours is greater than the threshold width
        cond_dist = self.nei.dist > self.par.width * self.par.neighbor_distance_factor
        
        # Create a condition to ensure that each neighbour is considered only once for a node
        sorted_id_bact = np.sort(self.nei.id_bact, axis=1)
        invert_sorted = np.argsort(np.argsort(self.nei.id_bact, axis=1), axis=1)
        cond_duplicate_neighbour = sorted_id_bact - np.roll(sorted_id_bact, shift=1, axis=1) == 0
        cond_duplicate_neighbour = np.take_along_axis(cond_duplicate_neighbour, invert_sorted, axis=1)
        
        # Reshape the neighbour direction vectors for bacteria and calculate their angles
        bacteria_nodes_direction = np.reshape(np.repeat(self.dir.neighbours_direction[:,:,0], self.par.kn), (2, int(self.par.n_bact * self.par.n_nodes), self.par.kn))
        bacteria_nodes_angle = np.arctan2(bacteria_nodes_direction[1, :, :], bacteria_nodes_direction[0, :, :])
        neighbours_nodes_angle = np.arctan2(self.dir.neighbours_direction[1, :, :], self.dir.neighbours_direction[0, :, :])
        
        # Calculate the polarity as the cosine of the angle difference between bacteria and their neighbours
        polarity = np.cos(bacteria_nodes_angle - neighbours_nodes_angle)
        
        # Set polarity to NaN where bacteria are the same, the distance is too large, or there are duplicate neighbours
        polarity[self.nei.cond_same_bact | cond_dist | cond_duplicate_neighbour] = np.nan
        
        # Set polarity to NaN where polarity is positive (we only care about negative polarity)
        polarity[polarity > 0] = np.nan
        
        # Reshape and compute the sum of the negative polarity for each bacterium
        self.signal = -np.nansum(polarity.reshape((self.par.n_bact, self.par.n_nodes * self.par.kn), order='F'), axis=1) / (self.par.n_nodes * self.par.max_signal_sum_neg_polarity)
        
        # Ensure that the signal is between 0 and 1
        self.signal[self.signal > 1] = 1


###################################################################################
##################################### DENSITY #####################################
###################################################################################


    # def set_local_density(self):
    #     """
    #     Measure the local density of bacteria and transform it into a signal between 0 and 1.
    #     The signal represents the number of neighbouring bacteria within a defined distance.
    #     """
    #     # Reshape the bacteria indices to count the number of neighbours for each node
    #     count_n = self.nei.id_bact.reshape((self.par.n_bact, int(self.par.n_nodes * self.par.kn)), order='F')
        
    #     # Define the condition for a neighbour within a certain distance threshold
    #     cond_dist = self.nei.dist < self.par.width * self.par.neighbor_distance_factor
    #     cond_dist = cond_dist.reshape((self.par.n_bact, int(self.par.n_nodes * self.par.kn)), order='F')
        
    #     # Condition to check if the bacteria are the same
    #     cond_same_bact = self.nei.cond_same_bact.reshape((self.par.n_bact, int(self.par.n_nodes * self.par.kn)), order='F')

    #     # Exclude the condition where the bacteria are the same
    #     cond_dist[cond_same_bact] = False
        
    #     # Count the number of neighbours for each bacterium
    #     cond_dist = np.sum(cond_dist, axis=1).astype('bool')

    #     # Compute the difference of bacteria indices to avoid counting the same bacterium twice
    #     cond_same_bact = np.take_along_axis(cond_same_bact, count_n.argsort(axis=1), axis=1)
    #     count_n = np.sort(count_n, axis=1)
    #     count_n = count_n[:, 1:] - count_n[:, :-1]
    #     count_n[count_n != 0] = 1
        
    #     # Set the difference equal to one for the same bacterium ID to zero
    #     count_n[cond_same_bact[:, :-1]] = 0
        
    #     # Sum the number of unique neighbours
    #     count_n = np.sum(count_n, axis=1)
        
    #     # Add one neighbour for bacteria with at least one neighbour
    #     count_n[cond_dist] += 1
        
    #     # Store the number of neighbours
    #     self.nb_neighbors = count_n
        
    #     # Compute the local density signal and ensure it stays between 0 and 1
    #     self.signal = self.nb_neighbors / self.par.max_neighbours_signal_density
    #     self.signal[self.signal > 1] = 1


    def _first_occurrence(self, row):
        """
        Get a boolean array indicating the first occurrences of elements in a given row.

        Parameters:
        - row (numpy.ndarray): One-dimensional array representing a row in the 2D array.

        Returns:
        - numpy.ndarray: Boolean array of the same length as the input row, 
        where True indicates the first occurrence of each unique element.
        """
        # Get unique elements and their indices of the first occurrences
        unique_elements, first_occurrence_indices = np.unique(row, return_index=True)
        
        # Create a boolean array with False, then set True at indices of the first occurrences
        result = np.full_like(row, fill_value=False, dtype=bool)
        result[first_occurrence_indices] = True
        
        return result


    def set_local_density(self):
        """
        Compute the local density of each bacterium by counting the number of unique neighbors 
        within a specified distance threshold. The density is normalized between 0 and 1.
        """
        # Flatten neighbor id array
        id_bact = self.nei.id_bact.reshape((self.par.n_bact, self.par.n_nodes * self.par.kn), order='F')
        cond_dist = (self.nei.dist < self.par.width * self.par.neighbor_distance_factor).reshape(id_bact.shape, order='F')
        cond_same = self.nei.cond_same_bact.reshape(id_bact.shape, order='F')

        # Mask self and distant neighbors
        id_bact_masked = np.where(cond_dist & ~cond_same, id_bact, np.nan)

        # First occurrence mask
        cond_unique = np.apply_along_axis(self._first_occurrence, axis=1, arr=id_bact_masked)

        # Count unique neighbors (excluding self)
        nb_neighbors = np.sum(cond_unique, axis=1) - 1

        # Normalize to [0, 1]
        self.nb_neighbors[:] = nb_neighbors
        signal = nb_neighbors / self.par.max_neighbours_signal_density
        signal[signal > 1] = 1
        self.signal = signal


    def set_local_nodes_density(self):
        """
        Count the number of unique neighbouring nodes (from other bacteria) per focal bacterium,
        within a given distance threshold. A neighbour node is counted only once per focal bacterium,
        even if it is adjacent to multiple nodes of that bacterium.

        The result is normalized into a [0, 1] signal using max_neighbours_signal_density.

        Sets
        -----
        - self.nb_neighbors : np.ndarray of shape (n_bact,)
            Number of unique neighbouring nodes (excluding self and same bacterium).
        - self.signal : np.ndarray of shape (n_bact,)
            Normalized density signal, clipped to 1.
        """
        n_bact = self.par.n_bact
        n_nodes = self.par.n_nodes
        kn = self.par.kn
        n_total_nodes = int(n_bact * n_nodes)

        # Boolean mask of valid neighbors (not same bacterium and within distance)
        cond_valid = (self.nei.dist < self.par.width * self.par.neighbor_distance_factor) & (~self.nei.cond_same_bact)
        cond_valid_flat = cond_valid.flatten(order='C')  # shape: (n_total_nodes * kn,)

        # Get global node IDs of valid neighbor nodes
        valid_nei_ids = self.nei.ind.flatten(order='C')[cond_valid_flat]  # shape: (?,)

        # Focal node IDs (from which we computed neighbors), mapped to bacterium ID
        focal_node_ids = np.repeat(np.arange(n_total_nodes), kn)[cond_valid_flat]
        focal_bact_ids = focal_node_ids % n_bact  # shape: (?,)

        # Encode pairs (focal bacterium, neighbor node) and extract unique
        pair_ids = focal_bact_ids * n_total_nodes + valid_nei_ids
        unique_pairs = np.unique(pair_ids)
        unique_focal_ids = unique_pairs // n_total_nodes

        # Count number of unique neighbor nodes per bacterium
        nb_neighbors = np.bincount(unique_focal_ids, minlength=n_bact)
        self.nb_neighbors[:] = nb_neighbors

        # Normalize to [0, 1]
        signal = nb_neighbors / self.par.max_neighbours_signal_density
        signal[signal > 1] = 1
        self.signal = signal


    def set_directional_density(self):
        """
        Compute the number of unique neighboring bacteria whose mean orientation 
        is opposite (cosine similarity < 0) to the focal bacterium.

        The orientation of each bacterium is computed as the average direction of its skeleton nodes.
        The polarity is then defined as the cosine of the difference between the mean angles of 
        the focal bacterium and its neighbors.

        Sets
        -----
        self.nb_neighbors : np.ndarray of shape (n_bact,)
            Number of distinct neighboring bacteria with opposite mean orientation.
        self.signal : np.ndarray of shape (n_bact,)
            Normalized signal based on the number of opposite neighbors (clipped to [0, 1]).
        """
        n_bact = self.par.n_bact
        n_nodes = self.par.n_nodes
        kn = self.par.kn

        # Reshape neighbor arrays to match bacterium-wise structure
        id_bact = self.nei.id_bact.reshape((n_bact, n_nodes * kn), order='F')
        cond_dist = (self.nei.dist < self.par.width * self.par.neighbor_distance_factor).reshape((n_bact, n_nodes * kn), order='F')
        cond_same = self.nei.cond_same_bact.reshape((n_bact, n_nodes * kn), order='F')
        valid = cond_dist & ~cond_same

        # Compute mean orientation angle for each bacterium
        mean_vec = np.mean(self.dir.nodes_direction, axis=1)  # shape: (2, n_bact)
        mean_angles = np.arctan2(mean_vec[1], mean_vec[0])    # shape: (n_bact,)

        # Retrieve neighbor angles and broadcast self angles for comparison
        neighbor_angles = mean_angles[id_bact]  # shape: (n_bact, n_nodes * kn)
        self_angles = np.repeat(mean_angles[:, np.newaxis], n_nodes * kn, axis=1)  # shape: (n_bact, n_nodes * kn)

        # Compute cosine similarity between focal and neighbor angles
        polarity = np.cos(self_angles - neighbor_angles)
        polarity[~valid] = np.nan  # Invalidate non-neighboring or self pairs

        # Identify first valid occurrence of each neighbor per bacterium
        id_bact_with_nan = np.where(valid, id_bact, np.nan)
        cond_unique = np.apply_along_axis(self._first_occurrence, axis=1, arr=id_bact_with_nan)

        # Extract negative polarity neighbors (cos(Δθ) < 0)
        is_negative = (polarity < 0) & cond_unique & ~np.isnan(id_bact_with_nan)
        id_bact_neg = np.where(is_negative, id_bact_with_nan, -1).astype(int)

        # Count unique negatively aligned neighbors per bacterium
        row_ids = np.repeat(np.arange(n_bact), n_nodes * kn)
        pair_ids = row_ids * n_bact + id_bact_neg.ravel()
        valid_pair_ids = pair_ids[id_bact_neg.ravel() != -1]
        unique_pairs = np.unique(valid_pair_ids)
        unique_bact_ids = unique_pairs // n_bact
        nb_opposite = np.bincount(unique_bact_ids, minlength=n_bact)

        # Store final neighbor count and normalized signal
        self.nb_neighbors[:] = nb_opposite
        signal = nb_opposite / self.par.max_neighbours_signal_density
        signal[signal > 1] = 1
        self.signal = signal


    def set_directional_nodes_density(self):
        """
        Count the number of unique neighbouring nodes (from other bacteria) whose orientation 
        is opposite (cosine similarity < 0) to at least one node of the focal bacterium.

        Each neighbour node is counted at most once per focal bacterium, even if multiple focal nodes 
        detect it as opposite-oriented.

        Sets
        -----
        - self.nb_neighbors : np.ndarray of shape (n_bact,)
            Number of unique opposite-oriented neighbouring nodes per bacterium.
        - self.signal : np.ndarray of shape (n_bact,)
            Normalized signal (clipped to [0, 1]) based on nb_neighbors.
        """
        n_bact = self.par.n_bact
        n_nodes = self.par.n_nodes
        kn = self.par.kn
        n_total_nodes = int(n_bact * n_nodes)

        # Reshape the neighbour direction vectors for bacteria and calculate their angles
        # Help: self.dir.neighbours_direction[:, :, k] == self.dir.nodes_direction[:, self.nei.ind[:, k]]
        bacteria_nodes_direction = np.reshape(np.repeat(self.dir.neighbours_direction[:, :, 0], self.par.kn), (2, int(self.par.n_bact * self.par.n_nodes), self.par.kn))
        bacteria_nodes_angle = np.arctan2(bacteria_nodes_direction[1, :, :], bacteria_nodes_direction[0, :, :])
        neighbours_nodes_angle = np.arctan2(self.dir.neighbours_direction[1, :, :], self.dir.neighbours_direction[0, :, :])
        
        # Calculate the polarity as the cosine of the angle difference between bacteria and their neighbours
        polarity = np.cos(bacteria_nodes_angle - neighbours_nodes_angle)

        # Boolean mask of valid neighbors (not same bacterium and within distance)
        cond_valid = (self.nei.dist < self.par.width * self.par.neighbor_distance_factor) & (~self.nei.cond_same_bact) & (polarity < 0)
        cond_valid_flat = cond_valid.flatten(order='C')  # shape: (n_total_nodes * kn,)

        # Get global node IDs of valid neighbor nodes
        valid_nei_ids = self.nei.ind.flatten(order='C')[cond_valid_flat]  # shape: (?,)

        # Focal node IDs (from which we computed neighbors), mapped to bacterium ID
        focal_node_ids = np.repeat(np.arange(n_total_nodes), kn)[cond_valid_flat]
        focal_bact_ids = focal_node_ids % n_bact  # shape: (?,)

        # Encode pairs (focal bacterium, neighbor node) and extract unique
        pair_ids = focal_bact_ids * n_total_nodes + valid_nei_ids
        unique_pairs = np.unique(pair_ids)
        unique_focal_ids = unique_pairs // n_total_nodes

        # Count number of unique neighbor nodes per bacterium
        nb_neighbors = np.bincount(unique_focal_ids, minlength=n_bact)
        self.nb_neighbors[:] = nb_neighbors

        # Normalize to [0, 1]
        signal = nb_neighbors / self.par.max_neighbours_signal_density
        signal[signal > 1] = 1
        self.signal = signal