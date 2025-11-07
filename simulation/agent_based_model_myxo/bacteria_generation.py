"""
Author: Jean-Baptiste Saulnier
Date: 2024-11-21
"""
import numpy as np
np.random.seed(42)  # Fixe la graine


class GenerationTypeError(Exception):
    """ 
    Custom exception raised when an invalid generation type is provided. 
    """
    pass


class GenerateBacteria:
    """
    Class to generate initial configurations for bacterial positions based on different generation types.

    The parameter `generation_type` determines how the bacteria are generated. Possible values are:
    - "disk_random_orientation": Generate bacteria in a disk-shaped region with random orientations.
    - "disk_alignment": Generate bacteria in a disk with aligned orientations.
    - "square_random_orientation": Generate bacteria in a square region with random orientations.
    - "square_alignment": Generate bacteria in a square with aligned orientations.
    - "rippling_swarming_transition": Generate bacteria for rippling and swarming simulations.
    - "choice": Generate bacteria based on predefined (x, y) coordinates and directions.
    
    Attributes:
    -----------
    par : object
        Instance of the `Parameters` class containing simulation parameters such as generation type, 
        bacterial dimensions, and floating-point precision.

    chosen_generation_function : method
        The selected function for generating bacterial configurations, based on `generation_type`.

    chosen_conditional_space_function : method
        The selected function for setting up conditional spaces, if required.

    data : ndarray
        Array storing the generated positions of bacteria nodes, shaped `(2, n_nodes, n_bact)`.

    U1, U2 : ndarray
        Uniform random variables used for positioning bacteria in the generation process.

    array_nodes : ndarray
        Array of node positions along each bacterium's body, based on the node spacing `d_n`.

    array_nodes_tile : ndarray
        A 2D array where each column represents the node positions along a bacterium's body.
        This array is created by repeating `array_nodes` for each bacterium, resulting in a shape 
        of (n_nodes, n_bact). Each column contains the same sequence of node positions, representing
        the nodes along the body of each bacterium in the population.

    cond_space_alignment : ndarray
        Boolean array indicating alignment conditions for rippling/swarming simulations.

    cond_space_eps : ndarray
        Boolean array indicating epsilon conditions for rippling/swarming simulations.

    middle_node : int
        Index of the node at the center of each bacterium.
    """
    def __init__(self, inst_par):
        # Store references to external class instances
        self.par = inst_par

        # Select the generation function based on the `generation_type` parameter
        if self.par.generation_type == 'disk_random_orientation':
            # Generate bacteria in a disk with random orientations
            self.chosen_generation_function = self.disk_random_orientation
            self.chosen_conditional_space_function = self.function_doing_nothing

        elif self.par.generation_type == 'disk_alignment':
            # Generate bacteria in a disk with aligned orientations
            self.chosen_generation_function = self.disk_alignment
            self.chosen_conditional_space_function = self.function_doing_nothing

        elif self.par.generation_type == 'square_random_orientation':
            # Generate bacteria in a square region with random orientations
            self.chosen_generation_function = self.square_random_orientation
            self.chosen_conditional_space_function = self.function_doing_nothing

        elif self.par.generation_type == 'square_alignment':
            # Generate bacteria in a square region with aligned orientations
            self.chosen_generation_function = self.square_alignment
            self.chosen_conditional_space_function = self.function_doing_nothing

        elif self.par.generation_type == 'rippling_swarming_transition':
            # Generate bacteria for rippling/swarming simulations
            self.chosen_generation_function = self.rippling_swarming_transition
            self.chosen_conditional_space_function = self.rippling_swarming_transition_condition

        elif self.par.generation_type == 'choice':
            # Generate bacteria based on predefined coordinates and orientations
            self.chosen_generation_function = self.choice
            self.chosen_conditional_space_function = self.function_doing_nothing

        else:
            # Raise an error if an invalid generation type is provided
            print('generation_type could be: "disk_random_orientation", "disk_alignment", "square_random_orientation", '
                  '"square_alignment", "rippling_swarming_transition", "choice"\n')
            raise GenerationTypeError()
        

        # Initialize the main data array to store positions of bacteria nodes
        self.data = np.zeros((2, self.par.n_nodes, self.par.n_bact), dtype=self.par.float_type)

        # Uniform random variables for generating initial positions
        self.U1 = np.random.uniform(low=0, high=1.0, size=self.par.n_bact).astype(self.par.float_type)
        self.U2 = np.random.uniform(low=0, high=1.0, size=self.par.n_bact).astype(self.par.float_type)

        # Arrays representing node positions along the length of each bacterium
        self.array_nodes = np.arange(0, self.par.n_nodes * self.par.d_n - 0.5 * self.par.d_n, self.par.d_n).astype(self.par.float_type)
        self.array_nodes_tile = np.tile(self.array_nodes, (self.par.n_bact, 1)).T

        # Conditional space arrays for rippling/swarming simulations (default: all true)
        self.cond_space_alignment = np.ones(self.par.n_bact).astype(bool)
        self.cond_space_eps = np.ones(self.par.n_bact).astype(bool)

        # Calculate the index of the middle node in a bacterium
        self.middle_node = int(self.par.n_nodes / 2)

        # In the case where there is bacterial prey
        # Determine the index where prey bacteria start.
        self.first_index_prey_bact = self.par.n_bact - self.par.n_bact_prey
        self.cond_prey = np.zeros(self.par.n_bact, dtype=bool)
        self.cond_prey[self.first_index_prey_bact:] = True
        
        # Correct the array_node_tile to take into account the length of the prey bacteria
        self.array_nodes_prey_bact = np.arange(0, self.par.n_nodes * self.par.d_n_prey - 0.5 * self.par.d_n_prey, self.par.d_n_prey).astype(self.par.float_type)
        self.array_nodes_tile[:, self.first_index_prey_bact:] = np.tile(self.array_nodes_prey_bact, (self.par.n_bact_prey, 1)).T


    def generate_bacteria(self):
        """
        Calls the selected bacteria generation function based on the chosen generation type.
        """
        self.chosen_generation_function()


    def update_conditional_space(self):
        """
        Calls the selected conditional space function to update space conditions, 
        especially relevant for rippling/swarming simulations.
        """
        self.chosen_conditional_space_function()


    def function_doing_nothing(self):
        """
        This function does nothing and is used as a placeholder for certain conditions.
        """
        pass


    def fill_data(self, x, y, direction):
        """
        Fills the bacteria position data array with the (x, y) coordinates and movement directions.
        
        Parameters:
            x (array): x positions of bacteria
            y (array): y positions of bacteria
            direction (array): directions of movement for each bacteria
        """
        self.data[0, :, :] = (np.tile(x, (self.par.n_nodes, 1))
                              + self.array_nodes_tile
                              * np.tile(np.cos(direction), (self.par.n_nodes, 1))
                              )
        self.data[1, :, :] = (np.tile(y, (self.par.n_nodes, 1))
                              + self.array_nodes_tile
                              * np.tile(np.sin(direction), (self.par.n_nodes, 1))
                              )


    def disk_random_orientation(self):
        """
        Generates n_bact bacteria inside a disk with random orientation.
        """
        x = 0.5 * (self.par.space_size + self.par.d_disk * np.sqrt(self.U2) * np.cos(2 * np.pi * self.U1))
        y = 0.5 * (self.par.space_size + self.par.d_disk * np.sqrt(self.U2) * np.sin(2 * np.pi * self.U1))
        direction = np.random.uniform(0, 2 * np.pi, size=self.par.n_bact).astype(self.par.float_type)
        self.fill_data(x, y, direction)


    def disk_alignment(self):
        """
        Generates n_bact bacteria inside a disk with the same nematic direction (alignment).
        """
        x = 0.5 * (self.par.space_size + self.par.d_disk * np.sqrt(self.U2) * np.cos(2 * np.pi * self.U1))
        y = 0.5 * (self.par.space_size + self.par.d_disk * np.sqrt(self.U2) * np.sin(2 * np.pi * self.U1))
        condition = np.random.binomial(1, 0.5 * np.ones(self.par.n_bact)).astype(bool)
        direction = np.zeros(condition.shape, dtype=self.par.float_type)
        direction[condition] = self.par.global_angle
        direction[~condition] = self.par.global_angle + np.pi
        self.fill_data(x, y, direction)


    def square_random_orientation(self):
        """
        Generates n_bact bacteria inside a square with random orientation.
        """
        x = self.par.space_size * self.U1
        y = self.par.space_size * self.U2
        direction = np.random.uniform(0, 2 * np.pi, size=self.par.n_bact).astype(self.par.float_type)
        self.fill_data(x, y, direction)


    def square_alignment(self):
        """
        Generates n_bact bacteria inside a square with aligned orientation.
        """
        x = self.par.space_size * self.U1
        y = self.par.space_size * self.U2
        condition = np.random.binomial(1, 0.5 * np.ones(self.par.n_bact)).astype(bool)
        direction = np.zeros(condition.shape)
        direction[condition] = self.par.global_angle
        direction[~condition] = self.par.global_angle + np.pi
        self.fill_data(x, y, direction)


    def rippling_swarming_transition(self):
        """
        Generates bacteria in both swarming and rippling configurations.
        """
        n_bact_rippling = int(self.par.percentage_bacteria_rippling * self.par.n_bact)
        n_bact_swarming = self.par.n_bact - n_bact_rippling
        U1_rippling = np.random.uniform(low=self.par.interval_rippling_space[0] * self.par.space_size,
                                        high=self.par.interval_rippling_space[1] * self.par.space_size,
                                        size=n_bact_rippling).astype(self.par.float_type)
        U2_rippling = np.random.uniform(low=0, high=self.par.space_size, size=n_bact_rippling).astype(self.par.float_type)
        U1_swarming = np.random.uniform(low=self.par.interval_rippling_space[1] * self.par.space_size,
                                        high=self.par.space_size, size=n_bact_swarming).astype(self.par.float_type)
        U2_swarming = np.random.uniform(low=0, high=self.par.space_size, size=n_bact_swarming).astype(self.par.float_type)
        x = np.concatenate((U1_rippling, U1_swarming))
        y = np.concatenate((U2_rippling, U2_swarming))
        direction = np.random.uniform(0, 2 * np.pi, size=self.par.n_bact).astype(self.par.float_type)
        self.fill_data(x, y, direction)
        cond_rippling_bact = np.random.binomial(1, 0.5 * np.ones(self.par.n_bact)).astype(bool)
        self.update_rippling_swarming_condition()
        direction[cond_rippling_bact & self.cond_space_alignment] = self.par.global_angle
        direction[~cond_rippling_bact & self.cond_space_alignment] = self.par.global_angle + np.pi
        self.fill_data(x, y, direction)


    def rippling_swarming_transition_condition(self):
        """
        Create the condition for the rippling part.
        
        This function sets up the condition for bacteria that are within the specified 
        rippling space interval. The bacteria are marked as aligned within this space.
        """
        self.cond_space_alignment[:] = (self.data[0, 0, :] > self.par.interval_rippling_space[0] * self.par.space_size) & \
                                      (self.data[0, 0, :] < self.par.interval_rippling_space[1] * self.par.space_size)
        self.cond_space_eps[:] = ~self.cond_space_alignment


    def choice(self):
        """
        Generate bacteria with predefined (x, y) coordinates and direction.
        This is used when the positions and directions of the bacteria are chosen based on user-defined input.
        """
        # Fill data using predefined x, y coordinates and directions
        self.fill_data(self.par.x, self.par.y, self.par.direction)