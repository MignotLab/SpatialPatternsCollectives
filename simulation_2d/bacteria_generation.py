import numpy as np

class GenerationTypeError(Exception):
    pass

class GenerateBacteria:
    """
    Generate initial configuration of the position of bacteria.
    generation_type could be: "gen_bact_disk", "gen_bact_disk_align", "gen_bact_square", "gen_bact_square_align", "gen_bact_rippling_swarming", "gen_bact_waves", "gen_bact_choice"; default is "gen_bact_square_align"
    
    """
    def __init__(self,inst_par):

        # Parameters from parameters.py
        self.par = inst_par
        if self.par.generation_type == 'gen_bact_disk':
            self.chosen_generation_fonction = self.gen_bact_disk
            self.chosen_conditional_space_fonction = self.function_doing_nothing
        elif self.par.generation_type == 'gen_bact_disk_align':
            self.chosen_generation_fonction = self.gen_bact_disk_align
            self.chosen_conditional_space_fonction = self.function_doing_nothing
        elif self.par.generation_type == 'gen_bact_square':
            self.chosen_generation_fonction = self.gen_bact_square
            self.chosen_conditional_space_fonction = self.function_doing_nothing
        elif self.par.generation_type == 'gen_bact_square_align':
            self.chosen_generation_fonction = self.gen_bact_square_align
            self.chosen_conditional_space_fonction = self.function_doing_nothing
        elif self.par.generation_type == 'gen_bact_rippling_swarming':
            self.chosen_generation_fonction = self.gen_bact_rippling_swarming
            self.chosen_conditional_space_fonction = self.gen_bact_rippling_swarming_cond
        elif self.par.generation_type == 'gen_bact_waves':
            self.chosen_generation_fonction = self.gen_bact_waves
            self.chosen_conditional_space_fonction = self.function_doing_nothing
        elif self.par.generation_type == 'gen_bact_choice':
            self.chosen_generation_fonction = self.gen_bact_choice
            self.chosen_conditional_space_fonction = self.function_doing_nothing
        else:
            # Print the different possible alignment when the class is call
            print('generation_type could be: "gen_bact_disk", "gen_bact_disk_align", "gen_bact_square", "gen_bact_square_align", "gen_bact_rippling_swarming", "gen_bact_waves", "gen_bact_choice"; default is "gen_bact_square_align"\n')
            raise GenerationTypeError()
        # Init data
        self.data = np.zeros((2,self.par.n_nodes,self.par.n_bact), dtype=self.par.float_type)
        # Generate of n_bact uniformaly distributed
        self.U1 = np.random.uniform(low=self.par.epsilon_float32, high=1.0-self.par.epsilon_float32, size=self.par.n_bact).astype(self.par.float_type)
        self.U2 = np.random.uniform(low=self.par.epsilon_float32, high=1.0-self.par.epsilon_float32, size=self.par.n_bact).astype(self.par.float_type)
        # Nodes generation
        self.array_nodes = np.arange(0, self.par.n_nodes * self.par.d_n - 0.5 * self.par.d_n, self.par.d_n).astype(self.par.float_type)
        # Conditional space for rippling and swarming simulation
        self.cond_space_alignment = np.ones(self.par.n_bact).astype(bool)
        self.cond_space_eps = np.ones(self.par.n_bact).astype(bool)
        # Node at the center of the bacteria
        self.middle_node = int(self.par.n_nodes / 2)

    def function_generation_type(self):
        """
        Function doing the movement type and the random movement depending on the choice of these movement in the parameter class
        
        """
        self.chosen_generation_fonction()

    def function_conditional_space(self):
        """
        Choose the space condition to know in which rippling is generate in case of swarming + rippling simulation
        
        """
        self.chosen_conditional_space_fonction()

    def function_doing_nothing(self):
        """
        Do nothing

        """
        pass

    def fill_data(self,x,y,direction):
        """
        Function to fill the data array from the distribution (x,y)
        
        """
        self.data[0,:,:] = (np.tile(x, (self.par.n_nodes,1))
                          + np.tile(self.array_nodes, (self.par.n_bact, 1)).T
                          * np.tile(np.cos(direction), (self.par.n_nodes, 1))
                          )
        self.data[1,:,:] = (np.tile(y, (self.par.n_nodes,1))
                          + np.tile(self.array_nodes, (self.par.n_bact, 1)).T
                          * np.tile(np.sin(direction), (self.par.n_nodes, 1))
                          )

    def gen_bact_disk(self):
        """
        Generate n_bact bacteria inside a disk of diameter d_disk in a square space of size space_size

        """
        x = 0.5 * (self.par.space_size + self.par.d_disk * np.sqrt(self.U2) * np.cos(2 * np.pi * self.U1))
        y = 0.5 * (self.par.space_size + self.par.d_disk * np.sqrt(self.U2) * np.sin(2 * np.pi * self.U1))
        direction = np.random.uniform(0, 2*np.pi, size = self.par.n_bact).astype(self.par.float_type)
        self.fill_data(x,y,direction=direction)

    def gen_bact_disk_align(self):
        """
        Generate n_bact bacteria with the same nematic direction 
        inside a disk of diameter d_disk in a square space of size space_size

        """
        x = 0.5 * (self.par.space_size + self.par.d_disk * np.sqrt(self.U2) * np.cos(2 * np.pi * self.U1))
        y = 0.5 * (self.par.space_size + self.par.d_disk * np.sqrt(self.U2) * np.sin(2 * np.pi * self.U1))
        condition = np.random.binomial(1, 0.5 * np.ones(self.par.n_bact)).astype(bool)
        direction = np.zeros(condition.shape, dtype=self.par.float_type)
        direction[condition] = self.par.global_angle
        direction[~condition] = self.par.global_angle + np.pi
        self.fill_data(x,y,direction=direction)

    def gen_bact_square(self):
        """
        Generate n_bact bacteria inside a square of size space_size

        """
        x = self.par.space_size * self.U1
        y = self.par.space_size * self.U2
        direction = np.random.uniform(0, 2*np.pi, size=self.par.n_bact).astype(self.par.float_type)
        self.fill_data(x,y,direction=direction)

    def gen_bact_square_align(self):
        """
        Generate n_bact bacteria inside a square of size space_size

        """
        x = self.par.space_size * self.U1
        y = self.par.space_size * self.U2
        condition = np.random.binomial(1, 0.5 * np.ones(self.par.n_bact)).astype(bool)
        direction = np.zeros(condition.shape)
        direction[condition] = self.par.global_angle
        direction[~condition] = self.par.global_angle + np.pi
        self.fill_data(x,y,direction=direction)

    def gen_bact_rippling_swarming(self):
        """
        Generate two field of bacteria for swarming and rippling
        
        """
        n_bact_rippling = int(self.par.percentage_bacteria_rippling * self.par.n_bact)
        n_bact_swarming = self.par.n_bact - n_bact_rippling
        U1_rippling = np.random.uniform(low=self.par.interval_rippling_space[0]*self.par.space_size, high=self.par.interval_rippling_space[1]*self.par.space_size, size=n_bact_rippling).astype(self.par.float_type)
        U2_rippling = np.random.uniform(low=0, high=self.par.space_size, size=n_bact_rippling).astype(self.par.float_type)
        U1_swarming = np.random.uniform(low=self.par.interval_rippling_space[1]*self.par.space_size, high=self.par.space_size, size=n_bact_swarming).astype(self.par.float_type)
        U2_swarming = np.random.uniform(low=0, high=self.par.space_size, size=n_bact_swarming).astype(self.par.float_type)
        x = np.concatenate((U1_rippling, U1_swarming))
        y = np.concatenate((U2_rippling, U2_swarming))
        direction = np.random.uniform(0, 2*np.pi, size=self.par.n_bact).astype(self.par.float_type)
        self.fill_data(x, y, direction=direction)
        cond_rippling_bact = np.random.binomial(1, 0.5 * np.ones(self.par.n_bact)).astype(bool)
        self.gen_bact_rippling_swarming_cond()
        direction[cond_rippling_bact & self.cond_space_alignment] = self.par.global_angle
        direction[~cond_rippling_bact & self.cond_space_alignment] = self.par.global_angle + np.pi
        self.fill_data(x, y, direction=direction)

    def gen_bact_rippling_swarming_cond(self):
        """
        Create the condition for the rippling part
        
        """
        self.cond_space_alignment[:] = (self.data[0,0,:] > self.par.interval_rippling_space[0]*self.par.space_size) & (self.data[0,0,:] < self.par.interval_rippling_space[1]*self.par.space_size)
        self.cond_space_eps[:] = ~self.cond_space_alignment

    def gen_bact_waves(self):
        """
        Generate bacteria with rippling waves already forms
        
        """
        waves_position = np.linspace(0.2,0.8,self.par.nb_waves)
        U1 = np.array([])
        direction = np.array([])
        for i in range(len(waves_position)):
            low = waves_position[i] - self.par.waves_width / self.par.space_size / 2
            high = waves_position[i] + self.par.waves_width / self.par.space_size / 2
            U1 = np.concatenate((U1, np.random.uniform(low=low, high=high, size=int(self.par.n_bact / self.par.nb_waves)).astype(self.par.float_type)))
            direction = np.concatenate((direction, np.ones(int(self.par.n_bact/self.par.nb_waves))*self.par.global_angle + (i+1)*np.pi))
        while len(U1) != self.par.n_bact:
            U1 = np.concatenate((U1, np.random.uniform(low=low, high=high, size=int(self.par.n_bact / self.par.nb_waves)).astype(self.par.float_type)))
            direction = np.concatenate((direction, np.ones(1) * self.par.global_angle + (i+1)*np.pi))
        x = self.par.space_size * U1 + self.par.bacteria_length
        y = self.par.space_size * self.U2 + self.par.bacteria_length
        self.fill_data(x,y,direction=direction)

    def gen_bact_choice(self):
        """
        Generate bacteria where coordinate and self.par.global_angle have to be done
        
        """

        self.fill_data(self.par.x,self.par.y,self.par.direction)
    