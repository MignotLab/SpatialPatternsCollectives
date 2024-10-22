import numpy as np
from scipy.ndimage import gaussian_filter

class EpsTypeError(Exception):
    
    pass

class Eps:
    """
    Create an eps road after the passage of the bacteria.
    eps_type could be: "igoshin_eps_road_follower", "follow_gradient", "no_eps"; default is "no_eps".
    
    """
    def __init__(self,inst_par,inst_gen,inst_pha,inst_dir,inst_nei,inst_ali,inst_tool):

        # Instance objects
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.dir = inst_dir
        self.nei = inst_nei
        self.ali = inst_ali
        self.tool = inst_tool

        if self.par.eps_follower_type == 'igoshin_eps_road_follower':
            self.chosen_eps_follower_fonction = self.igoshin_eps_road_follower
        elif self.par.eps_follower_type == 'follow_gradient':
            self.chosen_eps_follower_fonction = self.follow_gradient
        elif self.par.eps_follower_type == 'no_eps':
            self.chosen_eps_follower_fonction = self.function_doing_nothing
        else:
            # Print the different possible alignment when the class is call
            print('eps_type could be: "igoshin_eps_road_follower", "follow_gradient", "no_eps"; default is "no_eps"\n')
            raise EpsTypeError()

        # Parameters from parameters.py
        self.l = self.par.space_size
        self.edges_width = 2 * self.par.pili_length # length of the eps edges in µm
        self.l_eps = self.par.space_size + 2 * self.edges_width # length of the eps space in µm
        self.bins = int(self.l_eps / self.par.width_bins) # number of bins of the eps map
        self.edges_width_bins = int(self.edges_width * self.bins / self.l_eps) # length of the eps edges in bins
        self.r = int(self.par.pili_length * self.bins / self.l_eps) # length of pili in pixel
        self.ang_view = self.par.eps_angle_view
        self.n_sec = self.par.n_sections
        self.ang_sec = self.par.angle_section
        self.max_eps = self.par.max_eps_value
        self.g_x = np.tile(np.arange(self.bins),(self.bins,1)) # array of the x coordinate in pixel in the eps grid
        self.g_y = self.g_x.T
        self.rate_eps_evaporation = 1 / self.par.eps_mean_lifetime
        # # Change the shape of data to use the kdtree
        # self.x, self.y = self.gen.data[0,:,:].flatten(), self.gen.data[1,:,:].flatten()
        # self.coord = np.column_stack((self.x,self.y))
        # Object of the class
        self.eps_grid,__,__= np.histogram2d(self.gen.data[0,-1,:], self.gen.data[1,-1,:], bins=self.bins, range=[[-self.edges_width, self.l+self.edges_width], [-self.edges_width, self.l+self.edges_width]])
        self.eps_grid = self.eps_grid.astype(self.par.float_type) * self.par.deposit_amount * self.par.v0 * self.par.dt * np.exp(-self.rate_eps_evaporation * self.par.dt)
        self.eps_grid_blur = self.eps_grid.copy()
        self.eps_angle = np.zeros(self.par.n_bact, dtype=self.par.float_type)
        self.bact_angle = np.zeros(self.par.n_bact, dtype=self.par.float_type)
        self.eps_diff = np.ones(self.par.n_bact, dtype=self.par.float_type)
        if self.par.sigma_eps != 0:
            self.epsilon_eps_heterogeneity = np.random.normal(loc=self.par.epsilon_eps, scale=self.par.sigma_eps, size=self.par.n_bact).astype(self.par.float_type)
            self.epsilon_eps_heterogeneity[self.epsilon_eps_heterogeneity < 0] = 0

    def function_eps_follower_type(self):
        """
        Function doing the movement type and the random movement depending on the choice of these movement in the parameter class
        
        """
        self.chosen_eps_follower_fonction()

    def function_doing_nothing(self):
        """
        No eps production and follower
        
        """
        pass

    def igoshin_eps_road_follower(self):
        """
        The bacteria will follow the eps road as in 
        Rajesh Balagam, Oleg A. Igoshin; Mechanism for Collective Cell Alignment in Myxococcus xanthus Bacteria; 2015
        
        """
        self.eps_generation()
        self.eps_direction_igoshin()
        self.follow_eps()

    def igoshin_eps_road_follower_specific_space(self):
        """
        The bacteria will follow the eps road as in 
        Rajesh Balagam, Oleg A. Igoshin; Mechanism for Collective Cell Alignment in Myxococcus xanthus Bacteria; 2015
        
        """
        self.eps_generation()
        self.eps_direction_igoshin()
        self.follow_eps_specific_space()

    def compute_gradient_rotation_matrix(self,x,y,sigma):
        """
        The bacteria will follow the gradient of the eps
        
        """
        self.eps_grid_blur = gaussian_filter(self.eps_grid_blur, sigma=sigma)
        # Compute the eps gradient
        grad_x, grad_y = np.gradient(self.eps_grid_blur)
        # Select the indexes (i,j) of the bins where the head of the bacteria are
        x_bins = ((x + self.edges_width) * self.bins / self.l_eps).astype(int)
        y_bins = ((y + self.edges_width) * self.bins / self.l_eps).astype(int)
        # Find the values of the gradient vectors
        # The first element represent the column of the array (y) and the second element the rows (x)
        y_dir = grad_x[x_bins,y_bins]
        x_dir = grad_y[x_bins,y_bins]
        angle_gradient = np.arctan2(y_dir, x_dir)
        nodes_angle_head = np.arctan2(self.dir.nodes_direction[1,0,self.gen.cond_space_eps], self.dir.nodes_direction[0,0,self.gen.cond_space_eps])
        # Compute the power of alignment depending on the gradient
        epsilon_grad = np.linalg.norm(np.array([x_dir,y_dir]), axis=0)
        # Compute angle the cell have to rotate to align with its closest neighbour
        # angle_rotation = self.epsilon_eps_heterogeneity * np.sin(2 * (nodes_angle_head - angle_gradient)) * self.par.dt
        # angle_rotation = epsilon_grad * np.sin(2 * (nodes_angle_head - angle_gradient)) * self.par.dt
        direction_rotation = - np.sign(np.sum(np.array([self.dir.nodes_direction[0,0,self.gen.cond_space_eps], self.dir.nodes_direction[1,0,self.gen.cond_space_eps]])*np.array([x_dir,y_dir]), axis=0))
        # direction_rotation[direction_rotation==0] = ((np.random.binomial(1, np.ones(direction_rotation.shape) * 0.5) - 0.5) * 2)[direction_rotation==0]
        epsilon_grad[epsilon_grad>0] = 1.
        if self.par.sigma_eps != 0:
            angle_rotation = self.epsilon_eps_heterogeneity * epsilon_grad * direction_rotation * self.par.dt
        else:
            angle_rotation = self.par.epsilon_eps * epsilon_grad * direction_rotation * self.par.dt
        # Construct the matrice of rotation
        rotation_matrix = self.tool.rotation(theta=angle_rotation)

        return rotation_matrix

    def follow_gradient(self):
        """
        The bacteria will follow the gradient of the eps
        
        """
        self.eps_generation()
        rotation_matrix = self.compute_gradient_rotation_matrix(x=self.gen.data[0,0,self.gen.cond_space_eps],y=self.gen.data[1,0,self.gen.cond_space_eps])
        rotation_data = np.sum(rotation_matrix * (self.pha.data_phantom[:,0,self.gen.cond_space_eps]-self.pha.data_phantom[:,1,self.gen.cond_space_eps]), axis=1) + self.gen.data[:,1,self.gen.cond_space_eps]
        rotation_data_phantom = np.sum(rotation_matrix * (self.pha.data_phantom[:,0,self.gen.cond_space_eps]-self.pha.data_phantom[:,1,self.gen.cond_space_eps]), axis=1) + self.pha.data_phantom[:,1,self.gen.cond_space_eps]
        self.gen.data[:,0,self.gen.cond_space_eps] = rotation_data[:,:]
        self.pha.data_phantom[:,0,self.gen.cond_space_eps] = rotation_data_phantom[:,:]

    def eps_generation(self):
        """
        Generate the eps actual map

        """
        # eps_local,__,__ = np.histogram2d(x, y, bins=self.bins, range=[[-2*self.r, self.l-2*self.r], [-2*self.r, self.l-2*self.r]])
        eps_local,__,__= np.histogram2d(self.gen.data[0,-1,:], self.gen.data[1,-1,:], bins=self.bins, range=[[-self.edges_width, self.l+self.edges_width], [-self.edges_width, self.l+self.edges_width]])
        # Here self.eps_grid should stay as np.float_type even if eps_local is not
        self.eps_grid += self.par.deposit_amount * eps_local * self.par.v0 * self.par.dt * np.exp(-self.rate_eps_evaporation * self.par.dt)
        # Copy the edges symetrically in the edges width
        self.eps_grid[:,:self.edges_width_bins] = self.eps_grid[:,self.bins-2*self.edges_width_bins:self.bins-self.edges_width_bins]
        self.eps_grid[:,self.bins-self.edges_width_bins:] = self.eps_grid[:,self.edges_width_bins:2*self.edges_width_bins]
        self.eps_grid[:self.edges_width_bins,:] = self.eps_grid[self.bins-2*self.edges_width_bins:self.bins-self.edges_width_bins,:]
        self.eps_grid[self.bins-self.edges_width_bins:,:] = self.eps_grid[self.edges_width_bins:2*self.edges_width_bins,:]
        self.eps_grid_blur = self.eps_grid.copy()

    def eps_bisectors(self,bact_direction):
        """
        Find the bisector of each section in the angle view of the bacterium
        
        """
        bact_angle = np.arctan2(bact_direction[1], bact_direction[0])
        start = bact_angle - self.ang_view / 2 + self.ang_view / (self.n_sec * 2)
        end = bact_angle + self.ang_view / 2 - self.ang_view / (self.n_sec * 2)
        angles_bisectors_sections = np.linspace(start, end, self.n_sec)

        return angles_bisectors_sections, bact_angle

    def eps_values_sections(self,x,y,angles_bisectors_sections):
        """
        Compute the value of the total eps in each section
        
        """
        sum_eps_section = np.zeros(self.n_sec)
        max_eps_section = np.zeros(self.n_sec)
        average_eps_section = np.zeros(self.n_sec)
        if self.par.sigma_blur_eps > 0:
            self.eps_grid_blur = gaussian_filter(input=self.eps_grid, sigma=self.par.sigma_blur_eps)
        local_eps_grid = self.eps_grid_blur[x-self.r:x+self.r, y-self.r:y+self.r].T.copy()
        local_coord_grid_x = self.g_x[x-self.r:x+self.r, x-self.r:x+self.r].copy()
        local_coord_grid_y = self.g_y[y-self.r:y+self.r, y-self.r:y+self.r].copy()
        local_dir_x = local_coord_grid_x - x
        local_dir_y = local_coord_grid_y - y
        eps_bins_sector_x = []
        eps_bins_sector_y = []

        for sector, angle in enumerate(angles_bisectors_sections):

            norm_dir = np.linalg.norm(np.array([local_dir_x, local_dir_y]), axis=0)
            scalar_product = local_dir_x * np.cos(angle) + local_dir_y * np.sin(angle)
            cond_sphere = (norm_dir <= self.r) & (norm_dir > 0)
            cond_section_view = scalar_product >= np.cos(self.ang_sec / 2) * norm_dir
            local_section = local_eps_grid[cond_sphere & cond_section_view]
            sum_eps_section[sector] = np.sum(local_section)
            max_eps_section[sector] = np.max(local_section)
            # Compute the average of the bins values and add some noise in weights in case of weights == 0
            background_noise = np.ones(len(local_section)) * 1e-6
            weights = local_section + background_noise
            average_eps_section[sector] = np.average(local_section, axis=None, weights=weights)
            eps_bins_sector_x.append(local_coord_grid_x[cond_sphere & cond_section_view])
            eps_bins_sector_y.append(local_coord_grid_y[cond_sphere & cond_section_view])

        return sum_eps_section, max_eps_section, average_eps_section, np.array(eps_bins_sector_x,dtype=object), np.array(eps_bins_sector_y,dtype=object)

    def eps_direction(self):
        """
        Find the direction of the cell which follow the eps road

        """
        # Transform data heads coordinates into coordinates on the eps grid
        x_head = ((self.gen.data[0,0,:] + self.edges_width) * self.bins / self.l_eps).astype(int)
        y_head = ((self.gen.data[1,0,:] + self.edges_width) * self.bins / self.l_eps).astype(int)
        indices = np.where(self.gen.cond_space_eps)[0]
        for index in indices:

            angles_bisectors_sections, bact_angle_i = self.eps_bisectors(bact_direction=self.dir.nodes_direction[:,0,index].T)
            sum_eps_section, max_eps_section, average_eps_section, eps_bins_sector_x, eps_bins_sector_y = self.eps_values_sections(x=x_head[index],
                                                                                                                                   y=y_head[index],
                                                                                                                                   angles_bisectors_sections=angles_bisectors_sections)
            # Compute the difference of the values between the section with the higher
            # amouth of eps with the eps in the middle section
            idx_sector = np.argmax(sum_eps_section)
            self.eps_angle[index] = angles_bisectors_sections[idx_sector]
            self.bact_angle[index] = bact_angle_i
            self.eps_diff[index] = (average_eps_section[idx_sector] - average_eps_section[int(self.par.n_sections/2)]) / self.par.max_eps_value

    def eps_direction_igoshin(self, min_eps_factor=0.8):
        """
        Find the direction of the cell which follow the eps road

        """
        # head_direction = data[:,:,0] - data[:,:,1]
        # Transform data heads coordinates into coordinates on the eps grid
        x_head = ((self.gen.data[0,0,:] + self.edges_width) * self.bins / self.l_eps).astype(int)
        y_head = ((self.gen.data[1,0,:] + self.edges_width) * self.bins / self.l_eps).astype(int)
        # eps_dir = np.zeros(data[:,0,:].shape)
        # array_coord_x_sector = np.array([])
        # array_coord_y_sector = np.array([])
        # array_coord_x_sectors = np.array([])
        # array_coord_y_sectors = np.array([])
        indices = np.where(self.gen.cond_space_eps)[0]

        for index in indices:
            angles_bisectors_sections, bact_angle_i = self.eps_bisectors(bact_direction=self.dir.nodes_direction[:,0,index].T)
            sum_eps_section, max_eps_section, average_eps_section, eps_bins_sector_x, eps_bins_sector_y = self.eps_values_sections(x=x_head[index],
                                                                                                                                   y=y_head[index],
                                                                                                                                   angles_bisectors_sections=angles_bisectors_sections)
            # Compute the difference of the values between the section with the higher
            # amouth of eps with the eps in the middle section
            idx_sector = np.argmax(sum_eps_section)
            self.eps_angle[index] = angles_bisectors_sections[idx_sector]
            self.bact_angle[index] = bact_angle_i
            if np.max(max_eps_section) < self.max_eps / 100:
                # pass
                self.eps_angle[index] = bact_angle_i
                self.bact_angle[index] = bact_angle_i
                # eps_dir[:,index] = np.array([0, 0])
                # array_coord_x_sector = np.concatenate((array_coord_x_sector, eps_bins_sector_x[int(self.n_sec/2)]))
                # array_coord_y_sector = np.concatenate((array_coord_y_sector, eps_bins_sector_y[int(self.n_sec/2)]))
            else:
                # Select sections with more than min_eps_factor of the max section value
                cond = sum_eps_section >= min_eps_factor * np.max(sum_eps_section)
                selected_angles = angles_bisectors_sections[cond]
                # selected_averages = average_eps_section[cond]
                # selected_coord_sectors_x = eps_bins_sector_x[cond]
                # selected_coord_sectors_y = eps_bins_sector_y[cond]
                idx_sector = np.argmin(np.abs(selected_angles - bact_angle_i) * (1 + np.random.uniform(-0.001,0.001,len(selected_angles))))
                self.eps_angle[index] = selected_angles[idx_sector]
                self.bact_angle[index] = bact_angle_i
                # eps_dir[index,:] = np.array([np.cos(eps_angle), np.sin(eps_angle)])
                # eps_dir[index,:] = eps_dir[index,:] * selected_averages[idx_sector] / self.max_eps
                # array_coord_x_sector = np.concatenate((array_coord_x_sector, selected_coord_sectors_x[idx_sector]))
                # array_coord_y_sector = np.concatenate((array_coord_y_sector, selected_coord_sectors_y[idx_sector]))
            # array_coord_x_sectors = np.concatenate((array_coord_x_sectors, np.concatenate(eps_bins_sector_x)))
            # array_coord_y_sectors = np.concatenate((array_coord_y_sectors, np.concatenate(eps_bins_sector_y)))

        # return eps_angle, bact_angle

    def eps_direction_igoshin_parallel(self,index,x_head_bins,y_head_bins,head_direction,min_eps_factor=0.8):
        """
        Find the direction of the cell which follow the eps road

        """
        angles_bisectors_sections, bact_angle_i = self.eps_bisectors(bact_direction=head_direction[index])
        sum_eps_section, max_eps_section, __, __, __ = self.eps_values_sections(x=x_head_bins[index],
                                                                                y=y_head_bins[index],
                                                                                angles_bisectors_sections=angles_bisectors_sections,)
        # Compute the difference of the values between the section with the higher
        # amouth of eps with the eps in the middle section
        idx_sector = np.argmax(sum_eps_section)
        self.eps_angle[index] = angles_bisectors_sections[idx_sector]
        self.bact_angle[index] = bact_angle_i

        if np.max(max_eps_section) < self.max_eps / 100:

            self.eps_angle[index] = bact_angle_i
            self.bact_angle[index] = bact_angle_i
        else:
            # Select sections with more than min_eps_factor of the max section value
            cond = sum_eps_section >= min_eps_factor * np.max(sum_eps_section)
            selected_angles = angles_bisectors_sections[cond]
            idx_sector = np.argmin(np.abs(selected_angles - bact_angle_i) * (1 + np.random.uniform(-0.001,0.001,len(selected_angles))))
            self.eps_angle[index] = selected_angles[idx_sector]
            self.bact_angle[index] = bact_angle_i

    def follow_eps(self):
        """
        Align the cells with the eps direction
        
        """
        # Construct the rotation matrix
        if self.par.sigma_eps != 0:
            rotation_angle = self.epsilon_eps_heterogeneity * self.eps_diff * np.sin(2 * (self.eps_angle - self.bact_angle)) * self.par.dt
        else:
            rotation_angle = self.par.epsilon_eps * self.eps_diff * np.sin(2 * (self.eps_angle - self.bact_angle)) * self.par.dt
        rotation_matrix = self.tool.rotation(theta=rotation_angle[self.gen.cond_space_eps])
        # Update data and data_phanto by applying the rotation on the head node
        rotation_data = np.sum(rotation_matrix * (self.pha.data_phantom[:,0,self.gen.cond_space_eps]-self.pha.data_phantom[:,1,self.gen.cond_space_eps]), axis=1) + self.gen.data[:,1,self.gen.cond_space_eps]
        rotation_data_phantom = np.sum(rotation_matrix * (self.pha.data_phantom[:,0,self.gen.cond_space_eps]-self.pha.data_phantom[:,1,self.gen.cond_space_eps]), axis=1) + self.pha.data_phantom[:,1,self.gen.cond_space_eps]
        self.gen.data[:,0,self.gen.cond_space_eps] = rotation_data[:,:]
        self.pha.data_phantom[:,0,self.gen.cond_space_eps] = rotation_data_phantom[:,:]

