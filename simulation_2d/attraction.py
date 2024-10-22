import numpy as np

class AttractionTypeError(Exception):

    pass

class Attraction:
    """
    Compute attraction between cells.
    attraction_type could be: "attraction_head", "attraction_head_all_neighbours", "attraction_body", "no_attraction"; default is "no_attraction".
    
    """

    def __init__(self,inst_par,inst_gen,inst_pha,inst_dir,inst_nei):

        # Instance objects
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.dir = inst_dir
        self.nei = inst_nei

        if self.par.attraction_type == 'attraction_head':
            self.chosen_attraction_fonction = self.attraction_head
        elif self.par.attraction_type == 'attraction_head_all_neighbours':
            self.chosen_attraction_fonction = self.attraction_head_all_neighbours
        elif self.par.attraction_type == 'attraction_body':
            self.chosen_attraction_fonction = self.attraction_body
        elif self.par.attraction_type == 'no_attraction':
            self.chosen_attraction_fonction = self.function_doing_nothing
        else:
            # Print the different possible alignment when the class is call
            print('attraction_type could be: "attraction_head", "attraction_head_all_neighbours", "attraction_body", "no_attraction"; default is "no_attraction"\n')
            raise AttractionTypeError()
        
        # Parameters from parameters.py
        self.bact_enum = np.arange(self.par.n_bact).astype(self.par.int_type)
        self.bact_enum_body = np.arange(self.par.n_bact*self.par.n_nodes).astype(self.par.int_type)

    def function_attraction_type(self):
        """
        Function doing the movement type and the random movement depending on the choice of these movement in the parameter class
        
        """

        self.chosen_attraction_fonction()

    def f_at_norm(self,d,r):
        """
        Attractive force between two objects at distance r (the force is 0 if r > 2d or f < d)

        """

        x = np.minimum(np.maximum(r / d, 1), 2)

        return -self.par.k_a * (x-1)*(x-2) * (4 + (x-1.5)*(24-16*x))

    def f_at_pili_norm(self,r,k,w0,w1,w2):
        """
        Attractive force between two objects at distance r (the force is 0 if r < w0 or r > w2 and maximal if r = w1)

        """

        a = -((k * (w0**2 + 3*w1**2 - 3*w1*w2 + w2**2 + w0*(-3*w1 + w2))) / ((w0 - w1)**3 * (w1 - w2)**3))
        b = (k * (w0**2*(3*w1 - w2) - w0*(-3*w1 + w2)**2 + w1*(8*w1**2 - 9*w1*w2 + 3*w2**2)))/((w0 - w1)**3*(w1 - w2)**3)
        c = -((k * (w0*w1*(-8*w1**2 + 9*w1*w2 - 3*w2**2) + w0**2*(3*w1**2 - 3*w1*w2 + w2**2) + w1**2*(6*w1**2 - 8*w1*w2 + 3*w2**2))) / ((w0 - w1)**3 * (w1 - w2)**3))

        res = (r - w0) * (r - w2) * (a*r**2 + b*r +c)
        res[r < w0] = 0
        res[r > w2] = 0

        return res
    
    def function_doing_nothing(self):
        """
        No attraction
        
        """

        pass

    def attraction_head(self):
        """
        Attraction of the head of each bacteria with its closest neighbour
        
        """

        x_dir_n = self.dir.nodes_to_nei_dir_torus[0,:self.par.n_bact,:]
        y_dir_n = self.dir.nodes_to_nei_dir_torus[1,:self.par.n_bact,:]
        # Select the appropiate nodes direction
        xy_dir = np.reshape(np.repeat(self.dir.nodes_direction[:,0,:], self.par.kn), (2,self.par.n_bact,self.par.kn))

        # Create the condition in the angle view
        cond_angle_view = (xy_dir[0] * x_dir_n + xy_dir[1] * y_dir_n) > np.cos(self.par.at_angle_view / 2)

        # Force intensity of the head attraction from its closest neighbour
        # f_norm = self.f_at_norm(d=self.par.pili_length, r=self.nei.dist[:self.par.n_bact])
        f_norm = self.f_at_pili_norm(r=self.nei.dist[:self.par.n_bact], k=self.par.k_a, w0=self.par.width, w1=(self.par.pili_length+self.par.width)/2, w2=self.par.pili_length)
        f_norm[self.nei.cond_same_bact[:self.par.n_bact] | ~cond_angle_view] = 0.

        # Search the closest neigbour with non 0 values (return 0 if there is no neighbour)
        ind_first_non_zero = (f_norm != 0).argmax(axis=1)
        f_norm = f_norm[self.bact_enum,ind_first_non_zero]

        # Reshape direction and keep direction with the closest neighbour of the head
        f_x = x_dir_n[self.bact_enum,ind_first_non_zero] * f_norm
        f_y = y_dir_n[self.bact_enum,ind_first_non_zero] * f_norm

        # Apply the forces on the head and nec
        force = np.array([f_x,f_y])
        self.gen.data[:,0,:] += force * self.par.dt**2
        # self.data[:,1,:] += force * self.par.dt**2
        self.pha.data_phantom[:,0,:] += force * self.par.dt**2
        # self.data_phantom[:,1,:] += force * self.par.dt**2

    def attraction_head_all_neighbours(self):
        """
        Attraction of the head of each bacteria with its closest neighbour
        
        """

        x_dir_n = self.dir.nodes_to_nei_dir_torus[0,:,:]
        y_dir_n = self.dir.nodes_to_nei_dir_torus[1,:,:]

        # Select the appropiate nodes direction
        xy_dir = np.reshape(np.repeat(self.dir.nodes_direction[:,0,:], self.par.kn), (2,self.par.n_bact,self.par.kn))

        # Create the condition in the angle view
        cond_angle_view = (xy_dir[0] * x_dir_n + xy_dir[1] * y_dir_n) > np.cos(self.par.at_angle_view / 2)

        # Force intensity of the head attraction from its closest neighbour
        # f_norm = self.f_at_pili_norm(r=self.nei.dist[:self.par.n_bact],sigma=0.8,mu=0.5,coeff=1.3)
        f_norm = self.f_at_pili_norm(r=self.nei.dist[:self.par.n_bact], k=self.par.k_a, w0=self.par.width, w1=(self.par.pili_length+self.par.width)/2, w2=self.par.pili_length)
        # f_norm[self.nei.cond_same_bact | ~cond_angle_view] = 0.
        f_norm[~cond_angle_view] = 0.

        # Sum over all neighbours forces
        f_x = np.sum(x_dir_n * f_norm, axis=1)
        f_y = np.sum(y_dir_n * f_norm, axis=1)

        # Apply the forces on the head and nec
        force = np.array([f_x,f_y])
        self.gen.data[:,0,:] += force * self.par.dt**2
        # self.gen.data[:,1,:] += force * self.par.dt**2
        self.pha.data_phantom[:,0,:] += force * self.par.dt**2
        # self.pha.data_phantom[:,1,:] += force * self.par.dt**2

    def attraction_body(self):
        """
        Attraction of the head of each bacteria with its closest neighbour
        finish it
        
        """

        x_dir_n = self.dir.nodes_to_nei_dir_torus[0,:,:]
        y_dir_n = self.dir.nodes_to_nei_dir_torus[1,:,:]

        # Select the appropiate nodes direction
        nodes_direction_flat = np.reshape(self.dir.nodes_direction,(2,self.par.n_bact*self.par.n_nodes))
        xy_dir = np.reshape(np.repeat(nodes_direction_flat, self.par.kn), (2,self.par.n_bact*self.par.n_nodes,self.par.kn))

        # Create the condition in the angle view
        cond_angle_view = (xy_dir[0] * x_dir_n + xy_dir[1] * y_dir_n) > np.cos(self.par.at_angle_view / 2)

        # Force intensity of the head attraction from its closest neighbour
        f_norm = self.f_at_norm(d=self.par.width, r=self.nei.dist)
        f_norm[self.nei.cond_same_bact | ~cond_angle_view] = 0.

        # Search the closest neigbour with non 0 values (return 0 if there is no neighbour)
        ind_first_non_zero = (f_norm != 0).argmax(axis=1)
        f_norm = f_norm[self.bact_enum_body,ind_first_non_zero]

        # Reshape direction and keep direction with the closest neighbour of the head
        f_x = x_dir_n[self.bact_enum_body,ind_first_non_zero] * f_norm
        f_y = y_dir_n[self.bact_enum_body,ind_first_non_zero] * f_norm

        # Apply the forces on the points
        f_att = np.array([np.reshape(f_x,(self.par.n_nodes,self.par.n_bact)),np.reshape(f_y,(self.par.n_nodes,self.par.n_bact))])

        # External Parrallal forces
        f_att_norm_par = np.sum(f_att[:,:,:] * self.dir.nodes_direction[:,:,:], axis=0)
        f_att_par = f_att_norm_par[:,:] * self.dir.nodes_direction[:,:,:]

        # Compute the perpendicular force et set the norm of the force as the initial force
        f_att_per = f_att[:,:,:] - f_att_par[:,:,:]
        f_att_norm_per = np.linalg.norm(f_att_per, axis=0)
        # Apply on the perpendicular force the same intensity than the total force
        f_att_norm_per[f_att_norm_per == 0] = np.inf
        f_att_per = f_att_per / f_att_norm_per * np.linalg.norm(f_att, axis=0)

        # Apply all the attraction on the head and only the perpendicular on the body
        self.gen.data[:,0,:] += f_att[:,0,:] * self.par.dt**2
        self.pha.data_phantom[:,0,:] += f_att[:,0,:] * self.par.dt**2
        self.gen.data[:,1:,:] += f_att_per[:,1:,:] * self.par.dt**2
        self.pha.data_phantom[:,1:,:] += f_att_per[:,1:,:] * self.par.dt**2

