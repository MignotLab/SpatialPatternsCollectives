import numpy as np

class RepulsionTypeError(Exception):
    
    pass

class Repulsion:
    """
    Compute interactions between cells (repulsion, attraction)
    
    """

    def __init__(self,inst_par,inst_gen,inst_pha,inst_dir,inst_nei):
        
        # Instance objects
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.dir = inst_dir
        self.nei = inst_nei

        if self.par.repulsion_type == 'repulsion':
            self.chosen_repulsion_fonction = self.repulsion
        elif self.par.repulsion_type == 'repulsion_propagation':
            self.chosen_repulsion_fonction = self.repulsion_propagation
        elif self.par.repulsion_type == 'repulsion_perpendicular':
            self.chosen_repulsion_fonction = self.repulsion_perpendicular
        elif self.par.repulsion_type == 'no_repulsion':
            self.chosen_repulsion_fonction = self.no_repulsion
        else:
            # Print the different possible alignment when the class is call
            print('repulsion_type could be: "repulsion", "repulsion_propagation", "repulsion_perpendicular", "no_repulsion"; default is "repulsion"\n')
            raise RepulsionTypeError()

        ### Objects of the class
        # repulsion
        self.f_norm = np.zeros((int(self.par.n_bact * self.par.n_nodes), self.par.kn), dtype=self.par.float_type)
        self.rep_force = np.zeros((2, self.par.n_nodes, self.par.n_bact), dtype=self.par.float_type)
        self.f_rep_norm = np.zeros((self.par.n_bact * self.par.n_nodes, self.par.kn), dtype=self.par.float_type)
        # repulsion_propagation
        self.f_rep_norm_ext = np.zeros(self.f_rep_norm.shape, dtype=self.par.float_type)
        self.f_rep_norm_int = np.zeros(self.f_rep_norm.shape, dtype=self.par.float_type)
        self.f_rep = np.zeros(self.gen.data.shape, dtype=self.par.float_type)
        self.f_rep_ext = np.zeros(self.gen.data.shape, dtype=self.par.float_type)
        self.f_rep_int = np.zeros(self.gen.data.shape, dtype=self.par.float_type)

        # Define array to generate condition if same point
        self.array_same_point = np.repeat((np.ones((self.par.kn, self.par.n_nodes), dtype=self.par.float_type) * np.arange(self.par.n_nodes)).T.astype(self.par.int_type), np.ones(self.par.n_nodes, dtype=self.par.int_type)*self.par.n_bact, axis=0)
        self.right_neighbour_node = np.roll(self.array_same_point, shift=-self.par.n_bact, axis=0)
        self.left_neighbour_node = np.roll(self.array_same_point, shift=+self.par.n_bact, axis=0)

    def function_repulsion_type(self):
        """
        Function doing the movement type and the random movement depending on the choice of these movement in the parameter class
        
        """
        self.chosen_repulsion_fonction()

    def function_rep_norm(self,d,r,k_r):
        """
        Repulsive force between two objects at distance r (the force is 0 if r > d)

        """
        force = -np.minimum(r - d, 0)**2 * (1 / np.maximum(r - 0.9 * d, 2 / k_r))

        return force

    def function_rep_norm_overlap(self,r,max_force,mu1,mu2,sig):
        """
        Experimental
        Repulsive force that allow overlap, the force decreases when to cells are really close
        
        """
        force = max_force*(np.exp(-np.power(r-mu1,2)/(2*np.power(sig,2))) + np.exp(-np.power(r-mu2,2)/(2*np.power(sig,2))))
        force[(r>mu2) & (r<mu1)] = max_force

        return -force
    
    def no_repulsion(self):
        """
        No repulsion
        
        """
        pass

    def repulsion(self):
        """
        Compute the repulsion between nodes
        
        """
        # Compute the repulsion force and change the distance 
        self.f_norm = self.function_rep_norm(d=self.par.width,r=self.nei.dist,k_r=self.par.k_r)
        # For the node which are on the same bact
        self.f_norm[self.nei.cond_same_bact & (self.right_neighbour_node==self.nei.id_node)][:int(self.par.n_bact*(self.par.n_nodes-1))] = 0
        self.f_norm[self.nei.cond_same_bact & (self.left_neighbour_node==self.nei.id_node)][self.par.n_bact:] = 0

        # Sum the repulsion force of all neighbours and reshape data
        self.rep_force[:,:,:] = np.reshape(np.sum(self.dir.nodes_to_nei_dir_torus[:,:,:] * self.f_norm, axis=2), self.gen.data[:,:,:].shape)

        # Apply the forces on the points
        self.gen.data += self.rep_force * self.par.dt**2
        self.pha.data_phantom += self.rep_force * self.par.dt**2

    def repulsion_propagation(self):
        """
        Compute the force on the node, and propagate the force on the head along the body of each cell taking into account their
        own direction
        
        """
        self.repulsion()
        # Apply the forces on the nec also on the head
        # rep_force[:,0,:] += rep_force[:,1,:]

        # All Parrallal forces
        force_norm_parallel = np.sum(self.rep_force[:,:,:] * self.dir.nodes_direction[:,:,:], axis=0)
        # force_parallel = force_norm_parallel[:,:] * self.dir.nodes_direction[:,:,:]

        # Propagate the force of the head and tail along the other points
        self.gen.data[:,1:,:] += self.dir.nodes_direction[:,1:,:] * force_norm_parallel[0,:] * self.par.dt**2
        self.gen.data[:,:-1,:] += self.dir.nodes_direction[:,:-1,:] * force_norm_parallel[-1,:] * self.par.dt**2
        self.gen.data_phantom[:,1:,:] += self.dir.nodes_direction[:,1:,:] * force_norm_parallel[0,:] * self.par.dt**2
        self.gen.data_phantom[:,:-1,:] += self.dir.nodes_direction[:,:-1,:] * force_norm_parallel[-1,:] * self.par.dt**2

    def repulsion_perpendicular(self):
        """
        Repulsion perpendicular to the local shape body of the neighbour
        
        """
        x_dir_n = self.dir.nodes_to_nei_dir_torus[0,:,:]
        y_dir_n = self.dir.nodes_to_nei_dir_torus[1,:,:]
        # Initialize distance of interaction 
        int_dist = np.ones((self.par.n_bact*self.par.n_nodes,self.par.kn))*self.width
        # direction between neighbours
        dir_nei = np.array([x_dir_n,y_dir_n])

        # Find the number of separated nodes between two neighbour
        length = np.abs(self.nei.id_node - self.array_same_point) * self.d_n
        cond_dist = length < self.par.width
        int_dist[self.nei.cond_same_bact & cond_dist] = length[self.nei.cond_same_bact & cond_dist] * self.par.d_n

        # Force norm itself, and neighbours
        self.f_rep_norm = self.function_rep_norm(d=int_dist, r=self.nei.dist, k_r=self.par.k_r)
        # Put a maximum force if and averlap is attemp
        # f_rep_norm[f_rep_norm < self.function_rep_norm(d=self.width, r=self.width/overlap)] = self.f_rep_norm(d=self.width, r=self.width/overlap)
        self.f_rep_norm_ext[~self.nei.cond_same_bact] = self.f_rep_norm[~self.nei.cond_same_bact]
        self.f_rep_norm_int[self.nei.cond_same_bact] = self.f_rep_norm[self.nei.cond_same_bact]

        # Sum the repulsion force of all neighbours and reshape data
        self.f_rep = np.reshape(np.sum(dir_nei * self.f_rep_norm, axis=2), self.gen.data.shape)
        self.f_rep_ext = np.reshape(np.sum(dir_nei * self.f_rep_norm_ext, axis=2), self.gen.data.shape)
        self.f_rep_int = np.reshape(np.sum(dir_nei * self.f_rep_norm_int, axis=2), self.gen.data.shape)

        # All Parrallal forces
        f_rep_norm_par = np.sum(self.f_rep[:,:,:] * self.dir.nodes_direction[:,:,:], axis=0)
        f_rep_par = f_rep_norm_par[:,:] * self.dir.nodes_direction[:,:,:]

        # External Parrallal forces
        f_rep_norm_par_ext = np.sum(self.f_rep_ext[:,:,:] * self.dir.nodes_direction[:,:,:], axis=0)
        f_rep_par_ext = f_rep_norm_par_ext[:,:] * self.dir.nodes_direction[:,:,:]

        # Compute the perpendicular force et set the norm of the force as the initial force
        f_rep_per_ext = self.f_rep_ext[:,:,:] - f_rep_par_ext[:,:,:]
        f_rep_norm_per_ext = np.linalg.norm(f_rep_per_ext, axis=0)
        # Apply on the perpendicular force the same intensity than the total force
        f_rep_norm_per_ext[f_rep_norm_per_ext == 0] = np.inf
        f_rep_per_ext = f_rep_per_ext / f_rep_norm_per_ext * np.linalg.norm(self.f_rep_ext, axis=0)

        # Apply perpendicular for on the body nodes
        self.gen.data[:,1:-1,:] += (f_rep_per_ext + self.f_rep_int)[:,1:-1,:] * self.par.dt**2
        self.pha.data_phantom[:,1:-1,:] += (f_rep_per_ext + self.f_rep_int)[:,1:-1,:] * self.par.dt**2

        # Propagate the force of the head and tail along the other points
        self.gen.data[:,1:,:] += self.dir.nodes_direction[:,1:,:] * f_rep_norm_par[0,:] * self.par.dt**2
        self.pha.data_phantom[:,1:,:] += self.dir.nodes_direction[:,1:,:] * f_rep_norm_par[0,:] * self.par.dt**2
        self.gen.data[:,:-1,:] += self.dir.nodes_direction[:,:-1,:] * f_rep_norm_par[-1,:] * self.par.dt**2
        self.pha.data_phantom[:,:-1,:] += self.dir.nodes_direction[:,:-1,:] * f_rep_norm_par[-1,:] * self.par.dt**2

        # Apply center to center direction force on the extremities
        self.gen.data[:,0,:] += self.f_rep[:,0,:] * self.par.dt**2
        self.pha.data_phantom[:,0,:] += self.f_rep[:,0,:] * self.par.dt**2
        self.gen.data[:,-1,:] += self.f_rep[:,-1,:] * self.par.dt**2
        self.pha.data_phantom[:,-1,:] += self.f_rep[:,-1,:] * self.par.dt**2


