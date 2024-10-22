import numpy as np

class Direction:

    def __init__(self,inst_par,inst_gen,inst_pha,inst_nei):
        
        # Parameters from parameters.py
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.nei = inst_nei

        ### Object of the class
        # set_nodes_direction
        self.nodes_direction = np.zeros((2,self.par.n_nodes,self.par.n_bact), dtype=self.par.float_type)
        self.nodes_distance = np.zeros((self.par.n_nodes,self.par.n_bact), dtype=self.par.float_type)
        # set_neighbours_direction
        self.neighbours_direction = np.zeros((2,int(self.par.n_bact * self.par.n_nodes),self.par.kn), dtype=self.par.float_type)
        # nodes_to_neighbours_direction_torus
        self.nodes_to_nei_dir_torus = np.zeros((2,int(self.par.n_bact * self.par.n_nodes),self.par.kn), dtype=self.par.float_type)
        self.nodes_to_nei_dist = np.zeros((int(self.par.n_bact * self.par.n_nodes),self.par.kn), dtype=self.par.float_type)

    def torus_dist(self, a, b):
        """
        Define the distance in a tore in 2d
        
        """

        dx = (a[0] - b[0] + self.par.space_size) % self.par.space_size
        dy = (a[1] - b[1] + self.par.space_size) % self.par.space_size

        return np.sqrt(np.minimum(dx,self.par.space_size-dx)**2 + np.minimum(dy,self.par.space_size-dy)**2)

    def set_nodes_direction(self):
        """
        Compute the direction and the distance between each nodes
        
        """

        self.nodes_direction[:,1:,:] = self.pha.data_phantom[:,:-1,:] - self.pha.data_phantom[:,1:,:]
        self.nodes_direction[:,0,:] = self.nodes_direction[:,1,:].copy()
        self.nodes_distance[:,:] = np.linalg.norm(self.nodes_direction, axis=0)
        # Normalize nodes_direction
        self.nodes_direction[:,:,:] = self.nodes_direction[:,:,:] / self.nodes_distance[:,:]

    def set_neighbours_direction(self):
        """
        Apply a force which on the head to align it with its closest neighbour
        
        """

        self.neighbours_direction = np.reshape(self.nodes_direction, (2, self.par.n_nodes * self.par.n_bact))[:,self.nei.ind]

    def nodes_to_neighbours_euclidian_direction(self,x_nodes,y_nodes,ind):
        """
        Compute the direction between the nodes and their k-neighbours
        
        """

        # Flatten the array of the neighbours indices
        ind_flat = np.concatenate(ind)

        # Define the coordinate of each bacterium self.k times (for each neighbour)
        x, y = np.repeat(x_nodes, self.par.k), np.repeat(y_nodes, self.par.k)

        # Compute the direction between the bacterium and their k-neighbour
        x_dir = x_nodes[ind_flat] - x
        y_dir = y_nodes[ind_flat] - y

        # Normalise the previous directions
        norm_dir = np.linalg.norm(np.array([x_dir,y_dir]), axis=0)
        neighbours_distance = norm_dir.copy()
        # Direction is null vector for superposed neighbours
        norm_dir[norm_dir == 0] = np.inf

        return np.reshape(x_dir / norm_dir, ind.shape), np.reshape(y_dir / norm_dir, ind.shape), neighbours_distance

    def nodes_torus_direction(self,data):
        """
        Compute the direction between the each node of the bacteria on a tore space, ind and dist must be computed on the tore space
        
        """

        nodes_euclidian_direction, nodes_euclidian_distance = self.nodes_euclidian_direction(data)
        nodes_torus_distance = self.torus_dist(a=data[:,:-1,:], b=data[:,1:,:])
        nodes_torus_distance = np.concatenate((nodes_torus_distance[0,:][None,:],nodes_torus_distance), axis=1)

        cond_dist = ~np.isclose(nodes_euclidian_distance, nodes_torus_distance)

        # We don't take into account the last point because it never be teleported to compute a distance
        cond_bottom_diagonal = (data[0,:-1,:] + data[1,:-1,:]) < self.par.space_size
        cond_bottom_anti_diagonal = (np.abs(data[0,:-1,:] - self.par.space_size) + data[1,:-1,:]) < self.par.space_size

        data_tmp = data.copy()
        data_tmp[0,:-1,:][cond_dist & cond_bottom_diagonal & ~cond_bottom_anti_diagonal] += self.par.space_size
        data_tmp[0,:-1,:][cond_dist & ~cond_bottom_diagonal & cond_bottom_anti_diagonal] -= self.par.space_size
        data_tmp[1,:-1,:][cond_dist & cond_bottom_diagonal & cond_bottom_anti_diagonal] += self.par.space_size
        data_tmp[1,:-1,:][cond_dist & ~cond_bottom_diagonal & ~cond_bottom_anti_diagonal] -= self.par.space_size

        nodes_torus_direction, __ = nodes_euclidian_direction(data_tmp)

        nodes_torus_direction[:,~cond_dist] = nodes_euclidian_direction[:,~cond_dist]

        return nodes_torus_direction, nodes_torus_distance

    def set_nodes_to_neighbours_direction_torus(self):
        """
        Compute the direction between neighbours on a tore space, ind and dist must be computed on the tore space
        
        """

        # Flatten the array of the neighbours indices
        x_nei_coord = self.nei.coord[:,0][self.nei.ind_flat]
        y_nei_coord = self.nei.coord[:,1][self.nei.ind_flat]

        # Define the coordinate of each bacterium self.k times (for each neighbour)
        x, y = np.repeat(self.nei.coord[:,0], self.par.kn), np.repeat(self.nei.coord[:,1], self.par.kn)

        # Compute the direction between the bacterium and their k-neighbour
        dist_euclidian_flat = np.sqrt((x_nei_coord - x)**2 + (y_nei_coord - y)**2)

        cond_dist = ~np.isclose(dist_euclidian_flat, self.nei.dist_flat)

        cond_bottom_diagonal = (x_nei_coord + y_nei_coord) < self.par.space_size
        cond_bottom_anti_diagonal = (np.abs(x_nei_coord - self.par.space_size) + y_nei_coord) < self.par.space_size

        x_nei_coord[cond_dist & cond_bottom_diagonal & ~cond_bottom_anti_diagonal] += self.par.space_size
        x_nei_coord[cond_dist & ~cond_bottom_diagonal & cond_bottom_anti_diagonal] -= self.par.space_size
        y_nei_coord[cond_dist & cond_bottom_diagonal & cond_bottom_anti_diagonal] += self.par.space_size
        y_nei_coord[cond_dist & ~cond_bottom_diagonal & ~cond_bottom_anti_diagonal] -= self.par.space_size

        # Compute the direction between the bacterium and their kn-neighbours
        x_dir = x_nei_coord - x
        y_dir = y_nei_coord - y

        # Normalise the previous directions
        norm_dir = np.linalg.norm(np.array([x_dir,y_dir]), axis=0)
        self.nodes_to_nei_dist[:,:] = np.reshape(norm_dir, self.nodes_to_nei_dist.shape)
        # Direction is null vector for superposed neighbours
        norm_dir[norm_dir == 0] = np.inf
        self.nodes_to_nei_dir_torus[:,:,:] = np.array([np.reshape(x_dir / norm_dir, self.nodes_to_nei_dist.shape),np.reshape(y_dir / norm_dir, self.nodes_to_nei_dist.shape)])



