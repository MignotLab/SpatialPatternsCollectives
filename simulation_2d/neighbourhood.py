from scipy.spatial import KDTree as kdtree
import numpy as np

class Neighbours:
    """
    Search the neighbours of the bacteria
    
    """


    def __init__(self,inst_par,inst_gen):

        # Parameters from parameters.py
        self.par = inst_par
        self.gen = inst_gen

        # Change the shape of data to use the kdtree
        self.coord = np.zeros((int(self.par.n_bact * self.par.n_nodes),2), dtype=self.par.float_type)

        # Object class
        self.ind = np.zeros((int(self.par.n_bact * self.par.n_nodes),self.par.kn), dtype=self.par.int_type)
        self.dist = np.zeros(self.ind.shape)
        self.ind_flat = np.zeros(int(self.par.n_bact * self.par.n_nodes * self.par.kn), dtype=self.par.int_type)
        self.dist_flat = np.zeros(self.ind_flat.shape)
        self.id_node, self.id_bact = np.zeros(self.ind.shape, dtype=self.par.int_type), np.zeros(self.ind.shape, dtype=self.par.int_type)
        self.cond_same_bact = np.zeros(self.ind.shape, dtype=bool)
        # Define array to generate condition if same bacterium
        self.array_same_bact = np.tile((np.ones((self.par.kn, self.par.n_bact)) * np.arange(self.par.n_bact)).T.astype(self.par.int_type), (self.par.n_nodes,1))

    
    def tore_dist(self,a, b):
        """
        Define the distance in a tore in 2d
        
        """

        dx = (b[0] - a[0] + self.par.space_size) % self.par.space_size
        dy = (b[1] - a[1] + self.par.space_size) % self.par.space_size

        return np.sqrt(np.minimum(dx,self.par.space_size-dx)**2 + np.minimum(dy,self.par.space_size-dy)**2)


    def set_kn_nearest_neighbours_torus(self):
        """
        Detect the k closest neighbours in a torus space
        
        """
        self.coord[:,:] = np.column_stack((self.gen.data[0,:,:].flatten(), self.gen.data[1,:,:].flatten()))
        tree = kdtree(self.coord, boxsize=[self.par.space_size, self.par.space_size])
        self.dist[:,:], self.ind[:,:] = tree.query(self.coord, k=self.par.kn)
        self.dist_flat[:], self.ind_flat[:] = np.concatenate(self.dist), np.concatenate(self.ind)


    def set_kn_nearest_neighbours_euclidian(self):
        """
        Detect the k closest neighbours
        
        """

        self.coord[:,:] = np.column_stack((self.gen.data[0,:,:].flatten(), self.gen.data[1,:,:].flatten()))
        tree = kdtree(self.coord)
        self.dist[:,:], self.ind[:,:] = tree.query(self.coord, k=self.par.kn)
        self.dist_flat[:], self.ind_flat[:] = np.concatenate(self.dist), np.concatenate(self.ind)
        


    def set_bacteria_index(self):
        """
        Compute the index of the bacteria and the id of the nodes and also the condition to know
        if two neighbours of nodes are part of the same bacteria
        
        """

        self.id_node[:,:], self.id_bact[:,:] = np.divmod(self.ind, self.par.n_bact)
        self.cond_same_bact[:,:] = self.id_bact == self.array_same_bact
