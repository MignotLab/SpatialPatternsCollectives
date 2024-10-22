import numpy as np
import parameters


class MultiLayers:
    """
    Provide new dimension for the multi-layers in the simulation.
    The cell which are not in the same layer cannot interact by repulsive contact
    
    """

    def __init__(self):
        
        self.par = parameters.Parameters()

        # Class object
        self.cond_second_dim = np.zeros((self.par.n_nodes*self.par.n_bact,self.par.kn),dtype=bool)
        self.cond_different_layer = np.zeros((self.par.n_nodes*self.par.n_bact,self.par.kn),dtype=bool)


    def set_layer_condition(self,dist,ind,cond_same_bact):
        """
        Build an array of size (n_nodes*n_bact,kn) to know which bacteria are in
        the second dimension of the z axis.
        The condition to go in the second dimension for a node is to have a neighbor
        closer than 3/4*width. The priority is done for a cell having already nodes 
        in the second dimension, otherwise it is 50/50.
        The condition to pass from the second dimension to the first dimension is to
        have all neighbours far than 3/4*width.
        A cell which already have a node in the second dimension could put its
        adjacent nodes more easier in the second dimension also?
        2 voisin < 3/4 * width minimum pour monter dans le multi-layer ?
        
        """

        # Condition to know which bacteria are to close in the first dimension
        cond_dist_close = (dist < self.par.min_dist_multi_layer) & (~cond_same_bact) & self.cond_different_layer

        # Condition to know how many nodes a bacteria have in the second dimension

        # Condition to know if a node can return in the first dimension
        cond_return_first_layer = np.sum(cond_dist_close, axis=1) > 1
        cond_dist = (dist > self.par.min_dist_multi_layer) & (~cond_same_bact)

        # Condition to know if a node have to go to the second dimension

        # Condition to know if two neighbours are in different dimension
