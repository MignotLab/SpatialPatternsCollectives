from dataclasses import dataclass
import numpy as np
import directions


class BacteriaBody:
    """
    Manages the internal distances between the nodes of each bacteria
    
    """


    def __init__(self,inst_par,inst_gen,inst_pha,inst_dir):

        # Import parameters
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.dir = inst_dir

        # Object class
        self.stiffness_forces = np.zeros((2, self.par.n_nodes-1, self.par.n_bact), dtype=self.par.float_type)

    def head_follower(self):
        """
        Each node follow the node in front of it to keep the same distance 
        between each node after the head move
        
        """

        for i in range(1,self.par.n_nodes):
            
            self.dir.set_nodes_direction(data=self.pha.data_phantom)

            self.gen.data[:,i,:] += (self.dir.nodes_distance[i,:] - self.par.d_n) * self.dir.nodes_direction[:,i,:]
            self.pha.data_phantom[:,i,:] += (self.dir.nodes_distance[i,:] - self.par.d_n) * self.dir.nodes_direction[:,i,:]


    def nodes_spring(self):
        """
        Apply spring force between each nodes of the bacteria
        
        """

        for i in range(self.par.n_nodes*5):

            self.dir.set_nodes_direction()
            self.stiffness_forces[:,:,:] = -self.par.k_s * (self.dir.nodes_direction[:,1:,:] * self.dir.nodes_distance[1:,:] - self.dir.nodes_direction[:,1:,:] * self.par.d_n)
            # print(self.stiffness_forces)

            self.gen.data[:,:-1,:] += self.stiffness_forces * self.par.dt**2
            self.gen.data[:,1:,:] -= self.stiffness_forces * self.par.dt**2
            self.pha.data_phantom[:,:-1,:] += self.stiffness_forces * self.par.dt**2
            self.pha.data_phantom[:,1:,:] -= self.stiffness_forces * self.par.dt**2
            


