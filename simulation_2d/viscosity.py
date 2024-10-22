from parameters import Parameters
import numpy as np


class Viscosity:


    def __init__(self,inst_par,inst_gen,inst_pha):
        
        # Parameters from parameters.py
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha

        # Class object
        self.distance_t0 = np.ones((int(self.par.n_nodes*self.par.n_bact),self.par.kn))
        

    def viscosity_bact(self,distance_t1,x_dir_n,y_dir_n,nodes_direction,cond_same_bact,max_dist):
        """
        Apply a viscosity force between the closest nodes which are not from the same bacterium
        
        """

        # Re
        # Condition on the distance of the nodes to another
        cond_dist = self.distance_t0 > max_dist
        # Compute the change in the distance between t0 and t1
        distance_change = distance_t1 - self.distance_t0
        cond_dist_change = distance_change < 0

        # Reshape the nodes direction
        direction = np.reshape(nodes_direction,(2,int(self.par.n_nodes*self.par.n_bact)))
        direction = np.repeat(direction,self.par.kn).reshape(2,int(self.par.n_nodes*self.par.n_bact),self.par.kn)

        # Compute the attraction of each nodes
        # visc_force = np.array([x_dir_n,y_dir_n]) * distance_change
        visc_force = np.sign(np.sum(np.array([x_dir_n,y_dir_n])*direction, axis=0)) * direction * distance_change
        visc_force[:,cond_dist | cond_dist_change | cond_same_bact] = np.nan
        visc_force = np.nansum(visc_force,axis=2)
        visc_force = np.reshape(visc_force,(2,self.par.n_nodes,self.par.n_bact))

        # Apply the changement ont the positions of the nodes
        self.gen.data += self.par.epsilon_v * visc_force
        self.pha.data_phantom += self.par.epsilon_v * visc_force

        self.distance_t0 = distance_t1.copy()