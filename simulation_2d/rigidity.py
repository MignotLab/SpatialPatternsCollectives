import numpy as np

class RigidityTypeError(Exception):
    
    pass

class Rigidity:


    def __init__(self,inst_par,inst_gen,inst_pha):
        
        # Parameters from parameters.py
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha

        # Class object
        self.force_correction = np.zeros((2,self.par.n_nodes,self.par.n_bact), dtype=self.par.float_type)
        self.alpha = np.zeros((2,self.par.n_nodes-2,self.par.n_bact), dtype=self.par.float_type)

        if self.par.rigidity_type == True:
            self.chosen_rigidity_fonction = self.rigidity_force_nodes
        elif self.par.rigidity_type == False:
            self.chosen_rigidity_fonction = self.function_doing_nothing
        else:
            print('rigidity_type could be: True or False; default is False\n')
            raise RigidityTypeError()
        
    def function_rigidity_type(self):
        """
        Function doing the movement type and the random movement depending on the choice of these movement in the parameter class
        
        """

        self.chosen_rigidity_fonction()

    def function_doing_nothing(self):
        """
        No eps production and follower
        
        """
        pass

    def rigidity_force_nodes(self):
        """
        Compute the rigidity forces of the bacteria for each triplet of nodes
        
        """
        for i in range(self.par.rigidity_iteration):

            # We compute here w0 = (X_q - X_q+1) / d, w1 = (X_q+2 - X_q+1) / d and alpha = w0.w1
            w0 = (self.pha.data_phantom[:,:-2,:] - self.pha.data_phantom[:,1:-1,:]) / self.par.d_n
            w1 = (self.pha.data_phantom[:,2: ,:] - self.pha.data_phantom[:,1:-1,:]) / self.par.d_n
            self.alpha[0,:,:] = np.sum(w0*w1, axis=0)
            self.alpha[1,:,:] = np.sum(w0*w1, axis=0)

            # Here we compute all V^0, V^1 and V^2
            v0 = - self.par.k_rigid * (w1 - self.alpha * w0)
            v2 = - self.par.k_rigid * (w0 - self.alpha * w1)
            v1 = (- v0 - v2).copy()

            # Build an array of size (_N,_n_nodes) which compute the rigid force correction of each nodes
            self.force_correction[:,:-2,:] += v0
            self.force_correction[:,1:-1,:] += v1
            self.force_correction[:,2:,:] += v2

            # Apply the forces
            self.gen.data[:,self.par.rigidity_first_node:,:] += self.force_correction[:,self.par.rigidity_first_node:,:] * self.par.dt**2
            self.pha.data_phantom[:,self.par.rigidity_first_node:,:] += self.force_correction[:,self.par.rigidity_first_node:,:] * self.par.dt**2
