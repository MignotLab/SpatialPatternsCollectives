import numpy as np

class AlignmentTypeError(Exception):
    pass

class Alignment:
    """
    Align the cell globally or with neighbourhood.
    alignment_type could be: "head_alignment", "global_alignment", "head_alignment_and_global_alignment", "head_alignment_and_global_alignment_specific_space", "no_aligment"; default is "no_alignment".
    
    """
    def __init__(self,inst_par,inst_gen,inst_pha,inst_dir,inst_nei,inst_tool):
        
        # Instance objects
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.dir = inst_dir
        self.nei = inst_nei
        self.tool = inst_tool

        if self.par.alignment_type == 'head_alignment':
            self.chosen_alignment_fonction = self.head_alignment
        elif self.par.alignment_type == 'global_alignment':
            self.chosen_alignment_fonction = self.global_alignment
        elif self.par.alignment_type == 'head_alignment_and_global_alignment':
            self.chosen_alignment_fonction = self.head_alignment_and_global_alignment
        elif self.par.alignment_type == 'no_alignment':
            self.chosen_alignment_fonction = self.function_doing_nothing
        else:
            # Print the different possible alignment when the class is call
            print('alignment_type could be: "head_alignment", "global_alignment", "head_alignment_and_global_alignment", "head_alignment_and_global_alignment_specific_space", "no_aligment"; default is "no_alignment"\n')
            raise AlignmentTypeError()

        self.nodes_head_enum = np.arange(self.par.n_bact).astype(self.par.int_type)

    def function_alignment_type(self):
        """
        Function doing the movement type and the random movement depending on the choice of these movement in the parameter class
        
        """
        self.chosen_alignment_fonction()

    def nodes_angle(self,x,y):
        """
        Translate the direction of the nodes into angle between [-pi/2,pi/2]
        
        """
        return np.arctan2(y, x)
    
    def function_doing_nothing(self):
        """
        Do nothing

        """
        pass
    
    def head_alignment(self):
        """
        Rotate the head to align it with its closest neighbour
        
        """
        # Copy
        nei_dir = self.dir.neighbours_direction[:,:self.par.n_bact,:].copy()
        # Angle view condition
        xy_dir = np.reshape(np.repeat(self.dir.nodes_direction[:,0,:], self.par.kn), (2,self.par.n_bact,self.par.kn))
        cond_angle_view = (xy_dir[0] * self.dir.nodes_to_nei_dir_torus[0,:self.par.n_bact,:] + xy_dir[1] * self.dir.nodes_to_nei_dir_torus[1,:self.par.n_bact,:]) > np.cos(self.par.at_angle_view / 2)
        # Condition for the distance between nodes and neighbour
        cond_dist = self.nei.dist[:self.par.n_bact] > self.par.max_align_dist
        nei_dir[:,self.nei.cond_same_bact[:self.par.n_bact] | cond_dist | ~cond_angle_view] = 0.
        # Find the id of the closest neighbour to align with
        ind_first_non_zero = ((nei_dir[0] != 0) | (nei_dir[1] != 0)).argmax(axis=1)
        # Keep the direction of the first closest neighbours which is not imself and which is enough close
        nei_dir = nei_dir[:,self.nodes_head_enum,ind_first_non_zero]
        # Compute the angle of nodes and of the closest neighbour of the node
        neighbours_angle_head = self.nodes_angle(x=nei_dir[0,:],y=nei_dir[1,:])
        nodes_angle_head = np.arctan2(self.dir.nodes_direction[1,0,:], self.dir.nodes_direction[0,0,:])
        # Compute angle the cell have to rotate to align with its closest neighbour
        angle = self.par.j_t * np.sin(2 * (neighbours_angle_head - nodes_angle_head)) * self.par.dt
        # Put angle to rotate to 0 if the cell have not close neighbour
        cond_no_neighbours = (nei_dir[0,:] == 0.) & (nei_dir[1,:] == 0.)
        angle[cond_no_neighbours] = 0.
        # Construct the matrice of rotation
        rotation_matrix = self.tool.rotation(theta=angle)
        # Update data and data_phanto by applying the rotation on the head node
        self.gen.data[:,0,:] = np.sum(rotation_matrix * (self.pha.data_phantom[:,0,:]-self.pha.data_phantom[:,1,:]), axis=1) + self.gen.data[:,1,:]
        self.pha.data_phantom[:,0,:] = np.sum(rotation_matrix * (self.pha.data_phantom[:,0,:]-self.pha.data_phantom[:,1,:]), axis=1) + self.pha.data_phantom[:,1,:]

    def global_alignment(self):
        """
        The cells follow a global direction in a part of the space
        
        """
        nodes_angle_head = self.nodes_angle(x=self.dir.nodes_direction[0,0,self.gen.cond_space_alignment],y=self.dir.nodes_direction[1,0,self.gen.cond_space_alignment])
        angle = self.par.j_t * np.sin(2 * (self.par.global_angle - nodes_angle_head)) * self.par.dt
        rotation_matrix = self.tool.rotation(theta=angle)
        # Update data and data_phanto by applying the rotation on the head node
        rotation_data = np.sum(rotation_matrix * (self.pha.data_phantom[:,0,self.gen.cond_space_alignment]-self.pha.data_phantom[:,1,self.gen.cond_space_alignment]), axis=1) + self.gen.data[:,1,self.gen.cond_space_alignment]
        rotation_data_phantom = np.sum(rotation_matrix * (self.pha.data_phantom[:,0,self.gen.cond_space_alignment]-self.pha.data_phantom[:,1,self.gen.cond_space_alignment]), axis=1) + self.pha.data_phantom[:,1,self.gen.cond_space_alignment]
        self.gen.data[:,0,self.gen.cond_space_alignment] = rotation_data[:,:]
        self.pha.data_phantom[:,0,self.gen.cond_space_alignment] = rotation_data_phantom[:,:]

    def head_alignment_and_global_alignment(self):
        """
        Combination of the head_alignment and global_alignment functions
        
        """
        self.head_alignment()
        self.global_alignment()

