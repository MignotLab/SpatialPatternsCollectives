import numpy as np

class MovementTypeError(Exception):

    pass

class RandomMovementTypeError(Exception):

    pass

class Move:
    """
    Generate movement of bacteria.
    movement_type could be: "tracted", "pushed", "a_motility", "a_motility_dynamic", "s_motility", "a_s_motility", "a_s_motility_dynamic", "tracted_s_motility"; default is "tracted".
    
    """

    def __init__(self, inst_par, inst_gen, inst_pha, inst_dir, inst_nei, inst_eps, inst_att, inst_vel, inst_tool):

        # Instance objects
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.dir = inst_dir
        self.nei = inst_nei
        self.eps = inst_eps
        self.att = inst_att
        self.vel = inst_vel
        self.tool = inst_tool

        if self.par.movement_type == 'tracted':
            self.chosen_motility_fonction = self.tracted_movement
        elif self.par.movement_type == 'pushed':
            self.chosen_motility_fonction = self.pushed_movement
        elif self.par.movement_type == 'a_motility':
            self.chosen_motility_fonction = self.a_motility
        elif self.par.movement_type == 'a_motility_dynamic':
            self.chosen_motility_fonction = self.a_motility_dynamic_focal_adhesion_point
        elif self.par.movement_type == 's_motility':
            self.chosen_motility_fonction = self.s_motility_bact
        elif self.par.movement_type == 'a_s_motility':
            self.chosen_motility_fonction = self.a_s_motility
        elif self.par.movement_type == 'a_s_motility_dynamic':
            self.chosen_motility_fonction = self.a_s_motility_dynamic
        elif self.par.movement_type == 'tracted_s_motility':
            self.chosen_motility_fonction = self.tracted_s_motility
        else:
            # Print the different possible movement when the class is call
            print('movement_type could be: "tracted", "pushed", "a_motility", "a_motility_dynamic", "s_motility", "a_s_motility", "a_s_motility_dynamic", "tracted_s_motility"; default is "tracted"\n')
            raise MovementTypeError()
        
        if self.par.random_movement == True:
            self.chosen_random_fonction = self.random_movement
        elif self.par.random_movement == False:
            self.chosen_random_fonction = self.function_doing_nothing
        else:
            print('random_movement could be: True or False, default is False\n')
            raise RandomMovementTypeError()

        # For s motility part
        self.bact_enum = np.arange(self.par.n_bact).astype(self.par.int_type)
        
        # Object from class
        self.half_population = 0
        self.cond_s_motility = 0

        # Velocities
        self.v0_tracted = self.par.v0 * self.par.n_nodes

        ### Not used, tu use the related function, uncomment
        # self.v0_variable = np.zeros((2,self.par.n_nodes,self.par.n_bact))
        # self.v0_variable[:,:,0] = self.par.v0
        # self.v0_variable_tracted = self.v0_variable * self.par.n_nodes

        ### Not used, tu use the related function, uncomment
        # # A motility
        # self.last_nodes_focal = self.par.n_nodes - 2
        # self.corrected_velocity = self.par.n_nodes / self.par.nb_adhesion_points * self.par.v0
        # self.position_focal_adhesion_point = np.zeros((2,self.par.n_nodes,self.par.n_bact), dtype=bool)
        # pos_focal_adhesion_point_1 = np.random.randint(low=1,high=self.last_nodes_focal,size=self.par.n_bact, dtype=int)
        # pos_focal_adhesion_point_2 = np.zeros(self.par.n_bact, dtype=int)
        # # pos_focal_adhesion_point_2 = pos_focal_adhesion_point_1 + int(self.par.n_nodes/2)
        # # pos_focal_adhesion_point_2[pos_focal_adhesion_point_2>self.par.n_nodes-1] -= self.par.n_nodes
        
        # for i in range(self.par.n_bact):
            
        #     p1 = pos_focal_adhesion_point_1[i]
        #     p2 = pos_focal_adhesion_point_2[i]
        #     self.position_focal_adhesion_point[:,p1,i] = True
        #     self.position_focal_adhesion_point[:,p2,i] = True

        # self.parallel_displacement = np.zeros(self.par.n_bact)

    def function_movement_type(self):
        """
        Function doing the movement type and the random movement depending on the choice of these movement in the parameter class
        
        """
        self.chosen_motility_fonction()
        self.chosen_random_fonction()

    def function_doing_nothing(self):
        """
        Function doing nothing

        """
        pass

    def random_movement(self):
        """
        Apply the random displacement on the head of the bacteria
        
        """
        # Construct the matrice of rotation
        noise = np.random.normal(loc=0, scale=self.par.sigma_random, size=self.par.n_bact).astype(self.par.float_type)
        rotation_matrix = self.tool.rotation(theta=noise)

        # Update data and data_phanto by applying the rotation on the head node
        self.gen.data[:,0,:] = np.sum(rotation_matrix * (self.pha.data_phantom[:,0,:]-self.pha.data_phantom[:,1,:]), axis=1) + self.gen.data[:,1,:]
        self.pha.data_phantom[:,0,:] = np.sum(rotation_matrix * (self.pha.data_phantom[:,0,:]-self.pha.data_phantom[:,1,:]), axis=1) + self.pha.data_phantom[:,1,:]

    def a_motility_dynamic_focal_adhesion_point(self):
        """
        Velocity at norm v0 from the head and the focal adhesion points

        """
        ### Focal adhesion points dynamics
        # The displacement of the last node is sufficient to know
        self.parallel_displacement += np.mean(np.sum(self.dir.nodes_direction[:,:,:] * self.vel.displacement[:,:,:], axis=0), axis=0)
        # Condition to know if the cell provide a displacement of a lenght equal to a node
        cond_change_node = self.parallel_displacement > self.par.d_n
        # Roll the array of the position of the focal adhesion point if the previous condition is True, and roll only the second point
        self.position_focal_adhesion_point[:,1:self.last_nodes_focal,cond_change_node] = np.roll(self.position_focal_adhesion_point[:,1:self.last_nodes_focal,:], shift=1, axis=1)[:,:,cond_change_node]
        # Reinitialize the displacemeent to 0 when the nodes are change
        self.parallel_displacement[cond_change_node] = 0

        ### Movement
        self.gen.data[self.position_focal_adhesion_point] += self.corrected_velocity * self.dir.nodes_direction[self.position_focal_adhesion_point] * self.par.dt
        self.pha.data_phantom[self.position_focal_adhesion_point] += self.corrected_velocity * self.dir.nodes_direction[self.position_focal_adhesion_point] * self.par.dt

    def a_motility(self):
        """
        Velocity at norm v0 from the head and the focal adhesion points

        """
        step = int(self.par.n_nodes / self.par.nb_adhesion_points)
        self.gen.data[:,0::step,:] += self.corrected_velocity * self.dir.nodes_direction[:,0::step,:] * self.par.dt
        self.pha.data_phantom[:,0::step,:] += self.corrected_velocity * self.dir.nodes_direction[:,0::step,:] * self.par.dt

    def s_motility_bact(self):
        """
        Velocity triggers by the attraction force of the pili
        
        """
        dir_head_to_nei = self.dir.nodes_to_nei_dir_torus[:,:self.par.n_bact,:]
        # Weight the attraction force depending on the direction of the pili conpare the the head direction
        weight_direction = (self.dir.nodes_direction[0,0,:]*dir_head_to_nei[0,:,:].T + self.dir.nodes_direction[1,0,:]*dir_head_to_nei[1,:,:].T).T
        # Condition for the angle view of the pili
        cond_angle_view = weight_direction > np.cos(self.par.at_angle_view / 2)
        # Force of the pili attraction
        f_a_pili = self.att.f_at_pili_norm(r=self.nei.dist[:self.par.n_bact], k=self.par.k_a_pili, w0=self.par.width, w1=(self.par.pili_length+self.par.width)/2, w2=self.par.pili_length)
        f_a_pili *= weight_direction
        f_a_pili[~cond_angle_view] = 0.

        # Search the most visible neigbour with non 0 values (return 0 if there is no neighbour)
        ind_more_visible = weight_direction.argmax(axis=1)
        f_a_pili = f_a_pili[self.bact_enum,ind_more_visible]

        # Keep direction with the closest neighbour of the head weighted by the angle
        force = dir_head_to_nei[:,self.bact_enum,ind_more_visible] * f_a_pili

        # Apply the forces on the head
        self.gen.data[:,0,:] += force * self.par.dt**2
        # self.data[:,1,:] += force * self.par.dt**2
        self.pha.data_phantom[:,0,:] += force * self.par.dt**2


    def a_s_motility(self):
        """
        Combination of A and S motility
        
        """
        self.a_motility()
        self.s_motility_bact()

    def a_s_motility_dynamic(self):
        """
        Combination of A dynamic and S motility
        
        """
        self.a_motility_dynamic_focal_adhesion_point()
        self.s_motility_bact()

    def tracted_s_motility(self):
        """
        Combination of tracted and s_motility

        """
        self.tracted_movement()
        self.s_motility_bact()
    
    def pushed_movement(self):
        """
        Velocity at norm v0 in the direction of the head

        """
        # Nodes displacement
        self.gen.data[:,:,:] += self.par.v0 * self.dir.nodes_direction[:,:,:] * self.dt
        self.pha.data_phantom[:,:,:] += self.par.v0 * self.dir.nodes_direction[:,:,:] * self.dt

    def tracted_movement(self):
        """
        Velocity at norm v0 in the direction of the head

        """
        # Head displacement
        self.gen.data[:,0,:] += self.v0_tracted * self.dir.nodes_direction[:,0,:] * self.par.dt
        self.pha.data_phantom[:,0,:] += self.v0_tracted * self.dir.nodes_direction[:,0,:] * self.par.dt

    def desired_velocity_v0_variable(self, nodes_direction):
        """
        Velocity at norm v0 in the direction of the head

        """
        # Nodes displacement
        self.gen.data[:,:,:] += (self.v0_variable * self.dir.nodes_direction)[:,:,:] * self.par.dt
        self.pha.data_phantom[:,:,:] += (self.v0_variable * self.dir.nodes_direction)[:,:,:] * self.par.dt

    def desired_velocity_v0_variable_tracted(self, nodes_direction):
        """
        Velocity at norm v0 in the direction of the head

        """
        # Head displacement
        self.gen.data[:,0,:] += self.v0_variable_tracted[:,0,:] * self.dir.nodes_direction[:,0,:] * self.par.dt
        self.pha.data_phantom[:,0,:] += self.v0_variable_tracted[:,0,:] * self.dir.nodes_direction[:,0,:] * self.par.dt
