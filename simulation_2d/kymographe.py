from turtle import color
import tools
import numpy as np
import matplotlib.pyplot as plt

class Kymograph:

    def __init__(self,inst_par,inst_gen,T):
        
        self.par = inst_par
        self.gen = inst_gen
        self.tools = tools.Tools()
        self.T = T
        self.id_save_node = int(self.par.n_nodes / 2)
        # Class object
        self.save = np.zeros(self.par.n_bact).astype(bool)
        self.position_array = np.zeros((2,int(self.T/self.par.save_freq_kymo),self.par.n_bact), dtype=self.par.float_type)
        self.time_array = np.tile(np.arange(0,self.T,self.par.save_freq_kymo).astype(self.par.float_type),(self.par.n_bact,1)).T
        # Object for the computation of the density kymograph
        self.count = 0
        self.hist_kymo = None
        self.bins_kymo = self.par.space_size
        self.density_kymo = np.zeros((int(self.T/self.par.save_freq_kymo+2),self.bins_kymo), dtype=self.par.float_type)

    def build_kymograph_density(self, index, save_kymo):
        """
        Plot the kymograph along x or y axis
        
        """
        if save_kymo and (index % int(1/self.par.dt*self.par.save_freq_kymo) == 0):
            cond_slice = (self.gen.data[1, self.id_save_node, :] > self.par.space_size / 2 - self.par.slice_width / 2) & (self.gen.data[1, self.id_save_node, :] < self.par.space_size / 2 + self.par.slice_width / 2)
            # cond_slice = cond_slice[0, :] & cond_slice[1, :]
            self.hist_kymo, __, __ = np.histogram2d(self.gen.data[1, self.id_save_node, cond_slice], self.gen.data[0, self.id_save_node, cond_slice], bins=self.bins_kymo)
            self.density_kymo[self.count, :] = np.sum(self.hist_kymo, axis=0)
            self.count += 1

    def save_position(self,index):
        """
        Save position at each time step save_frequency
        
        """
        if index % int(1/self.par.dt*self.par.save_freq_kymo) == 0:
            self.position_array[:,int(index*self.par.dt/self.par.save_freq_kymo),:] = self.gen.data[:,self.id_save_node,:]
   
    def rectangle(self,axis_coord,width):
        """
        Find the position of the 4 corners of the rectangle where axis_coord cut into two parts the rectangle
        
        """
        axis_vector = axis_coord[1,:] - axis_coord[0,:]
        axis_vector = axis_vector / np.linalg.norm(axis_vector)
        rotation_matrix = self.tools.rotation(theta=np.pi/2)
        # perp_axis_vector = axis_vector * rotation_matrix
        perp_axis_vector = np.sum(rotation_matrix * axis_vector, axis=1)
        x1, y1 =  perp_axis_vector * width/2 + axis_coord[0,:]
        x2, y2 = -perp_axis_vector * width/2 + axis_coord[0,:]
        x3, y3 = -perp_axis_vector * width/2 + axis_coord[1,:]
        x4, y4 =  perp_axis_vector * width/2 + axis_coord[1,:]

        return x1, y1, x2, y2, x3, y3, x4, y4

    def triangle_area(self,x1,y1,x2,y2,x3,y3):
        """
        A utility function to calculate area of triangle formed by (x1, y1), (x2, y2) and (x3, y3)
        
        """
        return np.abs((x1 * (y2 - y3) +
                       x2 * (y3 - y1) +
                       x3 * (y1 - y2)) / 2.0)

    def inside_rectangle_check(self,X,x,y):
        """
        A function to check whether point P(x, y) lies inside the rectangle
        formed by A(X[0], X[1]), B(X[2], X[3]), C(X[4], X[5]) and D(X[6], X[7])
        formed by A(x1,   y1),   B(x2,   y2),   C(x3,   y3) and   D(x4,   y4)
        
        """
        # Calculate area of rectangle ABCD
        A = (self.triangle_area(X[0], X[1], X[2], X[3], X[4], X[5]) +
             self.triangle_area(X[0], X[1], X[6], X[7], X[4], X[5]))
        # Calculate area of triangle PAB
        A1 = self.triangle_area(x, y, X[0], X[1], X[2], X[3])
        # Calculate area of triangle PBC
        A2 = self.triangle_area(x, y, X[2], X[3], X[4], X[5])
        # Calculate area of triangle PCD
        A3 = self.triangle_area(x, y, X[4], X[5], X[6], X[7])
        # Calculate area of triangle PAD
        A4 = self.triangle_area(x, y, X[0], X[1], X[6], X[7])
        # Check if sum of A1, A2, A3 and A4 is same as A
        eps = 10e-3
        return ((A < A1 + A2 + A3 + A4 + eps) & (A > A1 + A2 + A3 + A4 - eps))

    def position_inside_rectangle(self,position_array,X,axis_coord):
        """
        Save the positions of the cells being inside the defines rectangle
        
        """
        cond_inside_rectangle = self.inside_rectangle_check(X=X,x=position_array[0,:,:],y=position_array[1,:,:])
        # Take into account the periodic boundaries, we want to cut the trajectories during the teleportation
        # cond_inside_rectangle = cond_inside_rectangle & (~self.bound.cond_boundary_0 | ~self.bound.cond_boundary_l)
        # Project position on the rectangle axis
        norm_axis = np.linalg.norm(axis_coord[1] - axis_coord[0])
        position_projection_x = (position_array[0,:,:] - axis_coord[0,0]) * (axis_coord[1,0] - axis_coord[0,0])
        position_projection_y = (position_array[1,:,:] - axis_coord[0,1]) * (axis_coord[1,1] - axis_coord[0,1])
        position_projection = (position_projection_x + position_projection_y) / norm_axis
        # Replace to nan position which are not inside the rectangle
        position_projection[~cond_inside_rectangle] = np.nan
        # Select trajectories one by one and fill them into a list
        # Also cut the trajectories which change bounds
        trajectories = []
        times = []
        for i in range(self.par.n_bact):
            trajectory_i = []
            time_i = []
            count = 1
            while count < int(self.T/self.save_freq):
                cond1 = ~np.isnan(position_projection[count,i])
                cond2 = np.abs(position_projection[count,i] - position_projection[count-1,i]) < (self.save_freq * self.par.v0)*2
                if cond1 & cond2:
                    trajectory_i.append(position_projection[count,i])
                    time_i.append(self.time_array[count,i])
                    count += 1
                else:
                    trajectories.append(trajectory_i)
                    times.append(time_i)
                    trajectory_i = []
                    time_i = []
                    count += 1
            trajectories.append(trajectory_i)
            times.append(time_i)
        return trajectories, times

    def display_kymograph(self,trajectories,times):
        """
        Display the kymograph where trajectories are project on the chosen axis
        
        """
        fig, ax = plt.subplots(figsize=(10,10))
        for i in range(len(trajectories)):
            ax.plot(trajectories[i], times[i], color="limegreen", alpha=0.1, linewidth=1)
