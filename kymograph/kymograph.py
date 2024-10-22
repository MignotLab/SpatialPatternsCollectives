# %%
import numpy as np
import pandas as pd
from tqdm import tqdm
from sympy import symbols, Eq, solve
import matplotlib.pyplot as plt
from nd2reader import ND2Reader
from skimage.io import imread, imsave
from tqdm import tqdm

import parameters
import tools


class Kymograph:


    def __init__(self):
         
         self.par = parameters.Parameters()
         self.tool = tools.Tools()


    def find_line_eq(self, xi, yi, xf, yf):
        """
        Find equation of a line
        
        """
        a = (yi-yf)/(xi-xf)
        b = yi-a*xi

        return a, b


    def solve_system(self, xi, yi, xf, yf, e):
        """
        Solve system of to equation
        
        """
        x, y = symbols('x y')
        eq1 = Eq((x-xi) * (xf-xi) + (y-yi) * (yf-yi), 0)
        eq2 = Eq((x-xi)**2 + (y-yi)**2 - e**2, 0)

        return solve((eq1,eq2), (x, y))
        
    
    def find_axe_equation(self, xi, yi, xf, yf, e):
        """
        Return the eqsuation of the main axe of the rectangle
        
        """
        self.xi = xi
        self.yi = yi
        self.xf = xf
        self.yf = yf
        self.e = e
        # Find equation of the axe of the rectangle
        self.a_axe, self.b_axe = self.find_line_eq(self.xi, self.yi, self.xf, self.yf)
        # Find the four point of the rectangle where the previous axe cut it into two equal part
        self.xb, self.yb = float(self.solve_system(self.xi, self.yi, self.xf, self.yf, self.e)[0][0]), float(self.solve_system(self.xi, self.yi, self.xf, self.yf, self.e)[0][1])
        self.xa, self.ya = float(self.solve_system(self.xi, self.yi, self.xf, self.yf, self.e)[1][0]), float(self.solve_system(self.xi, self.yi, self.xf, self.yf, self.e)[1][1])
        self.xc, self.yc = float(self.solve_system(self.xf, self.yf, self.xi, self.yi, self.e)[0][0]), float(self.solve_system(self.xf, self.yf, self.xi, self.yi, self.e)[0][1])
        self.xd, self.yd = float(self.solve_system(self.xf, self.yf, self.xi, self.yi, self.e)[1][0]), float(self.solve_system(self.xf, self.yf, self.xi, self.yi, self.e)[1][1])
        # Find equation of line of the rectangle (a, b, c, d)
        self.a_ab, self.b_ab = self.find_line_eq(self.xa, self.ya, self.xb, self.yb)
        self.a_bc, self.b_bc = self.find_line_eq(self.xb, self.yb, self.xc, self.yc)
        self.a_cd, self.b_cd = self.find_line_eq(self.xc, self.yc, self.xd, self.yd)
        self.a_da, self.b_da = self.find_line_eq(self.xd, self.yd, self.xa, self.ya)


    def visualize_rectangle(self, im, figsize, linewidth, color='Greys_r'):
        """
        Visualize the rectangle on your image
        
        """
        self.fig, ax = plt.subplots(figsize=figsize)
        cmap = plt.get_cmap(color)
        ax.imshow(im, cmap=cmap)
        ax.plot(np.array([self.xa, self.xb, self.xc, self.xd, self.xa]), np.array([self.ya, self.yb, self.yc, self.yd, self.ya]), color="r", linewidth=linewidth)
        # ax.plot(np.array([self.xi, self.xf]), np.array([self.yi, self.yf]), color="r", linewidth=linewidth, linestyle="--")
        ax.axis('off')
        self.fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
        self.fig.subplots_adjust(wspace=0,hspace=0)


    
    def area(self, x1, y1, x2, y2, x3, y3):
        """
        A utility function to calculate area of triangle formed by (x1, y1), (x2, y2) and (x3, y3)
        
        """
        return np.abs((x1 * (y2 - y3) +
                       x2 * (y3 - y1) +
                       x3 * (y1 - y2)) / 2.0)
    

    def check(self, x1, y1, x2, y2, x3, y3, x4, y4, x, y):
        """
        A function to check whether point P(x, y) lies inside the rectangle formed by A(x1, y1), B(x2, y2), C(x3, y3) and D(x4, y4)
        
        """     
        # Calculate area of rectangle ABCD
        A = (self.area(x1, y1, x2, y2, x3, y3) +
             self.area(x1, y1, x4, y4, x3, y3))
        # Calculate area of triangle PAB
        A1 = self.area(x, y, x1, y1, x2, y2)
        # Calculate area of triangle PBC
        A2 = self.area(x, y, x2, y2, x3, y3)
        # Calculate area of triangle PCD
        A3 = self.area(x, y, x3, y3, x4, y4)
        # Calculate area of triangle PAD
        A4 = self.area(x, y, x1, y1, x4, y4);
        # Check if sum of A1, A2, A3
        # and A4 is same as A
        eps = 10e-3

        return ((A < A1 + A2 + A3 + A4 + eps) & (A > A1 + A2 + A3 + A4 - eps))
    

    def extract_traj_inside_rectangle(self, df, width_bins, ti, tf):
        """
        Extract the trajectories that are inside a rectangle
        If a cell go outside the rectangle and come again the traj_id will be different
        
        """
        self.ti = ti
        self.tf = tf
        df.sort_values(by=[self.par.track_id_column, self.par.t_column], ignore_index=True, inplace=True)
        cond_time_min = df.loc[:, self.par.t_column] >= self.ti / self.par.tbf
        cond_time_max = df.loc[:, self.par.t_column] <= self.tf / self.par.tbf
        df_time = df.loc[cond_time_min & cond_time_max].copy()
        x = df_time.loc[:, self.par.x_column].values
        y = df_time.loc[:, self.par.y_column].values
        # t = df.loc[:, self.par.t_column].values

        cond_inside_rectangle = self.check(self.xa, self.ya, self.xb, self.yb, self.xc, self.yc, self.xd, self.yd, x, y)
        self.df_rectangle = df_time.loc[cond_inside_rectangle, [self.par.t_column, self.par.track_id_column, self.par.x_column, self.par.y_column, self.par.rev_column, self.par.len_traj_column]]

        # Project all the coordinate on the rectangle axis
        rectangle_axe_vector = np.array([self.xi-self.xf, self.yi-self.yf])
        rectangle_axe_vector = rectangle_axe_vector / np.linalg.norm(rectangle_axe_vector)
        x = self.df_rectangle.loc[:, self.par.x_column].values
        y = self.df_rectangle.loc[:, self.par.y_column].values
        coords_vector = np.array([self.xi-x, self.yi-y])
        self.xy_projection = rectangle_axe_vector.dot(coords_vector)

        self.len_axe = np.sqrt((self.xf-self.xi)**2 + (self.yf-self.yi)**2) * self.par.scale
        end = round(self.len_axe) + 2
        nb_bins_axe = len(np.arange(0, end, width_bins))
        coords_projection = rectangle_axe_vector.dot(coords_vector) / width_bins * self.par.scale

        self.frame_indices = np.unique(self.df_rectangle.loc[:, self.par.t_column].values)
        self.nb_frame = len(self.frame_indices)
        self.kymograph_map = np.zeros((self.nb_frame, nb_bins_axe))

        for count, frame in enumerate(tqdm(self.frame_indices)):
            cond_frame = self.df_rectangle.loc[:, self.par.t_column].values == frame
            hist, __ = np.histogram(coords_projection[cond_frame], bins=nb_bins_axe)
            self.kymograph_map[count, :] = hist
    

    def plot_kymograph(self, cmap,
                       vmin, vmax,
                       labelsize,
                       plot_traj=True,
                       plot_rev=True,
                       nb_traj=3,
                       idx_traj = None,
                       min_traj_length=501,
                       s=1,
                       s_rev=10,
                       c_traj='limegreen',
                       c_rev='purple',
                       linewidths_rev=1):
        """
        Plot the kymograph with superposed trajectories in option
        
        """
        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'cmr10'  # "Computer Modern Roman"
        self.fig, ax = plt.subplots(figsize=(8,8))
        im = ax.imshow(self.kymograph_map, 
                       cmap=cmap, 
                    #    extent=(0, self.len_axe*self.par.scale, self.tf, self.ti),
                       extent=(0, self.len_axe, self.tf, self.ti),
                       vmin=vmin, vmax=vmax,
                       aspect='auto')
        ax.set_ylabel('Time (min)', fontsize=labelsize)
        ax.set_xlabel('Space (µm)', fontsize=labelsize)
        # self.forceAspect(ax, aspect=1)
        ax.tick_params(labelsize=labelsize)
        cbar = self.fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=labelsize)
        cbar.set_ticks(np.arange(vmin, vmax+1, 1))  # Set the tick positions for the colorbar
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.05)

        cond_min_traj_length = self.df_rectangle.loc[:, self.par.len_traj_column] > min_traj_length
        traj_id = np.unique(self.df_rectangle.loc[cond_min_traj_length, self.par.track_id_column])
        if nb_traj:
            traj_id_selected = np.random.choice(traj_id, size=nb_traj, replace=False)
        elif idx_traj:
            traj_id_selected = idx_traj

        print(traj_id_selected)
        array_x_rev = []
        array_y_rev = []
        array_x_rev_proj = []
        array_y_rev_proj = []
        for count, traj in enumerate(tqdm(traj_id_selected)):
            cond_traj = self.df_rectangle.loc[:, self.par.track_id_column] == traj
            xy_proj = self.xy_projection[cond_traj] * self.par.scale
            time = self.df_rectangle.loc[cond_traj, self.par.t_column].values * self.par.tbf
            array_x_rev.append(xy_proj)
            array_y_rev.append(time)
            ax.scatter(xy_proj, time, s=s, c=c_traj)

            if plot_rev:
                cond_rev = self.df_rectangle.loc[cond_traj, self.par.rev_column].values.astype(bool)
                xy_rev_proj = xy_proj[cond_rev]
                time_rev = time[cond_rev]
                array_x_rev_proj.append(xy_rev_proj)
                array_y_rev_proj.append(time_rev)
                ax.scatter(xy_rev_proj, time_rev, s=s_rev, marker='o', facecolors='none', edgecolors=c_rev, linewidths=linewidths_rev)
        
        self.list_xy_proj = [array_x_rev, array_y_rev]
        self.array_xy_rev_proj = np.array([np.concatenate(array_x_rev_proj), np.concatenate(array_y_rev_proj)])
        plt.show()


    def forceAspect(self, ax, aspect):
        im = ax.get_images()
        extent =  im[0].get_extent()
        ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

kym = Kymograph()
tool = tools.Tools()

# %% EXPERIMENT
# Folders names
input_folder = ""
filename = "rippling_movie_1.nd2"
output_folder = ""
im = ND2Reader(input_folder+filename)
xi = 366
yi = 2532
xf = 2868
yf = 270
e = 300
kym.find_axe_equation(xi, yi, xf, yf, e)
figsize = (32, 32)
kym.visualize_rectangle(im=im, figsize=figsize, linewidth=10)
tool.save_fig(fig=kym.fig, name=output_folder+"kymo_area.png", dpi=100)

# Specify the folder where file_tracking_rippling_movie_1.csv is
input_folder = ""
filename = "file_tracking_rippling_movie_1.csv"
# Specify the folder for the save
output_folder = ""
df = pd.read_csv(input_folder+filename)

scale = 0.0646028 # exp
kym = Kymograph()
kym.find_axe_equation(xi*scale, yi*scale, xf*scale, yf*scale, e*scale)
start = 120
end = 150
kym.extract_traj_inside_rectangle(df=df, width_bins=4*scale, ti=start, tf=end)
idx_traj = [381, 8789, 5706] # You can choose the trajectories to display
kym.plot_kymograph(cmap='hot',
                   vmin=0., vmax=6,
                   labelsize=30, 
                   s=3,
                   s_rev=50, 
                   nb_traj=None,
                   idx_traj=idx_traj,
                   min_traj_length=50,
                   c_traj='lightgreen',
                   c_rev='blue',
                   linewidths_rev=2)

kym.fig.savefig(output_folder+'rippling_kymo.png', dpi=100, bbox_inches='tight')

# %% SIMULATION
# Folders names
input_folder = ""
filename = "73.png"
output_folder = ""
im = imread(input_folder+filename)
xi = 1
yi = 1600
xf = 3199
yf = 1601
e = 300
kym.find_axe_equation(xi, yi, xf, yf, e)
figsize = (32, 32)
kym.visualize_rectangle(im=im, figsize=figsize, linewidth=10)
tool.save_fig(fig=kym.fig, name=output_folder+"kymo_area.png", dpi=100)

# Specify the folder where simu__9000_bacts__tbf=1.8_secondes__space_size=195.csv is
input_folder = ""
filename = "simu__9000_bacts__tbf=1.8_secondes__space_size=195.csv"
# Specify the folder for the save
output_folder = ""
df = pd.read_csv(input_folder+filename)

scale = 195/3200 # simu
kym = Kymograph()
kym.find_axe_equation(xi*scale, yi*scale, xf*scale, yf*scale, e*scale)
start = 120
end = 150
kym.extract_traj_inside_rectangle(df=df, width_bins=4*scale, ti=start, tf=end)
idx_traj = [1141, 8246, 8513] # You can choose the trajectories to display
kym.plot_kymograph(cmap='hot',
                   vmin=0., vmax=6,
                   labelsize=30, 
                   s=3,
                   s_rev=50, 
                   nb_traj=None,
                   idx_traj=idx_traj,
                   min_traj_length=50,
                   c_traj='lightgreen',
                   c_rev='blue',
                   linewidths_rev=2)

kym.fig.savefig(output_folder+'rippling_kymo.png', dpi=100, bbox_inches='tight')
