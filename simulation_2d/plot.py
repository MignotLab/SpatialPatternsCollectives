import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import os


class Plot:

    def __init__(self, inst_par, inst_gen, inst_dir, inst_rev, inst_eps, inst_move, inst_kym, sample):

        # Parameters from parameters.py
        self.par = inst_par
        self.gen = inst_gen
        self.dir = inst_dir
        self.rev = inst_rev
        self.eps = inst_eps
        self.move = inst_move
        self.kym = inst_kym
        self.sample = sample
        isExist = os.path.exists(self.sample+'/')
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self.sample+'/')
            print("The new directory is created!")
        self.n_bact = self.par.n_bact
        self.n_nodes = self.par.n_nodes
        self.d_n = self.par.d_n
        self.size = self.par.space_size
        self.border = self.eps.edges_width
        self.max_eps_value = self.par.max_eps_value
        self.r = self.eps.r # length of pili in pixels
        # Size of the points for the plot
        # self.point_size = (self.par.width / (self.size+ 2 * self.border)) * (self.size + 2 * self.border) ** 2
        self.figsize = (32,32)
        self.point_size = self.par.param_point_size*20 / self.size * self.figsize[0] ** 2

        # Random color of the bacteria
        self.random_color = np.tile(np.random.rand(self.n_bact), (self.n_nodes,1))


    def plotty(self, path):
        """
        Plot data in a  scatter plot
        
        """

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        angle_color = np.tile(np.arctan2(self.dir.nodes_direction[1,0,:], self.dir.nodes_direction[0,0,:]),(self.par.n_nodes,1))
        plt.scatter(self.gen.data[0, :, :], self.gen.data[1, :, :], s=self.point_size, linewidths=0.15, c=angle_color, edgecolor='k', cmap='hsv', vmin=-np.pi, vmax=np.pi, zorder=1, alpha=0.35)#, cmap="Pastel2")
        # plt.scatter(self.gen.data[0,:,:], self.gen.data[1,:,:], s=self.point_size, c=angle_color, cmap='hsv', vmin=-np.pi, vmax=np.pi, zorder=1, alpha=0.35)#, cmap="Pastel2")
        # ax.scatter(self.gen.data[0,:,:], self.gen.data[1,:,:], s=self.point_size, edgecolors='k',facecolors='none', linewidths=0.3, zorder=2)
        # ax.scatter(self.gen.data[0,:,:], self.gen.data[1,:,:], s=self.point_size, edgecolors='none', c=angle_color, cmap='hsv', vmin=-np.pi, vmax=np.pi, zorder=3, alpha=0.85)
        if self.par.plot_position_focal_adhesion_point:
            plt.scatter(self.gen.data[0, self.move.position_focal_adhesion_point[0, :, :]], self.gen.data[1,self.move.position_focal_adhesion_point[1, :, :]], s=self.point_size/5, c='k', zorder=1, alpha=0.35)#
        if self.par.plot_eps_grid:
            cmap = plt.cm.get_cmap('Greys', int(self.max_eps_value))
            plt.imshow(np.rot90(self.eps.eps_grid),
                       extent=(-self.border, self.size+self.border, -self.border, self.size+self.border),
                       cmap=cmap,
                       vmin=0,
                       vmax=self.max_eps_value)
        ax.set_aspect('equal', adjustable='box')
        # plt.xlim(-self.border, self.size+self.border)
        # plt.ylim(-self.border, self.size+self.border)
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.axis('off')
        fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
        fig.subplots_adjust(wspace=0,hspace=0)
        # fig.savefig(path, bbox_inches='tight')
        fig.savefig(path, dpi=100)
        plt.close()
        gc.collect()


    def plotty_rippling_swarming(self, path, t):
        """
        Plot data in a  scatter plot
        
        """
        if t * self.par.dt < self.par.time_rippling_swarming_colored:
            # Condition for initial swarming and rippling patterns
            self.cond_rippling = self.gen.cond_space_alignment.copy()
            self.cond_swarming = self.gen.cond_space_eps.copy()

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        plt.scatter(self.gen.data[0, :, self.cond_rippling], self.gen.data[1, :, self.cond_rippling], s=self.point_size, linewidths=0.5, c=self.par.color_rippling, edgecolor='grey', zorder=1, alpha=self.par.alpha)#, cmap="Pastel2")
        plt.scatter(self.gen.data[0, :, self.cond_swarming], self.gen.data[1, :, self.cond_swarming], s=self.point_size, linewidths=0.5, c=self.par.color_swarming, edgecolor='grey', zorder=2, alpha=self.par.alpha)

        if self.par.plot_position_focal_adhesion_point:
            plt.scatter(self.gen.data[0, self.move.position_focal_adhesion_point[0, :, :]], self.gen.data[1,self.move.position_focal_adhesion_point[1,:,:]], s=self.point_size/5, c='k', zorder=1, alpha=0.35)#
        
        if self.par.plot_eps_grid:
            cmap = plt.cm.get_cmap('Greys', int(self.max_eps_value))
            plt.imshow(np.rot90(self.eps.eps_grid),
                       extent=(-self.border, self.size+self.border, -self.border, self.size+self.border),
                       cmap=cmap,
                       vmin=0,
                       vmax=self.max_eps_value)
            
        ax.set_aspect('equal', adjustable='box')
        # plt.xlim(-self.border, self.size+self.border)
        # plt.ylim(-self.border, self.size+self.border)
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.axis('off')
        fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
        fig.subplots_adjust(wspace=0,hspace=0)
        # fig.savefig(path, bbox_inches='tight')
        fig.savefig(path, dpi=100)
        plt.close()
        gc.collect()


    def plotty_reversing_and_non_reversing(self, path):
        """
        Plot data in a  scatter plot
        
        """
        plt.style.use('dark_background')
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        plt.scatter(self.gen.data[0, :, self.rev.cond_reversing], self.gen.data[1, :, self.rev.cond_reversing], s=self.point_size, linewidths=0.5, c=self.par.color_reversing, edgecolor='grey', zorder=1, alpha=self.par.alpha) #, cmap="Pastel2")
        plt.scatter(self.gen.data[0, :, ~self.rev.cond_reversing], self.gen.data[1, :, ~self.rev.cond_reversing], s=self.point_size, linewidths=0.5, c=self.par.color_non_reversing, edgecolor='grey', zorder=2, alpha=self.par.alpha)
        
        if self.par.plot_position_focal_adhesion_point:
            plt.scatter(self.gen.data[0, self.move.position_focal_adhesion_point[0, :, :]], self.gen.data[1, self.move.position_focal_adhesion_point[1, :, :]], s=self.point_size/5, c='k', zorder=1, alpha=0.35)#
        
        if self.par.plot_eps_grid:
            cmap = plt.cm.get_cmap('Greys', int(self.max_eps_value))
            plt.imshow(np.rot90(self.eps.eps_grid),
                       extent=(-self.border, self.size+self.border, -self.border, self.size+self.border),
                       cmap=cmap,
                       vmin=0,
                       vmax=self.max_eps_value)
        
        ax.set_aspect('equal', adjustable='box')
        # plt.xlim(-self.border, self.size+self.border)
        # plt.ylim(-self.border, self.size+self.border)
        ax.set_xlim(0, self.size)
        ax.set_ylim(0, self.size)
        ax.axis('off')
        fig.subplots_adjust(left=0,bottom=0,right=1,top=1)
        fig.subplots_adjust(wspace=0,hspace=0)
        # fig.savefig(path, bbox_inches='tight')
        fig.savefig(path, dpi=100)
        plt.close()
        gc.collect()


    def plotty_rev(self, x, y, path, point_size, fig, ax):
        """
        Plot the reversals on another frame than the cells
        
        """

        # fig = plt.figure(figsize=(15,15))
        # ax = fig.add_subplot(111)
        center_node = int(self.par.n_nodes / 2)
        x_rev = np.concatenate(x)
        y_rev = np.concatenate(y)
        ax.scatter(x_rev, y_rev, s=point_size, c="blue", zorder=1, alpha=0.05)
        ax.set_aspect('equal', adjustable='box')
        plt.xlim(-self.border, self.size+self.border)
        plt.ylim(-self.border, self.size+self.border)
        fig.savefig(path)
        plt.close()
        gc.collect()


    def plotty_eps(self, path):
        """
        Draw the eps map
        
        """

        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        cmap = plt.cm.get_cmap('RdPu', int(2 * (self.max_eps_value+1)))
        plt.imshow(np.rot90(self.eps.eps_grid),
                    extent=(-0.1*self.r, self.size+0.1*self.r, -0.1*self.r, self.size+0.1*self.r),
                    cmap=cmap,
                    vmin=0,
                    vmax=self.max_eps_value/2)
        ax.set_aspect('equal', adjustable='box')
        plt.xlim(-0.1*self.r, self.size+0.1*self.r)
        plt.ylim(-0.1*self.r, self.size+0.1*self.r)
        fig.savefig(path, bbox_inches='tight')
        plt.close()
        gc.collect()


    def compute_tbr(self, n_sample=1000):
        """
        Compute the mean of the tbr depending of the values of the reversal rate
        
        """

        tbr_list = []
        mean_tbr = []
        var_tbr = []
        clock_tbr = np.zeros(n_sample)
        reversal_rate = np.arange(0.1,30.1,1)
        for rr in tqdm(reversal_rate):

            for time in range(int(50/self.par.dt)):
                clock_tbr += self.par.dt
                prob = np.ones(n_sample) * (1 - np.exp(-rr))
                cond_rev = np.random.binomial(1, prob*self.par.dt).astype("bool")
                tbr_list.append(clock_tbr[cond_rev])
                clock_tbr[cond_rev] = 0

            mean_tbr.append(np.mean(np.concatenate((tbr_list))))
            var_tbr.append(np.std(np.concatenate((tbr_list))))

        fontsize = 25
        fig, ax = plt.subplots(figsize=(12,7))
        ax.plot(mean_tbr,reversal_rate)
        ax.set_xlabel("Time between reversals (min)",fontsize=fontsize)
        ax.set_ylabel("Reversal rate",fontsize=fontsize)
        ax.errorbar(mean_tbr, reversal_rate, xerr=var_tbr, fmt='.k')
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.show()


    def tbr(self, sample, T, duration=50, xmin=0, xmax=15):
        """
        Plot the time between reversals distribution
        
        """
        start = int(np.maximum(T-duration,0) / self.par.dt)
        tbr_plot = np.concatenate(self.rev.tbr_list[start:])
        tbr_position_x = np.concatenate(self.rev.tbr_position_x_list[start:])
        tbr_position_y = np.concatenate(self.rev.tbr_position_y_list[start:])
        # Définissez les noms des colonnes correspondantes
        column_names = ['tbr', 'x', 'y']
        # Concaténez les tableaux en une seule matrice
        data = np.column_stack((tbr_plot, tbr_position_x, tbr_position_y))
        # Sauvegardez la matrice dans un fichier CSV
        np.savetxt(sample+'/tbr.csv', data, delimiter=',', header=','.join(column_names), comments='')

        step = round((xmax-xmin) / self.par.save_freq_tbr) + 1
        bins_tbr = np.linspace(xmin, xmax, step)
        fontsize = 30
        fig, ax = plt.subplots(figsize=(8,6))
        hist, bins, __ = plt.hist(tbr_plot, bins=bins_tbr, color="royalblue", density=True, alpha=0.4, histtype='bar', ec='black')
        plt.xlabel("Time between reversals (min)",fontsize=fontsize)
        plt.ylabel("Density",fontsize=fontsize)
        plt.text(xmax-0.5,max(hist)-0.02,'Mean = '+str(format(np.mean(tbr_plot), '.3g'))+' min', 
                ha='right', va='center', fontsize=fontsize/1.5, bbox=dict(facecolor='none',edgecolor='k'))
        # plt.tick_par(axis="both", which="both", labelsize=fontsize)
        plt.xlim(xmin, xmax)
        plt.xticks(np.arange(xmin, xmax+1, step=5), fontsize=fontsize)
        plt.yticks(np.arange(0, max(hist), step=0.1), fontsize=fontsize)
        fig.tight_layout()
        plt.title(sample+' all',fontsize=fontsize)
        path = self.sample+'/'
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")
        fig.savefig(path+'tbr_distribution.png', bbox_inches='tight')


    def tbr_cond_space(self, sample, T, duration=50, xmin=0, xmax=15):
        """
        Plot the time between reversals distribution
        
        """
        start = int(np.maximum(T-duration,0) / self.par.dt)
        tbr_plot = np.concatenate(self.rev.tbr_list[start:])
        tbr_position_x = np.concatenate(self.rev.tbr_position_x_list[start:])
        cond_rippling = (tbr_position_x > self.par.interval_rippling_space[0]*self.par.space_size) & (tbr_position_x < self.par.interval_rippling_space[1]*self.par.space_size)
        step = round((xmax-xmin)/self.par.save_freq_tbr)+1
        bins_tbr = np.linspace(xmin,xmax,step)
        # max_tbr = np.max(tbr_array)
        # min_tbr = np.min(tbr_array)
        # bins_tbr = np.arange(min_tbr,max_tbr,self.par.save_freq_tbr)
        fontsize = 30
        fig1, ax1 = plt.subplots(figsize=(8,6))
        hist, __, __ = plt.hist(tbr_plot[cond_rippling], bins=bins_tbr, color="royalblue", density=True, alpha=0.4, histtype='bar', ec='black')
        plt.xlabel("Time between reversals (min)",fontsize=fontsize)
        plt.ylabel("Density",fontsize=fontsize)
        plt.text(xmax-0.5,max(hist)-0.02,'Mean = '+str(format(np.mean(tbr_plot[cond_rippling]), '.3g'))+' min', 
                ha='right', va='center', fontsize=fontsize/1.5, bbox=dict(facecolor='none',edgecolor='k'))
        # plt.tick_par(axis="both", which="both", labelsize=fontsize)
        plt.xlim(xmin,xmax)
        plt.xticks(np.arange(xmin, xmax+1, step=5), fontsize=fontsize)
        plt.yticks(np.arange(0,max(hist), step=0.1), fontsize=fontsize)
        fig1.tight_layout()
        plt.title(sample+' rippling',fontsize=fontsize)
        # plt.legend("Mean = "+"{:.1e}".format(np.mean(tbr_array)),fontsize=fontsize)
        # plt.setp(legend.get_title(),fontsize=fontsize)

        fig2, ax2 = plt.subplots(figsize=(8,6))
        hist, __, __ = plt.hist(tbr_plot[~cond_rippling], bins=bins_tbr, color="royalblue", density=True, alpha=0.4, histtype='bar', ec='black')
        plt.xlabel("Time between reversals (min)",fontsize=fontsize)
        plt.ylabel("Density",fontsize=fontsize)
        plt.text(xmax-0.5,max(hist)-0.02,'Mean = '+str(format(np.mean(tbr_plot[~cond_rippling]), '.3g'))+' min', 
                ha='right', va='center', fontsize=fontsize/1.5, bbox=dict(facecolor='none',edgecolor='k'))
        # plt.tick_par(axis="both", which="both", labelsize=fontsize)
        plt.xlim(xmin,xmax)
        plt.xticks(np.arange(xmin,xmax+1,step=5), fontsize=fontsize)
        plt.yticks(np.arange(0,max(hist),step=0.1), fontsize=fontsize)
        fig2.tight_layout()
        plt.title(sample+' swarming',fontsize=fontsize)
        # plt.legend("Mean = "+"{:.1e}".format(np.mean(tbr_array)),fontsize=fontsize)
        # plt.setp(legend.get_title(),fontsize=fontsize)
        # path = 'result/'+str(self.par.n_bact)+'bact_'+str(self.par.epsilon_eps)+'epsilon_eps_'+str(self.par.s0)+'s0_'+str(self.par.s1)+'s1_'+str(self.par.s2)+'s2_'+str(self.par.r_max)+'r_max'+'/'
        path = self.sample+'/'
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")
        fig1.savefig(path+'tbr_distribution_rippling.png', bbox_inches='tight')
        fig2.savefig(path+'tbr_distribution_swarming.png', bbox_inches='tight')


    def velocity(self,velocity_list,velocity_max,width_bin):
        """
        Plot the velocites
        
        """

        fontsize = 30
        vel = np.concatenate((velocity_list))
        vel = vel[vel<=velocity_max]
        bins = np.arange(0,velocity_max+1,width_bin)
        fig, ax = plt.subplots(figsize=(8,6))
        hist, bins, __  = ax.hist(vel, bins=bins, color="royalblue", density=True, alpha=0.4, histtype='bar', ec='black')
        ax.set_title("Velocities "+self.sample,fontsize=fontsize)
        ax.set_xlabel(r"Velocity ($\mu$m/min)",fontsize=fontsize)
        ax.set_ylabel("Density",fontsize=fontsize)
        plt.text(velocity_max-0.5,max(hist)-0.02,'Mean = '+str(format(np.mean(vel), '.3g'))+' min', 
                ha='right', va='center', fontsize=fontsize/1.5, bbox=dict(facecolor='none',edgecolor='k'))
        ax.set_xlim(0,velocity_max)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

        path = self.sample+'/'
        isExist = os.path.exists(path)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(path)
            print("The new directory is created!")
        fig.savefig(path+'velocities_distribution.png', bbox_inches='tight')


    def frustration(self,T,frustration,all_frustration=True,bact_id=0,width_bins=0.1):
        """
        Plot the all frustration during all the movie or the frustration in time for a specific bacterium
        
        """

        fontsize = 20
        fig, ax = plt.subplots(figsize=(8,6))
    
        if all_frustration:

            min_fru, max_fru = np.min(frustration), np.max(frustration)
            bins_fru = round((max_fru - min_fru) / width_bins)
            mean_fru = np.mean(frustration)
            ax.hist(frustration,bins=bins_fru,label='mean = '+str(round(mean_fru,3)),density=True,alpha=0.7,histtype='bar',ec='grey',color='lightblue')
            ax.set_xlim(min_fru, max_fru)
            ax.set_xlabel('Frustration',fontsize=fontsize)
            ax.tick_params(labelsize=fontsize/1.5)
            ax.legend(loc='best', fontsize=fontsize/1.5)

        else:

            tmp = np.reshape(frustration,(int(T/self.par.dt),self.par.n_bact))

            mean_cumul = tmp[:,bact_id]

            # Start plot
            start = int(self.par.time_memory / self.par.dt)
            plt.plot(np.arange(start*self.par.dt,T,self.par.dt),mean_cumul[start:])

        plt.show()


    def reversal_functions(self,sample,signal,refractory_period,reversal_rate):
        """
        Plot of the frz activity, the refractory period and the rate of reversals against the signal
        
        """
        fontsize = 20
        fig, ax1 = plt.subplots()
        ax2 = plt.twinx()
        ax1.plot(signal, refractory_period, label="refractory period",alpha=0.8, color="k")
        ax2.plot(signal, reversal_rate, label="reversal rate",alpha=0.8, color="limegreen")
        ax1.set_xlim(0, 2*self.par.s1)
        ax1.set_ylim(0, self.par.rp_max+1)
        ax2.set_ylim(0, self.par.r_max+1)
        ax1.set_title(sample, fontsize=fontsize)
        ax1.legend(loc='upper right')
        ax2.legend(loc='upper left')
        path = self.sample+'/reversal_functions_plot.png'
        fig.savefig(path, bbox_inches='tight')


    def kymograph_density(self,T,start):
        """
        Plot the kymograph density
        
        """

        path = self.sample+'/kymo_plot.png'
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111)
        plt.imshow(self.kym.density_kymo,
                    extent=(0, self.size, T, start),
                    cmap='Greys')
        ax.set_aspect('equal', adjustable='box')
        fig.savefig(path, bbox_inches='tight')