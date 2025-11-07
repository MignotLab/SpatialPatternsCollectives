# %%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm import tqdm
from igoshin_compute_eigeinvalues import Eigeinvalues
from tools import Tools
import pickle


class DiscretisationError:
    """
    Compute error coming from the discretisation step
    
    """
    def __init__(self, 
                 inst_par,
                 inst_eig,
                 function,
                 discretisation_parameters,
                 exp_name):
        
        self.par = inst_par
        self.eig = inst_eig
        self.tool = Tools()
        self.function = function
        self.ds_list = discretisation_parameters
        self.exp_name = exp_name
        self.path_save = "results_"+self.exp_name+"/pkl/"
        self.filename_dic_maps = "dic_eigenvalues_and_eigenvectors_igoshin_"+self.exp_name+"_q="+str(self.par.q_value_constant)+"_rho_w="+str(self.par.rho_w)+".pkl"
        self.filename_dic_error = "dic_error_eigenvalues_and_eigenvectors_igoshin_"+self.exp_name+"_q="+str(self.par.q_value_constant)+"_rho_w="+str(self.par.rho_w)+".pkl"

    
    def compute_eigenvalues_for_different_discretisation(self, only_plot, compute_dic_map=False):
        """
        Compute the eigenvalues for all parameters for different discretisation step size
        
        """
        eps = 1e-15
        if only_plot:
            pass

        else:
            # Test the save
            dic_test = {}
            self.tool.initialize_directory_or_file(path=self.path_save+"test.pkl")
            with open(self.path_save+"test.pkl", 'wb') as pickle_file:
                pickle.dump(dic_test, pickle_file)
            
            abs_error_map_eigenvalues = np.zeros(len(self.ds_list) - 1)
            abs_error_map_eigenvectors = np.zeros(len(self.ds_list) - 1)

            if compute_dic_map:
                print("COMPUTE EIGENVALUES AND EIGENVECTORS FOR DIFFERENT DS PARAMETER")
                list_map_eigenvalues = []
                list_map_eigenvectors = []
                for count, dp in enumerate(tqdm(self.ds_list)):
                    map_eigenvalues, map_eigenvectors = self.eig.compute_eigeinvalues(function=self.function, dp=dp)
                    list_map_eigenvalues.append(map_eigenvalues)
                    list_map_eigenvectors.append(map_eigenvectors)
                    if count > 0:
                        abs_error_map_eigenvalues[count-1] = np.mean(np.abs(list_map_eigenvalues[count] - list_map_eigenvalues[count-1]))
                        abs_error_map_eigenvectors[count-1] = np.mean(np.abs(list_map_eigenvectors[count] - list_map_eigenvectors[count-1]))
                        # print(abs_error_map_eigenvalues[count-1])
                # Save list_map
                dic_list_map = {"list_map_eigenvalues": list_map_eigenvalues,
                                "list_map_eigenvectors": list_map_eigenvectors}
                with open(self.path_save+self.filename_dic_maps, 'wb') as pickle_file:
                    pickle.dump(dic_list_map, pickle_file)
                # Save difference
                dic_error_map = {"abs_error_map_eigenvalues": abs_error_map_eigenvalues,
                                "abs_error_map_eigenvectors": abs_error_map_eigenvectors}
                with open(self.path_save+self.filename_dic_error, 'wb') as pickle_file:
                    pickle.dump(dic_error_map, pickle_file)

            else:
                with open(self.path_save+self.filename_dic_maps, 'rb') as pickle_file:
                    dic_list_map = pickle.load(pickle_file)
                list_map_eigenvalues = np.array(dic_list_map["list_map_eigenvalues"])
                list_map_eigenvectors = np.array(dic_list_map["list_map_eigenvectors"])
                for count in range(1, self.ds_list):
                    abs_error_map_eigenvalues[count-1] = np.nanmean(np.abs(list_map_eigenvalues[count] - list_map_eigenvalues[count-1]))
                    abs_error_map_eigenvectors[count-1] = np.nanmean(np.abs(list_map_eigenvectors[count] - list_map_eigenvectors[count-1]))
                # Save difference
                dic_error_map = {"abs_error_map_eigenvalues": abs_error_map_eigenvalues,
                                "abs_error_map_eigenvectors": abs_error_map_eigenvectors}
                with open(self.path_save+self.filename_dic_error, 'wb') as pickle_file:
                    pickle.dump(dic_error_map, pickle_file)

        with open(self.path_save+self.filename_dic_error, 'rb') as pickle_file:
                    dic_diff = pickle.load(pickle_file)
        abs_error_map_eigenvalues = dic_diff["abs_error_map_eigenvalues"]
        abs_error_map_eigenvectors = dic_diff["abs_error_map_eigenvectors"]

        fig, ax = plt.subplots(figsize=self.par.figsize)
        ax.plot(self.ds_list[1:], abs_error_map_eigenvalues, "+-", color="k", linewidth=3, markersize=10, alpha=1)
        ax.tick_params(labelsize=self.par.fontsize/1.5)
        ax.set_xlabel("Discretization step", fontsize=self.par.fontsize)
        ax.set_ylabel("Absolute error", fontsize=self.par.fontsize)
        self.tool.initialize_directory_or_file(path=self.path_save)
        fig.savefig(self.path_save+"abs_error_map_eigenvalues"+".png", bbox_inches="tight", dpi=100)
        fig.savefig(self.path_save+"abs_error_map_eigenvalues"+".svg", dpi=100)

        fig, ax = plt.subplots(figsize=self.par.figsize)
        ax.plot(self.ds_list[1:], abs_error_map_eigenvectors, "+-", color="k", linewidth=3, markersize=10, alpha=1)
        ax.tick_params(labelsize=self.par.fontsize/1.5)
        ax.set_xlabel("Discretization step", fontsize=self.par.fontsize)
        ax.set_ylabel("Absolute error", fontsize=self.par.fontsize)
        self.tool.initialize_directory_or_file(path=self.path_save)
        fig.savefig(self.path_save+"abs_error_map_eigenvectors"+".png", bbox_inches="tight", dpi=100)
        fig.savefig(self.path_save+"abs_error_map_eigenvectors"+".svg", dpi=100)



from parameters import Parameters
from igoshin_compute_eigeinvalues import Eigeinvalues
par = Parameters()
eig = Eigeinvalues(values=par.combined_array_1,
                   grid=par.S_grid_1,
                   inst_par=par)

# discretisation_parameters = [0.2, 0.19, 0.18, 0.17, 0.16, 0.15, 0.14, 0.13, 0.12, 0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01]
# discretisation_parameters = [0.012, 0.011, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005, 0.004, 0.003, 0.002, 0.001]
# discretisation_parameters = [0.11, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.019, 0.018, 0.017, 0.016, 0.015, 0.014, 0.013, 0.012, 0.011, 0.01, 0.009, 0.008, 0.007, 0.006, 0.005]
discretisation_parameters = [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.01]
exp_name = "xi_S_linear"
de = DiscretisationError(inst_par=par,
                         inst_eig=eig,
                         function=eig.compute_eigeinvalues_xi_S_linear,
                         discretisation_parameters=discretisation_parameters,
                         exp_name=exp_name)
de.compute_eigenvalues_for_different_discretisation(only_plot=False, compute_dic_map=True)        
# %%
