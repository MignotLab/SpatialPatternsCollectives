# %%
import sys, os
import multiprocessing
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from simulation.agent_based_model_myxo.main import Main
from simulation.agent_based_model_myxo.parameters import Parameters

"""
'reversal_type could be: (refractory_period_type, reversal_rate_type), "threshold_frustration", "periodic" or "off"; default is ("linear", "bilinear")

'reversal_rate_type could be: "bilinear", "bilinear_smooth", "linear", "sigmoidal", "exponential" or "constant"; default is "bilinear"

'refractory_period_type could be: "linear", "sigmoidal" or "constant"; default is "linear"

# EXAMPLE SWARMING
{   'generation_type':"square_random_orientation", 'n_bact':300, 'space_size':65,
    
    'repulsion_type':"repulsion", 'k_r':8e4,
    'attraction_type':"attraction_body", 'k_a':1e3,
    'alignment_type':"no_alignment",
    'eps_follower_type':"igoshin_eps_road_follower", 'plot_eps_grid':True,
    'reversal_type':("linear", "bilinear"),  's1':0.11, 's2':0.15, 'r_min':0, 'r_max':2, 'rp_min':0.5, 'rp_max':5,
    'alpha_sigmoid_rr':200, 'alpha_sigmoid_rp':100, 'alpha_bilinear_rr':10
    
},

# EXAMPLE RIPPLING
{   'generation_type':"square_alignment", 'n_bact':1000, 'space_size':65,
    
    'repulsion_type':"repulsion", 'k_r':8e4,
    'attraction_type':"attraction_body", 'k_a':1e3,
    'alignment_type':"global_alignment",
    'eps_follower_type':"no_eps", 'plot_eps_grid':True,
    'reversal_type':("linear", "bilinear"),  's1':0.11, 's2':0.15, 'r_min':0, 'r_max':2, 'rp_min':0.5, 'rp_max':5,
    'alpha_sigmoid_rr':200, 'alpha_sigmoid_rp':100, 'alpha_bilinear_rr':10
    
},

"""
# Number of cell in rippling 100X field of the paper : 6294
# Number of cell in swarming 100X field of the paper : 3305

T = 150 # minutes
# Liste des ensembles de paramètres pour chaque simulation
params_list = [
    ## RIPPLING CONTROL 1
    {'generation_type':"square_alignment", 'n_bact':1000, 'space_size':65, 'save_frequency_csv':10, 'param_point_size':2,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
     'r_max':5,
    },
    {'generation_type':"square_alignment", 'n_bact':1100, 'space_size':65, 'save_frequency_csv':10, 'param_point_size':2,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
     'r_max':5,
    },

    ## SWARMING CONTROL 4
    {'generation_type':"square_random_orientation", 'n_bact':370, 'space_size':65, 'save_frequency_csv':10, 'param_point_size':2,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'r_max':5,
     'epsilon_eps':11,
    },
    {'generation_type':"square_random_orientation", 'n_bact':370, 'space_size':65, 'save_frequency_csv':10, 'param_point_size':2,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'r_max':5,
     'epsilon_eps':9,
    },
    {'generation_type':"square_random_orientation", 'n_bact':370, 'space_size':65, 'save_frequency_csv':10, 'param_point_size':2,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'r_max':5,
     'epsilon_eps':4,
    },

    # Ajoutez d'autres ensembles de paramètres selon vos besoins
]


# Function executed in each simulation process
def simulate(params, sample):
    # Generate a unique seed using system entropy (guaranteed to differ between processes)
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    random.seed(seed)
    np.random.seed(seed)

    # Initialize simulation parameters
    par = Parameters()
    for key, value in params.items():
        setattr(par, key, value)

    # Launch the simulation
    ma = Main(inst_par=par, sample=sample, T=T)
    ma.start()

if __name__ == '__main__':
    # Force 'spawn' method on Linux to avoid shared RNG state across child processes
    multiprocessing.set_start_method('spawn')

    # Create and launch one process per simulation
    processes = []
    for i, params in enumerate(params_list):
        sample = 'output/agent_based_simulation_script_paper_2024_test_fit_tbr/sample' + str(i + 1)
        process = multiprocessing.Process(target=simulate, args=(params, sample))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()
