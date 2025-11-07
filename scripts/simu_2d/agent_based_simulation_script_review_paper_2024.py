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
{   'generation_type':"square_random_orientation", 'n_bact':300, 'space_size':65*2,
    
    'repulsion_type':"repulsion", 'k_r':8e4,
    'attraction_type':"attraction_body", 'k_a':1e3,
    'alignment_type':"no_alignment",
    'eps_follower_type':"igoshin_eps_road_follower", 'plot_eps_grid':True,
    'reversal_type':("linear", "bilinear"),  's1':0.11, 's2':0.15, 'r_min':0, 'r_max':2, 'rp_min':0.5, 'rp_max':5,
    'alpha_sigmoid_rr':200, 'alpha_sigmoid_rp':100, 'alpha_bilinear_rr':10
    
},

# EXAMPLE RIPPLING
{   'generation_type':"square_alignment", 'n_bact':1000, 'space_size':65*2,
    
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
    ### REVIEWER 1
    ## QUESTION 4.1 (RIPPLING LOOSE ALIGNMENT FORCE) sample1
    {'generation_type':"square_alignment", 'n_bact':1000, 'space_size':65, 'save_frequency_csv':1, 'param_point_size':1.5,
     'alignment_type':"global_alignment",
     'stop_alignment': 100,
     'eps_follower_type':"no_eps",
    },
    ## QUESTION 4.2 (SWARMING LOOSE EPS FOLLOWING) sample2
    {'generation_type':"square_random_orientation", 'n_bact':370, 'space_size':65, 'save_frequency_csv':1, 'param_point_size':1.,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'stop_eps_follower': 100,
    },
    ## QUESTION 6 (RIPPLING CELL DENSITY 1200) sample3
    {'generation_type':"square_alignment", 'n_bact':1200, 'space_size':65, 'save_frequency_csv':1, 'param_point_size':1.,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
    },
    ## QUESTION 6 (RIPPLING CELL DENSITY 1100) sample4
    {'generation_type':"square_alignment", 'n_bact':1100, 'space_size':65, 'save_frequency_csv':1, 'param_point_size':1.,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
    },
    ## QUESTION 6 (RIPPLING CELL DENSITY 1000) sample5
    {'generation_type':"square_alignment", 'n_bact':1000, 'space_size':65, 'save_frequency_csv':1, 'param_point_size':1.,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
    },
    ## QUESTION 6 (RIPPLING CELL DENSITY 900) sample6
    {'generation_type':"square_alignment", 'n_bact':900, 'space_size':65, 'save_frequency_csv':1, 'param_point_size':1.,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
    },
    ## QUESTION 6 (RIPPLING CELL DENSITY 800) sample7
    {'generation_type':"square_alignment", 'n_bact':800, 'space_size':65, 'save_frequency_csv':1, 'param_point_size':1.,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
    },
    ## QUESTION 6 (RIPPLING CELL DENSITY 600) sample8
    {'generation_type':"square_alignment", 'n_bact':600, 'space_size':65, 'save_frequency_csv':1, 'param_point_size':1.,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
    },
    ## QUESTION 10.2 Why the FrzE mutant does not form mesh-like structures?
    # Swarming WT sample 9
    {'generation_type':"square_random_orientation", 'n_bact':370*9, 'space_size':65*3, 'save_frequency_csv':1, 'param_point_size':1.,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'epsilon_eps':3
    },
    # Swarming FrzE mutant sample 10
    {'generation_type':"square_random_orientation", 'n_bact':370*9, 'space_size':65*3, 'save_frequency_csv':1, 'param_point_size':1.,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'reversal_type':"off",
     'epsilon_eps':3
    },

    # ## REVIEWER 4 (now in script agent_based_simulation_script_review_paper_2024_signaling_test.py)
    # ## QUESTION 2.1 different signalling type (rippling frustration) (launched 3 times to have 3 replicates)

    # ## Rippling frustration sample 1
    # {'generation_type':"square_alignment", 'n_bact':1000*9, 'space_size':65*3, 'param_point_size':0.5,
    #  'save_frequency_csv':2/60, 'save_in_csv_from_time': 200,
    #  'alignment_type':"global_alignment",
    #  'save_other_reversal_signals':True,
    # },
    # ## Rippling local density sample 2
    # {'generation_type':"square_alignment", 'n_bact':1000*9, 'space_size':65*3, 'param_point_size':0.5,
    #  'save_frequency_csv':2/60, 'save_in_csv_from_time': 200,
    #  'alignment_type':"global_alignment",
    #  'save_other_reversal_signals':True,
    #  'signal_type':"set_local_density",
    #  'max_neighbours_signal_density':40,
    # },
    # ## Rippling directional density sample 3
    # {'generation_type':"square_alignment", 'n_bact':1000*9, 'space_size':65*3, 'param_point_size':0.5,
    #  'save_frequency_csv':2/60, 'save_in_csv_from_time': 200,
    #  'alignment_type':"global_alignment",
    #  'save_other_reversal_signals':True,
    #  'signal_type':"set_directional_density",
    #  'max_neighbours_signal_density':20,
    # },

    # ## Swarming frustration sample 4
    # {'generation_type':"square_random_orientation", 'n_bact':370*9, 'space_size':65*3, 'param_point_size':0.5,
    #  'save_frequency_csv':2/60, 'save_in_csv_from_time': 200,
    #  'eps_follower_type':"igoshin_eps_road_follower",
    #  'save_other_reversal_signals':True,
    # },
    # ## Swarming local density sample 5
    # {'generation_type':"square_random_orientation", 'n_bact':370*9, 'space_size':65*3, 'param_point_size':0.5,
    #  'save_frequency_csv':2/60, 'save_in_csv_from_time': 200,
    #  'eps_follower_type':"igoshin_eps_road_follower",
    #  'save_other_reversal_signals':True,
    #  'signal_type':"set_local_density",
    #  'max_neighbours_signal_density':40,
    # },
    # ## Swarming directional density sample 6
    # {'generation_type':"square_random_orientation", 'n_bact':370*9, 'space_size':65*3, 'param_point_size':0.5,
    #  'save_frequency_csv':2/60, 'save_in_csv_from_time': 200,
    #  'eps_follower_type':"igoshin_eps_road_follower",
    #  'save_other_reversal_signals':True,
    #  'signal_type':"set_directional_density",
    #  'max_neighbours_signal_density':20,
    # },

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
        sample = 'output/agent_based_simulation_script_review_paper_2024/sample' + str(i + 1)
        process = multiprocessing.Process(target=simulate, args=(params, sample))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()
