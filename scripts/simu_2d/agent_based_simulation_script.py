# %%
import sys, os
import multiprocessing
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simulation.agent_based_model_myxo.main import Main
from simulation.agent_based_model_myxo.parameters import Parameters

"""
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

T = 500 # minutes
# List of parameter sets for each simulation
params_list = [
    ## RIPPLING AND SWARMING 0.5 times more bact in rippling 1
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+2000)*18, 'percentage_bacteria_rippling':0.333, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True 
    },
    ## RIPPLING AND SWARMING 0.5 times more bact in rippling 2
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+2000)*18, 'percentage_bacteria_rippling':0.333, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True 
    },
    ## RIPPLING AND SWARMING 0.5 times more bact in rippling 3
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+2000)*18, 'percentage_bacteria_rippling':0.333, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True 
    },
    ## RIPPLING AND SWARMING 0.5 times more bact in rippling 4
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+2000)*18, 'percentage_bacteria_rippling':0.333, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True 
    },
    ## RIPPLING AND SWARMING 0.5 times more bact in rippling 5
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+2000)*18, 'percentage_bacteria_rippling':0.333, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'tbr_cond_space_plot':True 
    },

    ## RIPPLING AND SWARMING 1 times more bact in rippling 6
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+1000)*18, 'percentage_bacteria_rippling':0.5, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True 
    },
    ## RIPPLING AND SWARMING 1 times more bact in rippling 7
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+1000)*18, 'percentage_bacteria_rippling':0.5, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True 
    },
    ## RIPPLING AND SWARMING 1 times more bact in rippling 8
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+1000)*18, 'percentage_bacteria_rippling':0.5, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True 
    },
    ## RIPPLING AND SWARMING 1 times more bact in rippling 9
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+1000)*18, 'percentage_bacteria_rippling':0.5, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True 
    },
    ## RIPPLING AND SWARMING 1 times more bact in rippling 10
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+1000)*18, 'percentage_bacteria_rippling':0.5, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True 
    },

    ## RIPPLING AND SWARMING 2 times more bact in rippling 11
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+500)*18, 'percentage_bacteria_rippling':0.666, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True 
    },
    ## RIPPLING AND SWARMING 2 times more bact in rippling 12
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+500)*18, 'percentage_bacteria_rippling':0.666, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING 2 times more bact in rippling 13
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+500)*18, 'percentage_bacteria_rippling':0.666, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING 2 times more bact in rippling 14
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+500)*18, 'percentage_bacteria_rippling':0.666, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING 2 times more bact in rippling 15
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+500)*18, 'percentage_bacteria_rippling':0.666, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },

    ## RIPPLING AND SWARMING 3 times more bact in rippling 16
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+333)*18, 'percentage_bacteria_rippling':0.75, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING 3 times more bact in rippling 17
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+333)*18, 'percentage_bacteria_rippling':0.75, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING 3 times more bact in rippling 18
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+333)*18, 'percentage_bacteria_rippling':0.75, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING 3 times more bact in rippling 19
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+333)*18, 'percentage_bacteria_rippling':0.75, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING 3 times more bact in rippling 20
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+333)*18, 'percentage_bacteria_rippling':0.75, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },

    ## RIPPLING AND SWARMING 4 times more bact in rippling 21
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+250)*18, 'percentage_bacteria_rippling':0.8, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING 4 times more bact in rippling 22
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+250)*18, 'percentage_bacteria_rippling':0.8, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING 4 times more bact in rippling 23
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+250)*18, 'percentage_bacteria_rippling':0.8, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING 4 times more bact in rippling 24
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+250)*18, 'percentage_bacteria_rippling':0.8, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING 4 times more bact in rippling 25
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1000+250)*18, 'percentage_bacteria_rippling':0.8, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },



    ## RIPPLING AND SWARMING NON-REVERSING BACTERIA 26
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1200+500)*18, 'percentage_bacteria_rippling':1-500/(1200+500), 'space_size':65*6, 
     'plot_movie':False, 'plot_reversing_and_non_reversing':True,'param_point_size':0.25, 'non_reversing':500/(1200+500),
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING NON-REVERSING BACTERIA 27
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1200+500)*18, 'percentage_bacteria_rippling':1-500/(1200+500), 'space_size':65*6, 
     'plot_movie':False, 'plot_reversing_and_non_reversing':True,'param_point_size':0.25, 'non_reversing':500/(1200+500),
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING NON-REVERSING BACTERIA 28
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1200+500)*18, 'percentage_bacteria_rippling':1-500/(1200+500), 'space_size':65*6, 
     'plot_movie':False, 'plot_reversing_and_non_reversing':True,'param_point_size':0.25, 'non_reversing':500/(1200+500),
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING NON-REVERSING BACTERIA 29
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1200+500)*18, 'percentage_bacteria_rippling':1-500/(1200+500), 'space_size':65*6, 
     'plot_movie':False, 'plot_reversing_and_non_reversing':True,'param_point_size':0.25, 'non_reversing':500/(1200+500),
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },
    ## RIPPLING AND SWARMING NON-REVERSING BACTERIA 30
    {'generation_type':"rippling_swarming_transition", 'n_bact':(1200+500)*18, 'percentage_bacteria_rippling':1-500/(1200+500), 'space_size':65*6, 
     'plot_movie':False, 'plot_reversing_and_non_reversing':True,'param_point_size':0.25, 'non_reversing':500/(1200+500),
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
     'tbr_cond_space_plot':True
    },



    ## RIPPLING 31
    {'generation_type':"square_alignment", 'n_bact':1000*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
     'save_frequency_csv':10,
    },
    ## RIPPLING 32
    {'generation_type':"square_alignment", 'n_bact':1000*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
     'save_frequency_csv':10,
    },
    ## RIPPLING 33
    {'generation_type':"square_alignment", 'n_bact':1000*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
     'save_frequency_csv':10,
    },
    ## RIPPLING  34
    {'generation_type':"square_alignment", 'n_bact':1000*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
     'save_frequency_csv':10,
    },
    ## RIPPLING 35
    {'generation_type':"square_alignment", 'n_bact':1000*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
     'save_frequency_csv':10,
    },



    ## SWARMING 36
    {'generation_type':"square_random_orientation", 'n_bact':370*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
    },
    ## SWARMING 37
    {'generation_type':"square_random_orientation", 'n_bact':370*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
    },
    ## SWARMING 38
    {'generation_type':"square_random_orientation", 'n_bact':370*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
    },
    ## SWARMING 39
    {'generation_type':"square_random_orientation", 'n_bact':370*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
    },
    ## SWARMING 40
    {'generation_type':"square_random_orientation", 'n_bact':370*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'save_frequency_csv':10,
    },

    # Add other parameter sets as needed
]


# Function for each simulation
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
        sample = 'output/agent_based_simulation_script/sample' + str(i + 1)
        process = multiprocessing.Process(target=simulate, args=(params, sample))
        processes.append(process)
        process.start()

    # Wait for all processes to complete
    for process in processes:
        process.join()
