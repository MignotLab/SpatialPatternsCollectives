# %%
import main
import parameters
import multiprocessing

"""
# EXAMPLE SWARMING
{   'generation_type':"gen_bact_square", 'n_bact':300, 'space_size':65,
    
    'repulsion_type':"repulsion", 'k_r':9e4,
    'attraction_type':"attraction_body", 'k_a':1e3,
    'alignment_type':"no_alignment",
    'eps_follower_type':"igoshin_eps_road_follower", 'plot_eps_grid':True,
    'reversal_type':("linear", "bilinear"),  's1':0.11, 's2':0.15, 'r_min':0, 'r_max':2, 'rp_min':0.5, 'rp_max':5,
    'alpha_sigmoid_rr':200, 'alpha_sigmoid_rp':100, 'alpha_bilinear_rr':10
    
},

# EXAMPLE RIPPLING
{   'generation_type':"gen_bact_square_align", 'n_bact':1000, 'space_size':65,
    
    'repulsion_type':"repulsion", 'k_r':9e4,
    'attraction_type':"attraction_body", 'k_a':1e3,
    'alignment_type':"global_alignment",
    'eps_follower_type':"no_eps", 'plot_eps_grid':True,
    'reversal_type':("linear", "bilinear"),  's1':0.11, 's2':0.15, 'r_min':0, 'r_max':2, 'rp_min':0.5, 'rp_max':5,
    'alpha_sigmoid_rr':200, 'alpha_sigmoid_rp':100, 'alpha_bilinear_rr':10
    
},

"""

T = 50
# Liste des ensembles de paramètres pour chaque simulation
params_list = [
    ## EXAMPLE SIMULATION RIPPLING
    {'generation_type':"gen_bact_square_align", 'n_bact':1000*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"global_alignment",
     'eps_follower_type':"no_eps",
    },

    ## EXAMPLE SIMULATION SWARMING 36
    {'generation_type':"gen_bact_square", 'n_bact':370*9, 'space_size':65*3, 'param_point_size':0.5,
     'alignment_type':"no_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
    },

    ## EXAMPLE SIMULATION RIPPLING AND SWARMING TRANSITION
    {'generation_type':"gen_bact_rippling_swarming", 'n_bact':(1000+500)*18, 'percentage_bacteria_rippling':0.666, 'space_size':65*6, 'plot_rippling_swarming_color':True,'param_point_size':0.25,
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'tbr_cond_space_plot':True 
    },

    ## EXAMPLE SIMULATION RIPPLING AND SWARMING TRANSITION WITH NON-REVERSING BACTERIA
    {'generation_type':"gen_bact_rippling_swarming", 'n_bact':(1200+500)*18, 'percentage_bacteria_rippling':1-500/(1200+500), 'space_size':65*6, 
     'plot_movie':False, 'plot_reversing_and_non_reversing':True,'param_point_size':0.25, 'non_reversing':500/(1200+500),
     'alignment_type':"global_alignment",
     'eps_follower_type':"igoshin_eps_road_follower",
     'tbr_cond_space_plot':True
    },
    # Add other parameter sets as needed
]

# Fonction pour chaque simulation
def simulate(params, sample):
    par = parameters.Parameters()
    for key, value in params.items():
        setattr(par, key, value)
    ma = main.Main(inst_par=par, sample=sample, T=T)
    ma.start()

if __name__ == '__main__':
    # Forcer l'utilisation de 'spawn' sur Linux pour que les simulations avec les même paramètre
    # ai un seed différent les une des autres
    multiprocessing.set_start_method('spawn')
    # Création et lancement des processus pour chaque simulation
    processes = []
    for i, params in enumerate(params_list):
        sample = 'sample' + str(i+1)
        process = multiprocessing.Process(target=simulate, args=(params, sample))
        processes.append(process)
        process.start()

    # Attente de la fin de tous les processus
    for process in processes:
        process.join()

# %%
