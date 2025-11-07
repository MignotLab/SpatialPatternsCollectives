"""
Author: Jean-Baptiste Saulnier
Date: 2024-11-21

This module contains the parameters class for the simulation. It defines various
parameters for bacteria characteristics, simulation settings, plotting options,
and saving intervals.
"""
import numpy as np


class Parameters:


    def __init__(self):
        ### OBJECT TYPE ###
        self.float_type = np.float64  # Type for floating-point values
        self.int_type = np.int32  # Type for integer values
        

        ### SIMULATION FEATURES ###
        self.alignment_type = "no_alignment"
        self.eps_follower_type = "no_eps"  # eps follower behavior (none by default)
        self.prey_follower_type = 'no_prey_ecm'  # ecm prey follower behavior (none by default)
        self.generation_type = "square_random_orientation"  # Method of bacteria alignment generation
        self.movement_type = "tracted"  # Type of movement for the bacteria
        self.neighbour_detection_type = "torus"  # Method of bacteria neighbours detection
        self.repulsion_type = "repulsion"  # Repulsion method between bacteria
        self.attraction_type = "no_attraction"  # Attraction method between bacteria
        self.random_movement = False  # Enable random movement of bacteria if True
        self.reversal_type = ("sigmoidal", "sigmoidal") # Reversal method
        self.signal_type = "set_frustration_memory_exp_decrease" # Signal method
        self.new_romr = False # Refer to Guzzo et al. 2018 and the method clock_reset in reversals.py to understand
        self.rigidity_type = False

        ### SIMULATION FEATURES LOOSES AT A SPECIFIC TIME ###
        self.stop_alignment = None
        self.stop_eps_follower = None

        ### PLOT ###
        self.tbr_plot = True  # Whether to plot the TBR (time before reversal) distribution.
        self.tbr_cond_space_plot = False  # Whether to plot the conditional space of TBR values (not activated here).
        self.plot_movie = True  # Whether to generate a movie of the simulation.
        self.plot_rippling_swarming_color = False  # Whether to color code rippling and swarming dynamics.
        self.plot_reversing_and_non_reversing = False  # Whether to distinguish between reversing and non-reversing cells in plots.
        self.plot_colored_nb_neighbors = False  # Whether to color code the number of neighbours.
        self.kymograph_plot = False  # Whether to generate a kymograph plot (spatial-temporal density).
        self.rev_function_plot = True  # Whether to plot the reversal functions (e.g., reversal rate vs. signal).
        self.velocity_plot = True  # Whether to plot the velocity distribution histogram.
        self.plot_ecm_grid = False  # Whether to plot the EPS grid (extracellular polymeric substances).
        self.plot_position_focal_adhesion_point = False  # Whether to plot the position of focal adhesion points (cell anchoring).
        self.param_point_size = 0.2  # Size of points used in plots, such as cell positions.
        self.time_rippling_swarming_colored = 60  # Time (in minutes) until which rippling patterns are emphasized in plots.

        # Colors and plot size
        self.alpha = 0.7
        self.size_height_figure = 7
        self.figsize = (self.size_height_figure, self.size_height_figure-1)
        self.dpi = 300
        self.dpi_simu = 25
        self.fontsize = 30
        self.fontsize_ticks = self.fontsize / 1.5
        self.color_rippling = 'mediumturquoise'
        self.color_swarming = 'darkorange'
        self.color_reversing = 'red'
        self.color_non_reversing = 'limegreen'


        ### SAVE ###
        self.save_frequency_image = 2  # Frequency of image saving (in minutes)
        self.save_frequency_csv = 10  # Frequency of CSV data saving (in minutes)
        self.save_in_csv_from_time = 0  # Starting time for saving CSV data
        self.save_freq_velo = 1  # (in minutes)
        self.node_velocity_measurement = 0  # Id of the node you extract the velocity when `velocity_plot` is True
        self.save_freq_tbr = 1/3  # (in minutes)
        self.save_freq_kymo = 1/10  # (in minutes)
        self.dpi = 100
        self.save_other_reversal_signals = None # Save all reversal signals values even if they are not used


        ### BACTERIA CHARACTERISTICS ###
        self.n_bact = 1000  # Number of total bacteria in the simulation
        self.n_bact_prey = 0  # Number of prey bacteria; WARNING: if this parameter is not 0, this will generate non motile prey bacterium in the simulation
        self.n_nodes = 10  # Number of nodes per bacterium
        self.d_n = 0.5  # Distance between nodes in a bacterium (µm)
        self.d_n_prey = 0.15 # Distance between nodes in a prey bacterium (µm)
        self.bacteria_length = self.d_n * self.n_nodes  # Total length of a bacterium (µm)
        self.width = 0.7  # Width of a bacterium (µm)
        self.v0 = 4  # Speed of bacteria (µm/minute)
        self.sigma_random = 0.01  # Standard deviation for random movement of the bacterial head


        ### TIME STEP ###
        self.dt = 0.005  # Time step in minutes for the simulation


        ### SPACE ###
        self.d_disk = 55  # Diameter of disk for bacteria placement (µm)
        self.space_size = 65  # Total size of the simulation space


        ### ECM ###
        self.pili_length = 5 # lenght of the pili of the bacteria in µm
        self.width_bins = 0.3  # Bin width for eps grid (µm)
        self.eps_angle_view = np.pi  # Angle of view for pili (in radians)
        self.n_sections = 5  # Number of sections in the eps grid's angle view
        self.angle_section = self.eps_angle_view / self.n_sections  # Angle per section
        self.epsilon_eps = 11  # Intensity of eps attraction
        self.epsilon_prey = 11  # Intensity of ECM prey attraction
        self.sigma_gaussian_eps = 2  # Standart deviation of the ECM prey gaussian 
        self.radius_prey_ecm_effect = 3 # Radius of the distance effect of the prey ECM
        self.max_eps_value = 10  # Maximum eps value per bin
        self.eps_mean_lifetime = 60 * 100  # Mean lifetime of eps (minutes)
        self.deposit_amount = 2  # Amount of eps deposited per minute
        self.sigma_blur_eps_map = 0  # Standard deviation for eps blur effect


        ### NEIGHBOURS ###
        self.i_dist = 2 * self.width  # Maximal interaction distance between bacteria (µm)
        self.kn = round((self.i_dist + self.d_n)**2 / self.d_n**2 * np.pi / (2*np.sqrt(3)))  # Max neighbours per disk


        ### REPULSION ###
        # self.k_r = 225  # Repulsion constant (µm/minute)
        self.k_r = 450  # Repulsion constant (µm/minute)
        # self.k_r = 9e4  # Repulsion constant (µm/minute)


        ### ATTRACTION ###
        self.k_a = 5  # The attraction constant (µm/minute) for the overall attraction force between bacteria. 
                           # This defines the strength of the attraction, with a higher value indicating stronger attraction.
        self.k_a_pili = 3  # The attraction constant (µm/minute) specific to pili-mediated interactions between bacteria. 
                                # This defines the strength of attraction through pili (hair-like structures), with a typical value around 3 to simulate 2 µm/min for maximal attraction.
        self.at_angle_view = np.pi  # The angular range within which bacteria can "see" each other for attraction purposes. 
                                    # Set to π (180°), meaning bacteria can attract each other in a full hemisphere around their body.


        ### ALIGNMENT ###
        self.j_t = 11  # Global alignment strength (µm/minute), controls the intensity of the alignment force applied to the bacteria's orientation.
        self.max_align_dist = self.width  # Maximum distance at which bacteria are considered for alignment with their neighbors.
        self.global_angle = 0  # The global alignment direction in radians, used when bacteria are aligned to a fixed direction.
        self.interval_rippling_space = [0, 1/2]  # [min, max] ratio of the space where alignment (rippling) will be enforced.
        self.percentage_bacteria_rippling = 0.666  # Percentage of bacteria within the `interval_rippling_space` that are aligned (i.e., experience rippling).


        ### NODES SPRING STIFFNESS CONSTANT ###
        self.k_s = 50  # Spring stiffness constant for nodes (µm/minute)
        # self.k_s = 1e4  # Spring stiffness constant for nodes (µm/minute)


        ### VISCOSITY ###
        self.epsilon_v = 0.3  # Viscosity coefficient (µm/minute) for the attraction or repulsion between the nodes of the bacteria. 
                                   # This parameter controls the intensity of the viscosity effect, with higher values 
                                   # increasing the attraction between nodes and affecting their relative movement.


        ### RIGIDITY ###
        self.k_rigid = 10  # Rigidity coefficient (µm/minute) used to calculate the resistance to deformation between the nodes in the rigidity model. 
                                # A higher value increases the force opposing the deformation or separation of nodes.
        self.rigidity_iteration = 1  # Number of iterations for computing the rigidity forces between triplets of nodes. 
                                     # A higher value helps to better stabilize the nodes but increases computational cost.
        self.rigidity_first_node = 0  # Index of the first node where the rigidity forces are applied.
                                      # This determines which node positions the rigidity forces will act upon.


        ### REVERSALS ###
        self.save_frustration = False  # Flag to save or log the frustration values during the reversal process.
        
        # Reversal activity
        self.a_max = 1  # Maximum activity level during a reversal. Defines the peak activity when the reversal is triggered.
        self.a_med = 0.5  # Medium activity level at the signal threshold (typically when the signal is in the mid-range).
        self.a_min = 0.0  # Minimum activity level when the signal is 0, meaning no reversal activity.
        self.s0 = 0.0  # The baseline or starting signal value.
        self.s1 = 0.08  # The first threshold for the signal; reversal activity begins to increase when the signal exceeds this value.
        self.s2 = 1  # The minute threshold for the signal; when the signal is above this value, the activity is maximal (a_max).
        self.dec_s1 = 0  # Decrement for the first threshold, possibly used to modify the threshold dynamically.
        
        # Refractory period
        self.rp_max = 5  # The maximum refractory period (in minutes) after a reversal, during which the system cannot reverse again.
        self.rp_min = 1/3  # The minimum refractory period (in minutes) after a reversal.

        # Rate of reversal
        # self.r_max = 1  # The maximum rate of reversal (in minutes^-1), defines how quickly the reversal process can occur.
        # self.r_max = 3  # The maximum rate of reversal (in minutes^-1), defines how quickly the reversal process can occur.
        # self.r_max = 1  # The maximum rate of reversal (in minutes^-1), defines how quickly the reversal process can occur.
        self.r_max = 3  # The maximum rate of reversal (in minutes^-1), defines how quickly the reversal process can occur.
        self.r_min = 0  # The minimum rate of reversal (in minutes^-1), indicates that no reversal is happening at this rate.
        self.c_r = 30  # 'curvature' of the curve to reach r_max
        self.alpha_sigmoid_rp = 100  # The steepness (slope) of the sigmoid function governing the refractory period.
        self.alpha_sigmoid_rr = 75  # The steepness (slope) of the sigmoid function governing the reversal rate.
        self.alpha_bilinear_rr = 10  # Smoothing factor for the bilinear function used to define the reversal rate.

        # Percentage of non-reversing cells
        self.non_reversing = False  # Flag to define the percentage of cells that do not reverse, i.e., the cells that are non-reversing.

        # Noise of the internal clock
        self.epsilon_clock = 0.  # Noise level in the internal clock, possibly introducing variability in time-related processes.

        # Signal max, max number of neighbors
        self.max_signal_sum_neg_polarity = 1.4  # Maximum sum of negative polarity signals for a bacterium to consider in the reversal process.
        self.max_dist_signal_density = self.width  # The maximum distance at which signal density is considered for reversal activity.
        self.max_neighbours_signal_density = 55  # The maximum number of neighboring bacteria that influence the signal density for a reversal event.
        self.neighbor_distance_factor = 1. # Defines how far to search for neighbors relative to bacterial width

        # Frustration
        self.max_frustration = 2  # Maximum level of frustration that can accumulate in the system.
        self.time_memory = 1/3  # The time window (in minutes) for remembering past frustrations, affecting current decisions.
        self.rate = 2  # The rate of frustration accumulation per unit of time (in minutes^-1).
        self.frustration_threshold_signal = 0.4  # The threshold of the signal value at which frustration will be triggered, influencing reversal behavior.

        ### SPECIFIC TO THE PREY INTERACTION ###
        self.rate_stop_at_prey = 10
        self.rate_prey_death = 5 # The rate of prey death when a predator is at contact (in minutes)

        ### SPECIFIC INITIAL POSITIONS ###
        ## Choose the coordinates and directions for the initial positions of the bacteria
        bl = self.bacteria_length
        wi = self.width
        dn = self.d_n
        ss = self.space_size
        ## Two bacteria
        self.x = np.array([ss/2.1 - 2*wi, ss/2.1])
        self.y = np.array([ss/2, ss/2 - 0.5*bl])
        self.direction = np.array([np.pi/2, np.pi])  # Initial directions (radians)
