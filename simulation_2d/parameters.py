import numpy as np


class Parameters:

    def __init__(self):
        
        ### SIMULATION TYPE
        self.generation_type = "gen_bact_square_align"
        self.movement_type = "tracted"
        self.alignment_type = "no_alignment"
        self.repulsion_type = "repulsion"
        self.attraction_type = "no_attraction"
        self.signal_type = "set_frustration_memory"
        self.reversal_type = ("sigmoidal", "sigmoidal")
        self.eps_follower_type = "no_eps"
        self.new_romr = False
        self.save_reversals = False
        self.random_movement = False
        self.rigidity_type = False

        ### ARRAY TYPE ###
        self.float_type = np.float64
        self.int_type = np.int32
        # Test for torus boundary condition (do not work from now)
        self.epsilon_float32 = 0

        ### PARALLELISATION ###
        self.n_jobs = 1

        ### KYMOGRAPH
        self.slice_width = 18 # µm

        ### PLOT
        self.tbr_plot = True
        self.tbr_cond_space_plot = False
        self.plot_movie = True
        self.plot_rippling_swarming_color = False
        self.plot_reversing_and_non_reversing = False
        self.kymograph_plot = False
        self.rev_function_plot = True
        self.velocity_plot = True
        self.plot_eps_grid = False
        self.plot_position_focal_adhesion_point = False
        self.param_point_size = 0.7
        self.time_rippling_swarming_colored = 60 # time until which patterns of rippling begin to form in minutes
        # Colors
        self.color_rippling = 'mediumturquoise'
        self.color_swarming = 'darkorange'
        self.color_reversing = 'red'
        self.color_non_reversing = 'limegreen'
        self.alpha = 0.5

        ### SAVE ###
        self.save_frequency = 2 # in minutes
        self.save_frequency_csv = 10 # minutes
        self.save_freq_velo = 1
        self.node_velocity_measurement = 0
        self.save_freq_tbr = 1/3 # in minutes
        self.save_freq_kymo = 1/10 # in minutes

        ### BACTERIA ###
        self.n_bact = 1000 # number of bacteria
        self.n_nodes = 10 # number of nodes which compose the bacteria
        self.d_n = 0.5 # distance between consecutive nodes
        self.bacteria_length = self.d_n * self.n_nodes
        self.width = 0.7
        self.v0 = 4
        self.nb_adhesion_points = 2
        self.sigma_v0 = 0.
        self.epsilon_velocity = 0.6
        self.sigma_random = 0.01

        ### TIME ###
        self.dt = 0.005 # min

        ### SPACE ###
        self.pili_length = self.n_nodes * self.d_n # lenght of the pili of the bacteria in µm
        self.d_disk = 55 # diameter of the disk where the bacteria will be plot if you choose the condition disk in the method generate bacteria
        self.space_size = 65

        ### EPS ###
        self.width_bins = 0.2 # in µm
        # eps attraction
        self.eps_angle_view = np.pi # angle view of the pili
        self.n_sections = 5 # number of section created inside the angle view of the pili
        self.angle_section = self.eps_angle_view / self.n_sections # angle view of each section
        self.epsilon_eps = 11 # intensity of the eps attraction
        self.sigma_eps = 0 # Heterogeneity in the eps following
        self.max_eps_value = 10 # maximum stack of eps in each bins of the eps grid
        self.eps_mean_lifetime = 60 * 100 # mean lifetime of the eps in minute
        self.deposit_amount = 2
        self.sigma_blur_eps = 0

        ### NEIGHBOURS ###
        # Maximal interaction distance
        self.i_dist = 2 * self.width
        # Maximal number of neighbours we can have
        self.kn = round((self.i_dist + self.d_n)**2 / self.d_n**2 * np.pi / (2*np.sqrt(3)))

        ### REPULSION ###
        self.k_r = 9e4 # repulsion intensity
        
        ### ATTRACTION ###
        self.k_a = 1e3
        self.k_a_pili = 6e2 # 6e2 ~ 2µm/min for maximal attraction
        self.at_angle_view = np.pi

        ### VISCOSITY ###
        self.epsilon_v = 0.3

        ### NODES SPRING STIFFNESS CONSTANT ###
        self.k_s = 1e4

        ### RIGIDITY ###
        self.k_rigid = 10
        self.rigidity_iteration = 1
        self.rigidity_first_node = 0

        ### ALIGNMENT ###
        self.j_t = 11
        self.max_align_dist = self.width
        self.global_angle = 0
        # Aligned and non aligned minimum and maximum in case of only a part of the space is aligned
        self.interval_rippling_space = [0, 1/2] # [min, max]
        self.percentage_bacteria_rippling = 0.666 # % of bacteria in interval_rippling_space

        ### REVERSALS ###
        self.save_frustration = False
        # Reversal activity
        self.a_max = 1 # maximal activity
        self.a_med = 0.5 # medium activity at threshold
        self.a_min = 0.0 # minimal activity when signal = 0
        self.s0 = 0.0
        self.s1 = 0.08 # first threshold of the signal
        self.s2 = 1 # second threshold, the activity is maximal when signal > s2
        self.dec_s1 = 0

        # Refractory period
        self.rp_max = 5 # maximal refractory period in minutes
        self.rp_min = 1/3 # minimal refractory period in minutes

        # Rate of reversal
        self.r_max = 3 # maximal reversal rate
        self.r_min = 0 # minimal rate of reversal
        self.c_r = 30 # 'curvature' of the curve to reach r_max
        self.alpha_sigmoid_rp = 100 # slope of the sigmoide function of the refractory period
        self.alpha_sigmoid_rr = 75 # slope of the sigmoide function of the reversal rate
        self.alpha_bilinear_rr = 10 # smoothing of the bilinear reversal function

        # Percentage of non reversing cells
        self.non_reversing = False # Percentage of False

        # Noise of the internal clock
        self.epsilon_clock = 0.

        # Signal max, max number of neighbour
        self.max_signal_sum_neg_polarity = 1.9
        self.max_dist_signal_density = self.width
        self.max_neighbours_signal_density = 25

        # Frustration
        self.max_frustration = 2
        self.time_memory = 1 / 3 # in minutes
        self.rate = 2 # in 1 / minutes
        self.frustration_threshold_signal = 0.4

        # Pre-waves
        self.waves_width = 30
        self.nb_waves = 2

        # Multi-layer
        self.min_dist_multi_layer = 3/4 * self.width

        # Choose the position of the initial bacteria
        ## Choose the coordinates and the directions of the bacteria
        bl = self.bacteria_length
        wi = self.width
        dn = self.d_n
        ss = self.space_size
        ## 5 bact
        # self.x = np.array([ss/1.9, ss/1.9, ss/2.1, ss/2.6, ss/3])
        # self.y = np.array([ss/2+3*dn, ss/2+7*dn, ss/2+bl-dn/2, ss/2, ss/2])
        # self.direction = np.array([0.02, -0.02, -np.pi/2, np.pi/2, np.pi/2])
        ## 4 bact
        # self.x = np.array([ss/2.1-1.5*bl, ss/2.1, ss/2.1+2*bl, ss/2.1])
        # self.y = np.array([ss/2-bl/2, ss/2-wi, ss/2, ss/2+2*bl])
        # self.direction = np.array([np.pi/2,np.pi,0,np.pi])
        ## Three bact
        # self.x = np.array([ss/2.1-2*wi, ss/2.1+6*wi, ss/2.1-3*wi])
        # self.y = np.array([ss/2, ss/2-bl/2, ss/2-bl/2+2*wi])
        # self.direction = np.array([np.pi/2.1, np.pi, 0])
        ## Two bact
        self.x = np.array([ss/2.1-2*wi, ss/2.1])
        self.y = np.array([ss/2, ss/2-0.5*bl])
        self.direction = np.array([np.pi/2, np.pi])
        ## One bact
        # self.x = np.array([ss/2])
        # self.y = np.array([ss/2])
        # self.direction = np.array([np.pi/4])
