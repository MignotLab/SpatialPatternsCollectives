import numpy as np

class ReversalTypeError(Exception):
    pass

class RefractoryPeriodTypeError(Exception):
    pass

class ReversalRateTypeError(Exception):
    pass

class SaveReversalsTypeError(Exception):
    pass

class Reversal():
    """
    Generate reversal depending on a signal and the choosen reversal mechanism.
    reversal_type could be: "reversal_guzzo", "reversal_guzzo_r_bilinear", "reversal_rate_linear", "reversal_rp_sigmoid_r_linear", "reversal_rp_constant", "reversals_periodic", "reversal_r_constant", "reversal_rate_sigmoidal", "reversal_threshold_frustration", "no_reversal"; default is "reversal_guzzo"
    
    """
    def __init__(self, inst_par, inst_gen, inst_pha, inst_sig):
        
        # Parameters from parameters.py
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.sig = inst_sig

        if type(self.par.reversal_type) == tuple:

            if self.par.reversal_type[0] == 'linear':
                self.chosen_rp_function = self.refractory_period_linear
            elif self.par.reversal_type[0] == 'sigmoidal':
                self.chosen_rp_function = self.refractory_period_sigmoidal
            elif self.par.reversal_type[0] == 'constant':
                self.chosen_rp_function = self.refractory_period_constant
            else:
                # Print the different possible alignment when the class is call
                raise RefractoryPeriodTypeError('refractory_period_type could be: "linear", "sigmoidal" or "constant"; default is "linear"\n')
            
            if self.par.reversal_type[1] == 'bilinear':
                self.chosen_rr_function = self.reversal_rate_bilinear
            elif self.par.reversal_type[1] == "bilinear_smooth":
                self.chosen_rr_function = self.reversal_rate_bilinear_smooth
            elif self.par.reversal_type[1] == 'linear':
                self.chosen_rr_function = self.reversal_rate_linear
            elif self.par.reversal_type[1] == 'sigmoidal':
                self.chosen_rr_function = self.reversal_rate_sigmoidal
            elif self.par.reversal_type[1] == 'exponential':
                self.chosen_rr_function = self.reversal_rate_exponential
            elif self.par.reversal_type[1] == 'constant':
                self.chosen_rr_function = self.reversal_rate_constant
            else:
                raise ReversalRateTypeError('reversal_rate_type could be: "bilinear", "bilinear_smooth", "linear", "sigmoidal", "exponential" or "constant"; default is "bilinear"\n')
            
            self.chosen_reversal_function = self.reversal_rp_rr
            
        elif self.par.reversal_type == 'threshold_frustration':
            self.chosen_reversal_function = self.reversal_threshold_frustration

        elif self.par.reversal_type == 'periodic':
            self.chosen_reversal_function = self.reversals_periodic
        
        elif self.par.reversal_type == 'random':
            self.chosen_reversal_function = self.reversals_random
            
        elif self.par.reversal_type == 'off':
            self.chosen_reversal_function = self.function_doing_nothing

        else:
            raise ReversalTypeError('reversal_type could be: (refractory_period_type, reversal_rate_type), "threshold_frustration", "periodic" , "random" or "off"; default is ("linear", "bilinear")\n')
        
        if self.par.save_reversals == True:
            self.chosen_save_function = self.save_reversals
        elif self.par.save_reversals == False:
            self.chosen_save_function = self.function_doing_nothing
        else:
            raise SaveReversalsTypeError('save_reversals could be: True or False, default is False\n')

        # Coefficient for the refractory period and the rate of reversals
        self.a_rp = (self.par.rp_min - self.par.rp_max) / (self.par.s2 - self.par.s1) # director coefficient
        self.b_rp = self.par.rp_max - self.a_rp * self.par.s1 # constant coefficient
        self.a_r = np.log(self.par.r_max * self.par.c_r + 1) / self.par.s1 # director coefficient
        self.b_r = -1 # constant coefficient

        # class objects
        self.clock = np.random.randint(low=0, high=self.par.rp_max/self.par.dt, size=self.par.n_bact) * self.par.dt
        self.clock_tbr = self.clock.copy()
        self.tbr_list = []
        self.tbr_position_x_list = []
        self.tbr_position_y_list = []
        self.A = np.zeros(self.par.n_bact)
        self.P = np.zeros(self.par.n_bact)
        self.R = np.zeros(self.par.n_bact)
        self.cond_rev = np.zeros(self.par.n_bact, dtype=bool)
        self.rev_to_plot_x = []
        self.rev_to_plot_y = []

        # Non reversing cells
        self.cond_reversing = np.ones(self.par.n_bact, dtype=bool)
        if self.par.non_reversing:
            # Number of elements to set to False (self.par.non_reversing %)
            num_elements_false = int(self.par.non_reversing * len(self.cond_reversing))
            # Generate random indices to select elements to modify
            indices_to_false = np.random.choice(len(self.cond_reversing), num_elements_false, replace=False)
            # Set the corresponding elements to False
            self.cond_reversing[indices_to_false] = False


    def function_reversal_type(self):
        """
        Function doing the movement type and the random movement depending on the choice of these movement in the parameter class
        
        """
        self.chosen_reversal_function()
        self.chosen_save_function()

    def function_doing_nothing(self):
        pass

    def clock_advenced(self):
        """
        Increase the time of the internal clock of each bacteria
        
        """
        noise = np.random.uniform(low=-self.par.epsilon_clock,high=self.par.epsilon_clock,size=self.par.n_bact).astype(self.par.float_type)
        self.clock[:] += self.par.dt * (1 + noise)
        self.clock_tbr[:] += self.par.dt # the tbr clock measure the real time

    def clock_reset(self):
        """
        Reset to rp_max - clock (RomR rest) the time of the internal clock of each bacteria
        
        """
        if self.par.new_romr:
            # NEW RomR
            reset = np.maximum(self.par.rp_max - self.clock, 0)
            self.clock[self.cond_rev] = reset[self.cond_rev]
        else:
            # OLD
            self.clock[self.cond_rev] = 0

    def save_tbr(self):
        """
        Save the time between reversals
        
        """
        self.tbr_list.append(self.clock_tbr[self.cond_rev])
        self.tbr_position_x_list.append(self.gen.data[0, 0, self.cond_rev])
        self.tbr_position_y_list.append(self.gen.data[1, 0, self.cond_rev])
        self.clock_tbr[self.cond_rev] = 0

    def frz_activity(self,signal):
        """
        Compute the activity of a cell depending of its neighborhood polarity
        The signal need to be between 0 and 1, 0 mean minimum signal and 1 maximum signal

        """
        self.A[:] = signal.copy()
        self.A[self.A < self.par.s0] = self.par.s0
        self.A[self.A > self.par.s2] = self.par.s2

    def refractory_period_linear(self):
        """
        Refractory period linear
        
        """
        self.P[:] = self.a_rp * self.A + self.b_rp
        self.P[self.A < self.par.s1] = self.par.rp_max

    def refractory_period_sigmoidal(self):
        """
        Refractory period linear
        
        """
        self.P[:] = self.par.rp_max + (self.par.rp_min-self.par.rp_max) / (1 + np.exp(-self.par.alpha_sigmoid_rp * (self.A-(self.par.s1+self.par.dec_s1))))

    def refractory_period_constant(self):
        """
        Refractory period linear
        
        """
        self.P[:] = self.par.rp_max

    def reversal_rate_exponential(self):
        """
        Rate of reversal
        
        """
        self.R[:] = np.exp(self.a_r * self.A) + self.b_r
        self.R[self.R > self.par.r_max * self.par.c_r] = self.par.r_max * self.par.c_r
        self.R[:] /= self.par.c_r

    def reversal_rate_bilinear(self):
        """
        Rate of reversal
        
        """
        self.R[:] = self.par.r_max / self.par.s1 * self.A
        self.R[self.A > self.par.s1] = self.par.r_max

    def reversal_rate_bilinear_smooth(self):
        """
        Rate of reversal
        
        """
        self.R[:] = self.par.r_min + (self.par.r_max-self.par.r_min) / (1 + ((self.A+1e-6)/self.par.s1)**(-self.par.alpha_bilinear_rr)) ** (1/self.par.alpha_bilinear_rr)

    def reversal_rate_linear(self):
        """
        Rate of reversal
        
        """
        self.R[:] = self.par.r_max / self.par.s2 * self.A

    def reversal_rate_constant(self):
        """
        Rate of reversal
        
        """
        self.R[:] = self.par.r_max

    def reversal_rate_sigmoidal(self):
        """
        Rate of reversal
        
        """
        self.R[:] = self.par.r_min - (self.par.r_min-self.par.r_max) / (1 + np.exp(-self.par.alpha_sigmoid_rr * (self.A-self.par.s1)))

    def reversal_rp_rr(self):
        """
        Function making reversal with a defined refractory period and reversal rate

        """
        # Update clock, activity, refractory period and reversal rate
        self.clock_advenced()
        self.frz_activity(signal=self.sig.signal)
        # Chosen refractory period
        self.chosen_rp_function()
        # Chosen reversal rate
        self.chosen_rr_function()
        # Compute the probability to reverse and create a boolean array which is true for cell which 
        # have to reverse
        prob = 1 - np.exp(-self.R * np.maximum(np.sign(self.clock - self.P), 0))
        self.cond_rev[:] = np.random.binomial(1, prob*self.par.dt) & self.cond_reversing
        # Flip the node dimension for cell which reverse
        self.gen.data[:,:,self.cond_rev] = np.flip(self.gen.data[:,:,self.cond_rev], axis=1)
        self.pha.data_phantom[:,:,self.cond_rev] = np.flip(self.pha.data_phantom[:,:,self.cond_rev], axis=1)
        # Reset the clock
        self.clock_reset()
        self.save_tbr()

    def reversals_periodic(self):
        """
        Periodic reversals
        
        """
        self.clock_advenced()
        self.cond_rev[:] = self.clock >= self.rp_max
        # Flip the node dimension for cell which reverse
        self.gen.data[:,:,self.cond_rev] = np.flip(self.gen.data[:,:,self.cond_rev], axis=1)
        self.pha.data_phantom[:,:,self.cond_rev] = np.flip(self.pha.data_phantom[:,:,self.cond_rev], axis=1)
        # Reset the clock
        self.clock_reset()
        self.save_tbr()

    def reversals_random(self):
        """
        Random reversals
        
        """
        prob = (1 - np.exp(-self.par.r_max)) * np.ones(self.par.n_bact)
        self.cond_rev[:] = np.random.binomial(1, prob*self.par.dt)
        # Flip the node dimension for cell which reverse
        self.gen.data[:,:,self.cond_rev] = np.flip(self.gen.data[:,:,self.cond_rev], axis=1)
        self.pha.data_phantom[:,:,self.cond_rev] = np.flip(self.pha.data_phantom[:,:,self.cond_rev], axis=1)
        self.clock_advenced()
        self.save_tbr()

    def reversal_threshold_frustration(self):
        """
        Frustration as Michèle measure it in the experiment
        
        """
        # Update clock, activity, refractory period and reversal rate
        self.clock_advenced()
        self.cond_rev[:] = (self.sig.signal > self.par.frustration_threshold_signal) & (self.clock > self.par.rp_max)
        self.gen.data[:,:,self.cond_rev] = np.flip(self.gen.data[:,:,self.cond_rev], axis=1)
        self.pha.data_phantom[:,:,self.cond_rev] = np.flip(self.pha.data_phantom[:,:,self.cond_rev], axis=1)
        # Reset the clock
        self.clock_reset()
        self.save_tbr()

    def save_reversals(self):
        """
        Save the position of the reversals
        
        """
        self.rev_to_plot_x.append(self.gen.data[0,int(self.par.n_nodes/2),self.cond_rev])
        self.rev_to_plot_y.append(self.gen.data[1,int(self.par.n_nodes/2),self.cond_rev])
