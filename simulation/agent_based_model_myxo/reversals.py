"""
Author: Jean-Baptiste Saulnier
Date: 2024-12-06
"""
import numpy as np


class ReversalTypeError(Exception):
    pass


class RefractoryPeriodTypeError(Exception):
    pass


class ReversalRateTypeError(Exception):
    pass


class Reversal():
    """
    This class generates reversal behaviors based on a signal and a chosen reversal mechanism.
    
    The `reversal_type` parameter can be one of the following options:
    - "reversal_guzzo"
    - "reversal_guzzo_r_bilinear"
    - "reversal_rate_linear"
    - "reversal_rp_sigmoid_r_linear"
    - "reversal_rp_constant"
    - "reversals_periodic"
    - "reversal_r_constant"
    - "reversal_rate_sigmoidal"
    - "reversal_threshold_frustration"
    - "off" (default: "reversal_guzzo")
    
    The class also handles different mechanisms for the refractory period and reversal rates,
    based on the specified parameters.

    Attributes:
    -----------
    par : object
        Instance of the Parameters class containing simulation settings such as reversal type, constants for refractory period and reversal rates.
    
    gen : object
        Instance of the class managing general data and behaviors in the simulation.
    
    pha : object
        Instance of the class managing phase-related data for simulation purposes.
    
    sig : object
        Instance of the class managing signal-related data for simulation purposes.
    
    chosen_rp_function : method
        The selected function for calculating the refractory period based on the `reversal_type` parameter.
    
    chosen_rr_function : method
        The selected function for calculating the reversal rate based on the `reversal_type` parameter.
    
    chosen_reversal_function : method
        The function that handles both refractory period and reversal rate, depending on the selected `reversal_type`.
    
    a_rp : float
        The coefficient used in the calculation of the refractory period. It is 
        typically a constant that defines the linear relationship between the 
        bacteria's activity level and the refractory period.
        
    b_rp : float
        The constant term in the refractory period calculation, modifying the 
        overall refractory period based on the bacteria's activity.
        
    a_r : float
        The coefficient used in the calculation of the reversal rate. It modifies 
        the relationship between the bacteria's activity and the rate at which 
        reversals occur.
        
    b_r : float
        The constant term in the reversal rate calculation, influencing the 
        maximum rate of reversal irrespective of the bacteria's activity level.
        
    clock : numpy.ndarray
        A numpy array representing the internal clock of each bacterium, 
        which tracks the time elapsed since the last reversal event. This clock 
        is used to determine when a bacterium can reverse again based on the 
        refractory period.
        
    clock_tbr : numpy.ndarray
        A copy of the `clock` array used to track the time between reversals 
        (`tbr`). This array is useful for tracking the specific time each 
        bacterium has spent in the refractory state since the last reversal.
        
    tbr_list : list
        A list that stores the time between reversals (TBR) for each bacterium 
        involved in a reversal event. This list is useful for tracking and 
        analyzing reversal behaviors over time.
        
    tbr_position_x_list : list
        A list that stores the x-coordinates of the bacteria involved in reversal 
        events. This list helps in visualizing or analyzing the spatial patterns 
        of reversals.
        
    tbr_position_y_list : list
        A list that stores the y-coordinates of the bacteria involved in reversal 
        events. Similar to `tbr_position_x_list`, this list helps in analyzing 
        the spatial distribution of reversal events.
        
    A : numpy.ndarray
        A placeholder array used to track additional variables associated with 
        each bacterium, such as activity or other properties that influence 
        the reversal process.
        
    P : numpy.ndarray
        A placeholder array used to store the values associated with the 
        refractory period for each bacterium, based on the selected refractory 
        period function (`chosen_rp_function`).
        
    R : numpy.ndarray
        A placeholder array used to store the reversal rates for each bacterium. 
        The reversal rate is calculated based on the selected reversal rate 
        function (`chosen_rr_function`) and represents the probability of a 
        reversal occurring in a given time step.
        
    cond_rev : numpy.ndarray
        A boolean array indicating whether each bacterium is currently in a 
        reversal state (i.e., whether it is reversing or not). This array is 
        updated during each simulation step based on the reversal conditions.
        
    rev_to_plot_x : list
        A list that stores the x-coordinates of the bacteria that have reversed. 
        This data is typically used for visualization purposes, such as plotting 
        the positions of reversals over time or space.
        
    rev_to_plot_y : list
        A list that stores the y-coordinates of the bacteria that have reversed. 
        Similar to `rev_to_plot_x`, this data is useful for visualizing the 
        locations of reversals in the simulation.
        
    cond_reversing : numpy.ndarray
        A boolean array indicating whether each bacterium is currently in the 
        process of reversing. This is used to manage and track the reversal 
        state of each bacterium across time steps.
    """
    def __init__(self, inst_par, inst_gen, inst_pha, inst_sig):
        # Store references to external class instances
        self.par = inst_par
        self.gen = inst_gen
        self.pha = inst_pha
        self.sig = inst_sig

        # Handling different reversal mechanisms based on `reversal_type`
        if type(self.par.reversal_type) == tuple:
            # If `reversal_type` is a tuple, the first element specifies the refractory period function,
            # and the second specifies the reversal rate function.

            # Select the appropriate refractory period function based on the given type
            if self.par.reversal_type[0] == 'linear':
                self.chosen_rp_function = self.refractory_period_linear
            elif self.par.reversal_type[0] == 'sigmoidal':
                self.chosen_rp_function = self.refractory_period_sigmoidal
            elif self.par.reversal_type[0] == 'constant':
                self.chosen_rp_function = self.refractory_period_constant
            else:
                # Error handling if an invalid type is passed
                raise RefractoryPeriodTypeError('refractory_period_type could be: "linear", "sigmoidal" or "constant"; default is "linear"\n')
            
            # Select the appropriate reversal rate function based on the given type
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
                # Error handling for invalid reversal rate types
                raise ReversalRateTypeError('reversal_rate_type could be: "bilinear", "bilinear_smooth", "linear", "sigmoidal", "exponential" or "constant"; default is "bilinear"\n')
            
            # Set the chosen function for both refractory period and reversal rate
            self.chosen_reversal_function = self.reversal_rp_rr

        elif self.par.reversal_type == 'threshold_frustration':
            # If the reversal type is 'threshold_frustration', set the corresponding reversal function
            self.chosen_reversal_function = self.reversal_threshold_frustration

        elif self.par.reversal_type == 'periodic':
            # If the reversal type is 'periodic', set the corresponding reversal function
            self.chosen_reversal_function = self.reversals_periodic
        
        elif self.par.reversal_type == 'random':
            # If the reversal type is 'random', set the corresponding reversal function
            self.chosen_reversal_function = self.reversals_random
            
        elif self.par.reversal_type == 'off':
            # If the reversal type is 'off', set the function to do nothing (i.e., no reversal)
            self.chosen_reversal_function = self.function_doing_nothing

        else:
            # Error handling for invalid reversal types
            raise ReversalTypeError('reversal_type could be: (refractory_period_type, reversal_rate_type), "threshold_frustration", "periodic" , "random" or "off"; default is ("linear", "bilinear")\n')

        # Coefficients for the refractory period and the rate of reversals
        self.a_rp = (self.par.rp_min - self.par.rp_max) / (self.par.s2 - self.par.s1)  # Coefficient for refractory period
        self.b_rp = self.par.rp_max - self.a_rp * self.par.s1  # Constant term for refractory period
        self.a_r = np.log(self.par.r_max * self.par.c_r + 1) / self.par.s1  # Coefficient for reversal rate
        self.b_r = -1  # Constant term for reversal rate

        # Initialize simulation parameters
        self.clock = np.random.randint(low=0, high=self.par.rp_max/self.par.dt, size=self.par.n_bact) * self.par.dt
        self.clock_tbr = self.clock.copy()  # Copy of the clock for tracking
        self.tbr_list = []  # List for tracking reversals
        self.tbr_position_x_list = []  # List for tracking x positions of reversals
        self.tbr_position_y_list = []  # List for tracking y positions of reversals
        self.A = np.zeros(self.par.n_bact)  # Placeholder array for reversal signal
        self.P = np.zeros(self.par.n_bact)  # Placeholder array for refractory period
        self.R = np.zeros(self.par.n_bact)  # Placeholder array for reversal rate
        self.cond_rev = np.zeros(self.par.n_bact, dtype=bool)  # Condition for reversals
        self.rev_to_plot_x = []  # List for plotting x positions of reversals
        self.rev_to_plot_y = []  # List for plotting y positions of reversals

        # Handling non-reversing cells
        self.cond_reversing = np.ones(self.par.n_bact, dtype=bool)  # Initialize all cells as reversing
        if self.par.non_reversing:
            # Randomly select a percentage of cells to be non-reversing
            num_elements_false = int(self.par.non_reversing * len(self.cond_reversing))
            indices_to_false = np.random.choice(len(self.cond_reversing), num_elements_false, replace=False)
            self.cond_reversing[indices_to_false] = False  # Set selected cells to non-reversing


    def function_reversal_type(self):
        """
        Executes the reversal behavior and applies the chosen reversal function 
        based on the `reversal_type` parameter.

        The reversal type mechanism are selected based on the 
        `chosen_reversal_function` and `chosen_save_function` respectively.
        """
        self.chosen_reversal_function()  # Executes the selected reversal mechanism


    def function_doing_nothing(self):
        """
        A placeholder function that does nothing. It is used when no action is required
        when reversal_type == 'off'.
        """
        pass


    def clock_advenced(self):
        """
        Advances the time of the internal clock for each bacterium. The clock is 
        incremented by a small noise term to simulate fluctuations in timekeeping. 
        This is useful for modeling bacteria with slightly randomized internal clock 
        behaviors.

        The clock is updated with noise, which is uniformly distributed within a 
        range defined by `epsilon_clock`, and scaled by the simulation timestep `dt`.
        """
        noise = np.random.uniform(low=-self.par.epsilon_clock, high=self.par.epsilon_clock, size=self.par.n_bact).astype(self.par.float_type)
        self.clock[:] += self.par.dt * (1 + noise)  # Update clock with noise
        self.clock_tbr[:] += self.par.dt  # Update tbr clock, which measures the real time


    def clock_reset(self):
        """
        Resets the internal clock of each bacterium based on the maximum refractory period 
        (`rp_max`). The clock is reset either to the time remaining in the refractory period 
        (if `new_romr` is enabled) or set to zero (for the older method). This allows each 
        bacterium to "start fresh" in terms of its refractory period after the reset.

        The new RomR approach uses `np.maximum` to ensure that the clock value never goes 
        below zero.
        """
        if self.par.new_romr:
            # NEW RomR method (reset clock based on remaining refractory period)
            reset = np.maximum(self.par.rp_max - self.clock, 0)
            self.clock[self.cond_rev] = reset[self.cond_rev]
        else:
            # OLD method (reset clock to zero for reversing bacteria)
            self.clock[self.cond_rev] = 0


    def save_tbr(self):
        """
        Saves the time between reversals (TBR) for each bacterium that is currently 
        in a reversal state. The TBR is appended to a list, along with the x and y 
        positions of the bacteria involved in the reversal. This allows tracking 
        of reversal intervals and positions for further analysis or visualization.

        After saving the data, the TBR clock is reset to zero for these bacteria.
        """
        self.tbr_list.append(self.clock_tbr[self.cond_rev])  # Save the TBR times for reversed bacteria
        self.tbr_position_x_list.append(self.gen.data[0, 0, self.cond_rev])  # Save x positions of reversing bacteria
        self.tbr_position_y_list.append(self.gen.data[1, 0, self.cond_rev])  # Save y positions of reversing bacteria
        self.clock_tbr[self.cond_rev] = 0  # Reset the TBR clock for these bacteria


    def frz_activity(self, signal):
        """
        Computes the activity level of a bacterium based on its neighborhood polarity. 
        The input `signal` must be a value between 0 and 1, where 0 indicates 
        minimum activity and 1 indicates maximum activity. The method ensures that the 
        activity remains within the bounds of `[s0, s2]` by clamping values below `s0` 
        to `s0`, and values above `s2` to `s2`. This simulates bacteria with varying 
        levels of activity depending on their environment.

        Parameters:
        -----------
        signal : float
            The activity signal between 0 and 1 that determines the bacterium's activity level.
        """
        self.A[:] = signal.copy()  # Assign the signal to the activity array
        self.A[self.A < self.par.s0] = self.par.s0  # Clamp values below s0 to s0
        self.A[self.A > self.par.s2] = self.par.s2  # Clamp values above s2 to s2


    def refractory_period_linear(self):
        """
        Computes the refractory period using a linear relationship between the 
        activity level (`A`) and the refractory period (`P`). The linear function 
        is defined by the coefficients `a_rp` and `b_rp`. If the activity level 
        is below the threshold `s1`, the refractory period is set to the maximum value `rp_max`.

        This function is used when the chosen refractory period mechanism is "linear".

        The formula for the refractory period is:
            P = a_rp * A + b_rp
        """
        self.P[:] = self.a_rp * self.A + self.b_rp  # Calculate refractory period using the linear formula
        self.P[self.A < self.par.s1] = self.par.rp_max  # Set refractory period to maximum for low activity levels


    def refractory_period_sigmoidal(self):
        """
        Computes the refractory period using a sigmoidal function, which models 
        a smooth transition between `rp_max` and `rp_min` depending on the activity 
        level (`A`). The sigmoid function is controlled by the parameter `alpha_sigmoid_rp`, 
        which determines the steepness of the transition.

        The formula for the refractory period is:
            P = rp_max + (rp_min - rp_max) / (1 + exp(-alpha_sigmoid_rp * (A - (s1 + dec_s1))))
        
        This function is used when the chosen refractory period mechanism is "sigmoidal".
        """
        self.P[:] = self.par.rp_max + (self.par.rp_min - self.par.rp_max) / (1 + np.exp(-self.par.alpha_sigmoid_rp * (self.A - (self.par.s1 + self.par.dec_s1))))


    def refractory_period_constant(self):
        """
        Sets the refractory period to a constant value (`rp_max`), regardless of 
        the activity level (`A`). This function is used when the chosen refractory 
        period mechanism is "constant".

        The formula for the refractory period is:
            P = rp_max
        """
        self.P[:] = self.par.rp_max  # Set the refractory period to the maximum value for all bacteria


    def reversal_rate_exponential(self):
        """
        Calculates the reversal rate based on an exponential function of the 
        activity level (`A`). The rate increases exponentially with activity, 
        with the parameters `a_r` and `b_r` controlling the shape of the curve. 
        If the calculated reversal rate exceeds the maximum allowable value 
        (`r_max * c_r`), it is capped at that value. The result is then scaled 
        by `c_r` to ensure the reversal rate is within a desired range.

        The formula used is:
            R = exp(a_r * A) + b_r
            R is capped to r_max * c_r, then scaled by c_r.
        """
        self.R[:] = np.exp(self.a_r * self.A) + self.b_r  # Exponential calculation of the reversal rate
        self.R[self.R > self.par.r_max * self.par.c_r] = self.par.r_max * self.par.c_r  # Cap the rate at r_max * c_r
        self.R[:] /= self.par.c_r  # Scale the rate by c_r


    def reversal_rate_bilinear(self):
        """
        Calculates the reversal rate using a bilinear function based on the 
        activity level (`A`). The rate is proportional to the activity level 
        up to the threshold `s1`, beyond which the rate is capped at the maximum 
        value (`r_max`). This allows for a linear increase in the reversal rate 
        with activity until the threshold is reached, after which the rate becomes constant.

        The formula used is:
            R = r_max / s1 * A, for A <= s1
            R = r_max, for A > s1
        """
        self.R[:] = self.par.r_max / self.par.s1 * self.A  # Bilinear calculation for reversal rate
        self.R[self.A > self.par.s1] = self.par.r_max  # Cap the rate at r_max for high activity


    def reversal_rate_bilinear_smooth(self):
        """
        Calculates the reversal rate using a smooth bilinear function based on 
        the activity level (`A`). The function transitions smoothly between 
        the minimum (`r_min`) and maximum (`r_max`) reversal rates using a 
        sigmoid-like curve controlled by the parameter `alpha_bilinear_rr`. 
        This ensures a gradual increase in reversal rate as the activity level 
        increases, without abrupt jumps.

        The formula used is:
            R = r_min + (r_max - r_min) / (1 + ((A + 1e-6) / s1)^(-alpha_bilinear_rr))^(1/alpha_bilinear_rr)
        """
        self.R[:] = self.par.r_min + (self.par.r_max - self.par.r_min) / (1 + ((self.A + 1e-6) / self.par.s1) ** (-self.par.alpha_bilinear_rr)) ** (1 / self.par.alpha_bilinear_rr)  # Smooth bilinear calculation


    def reversal_rate_linear(self):
        """
        Calculates the reversal rate using a linear function based on the 
        activity level (`A`). The rate increases linearly with activity up to 
        a threshold `s2`, after which it becomes constant. This approach is 
        useful when the reversal rate should scale directly with activity until 
        a saturation point.

        The formula used is:
            R = r_max / s2 * A
        """
        self.R[:] = self.par.r_max / self.par.s2 * self.A  # Linear reversal rate calculation


    def reversal_rate_constant(self):
        """
        Sets the reversal rate to a constant value (`r_max`), regardless of 
        the activity level (`A`). This is useful when the reversal rate is 
        intended to remain constant throughout the simulation.

        The formula used is:
            R = r_max
        """
        self.R[:] = self.par.r_max  # Constant reversal rate


    def reversal_rate_sigmoidal(self):
        """
        Calculates the reversal rate using a sigmoidal function of the 
        activity level (`A`). The sigmoid function smoothly transitions between 
        the minimum (`r_min`) and maximum (`r_max`) reversal rates as the 
        activity level passes through the threshold `s1`. The sharpness of the 
        transition is controlled by the parameter `alpha_sigmoid_rr`.

        The formula used is:
            R = r_min - (r_min - r_max) / (1 + exp(-alpha_sigmoid_rr * (A - s1)))
        """
        self.R[:] = self.par.r_min - (self.par.r_min - self.par.r_max) / (1 + np.exp(-self.par.alpha_sigmoid_rr * (self.A - self.par.s1)))  # Sigmoidal calculation


    def reversal_rp_rr(self):
        """
        Performs a reversal operation considering both the refractory period 
        and the reversal rate. This method first updates the clocks, activity, 
        refractory period, and reversal rate. It then computes the probability 
        of a reversal for each bacterium based on the calculated reversal rate 
        and the time spent in the refractory period.

        A binomial distribution is used to determine which cells reverse, 
        and the nodes of the reversing bacteria are flipped. The clock is then 
        reset, and the time between reversals is saved for future analysis.

        The process includes the following steps:
        -----------
        1. Advance the clock.
        2. Calculate the activity of the cells based on the signal.
        3. Apply the chosen refractory period function.
        4. Apply the chosen reversal rate function.
        5. Calculate the probability of reversal and apply it using a binomial distribution.
        6. Flip the node dimensions for cells that reverse.
        7. Reset the clock for reversed cells and save the time between reversals.
        """
        # Update clock, activity, refractory period and reversal rate
        self.clock_advenced()  # Advance the clock
        self.frz_activity(signal=self.sig.signal)  # Calculate activity
        # Chosen refractory period
        self.chosen_rp_function()  # Apply the selected refractory period function
        # Chosen reversal rate
        self.chosen_rr_function()  # Apply the selected reversal rate function
        # Compute the probability to reverse and create a boolean array which is true for cells which have to reverse
        # prob = 1 - np.exp(-self.R * np.maximum(np.sign(self.clock - self.P), 0) * self.par.dt)  # Probability of reversal
        # self.cond_rev[:] = np.random.binomial(1, prob) & self.cond_reversing  # Determine which cells reverse
        prob = 1 - np.exp(-self.R * np.maximum(np.sign(self.clock - self.P), 0))  # Probability of reversal
        self.cond_rev[:] = np.random.binomial(1, prob * self.par.dt) & self.cond_reversing  # Determine which cells reverse
        # prob = 1 - np.exp(-self.R * np.maximum(np.sign(self.clock - self.P), 0))  # Probability of reversal
        # self.cond_rev[:] = np.random.binomial(1, prob * self.par.dt) & self.cond_reversing  # Determine which cells reverse
        # Flip the node dimension for cells which reverse
        self.gen.data[:, :, self.cond_rev] = np.flip(self.gen.data[:, :, self.cond_rev], axis=1)  # Flip positions of reversing cells
        self.pha.data_phantom[:, :, self.cond_rev] = np.flip(self.pha.data_phantom[:, :, self.cond_rev], axis=1)  # Flip phantom data for reversing cells
        # Reset the clock
        self.clock_reset()  # Reset the clock for reversed cells
        self.save_tbr()  # Save the time between reversals


    def reversals_periodic(self):
        """
        Performs periodic reversals, where the clock of each bacterium is checked 
        against the maximum refractory period (`rp_max`). If the clock has exceeded 
        this value, the bacterium performs a reversal. After the reversal, the nodes 
        are flipped and the clock is reset.

        This method simulates bacteria that reverse periodically based on a fixed 
        time schedule, independent of their activity level.

        The process includes the following steps:
        -----------
        1. Advance the clock.
        2. Check if the bacterium has exceeded the maximum refractory period.
        3. Flip the node dimensions for cells that reverse.
        4. Reset the clock for reversed cells and save the time between reversals.
        """
        self.clock_advenced()  # Advance the clock
        self.cond_rev[:] = self.clock >= self.rp_max  # Check if the clock exceeds the maximum refractory period
        # Flip the node dimension for cells which reverse
        self.gen.data[:, :, self.cond_rev] = np.flip(self.gen.data[:, :, self.cond_rev], axis=1)  # Flip positions of reversing cells
        self.pha.data_phantom[:, :, self.cond_rev] = np.flip(self.pha.data_phantom[:, :, self.cond_rev], axis=1)  # Flip phantom data for reversing cells
        # Reset the clock
        self.clock_reset()  # Reset the clock for reversed cells
        self.save_tbr()  # Save the time between reversals


    def reversals_random(self):
        """
        Performs random reversals for the bacteria. The probability of a reversal 
        for each bacterium is determined by a fixed value based on the maximum 
        reversal rate (`r_max`). The reversal events are modeled using a binomial 
        distribution, where each bacterium has an independent probability to reverse 
        during each time step. After a reversal, the nodes of the bacterium are flipped.

        The process includes the following steps:
        -----------
        1. Compute a constant probability for all bacteria based on r_max.
        2. Use a binomial distribution to decide which bacteria reverse.
        3. Flip the node dimensions for bacteria that reverse.
        4. Advance the clock.
        5. Save the time between reversals.
        
        The formula used for probability:
            prob = (1 - exp(-r_max)) for all bacteria.
        """
        prob = (1 - np.exp(-self.par.r_max * self.par.dt)) * np.ones(self.par.n_bact)  # Calculate the reversal probability
        self.cond_rev[:] = np.random.binomial(1, prob)  # Apply the binomial distribution for each bacterium
        # Flip the node dimension for cells which reverse
        self.gen.data[:, :, self.cond_rev] = np.flip(self.gen.data[:, :, self.cond_rev], axis=1)  # Flip positions of reversing cells
        self.pha.data_phantom[:, :, self.cond_rev] = np.flip(self.pha.data_phantom[:, :, self.cond_rev], axis=1)  # Flip phantom data for reversing cells
        self.clock_advenced()  # Advance the clock
        self.save_tbr()  # Save the time between reversals


    def reversal_threshold_frustration(self):
        """
        Performs reversals based on the frustration measure as described by Michèle 
        in the experiment. A reversal occurs when the signal exceeds a threshold 
        (`frustration_threshold_signal`) and the clock has passed a minimum refractory 
        period (`rp_max`). This represents a condition where the bacteria reverse 
        due to accumulated frustration, measured through the signal.

        The process includes the following steps:
        -----------
        1. Advance the clock.
        2. Check if the frustration signal exceeds the threshold and if the clock 
           exceeds the refractory period.
        3. If both conditions are true, reverse the bacteria by flipping their nodes.
        4. Reset the clock.
        5. Save the time between reversals.
        
        The formula used for the condition is:
            condition = (signal > frustration_threshold_signal) AND (clock > rp_max)
        """
        # Update clock, activity, refractory period, and reversal rate
        self.clock_advenced()  # Advance the clock
        self.cond_rev[:] = (self.sig.signal > self.par.frustration_threshold_signal) & (self.clock > self.par.rp_max)  # Frustration condition
        self.gen.data[:, :, self.cond_rev] = np.flip(self.gen.data[:, :, self.cond_rev], axis=1)  # Flip positions of reversing cells
        self.pha.data_phantom[:, :, self.cond_rev] = np.flip(self.pha.data_phantom[:, :, self.cond_rev], axis=1)  # Flip phantom data for reversing cells
        self.clock_reset()  # Reset the clock for reversed cells
        self.save_tbr()  # Save the time between reversals