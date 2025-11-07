import numpy as np


class IgoshinMatrix:
    """
    Igoshin matrices
    
    """
    def __init__(self, inst_par):
        
        self.par = inst_par


    def w_1(self, rho_bar, q):
        """
        Function \omega_1
        
        """
        return self.par.w_n * rho_bar**q / (rho_bar**q + self.par.rho_w**q)
    

    def K_1(self, S, q):
        """
        Function K_1
        
        """
        num = q * S * (1 - S / self.par.signal_max)
        den = S * self.par.delta_phi_r + np.pi

        return num / den


    def K_1_linear(self, S):
        """
        Function K_1
        
        """
        num = S
        den = S * self.par.delta_phi_r + np.pi

        return num / den
    

    def main_matrix(self, xi, S, K_1, dp):
        """
        Construct the matrix from the igoshin linearised model

        """
        n = int(self.par.phi_max / dp)
        n_rp = int(self.par.delta_phi_r / dp)
        g_plus = np.zeros((n, n), dtype='cfloat')
        g_a = np.zeros((n, n), dtype='cfloat')

        # Diag G+
        g_plus[self.par.index_diag[:n_rp], self.par.index_diag[:n_rp]] += -1j*self.par.v*xi - 1 / self.par.dp
        g_plus[self.par.index_diag[n_rp:], self.par.index_diag[n_rp:]] += -1j*self.par.v*xi - (1 + S) / self.par.dp
        g_plus[0, :] += K_1

        # Subdiag G+
        g_plus[self.par.index_sub_diag+1, self.par.index_sub_diag] += 1 / self.par.dp
        g_plus[(self.par.index_sub_diag+1)[n_rp:], self.par.index_sub_diag[n_rp:]] *= (1 + S)

        # G-
        g_minus = g_plus.copy()
        g_minus[self.par.index_diag, self.par.index_diag] += 2j*self.par.v*xi

        # G_A
        g_a[0, -1] += (1 + S) / self.par.dp
        g_a[n_rp, :] += - K_1
        
        g = np.block([[g_plus, g_a], [g_a, g_minus]])

        return g, g_plus, g_minus, g_a
    

    def phi_r(self, rho_bar):
        """
        Modulated refractory period for igoshin
        
        """
        return self.par.q_value_constant * rho_bar
    

    def K_1_rp(self, S):
        """
        Form of K_1 when modulate the refractory period
        
        """
        return S / (S * self.par.delta_phi_r + np.pi)
    

    def rp_matrix_L(self, xi, S, K_1, dp):
        """
        Construct the matrix from the igoshin linearised model

        """
        n = int(self.par.phi_max / dp)
        n_rp = int(self.par.delta_phi_r / dp)

        l_plus = np.zeros((n, n), dtype='cfloat')
        l_a_plus = np.zeros((n, n), dtype='cfloat')

        # Diag L+
        l_plus[self.par.index_diag[:n_rp], self.par.index_diag[:n_rp]] = -1j*self.par.v*xi - 1 / self.par.dp
        l_plus[self.par.index_diag[n_rp:], self.par.index_diag[n_rp:]] = -1j*self.par.v*xi - (1 + S) / self.par.dp

        # Subdiag L+
        l_plus[self.par.index_sub_diag+1, self.par.index_sub_diag] = 1 / self.par.dp
        l_plus[(self.par.index_sub_diag+1)[n_rp:], self.par.index_sub_diag[n_rp:]] *= (1 + S)

        # Diag L-
        l_minus = l_plus.copy()
        l_minus[self.par.index_diag, self.par.index_diag] += 2j*self.par.v*xi

        # L_A+
        l_a_plus[0, -1] = (1 + S) / self.par.dp
        l_a_plus[n_rp, :] = K_1 * self.par.v * 1j * xi

        # L_A-
        l_a_minus = l_a_plus.copy()
        l_a_minus[n_rp, :] *= -1

        
        l = np.block([[l_plus, l_a_plus], [l_a_minus, l_minus]])

        return l, l_plus, l_minus, l_a_plus, l_a_minus
    

    def rp_matrix_B(self, K_1, dp):
        """
        Construct the matrix from the igoshin linearised model

        """
        n = int(self.par.phi_max / dp)
        n_rp = int(self.par.delta_phi_r / dp)

        b_id = np.zeros((n, n), dtype='cfloat')
        b_a = np.zeros((n, n), dtype='cfloat')

        # Diag identity
        b_id[self.par.index_diag[:], self.par.index_diag[:]] = 1

        # b_a
        b_a[n_rp, :] = - K_1

        b = np.block([[b_id, b_a], [b_a, b_id]])

        return b, b_id, b_a
    

# %%
# # EXAMPLE FOR G_+-
# import numpy as np
# n = 10
# n_rp = 5
# g = np.zeros((n, n))
# index_diag = np.arange(n)
# index_sub_diag = np.arange(n-1)
# g[index_diag[:n_rp-1], index_diag[:n_rp-1]] = 1
# g[index_diag[n_rp-1:], index_diag[n_rp-1:]] = 2
# g[0, :] += 0.5
# g[index_sub_diag+1, index_sub_diag] = 3
# g[(index_sub_diag+1)[n_rp-1:], index_sub_diag[n_rp-1:]] *= 2
# print('n_rp=', n_rp)
# print(g)

# g_a = np.zeros((n, n))
# g_a[0, -1] = 0.61
# g_a[n_rp-1, :] = -10
# print(g_a)