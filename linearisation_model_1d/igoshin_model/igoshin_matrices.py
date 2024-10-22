import numpy as np


class IgoshinMatrix:
    """
    Igoshin matrices
    
    """
    def __init__(self, inst_par):
        
        self.par = inst_par


    def w_1_sigmoid(self, rho_bar, q):
        """
        Function \omega_1
        
        """
        return self.par.w_star * rho_bar**q / (rho_bar**q + self.par.rho_w**q)
    

    def w_1_linear(self, rho_bar):
        """
        Function \omega_1
        
        """
        return self.par.w_star * rho_bar / self.par.rho_t
    

    def C_sigmoid(self, S, q):
        """
        Function C
        
        """
        num = q * S * (1 - S / self.par.signal_max)
        den = S * self.par.Phi_R_star + np.pi

        return num / den


    def C_linear(self, S, Phi_R):
        """
        Function C
        
        """
        num = S
        den = S * Phi_R + np.pi

        return num / den
    

    def main_matrix(self, xi, S, Phi_R):
        """
        Construct the matrix from the igoshin linearised model

        """
        n = self.par.n
        n_rp = int(Phi_R / self.par.dp - 1)
        g_plus = np.zeros((n, n), dtype='cfloat')
        g_a = np.zeros((n, n), dtype='cfloat')

        # Diag G+
        g_plus[self.par.index_diag[:n_rp], self.par.index_diag[:n_rp]] += -1j*xi - 1 / self.par.dp
        g_plus[self.par.index_diag[n_rp:], self.par.index_diag[n_rp:]] += -1j*xi - (1 + S) / self.par.dp
        g_plus[0, :] += self.C_linear(S, Phi_R)

        # Subdiag G+
        g_plus[self.par.index_sub_diag+1, self.par.index_sub_diag] += 1 / self.par.dp
        g_plus[(self.par.index_sub_diag+1)[n_rp:], self.par.index_sub_diag[n_rp:]] *= (1 + S)

        # G-
        g_minus = g_plus.copy()
        g_minus[self.par.index_diag, self.par.index_diag] += 2j*xi

        # G_A
        g_a[0, -1] += (1 + S) / self.par.dp
        g_a[n_rp, :] += -self.C_linear(S, Phi_R)
        
        g = np.block([[g_plus, g_a], [g_a, g_minus]])

        return g, g_plus, g_minus, g_a
    

    def Phi_R_bar(self, rho_bar):
        """
        Modulated refractory period for igoshin
        
        """
        return self.par.Phi_R_star * self.par.rho_t / rho_bar
    

    def C_linear_rp(self, S, Phi_R):
        """
        Function C
        
        """
        num = Phi_R * S
        den = S * Phi_R + np.pi

        return num / den
    

    def rp_matrix_L(self, xi, S, Phi_R):
        """
        Construct the matrix from the igoshin linearised model

        """
        n = self.par.n
        n_rp = int(Phi_R / self.par.dp - 1)

        l_plus = np.zeros((n, n), dtype='cfloat')
        l_a_plus = np.zeros((n, n), dtype='cfloat')

        # Diag L+
        l_plus[self.par.index_diag[:n_rp], self.par.index_diag[:n_rp]] = -1j*xi - 1 / self.par.dp
        l_plus[self.par.index_diag[n_rp:], self.par.index_diag[n_rp:]] = -1j*xi - (1 + S) / self.par.dp

        # Subdiag L+
        l_plus[self.par.index_sub_diag+1, self.par.index_sub_diag] = 1 / self.par.dp
        l_plus[(self.par.index_sub_diag+1)[n_rp:], self.par.index_sub_diag[n_rp:]] *= (1 + S)

        # Diag L-
        l_minus = l_plus.copy()
        l_minus[self.par.index_diag, self.par.index_diag] += 2j*xi

        # L_A+
        l_a_plus[0, -1] = (1 + S) / self.par.dp
        l_a_plus[n_rp, :] = self.C_linear_rp(S, Phi_R) * 1j * xi

        # L_A-
        l_a_minus = l_a_plus.copy()
        l_a_minus[n_rp, :] *= -1

        
        l = np.block([[l_plus, l_a_plus], [l_a_minus, l_minus]])

        return l, l_plus, l_minus, l_a_plus, l_a_minus
    

    def rp_matrix_B(self, S, Phi_R):
        """
        Construct the matrix from the igoshin linearised model

        """
        n = self.par.n
        n_rp = int(Phi_R / self.par.dp - 1)

        b_id = np.zeros((n, n), dtype='cfloat')
        b_a = np.zeros((n, n), dtype='cfloat')

        # Diag identity
        b_id[self.par.index_diag[:], self.par.index_diag[:]] = 1

        # b_a
        b_a[n_rp, :] = -self.C_linear_rp(S, Phi_R)

        b = np.block([[b_id, b_a], [b_a, b_id]])

        return b, b_id, b_a