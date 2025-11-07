import numpy as np
from parameters import Parameters


class IgoshinMatrix:
    """
    Igoshin matrices
    
    """

    def __init__(self):
        
        self.par = Parameters()


    def w_1(self, rho_bar, q):
        """
        Function \omega_1
        
        """
        return self.par.w_n * rho_bar**q / (rho_bar**q + self.par.rho_w**q)
    

    # def K_1(self, rho_bar, q):
    #     """
    #     Function K_1
        
    #     """
    #     num = q * self.w_1(rho_bar, q) * self.par.w_0**2 * (1 - self.w_1(rho_bar, q) / self.par.w_n)
    #     den = (self.w_1(rho_bar, q) * self.par.delta_phi + np.pi * self.par.w_0) * (self.par.w_0 + self.w_1(rho_bar, q))

    #     return num / den
    

    def K_1(self, rho_bar, q):
        """
        Function K_1
        
        """
        num = self.w_1(rho_bar, q) * q * (1 - self.w_1(rho_bar, q) / self.par.w_n)
        den = (self.par.w_0 + self.w_1(rho_bar, q)) * (np.pi + self.par.delta_phi_r * self.w_1(rho_bar, q) / self.par.w_0)

        return num / den
    

    def main_matrix(self, xi, w_1, K_1):
        """
        Construct the matrix from the igoshin linearised model

        """
        g_plus = np.zeros((self.par.n, self.par.n), dtype='cfloat')
        g_a = np.zeros((self.par.n, self.par.n), dtype='cfloat')

        # Diag G+
        g_plus[self.par.index_diag[:self.par.n_rp], self.par.index_diag[:self.par.n_rp]] += -1j*self.par.v*xi - 1 / (self.par.d_phi * w_1 / self.par.w_0)
        g_plus[self.par.index_diag[self.par.n_rp:], self.par.index_diag[self.par.n_rp:]] += -1j*self.par.v*xi - 1 / (self.par.d_phi * w_1 / self.par.w_0) * (1 + w_1 / self.par.w_0)
        g_plus[0, :] += (1 + w_1 / self.par.w_0) * K_1

        # Subdiag G+
        g_plus[self.par.index_sub_diag+1, self.par.index_sub_diag] += 1 / (self.par.d_phi * w_1 / self.par.w_0)
        g_plus[(self.par.index_sub_diag+1)[self.par.n_rp:], self.par.index_sub_diag[self.par.n_rp:]] *= (1 + w_1 / self.par.w_0)

        # G-
        g_minus = g_plus.copy()
        g_minus[self.par.index_diag, self.par.index_diag] += 2j*self.par.v*xi

        # G_A
        g_a[0, -1] += (1 + w_1 / self.par.w_0) / (self.par.d_phi * w_1 / self.par.w_0)
        g_a[self.par.n_rp, :] += -(1 + w_1 / self.par.w_0) * K_1
        
        g = np.block([[g_plus, g_a], [g_a, g_minus]])

        return g, g_plus, g_minus, g_a
    
    
    def phi_r(self, rho_bar):
        """
        Modulated refractory period for igoshin
        
        """
        return self.par.delta_phi * self.par.rho_t / (rho_bar+1e-8)
    

    def K_2(self, phi_r):
        """
        Function K
        
        """
        num = phi_r * self.par.w_n**2 * self.par.w_0
        den = (self.par.w_n * phi_r + np.pi * self.par.w_0) * (self.par.w_0 + self.par.w_n)

        return num / den
    

    def rp_matrix(self, xi, phi_r, K_2):
        """
        Construct the matrix from the igoshin linearised model

        """
        n_rp_local = int(phi_r / self.par.d_phi)
        l_plus = np.zeros((self.par.n, self.par.n), dtype='cfloat')
        l_a = np.zeros((self.par.n, self.par.n), dtype='cfloat')
        l_minus = np.zeros((self.par.n, self.par.n), dtype='cfloat')
        index = np.arange(self.par.n - 1)

        # Diag
        diag = np.zeros(self.par.n, dtype='cfloat')
        diag[:] = -1j*self.par.v*xi - self.par.w_0 / self.par.d_phi
        diag[n_rp_local-1:] -= self.par.w_n / self.par.d_phi
        diag[n_rp_local-1] -= self.par.w_n / self.par.d_phi

        # g_plus
        np.fill_diagonal(l_plus, diag)
        l_plus[index+1, index] = self.par.w_0 / self.par.d_phi
        l_plus[(index+1)[n_rp_local-1:], index[n_rp_local-1:]] += self.par.w_n / self.par.d_phi # -1 because Python start index to 0

        # g_minus
        diag[:] += 2j*self.par.v*xi # update for g_minus
        np.fill_diagonal(l_minus, diag)
        l_minus[index+1, index] = self.par.w_0 / self.par.d_phi
        l_minus[(index+1)[n_rp_local-1:], index[n_rp_local-1:]] += self.par.w_n / self.par.d_phi # -1 because Python start index to 0

        # g_a
        l_a[0, -1] = self.par.w_0 / self.par.d_phi + self.par.w_n / self.par.d_phi
        l_a[n_rp_local-1, :] = K_2 / self.par.d_phi
        
        l = np.vstack((np.hstack((l_plus, l_a)), np.hstack((l_a, l_minus))))

        return l, l_plus, l_minus, l_a