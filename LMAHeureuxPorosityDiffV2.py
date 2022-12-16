import numpy as np
from pde import FieldCollection, PDEBase, ScalarField, FieldBase
from numba import njit, prange
np.seterr(divide="raise", over="raise", under="warn", invalid="raise")
from scipy.sparse import csr_matrix, find   
from Compute_jacobian import Jacobian
from pde.grids.operators.cartesian import _make_derivative

class LMAHeureuxPorosityDiff(PDEBase):
    """SIR-model with diffusive mobility"""

    def __init__(self, Depths, slices_for_all_fields, CA0, CC0, cCa0, cCO30, Phi0, 
                sedimentationrate, Xstar, Tstar, k1, k2, k3, k4, m1, m2, n1, n2, 
                b, beta, rhos, rhow, rhos0, KA, KC, muA, D0Ca, PhiNR, PhiInfty, 
                DCa, DCO3, not_too_shallow, not_too_deep):  
        self.no_fields = 5
        self.Depths = Depths    
        self.delta_x = self.Depths._axes_coords[0][1] - \
                       self.Depths._axes_coords[0][0]
        self.slices_for_all_fields = slices_for_all_fields
        self.bc_CA = [{"value": CA0}, {"curvature" : 0}]
        self.bc_CC = [{"value": CC0}, {"curvature": 0}]
        self.bc_cCa = [{"value": cCa0}, {"derivative": 0}]
        self.bc_cCO3 = [{"value": cCO30}, {"derivative": 0}]
        self.bc_Phi = [{"value": Phi0}, {"derivative": 0}]   
        self.sedimentationrate = sedimentationrate
        self.Xstar = Xstar
        self.Tstar = Tstar
        self.k1 = k1
        self.k2 = k2
        self.nu1 = k1/k2
        self.k3 = k3
        self.k4 = k4
        self.nu2 = k4/k3
        self.m1 = m1
        self.m2 = m2
        self.n1 = n1
        self.n2 = n2
        self.b = b
        self.beta = beta
        self.rhos = rhos
        self.rhow = rhow
        self.rhos0 = rhos0
        self.KA = KA
        self.KC = KC
        self.KRat = self.KC/self.KA
        self.muA = muA
        self.D0Ca = D0Ca
        self.PhiNR = PhiNR
        self.PhiInfty = PhiInfty 
        self.Phi0 = Phi0
        self.DCa = DCa
        self.DCO3 = DCO3
        self.g = 100 * 9.81
        self.dCa = self.DCa / self.D0Ca
        self.dCO3 = self.DCO3 / self.D0Ca
        self.delta = self.rhos / (self.muA * np.sqrt(self.KC))
        self.Da = self.k2 * self.Tstar
        self.lambda_ = self.k3 / self.k2
        self.auxcon = self.beta / (self.D0Ca * self.b * self.g * self.rhow * \
                 (self.PhiNR - self.PhiInfty))
        self.rhorat0 = (self.rhos0 / self.rhow - 1) * self.beta / \
                  self.sedimentationrate
        self.rhorat = (self.rhos / self.rhow - 1) * self.beta / \
                 self.sedimentationrate
        self.presum = 1 - self.rhorat0 * self.Phi0 ** 3 * \
                 (1 - np.exp(10 - 10 / self.Phi0)) / (1 - self.Phi0)  
        self.not_too_shallow = not_too_shallow.data
        self.not_too_deep = not_too_deep.data      
        self.CA_sl= self.slices_for_all_fields[0]
        self.CC_sl = self.slices_for_all_fields[1]
        self.cCa_sl = self.slices_for_all_fields[2]
        self.cCO3_sl = self.slices_for_all_fields[3]
        self.Phi_sl = self.slices_for_all_fields[4]
        
        self.backward_diff = _make_derivative(Depths, method="backward")
        self.forward_diff = _make_derivative(Depths, method="forward")

        # Fiadeiro-Veronis integration involves a coth and a reciprocal, which can
        # easily lead to FloatingPointError: overflow encountered in double_scalars.
        # To avoid this, better revert to either backwards or central differencing 
        # the Peclet number is very large or very small.
        self.Peclet_min = 1e-2
        self.Peclet_max = 1/self.Peclet_min

        # Make sure integration stops when we field values become less than zero
        # or more than one, in some cases. Or just register that this is happening
        # and continue integration, which corresponds to "False".
        setattr(self.zeros.__func__, "terminal", False)
        setattr(self.zeros_CA.__func__, "terminal", False)
        setattr(self.zeros_CC.__func__, "terminal", False)
        setattr(self.ones_CA_plus_CC.__func__, "terminal", False)
        setattr(self.ones_Phi.__func__, "terminal", False)
        setattr(self.zeros_U.__func__, "terminal", False)
        setattr(self.zeros_W.__func__, "terminal", False)

    def get_state(self, AragoniteSurface, CalciteSurface, CaSurface, CO3Surface, 
                  PorSurface):
        """generate a suitable initial state"""
        AragoniteSurface.label = "ARA"
        CalciteSurface.label = "CAL"
        CaSurface.label = "Ca"
        CO3Surface.label = "CO3"
        PorSurface.label = "Po"

        return FieldCollection([AragoniteSurface, CalciteSurface, CaSurface, 
                                CO3Surface, PorSurface])

    def evolution_rate(self, state: FieldBase, t: float = 0) -> FieldBase:
        return super().evolution_rate(state, t)                            

    def fun(self, t, y, pbar, state):
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress reports.
        However, for this rhs calculation, they are redundant. """    
        CA = ScalarField(self.Depths, y[self.slices_for_all_fields[0]])
        CC = ScalarField(self.Depths, y[self.slices_for_all_fields[1]])
        cCa = ScalarField(self.Depths, y[self.slices_for_all_fields[2]])
        cCO3 = ScalarField(self.Depths, y[self.slices_for_all_fields[3]])
        Phi = ScalarField(self.Depths, y[self.slices_for_all_fields[4]])

        # dPhislash = (self.auxcon * (Phi / ((1 - Phi) ** 2)) * (np.exp(10 - 10 / Phi) * 
        #            (2 * Phi ** 2 + 7 * Phi - 10) + Phi * (3 - 2 * Phi)))    

        two_factors = cCa * cCO3
        two_factors_upp_lim = two_factors.to_scalar(lambda f: np.fmin(f,1))
        two_factors_low_lim = two_factors.to_scalar(lambda f: np.fmax(f,1))

        three_factors = two_factors * self.KRat
        three_factors_upp_lim = three_factors.to_scalar(lambda f: np.fmin(f,1))
        three_factors_low_lim = three_factors.to_scalar(lambda f: np.fmax(f,1))

        coA = CA * (((1 - three_factors_upp_lim) ** self.m2) * \
                    (self.not_too_deep * self.not_too_shallow) - self.nu1 * \
                    (three_factors_low_lim - 1) ** self.m1)
 
        # coA = CA * (((np.amax(0,1 - cCa * cCO3 * self.KRat) ** self.m2) * \
        #      (Depths.data * Xstar <= self.DeepLimit and \
        #      Depths.data * Xstar >= self.ShallowLimit)) - self.nu1 * \
        #      (np.amax(0,cCa * cCO3 * self.KRat - 1) ** self.m1))

        coC = CC * (((two_factors_low_lim - 1) ** self.n1) - self.nu2 * \
                    (1 - two_factors_upp_lim) ** self.n2)

        # coC = CC * ((np.amax(0,cCa * cCO3 - 1) ** self.n1) - self.nu2 * \
        #      (np.amax(0,1 - cCa * cCO3) ** self.n2))

        F = 1 - np.exp(10 - 10 / Phi)

        U = self.presum + self.rhorat * Phi ** 3 * F / (1 - Phi)
        
        W = self.presum - self.rhorat * Phi ** 2 * F
        
        # Wslash = - self.rhorat * 2 * (Phi - (Phi + 5) * np.exp(10 - 10 / Phi))

        dCA_dt = - U * CA.gradient(self.bc_CA)[0] - self.Da * ((1 - CA) * coA + self.lambda_ * CA * coC)

        dCC_dt = - U * CC.gradient(self.bc_CC)[0] + self.Da * (self.lambda_ * (1 - CC) * coC + CC * coA)

        # Implementing equation 6 from l'Heureux.
        denominator = 1 - 2 * ScalarField(self.Depths, np.log(Phi.data))
        Phi_denom = Phi/denominator
        dPhi_dx = Phi.gradient(self.bc_Phi)[0]
        grad_Phi_denom = dPhi_dx * (denominator + 2) / denominator ** 2

        common_helper = coA - self.lambda_ * coC

        dcCa_dx = cCa.gradient(self.bc_cCa)[0]
        # dcCa_dt = ((Phi * self.dCa * dcCa_dx/denominator).gradient(self.bc_cCa))/Phi -W * dcCa_dx \
        #          + self.Da * (1 - Phi) * (self.delta - cCa) * (coA - self.lambda_ * coC)/Phi
       
        dcCa_dt = (dcCa_dx * grad_Phi_denom + Phi_denom * cCa.laplace(self.bc_cCa)) \
                  * self.dCa /Phi -W * dcCa_dx \
                  + self.Da * (1 - Phi) * (self.delta - cCa) * common_helper / Phi

        dcCO3_dx = cCO3.gradient(self.bc_cCO3)[0]
        # dcCO3_dt = (Phi * self.dCO3 * dcCO3_dx/denominator).gradient(self.bc_cCO3)/Phi \
        #           -W * dcCO3_dx + self.Da * (1 - Phi) * (self.delta - cCO3) * \
        #           (coA - self.lambda_ * coC)/Phi

        dcCO3_dt = (dcCO3_dx * grad_Phi_denom + Phi_denom * cCO3.laplace(self.bc_cCO3)) \
                   * self.dCO3/Phi -W * dcCO3_dx \
                   + self.Da * (1 - Phi) * (self.delta - cCO3) * common_helper / Phi


        dPhi = self.auxcon * F * (Phi ** 3) / (1 - Phi)

        dW_dx = -self.rhorat * dPhi_dx * (2 * Phi * F + 10 * (F - 1))

        # dPhi_dt = ((self.auxcon * ((Phi ** 3) / (1 - Phi)) * \
        #           (1 - np.exp(10 - 10 / Phi))) * dPhi_dx).gradient(self.bc_Phi) \
        #          + self.Da * (1 - Phi) * (coA - self.lambda_ * coC) - dPhi_dx * \
        #            (W + Wslash * Phi + dPhi_dx * dPhislash)


        # This is closer to the original form of (43) from l' Heureux than
        # the Matlab implementation.
        dPhi_dt = - (Phi * dW_dx + W * dPhi_dx) \
                  + dPhi * Phi.laplace(self.bc_Phi) \
                  + self.Da * (1 - Phi) * common_helper

        return FieldCollection([dCA_dt, dCC_dt, dcCa_dt, dcCO3_dt, dPhi_dt]).data.ravel()

    def fun_numba(self, t, y, pbar, state):
        """ For tqdm to monitor progress. """
        """ From 
        https://stackoverflow.com/questions/59047892/how-to-monitor-the-process-of-scipy-odeint """
        last_t, dt = state
        n = int((t - last_t)/dt)
        pbar.update(n)
        # this we need to take into account that n is a rounded number.
        state[0] = last_t + dt * n

        """ the numba-accelerated evolution equation """     
        CA = ScalarField(self.Depths, y[self.slices_for_all_fields[0]])
        CC = ScalarField(self.Depths, y[self.slices_for_all_fields[1]])
        cCa = ScalarField(self.Depths, y[self.slices_for_all_fields[2]])
        cCO3 = ScalarField(self.Depths, y[self.slices_for_all_fields[3]])
        Phi = ScalarField(self.Depths, y[self.slices_for_all_fields[4]])

        # Instead of the default central differenced gradient from py-pde
        # construct forward and backward differenced gradients and apply
        # either one of them, based on the sign of U.
        CA_grad_back = CA.copy()
        CA_grad_forw = CA.copy()
        CA_grad_back.data[:] = 0
        CA_grad_forw.data[:] = 0
        CA.set_ghost_cells(self.bc_CA)
        self.backward_diff(CA._data_full, out = CA_grad_back.data)
        self.forward_diff(CA._data_full, out = CA_grad_forw.data)

        # Instead of the default central differenced gradient from py-pde
        # construct forward and backward differenced gradients and apply
        # either one of them, based on the sign of U.
        CC_grad_back = CC.copy()
        CC_grad_forw = CC.copy()
        CC_grad_back.data[:] = 0
        CC_grad_forw.data[:] = 0
        CC.set_ghost_cells(self.bc_CC)
        self.backward_diff(CC._data_full, out = CC_grad_back.data)
        self.forward_diff(CC._data_full, out = CC_grad_forw.data)

        # Instead of the default central differenced gradient from py-pde
        # construct forward and backward differenced gradients and give them
        # appropriate weights according to a Fiadeiro-Veronis scheme.
        cCa_grad_back = cCa.copy()
        cCa_grad_forw = cCa.copy()
        cCa_grad_back.data[:] = 0
        cCa_grad_forw.data[:] = 0
        cCa.set_ghost_cells(self.bc_cCa)
        self.backward_diff(cCa._data_full, out = cCa_grad_back.data)
        self.forward_diff(cCa._data_full, out = cCa_grad_forw.data)
        cCa_laplace = cCa.laplace(self.bc_cCa)

        # Instead of the default central differenced gradient from py-pde
        # construct forward and backward differenced gradients and give them
        # appropriate weights according to a Fiadeiro-Veronis scheme.
        cCO3_grad_back = cCO3.copy()
        cCO3_grad_forw = cCO3.copy()
        cCO3_grad_back.data[:] = 0
        cCO3_grad_forw.data[:] = 0
        cCO3.set_ghost_cells(self.bc_cCO3)
        self.backward_diff(cCO3._data_full, out = cCO3_grad_back.data)
        self.forward_diff(cCO3._data_full, out = cCO3_grad_forw.data)
        cCO3_laplace = cCO3.laplace(self.bc_cCO3)

        # Instead of the default central differenced gradient from py-pde
        # construct forward and backward differenced gradients and give them
        # appropriate weights according to a Fiadeiro-Veronis scheme.
        Phi_grad_back = Phi.copy()
        Phi_grad_forw = Phi.copy()
        Phi_grad_back.data[:] = 0
        Phi_grad_forw.data[:] = 0
        Phi.set_ghost_cells(self.bc_Phi)
        self.backward_diff(Phi._data_full, out = Phi_grad_back.data)
        self.forward_diff(Phi._data_full, out = Phi_grad_forw.data)
        Phi_laplace = Phi.laplace(self.bc_Phi)

        rhs = LMAHeureuxPorosityDiff.pde_rhs(CA.data, CC.data, cCa.data, \
            cCO3.data, Phi.data, self.KRat, \
            self.m1, self.m2, self.n1, self.n2, self.nu1, self.nu2, \
            self.not_too_deep, self.not_too_shallow, self.presum, self.rhorat, \
            self.lambda_, self.Da, self.dCa, self.dCO3, self.delta, self.auxcon, \
            CA_grad_back.data, CA_grad_forw.data, CC_grad_back.data, \
            CC_grad_forw.data, cCa_grad_back.data, cCa_grad_forw.data, \
            cCa_laplace.data, cCO3_grad_back.data, cCO3_grad_forw.data,\
            cCO3_laplace.data, Phi_grad_back.data, Phi_grad_forw.data, \
            Phi_laplace.data, self.delta_x, self.Peclet_min, self.Peclet_max, \
            no_depths = self.Depths.shape[0])

        return rhs

    @njit(nogil = True, parallel = True, fastmath = True)
    def pde_rhs(CA, CC, cCa, \
            cCO3, Phi, KRat, \
            m1, m2, n1, n2, nu1, nu2, \
            not_too_deep, not_too_shallow, presum, rhorat, \
            lambda_, Da, dCa, dCO3, delta, auxcon, \
            CA_grad_back, CA_grad_forw, CC_grad_back, \
            CC_grad_forw, cCa_grad_back, cCa_grad_forw, \
            cCa_laplace, cCO3_grad_back, cCO3_grad_forw,\
            cCO3_laplace, Phi_grad_back, Phi_grad_forw, \
            Phi_laplace, delta_x, Peclet_min, Peclet_max, no_depths):
        """ compiled helper function evaluating right hand side """

        denominator = np.empty(no_depths)
        common_helper1 = np.empty(no_depths)
        common_helper2 = np.empty(no_depths)
        helper_cCa_grad = np.empty(no_depths)
        helper_cCO3_grad = np.empty(no_depths)
        F = np.empty(no_depths)
        U = np.empty(no_depths)
        W = np.empty(no_depths)
        two_factors = np.empty(no_depths)
        two_factors_upp_lim = np.empty(no_depths)
        two_factors_low_lim = np.empty(no_depths)
        three_factors = np.empty(no_depths)
        three_factors_upp_lim = np.empty(no_depths)
        three_factors_low_lim = np.empty(no_depths)
        coA = np.empty(no_depths)
        coC = np.empty(no_depths)
        common_helper3 = np.empty(no_depths)
        rate = np.empty(5 * no_depths)
        dPhi = np.empty(no_depths)
        dW_dx = np.empty(no_depths)
        one_minus_Phi = np.empty(no_depths)
        CA_grad = np.empty(no_depths)
        CC_grad = np.empty(no_depths)
        cCa_grad = np.empty(no_depths)
        cCO3_grad = np.empty(no_depths)
        Phi_grad = np.empty(no_depths)

        for i in prange(no_depths):
            F[i] = 1 - np.exp(10 - 10 / Phi[i])

            U[i] = presum + rhorat * Phi[i] ** 3 * F[i]/ (1 - Phi[i])

            if U[i] > 0:
                CA_grad[i] = CA_grad_back[i]
                CC_grad[i] = CC_grad_back[i]
            else:
                CA_grad[i] = CA_grad_forw[i]
                CC_grad[i] = CC_grad_forw[i]

            W[i] = presum - rhorat * Phi[i] ** 2 * F[i]

            # Implementing equation 6 from l'Heureux.
            denominator[i] = 1 - 2 * np.log(Phi[i])

            # Fiadeiro-Veronis scheme for equations 42 and 43
            # from l'Heureux. 
            Peclet_cCa = W[i] * delta_x * denominator[i]/ (2. * dCa )
            if np.abs(Peclet_cCa) < Peclet_min:
                sigma_cCa = 0
            elif np.abs(Peclet_cCa) > Peclet_max:
                sigma_cCa = np.sign(W[i])
            else:
                 sigma_cCa = np.cosh(Peclet_cCa)/np.sinh(Peclet_cCa) - 1/Peclet_cCa

            Peclet_cCO3 = W[i] * delta_x * denominator[i]/ (2. * dCO3)
            if np.abs(Peclet_cCO3) < Peclet_min:
                sigma_cCO3 = 0
            elif np.abs(Peclet_cCO3) > Peclet_max:
                sigma_cCO3 = np.sign(W[i])
            else:
                sigma_cCO3 = np.cosh(Peclet_cCO3)/np.sinh(Peclet_cCO3) - 1/Peclet_cCO3

            one_minus_Phi[i] = 1 - Phi[i]                 
            dPhi[i] = auxcon * F[i] * (Phi[i] ** 3) / one_minus_Phi[i]
            Peclet_Phi = W[i] * delta_x * denominator[i] / (2. * dPhi[i])
            if np.abs(Peclet_Phi) < Peclet_min:
                sigma_Phi = 0
            elif np.abs(Peclet_Phi) > Peclet_max:
                sigma_Phi = np.sign(W[i])
            else:
                sigma_Phi = np.cosh(Peclet_Phi)/np.sinh(Peclet_Phi) - 1/Peclet_Phi

            cCa_grad[i] = 0.5 * ((1-sigma_cCa) * cCa_grad_forw[i] + \
                          (1+sigma_cCa) * cCa_grad_back[i])
            cCO3_grad[i] = 0.5 * ((1-sigma_cCO3) * cCO3_grad_forw[i] + \
                          (1+sigma_cCO3) * cCO3_grad_back[i])
            Phi_grad[i] = 0.5 * ((1-sigma_Phi) * Phi_grad_forw [i]+ \
                          (1+sigma_Phi) * Phi_grad_back[i])

            common_helper1[i] = Phi[i]/denominator[i]
            common_helper2[i] = Phi_grad[i] * (2 + denominator[i]) \
                                / denominator[i] ** 2
            helper_cCa_grad[i] = dCa * (common_helper2[i] * cCa_grad[i] \
                                 + common_helper1[i] * cCa_laplace[i])
            helper_cCO3_grad[i] = dCO3 * (common_helper2[i] * cCO3_grad[i] \
                                 + common_helper1[i] * cCO3_laplace[i])            

            two_factors[i] = cCa[i] * cCO3[i]
            two_factors_upp_lim[i] = min(two_factors[i],1)
            two_factors_low_lim[i] = max(two_factors[i],1)
            three_factors[i] = two_factors[i] * KRat
            three_factors_upp_lim[i] = min(three_factors[i],1)
            three_factors_low_lim[i] = max(three_factors[i],1)

            coA[i] = CA[i] * (((1 - three_factors_upp_lim[i]) ** m2) * \
                (not_too_deep[i] * not_too_shallow[i]) - nu1 * \
                (three_factors_low_lim[i] - 1) ** m1)

            coC[i] = CC[i] * (((two_factors_low_lim[i] - 1) ** n1) - nu2 * \
                (1 - two_factors_upp_lim[i]) ** n2)         

            common_helper3[i] = coA[i] - lambda_* coC[i]
               
            dW_dx[i] = -rhorat * Phi_grad[i] * (2 * Phi[i] * F[i] + 10 * (F[i] - 1))    
       
            # This is dCA_dt
            rate[i] = - U[i] * CA_grad[i] - Da * ((1 - CA[i]) \
                                    * coA[i] + lambda_ * CA[i] * coC[i])

            # This is dCC_dt
            rate[no_depths + i] = - U[i] * CC_grad[i] + Da * (lambda_ * \
                                    (1 - CC[i]) * coC[i] + CC[i] * coA[i])              

            # This is dcCa_dt
            rate[2 * no_depths + i] =  helper_cCa_grad[i]/Phi[i] - W[i] * \
                                        cCa_grad[i] + Da * one_minus_Phi[i] * \
                                        (delta - cCa[i]) * common_helper3[i] \
                                        /Phi[i]                                 

            # This is dcCO3_dt
            rate[3 * no_depths + i] =  helper_cCO3_grad[i]/Phi[i] - W[i] * \
                                        cCO3_grad[i] + Da * one_minus_Phi[i] * \
                                        (delta - cCO3[i]) * common_helper3[i] \
                                        /Phi[i]                       

            # This is dPhi_dt
            rate[4 * no_depths + i] = - (dW_dx[i] * Phi[i] + W[i] * Phi_grad[i]) \
                                      + dPhi[i] * Phi_laplace[i] + Da * one_minus_Phi[i] \
                                      * common_helper3[i] 

        return rate

    def jacobian_sparsity(self):
        no_depths = self.Depths.shape[0]
        n = self.no_fields * no_depths
        jacob_csr = csr_matrix((n, n))
        data = np.ones(no_depths)
        row = np.arange(no_depths)
        col = np.arange(no_depths)
        for i in range(self.no_fields):
            for j in range(self.no_fields):
                if i < self.no_fields - 1 or j > 1:
                    jacob_csr += csr_matrix((data, (i * no_depths + row, \
                        j * no_depths + col)), shape = (n, n))

        return jacob_csr

    def zeros(self, t, y, pbar, state):
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        return np.amin(y)

    def zeros_CA(self, t, y, pbar, state):
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        return np.amin(y[self.CA_sl])

    def zeros_CC(self, t, y, pbar, state):
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        return np.amin(y[self.CC_sl])

    def ones_CA_plus_CC(self, t, y, pbar, state): 
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        CA = y[self.CA_sl]
        CC = y[self.CC_sl]
        return np.amax(CA + CC) - 1

    def ones_Phi(self, t, y, pbar, state): 
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        Phi = y[self.Phi_sl]   
        return np.amax(Phi) - 1

    def zeros_U(self, t, y, pbar, state): 
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        Phi = y[self.Phi_sl]   
        F = 1 - np.exp(10 - 10 / Phi)
        U = self.presum + self.rhorat * Phi ** 3 * F / (1 - Phi)
        # Assume that U is positive at every depth at the beginning
        # of the integration, so U will become zero first where it
        # is smallest.
        return np.amin(U)

    def zeros_W(self, t, y, pbar, state): 
        """ solve_ivp demands that I add these two extra aguments, i.e.
        pbar and state, as in jac, where I need them for 
        tqdm progress display.
        However, for this rhs calculation, they are redundant. """

        Phi = y[self.Phi_sl]   
        F = 1 - np.exp(10 - 10 / Phi)
        W = self.presum - self.rhorat * Phi ** 2 * F
        # Assume that W is negative at every depth at the beginning
        # of the integration, so U will become zero first where it is
        # largest or least negative.
        return np.amax(W)


    def jac(self, t, y, pbar, state):
        """ For tqdm to monitor progress. """
        """ From 
        https://stackoverflow.com/questions/59047892/how-to-monitor-the-process-of-scipy-odeint """
        last_t, dt = state
        n = int((t - last_t)/dt)
        pbar.update(n)
        # this we need to take into account that n is a rounded number.
        state[0] = last_t + dt * n
        
        CA = y[self.CA_sl]
        CC = y[self.CC_sl]
        cCa = y[self.cCa_sl]
        cCO3 = y[self.cCO3_sl]
        Phi = y[self.Phi_sl]  

        jacob_csr = csr_matrix(Jacobian(CA, CC, cCa, cCO3, Phi, \
            self.KRat, self.m1, self.m2, \
            self.n1, self.n2, self.nu1, self.nu2, self.not_too_deep, \
            self.not_too_shallow, self.lambda_, \
            self.Da, self.delta, \
            no_depths = self.Depths.shape[0], no_fields = self.no_fields))

        # jacob_csr = Jacobian(CA, CC, cCa, cCO3, Phi, \
        #     self.KRat, self.m1, self.m2, \
        #     self.n1, self.n2, self.nu1, self.nu2, self.not_too_deep, \
        #     self.not_too_shallow, self.lambda_, \
        #     self.Da, self.delta, \
        #     no_depths = self.Depths.shape[0], no_fields = self.no_fields)
        # Check that we should have 5**2 sets of Jacobian values for 5 fields,
        # with each set comprised of no_depth numbers.
        """     no_sets_of_jacobian_values = len(all_jac_values_rows_and_cols)
        assert no_sets_of_jacobian_values == self.no_fields ** 2
       
        no_depths = self.Depths.shape[0]
        n = self.no_fields * no_depths
        jacob_csr = csr_matrix((n, n))
 
        for item in all_jac_values_rows_and_cols: """
            # item[0] are the self.Depths.shape[0] Jacobian values.
            # item[1] is a tuple of self.Depths.shape[0] row indices and
            # self.Depths.shape[0] column indices.
            # This loop fills the sparse matrix.
        """     data, row_and_col = item
            jacob_csr += csr_matrix((data, row_and_col), shape = (n, n)) """

        # Check that the non-zero values in the Jacobian occur ath the same 
        # indices as the Jacobian sparsity matrix.
        # Perhaps we do not need this check for every computation of the
        # Jacobian.
        """ sparsity_matrix = self.jacobian_sparsity()
        nonzero_sparsity = find(sparsity_matrix)
        nonzero_jacobian = find(jacob_csr) """
        # print("Jacobian matrix evaluated")
        # assert np.allclose(nonzero_sparsity[0], nonzero_jacobian[0]) 
        # assert np.allclose(nonzero_sparsity[1], nonzero_jacobian[1])       

        return jacob_csr

