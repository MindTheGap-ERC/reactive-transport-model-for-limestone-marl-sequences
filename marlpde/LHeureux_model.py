import numpy as np
from pde import FieldCollection, PDEBase, ScalarField
from numba import njit

np.seterr(divide="raise", over="raise", under="raise", invalid="raise")

# Control what pdoc will document, i.e. output to .html.
__all__ = ['LMAHeureuxPorosityDiff']


class LMAHeureuxPorosityDiff(PDEBase):
    """Sets all the physical parameters that define the model."""

    def __init__(self, AragoniteSurface, CalciteSurface, CaSurface,
                CO3Surface, PorSurface, not_too_shallow, not_too_deep, CA0, CC0,
                cCa0, cCO30, Phi0, sedimentationrate, Xstar, Tstar, k1, k2, k3,
                k4, m1, m2, n1, n2, b, beta, rhos, rhow, rhos0, KA, KC, muA,
                D0Ca, PhiNR, PhiInfty, PhiIni, DCa, DCO3):
        """@private"""

        self.AragoniteSurface = AragoniteSurface
        """  (*pde.ScalarField*):   
        Initial distribution of aragonite composition along all depths
        (dimensionless)."""
        self.CalciteSurface = CalciteSurface
        """(*pde.ScalarField*):   
        Initial distribution of calcite composition along all depths
        (dimensionless)."""
        
        self.CaSurface = CaSurface
        """(*pde.Scalarfield*):   
        Initial distribution of calcium ions concentration along all depths
        (dimensionless)."""
        
        self.CO3Surface = CO3Surface
        """(*pde.ScalarField*):   
        Initial distribution of carbonate ions concentration along all depths
        (dimensionless)."""
        
        self.PorSurface = PorSurface
        """(*pde.ScalarField*):   
        Initial distribution of porosity along all depths (dimensionless)."""
        
        self.bc_CA = [{"value": CA0}, {"curvature": 0}]
        """Top and bottom  boundary condition for aragonite.  
        **CA0** (*float*):   
        Aragonite composition at the top of the system, i.e. at the 
        water-sediment interface (dimensionless)."""
        
        self.bc_CC = [{"value": CC0}, {"curvature": 0}]        
        """Top and bottom  boundary condition for calcite.  
        **CC0** (*float*):   
        Calcite composition at the top of the system, i.e. at the 
        water-sediment interface (dimensionless)."""

        self.bc_cCa = [{"value": cCa0}, {"derivative": 0}]
        """Top and bottom  boundary condition for the concentration of
        calcium ions.  
        **cCa0** (*float*):   
        Calcium ion concentration at the top of the system, i.e. at the 
        water-sediment interface (dimensionless)."""

        self.bc_cCO3 = [{"value": cCO30}, {"derivative": 0}]
        """Top and bottom  boundary condition for the concentration of
        carbonate ions.  
        **cCO30** (*float*):   
        Carbonate ion concentration at the top of the system, i.e. at the
        water-sediment interface (dimensionless)."""
        
        self.bc_Phi = [{"value": Phi0}, {"derivative": 0}]
        """Top and bottom  boundary condition for the porosity.  
        **Phi0** (*float*):   
        Porosity at the top of the system, i.e. at the water-sediment 
        interface (dimensionless)."""
        
        self.sedimentationrate = sedimentationrate
        """(*float*):   
        Sedimentation rate (cm/a)."""
        
        self.Xstar = Xstar
        """(*float*):   
        Scaling factor between physical depths and dimensionless depths as
        used in the differential equations, in cm."""
        
        self.Tstar = Tstar
        """(*float*):   
        Scaling factor between physical time and dimensionless time as
        used in the differential equations, in years."""
        
        self.k1 = k1
        """(*float*):   
        rate coefficient (1/a)."""
        
        self.k2 = k2
        """(*float*):   
        rate coefficient (1/a)."""
        
        self.nu1 = k1/k2
        """@private"""

        self.k3 = k3
        """(*float*):   
        rate coefficient (1/a)."""
        
        self.k4 = k4
        """(*float*):   
        rate coefficient (1/a)."""
        
        self.nu2 = k4/k3
        """@private"""
        
        self.m1 = m1
        """(*float*):       
        Exponent of the reaction term for aragonite precipation."""
        
        self.m2 = m2
        """(*float*):   
        Exponent of the reaction term for aragonite dissolution."""
        
        self.n1 = n1
        """(*float*):   
        Exponent of the reaction term for calcite precipation."""
        
        self.n2 = n2
        """(*float*):   
        Exponent of the reaction term for calcite dissolution."""
        
        self.b = b
        """(*float*):   
        Sediment compressibility (1/(kPa) in SI or 0.1/kBa in cgs)."""
        
        self.beta = beta
        """(*float*):   
        Constant in the hydraulic conductivity (cm/a)."""
        
        self.rhos = rhos
        """(*float*):   
        Total solid density (g/cm³)."""
        
        self.rhow = rhow
        """(*float*):   
        Density of water (g/cm³)."""
        
        self.rhos0 = rhos0
        """(*float*):   
        Initial value of the total solid density (g/cm³)."""
        
        self.KA = KA
        """(*float*):   
        Solubility of aragonite (M²)."""
        
        self.KC = KC
        """(*float*):   
        Solubility of calcite (M²)."""
        
        self.KRat = self.KC/self.KA
        """@private"""
        
        self.muA = muA
        """(*float*):   
        Molar mass of calcite and aragonite (g/mol)."""
        
        self.D0Ca = D0Ca
        """(*float*):   
        Molecular diffusion coefficient of calcium ions in water (cm²/a)."""
        
        self.PhiNR = PhiNR
        """(*float*):   
        Stationary unreactive system porosity (dimensionless)."""
        
        self.PhiInfty = PhiInfty 
        """(*float*):   
        Parameter in equation 23 from L'Heureux (dimensionless)."""
        
        self.Phi0 = Phi0
        """@private"""

        self.DCa = DCa
        """(*float*):   
        Molecular diffusion coefficient of calcium ions in the sediment 
        (cm²/a)."""
        
        self.DCO3 = DCO3
        """(*float*):   
        Molecular diffusion coefficient of carbonate ions in the sediment 
        (cm²/a)."""
        
        self.not_too_shallow = not_too_shallow
        """(*pde.Scalarfield*):   
        Heaviside step function that is zero above the upper edge of the 
        aragonite dissolution zone (ADZ) and 1 below that edge."""
        
        self.not_too_deep = not_too_deep
        """(*pde.Scalarfield*):   
        Heaviside step function that is zero below the lower edge of the ADZ
        and 1 above it."""
        

        self.g = 100 * 9.81
        """@private"""
        self.dCa = self.DCa / self.D0Ca
        """@private"""
        self.dCO3 = self.DCO3 / self.D0Ca
        """@private"""
        self.delta = self.rhos / (self.muA * np.sqrt(self.KC))
        """@private"""
        self.Da = self.k2 * self.Tstar
        """@private"""
        self.lambda_ = self.k3 / self.k2
        """@private"""
        self.auxcon = self.beta / (self.D0Ca * self.b * self.g * self.rhow * \
                 (self.PhiNR - self.PhiInfty))
        """@private"""
        self.rhorat0 = (self.rhos0 / self.rhow - 1) * self.beta / \
                  self.sedimentationrate
        """@private"""
        self.rhorat = (self.rhos / self.rhow - 1) * self.beta / \
                 self.sedimentationrate
        """@private"""
        self.presum = 1 - self.rhorat0 * self.Phi0 ** 3 * \
                 (1 - np.exp(10 - 10 / self.Phi0)) / (1 - self.Phi0)     
        """@private"""

        # Fiadeiro-Veronis differentiation involves a coth and a reciprocal, which can
        # easily lead to FloatingPointError: overflow encountered in double_scalars.
        # To avoid this, better revert to either backwards or central differencing 
        # the Peclet number is very large or very small.
        self.Peclet_min = 1e-2
        """@private"""
        self.Peclet_max = 1/self.Peclet_min
        """@private"""
        # Need this number for Fiadeiro-Veronis differentiation.
        self.delta_x = self.AragoniteSurface.grid._axes_coords[0][1] - \
                       self.AragoniteSurface.grid._axes_coords[0][0]
        """@private"""
        self.PhiIni = PhiIni
        """**PhiIni** (*float*):   
        Initial homogeneous porosity (dimensionless)"""
        self.F_fixed = 1 - np.exp(10 - 10 / self.PhiIni)
        """@private"""
        self.dPhi_fixed = self.auxcon * self.F_fixed *\
                          self.PhiIni ** 3 / (1 - self.PhiIni) 
        """@private"""
                          

    def get_state(self, AragoniteSurface, CalciteSurface, CaSurface, CO3Surface, 
                  PorSurface):
        """@private"""
        # Return initial state and register forward and backward difference
        # operators. 
        AragoniteSurface.label = "ARA"
        CalciteSurface.label = "CAL"
        CaSurface.label = "Ca"
        CO3Surface.label = "CO3"
        PorSurface.label = "Po"

        return FieldCollection([AragoniteSurface, CalciteSurface, CaSurface, 
                                CO3Surface, PorSurface])


    def track_U_at_bottom(self, state, t):
        """For recording U, the solid matrix advection field.
        Although U is computed as part of the integration, for recording it,
        we need to rederive it from the porosity at the bottom of the depth 
        grid.
          """
        # First, extract the porosity at the bottom of the system.
        # The derived quantities will then also be at the bottom of the system.
        Phi = state.data[4][-1]
        F = 1 - np.exp(10 - 10 / Phi)
        one_minus_Phi = 1 - Phi
        U_bottom = self.presum + self.rhorat * Phi ** 3 * F /one_minus_Phi
        return {"U at bottom": U_bottom}

    @staticmethod
    @njit
    def calculate_sigma(Peclet, W_data, Peclet_min, Peclet_max):
        """ Calculate sigma following formula 8.73 from Boudreau:
        'Diagenetic Models and their implementation'.

        Args:
        -  **Peclet** (*ndarray(dtype=float, ndim=1)*):
           Array along depth of the Peclet numbers.
        -  **W-data** (*ndarray(dtype=float, ndim=1)*):
           Array along depth of the velocity of the pore water 
           (counted positive downwards)
        -  **Peclet_min** (*float*):
           Lower limit for calculating 1/tanh(Peclet)
        -  **Peclet_max** (*float*):
           Upper limit for calculating 1/tanh(Peclet)

        Returns
        -  **sigma** (*ndarray(dtype=float, ndim=1)*):
           Array along depth with sigma values
        """
        sigma = np.empty(Peclet.size)
        for i in range(sigma.size):
            if np.abs(Peclet[i]) < Peclet_min:
                sigma[i] = 0
            elif np.abs(Peclet[i]) > Peclet_max:
                sigma[i] = np.sign(W_data[i])
            else:
                sigma[i] = np.cosh(Peclet[i])/np.sinh(Peclet[i]) - \
                    1/Peclet[i]
        return sigma

    def evolution_rate(self, state, t=0):
        CA, CC, cCa, cCO3, Phi = state   

        two_factors = cCa * cCO3
        two_factors_upp_lim = two_factors.to_scalar(lambda f: np.fmin(f,1))
        two_factors_low_lim = two_factors.to_scalar(lambda f: np.fmax(f,1))

        three_factors = two_factors * self.KRat
        three_factors_upp_lim = three_factors.to_scalar(lambda f: np.fmin(f,1))
        three_factors_low_lim = three_factors.to_scalar(lambda f: np.fmax(f,1))

        coA = CA * (((1 - three_factors_upp_lim) ** self.m2) * \
                    (self.not_too_deep * self.not_too_shallow) - self.nu1 * \
                    (three_factors_low_lim - 1) ** self.m1)
 
        coC = CC * (((two_factors_low_lim - 1) ** self.n1) - self.nu2 * \
                    (1 - two_factors_upp_lim) ** self.n2)

        F = 1 - np.exp(10 - 10 / Phi)
        one_minus_Phi = 1 - Phi
        U = self.presum + self.rhorat * Phi ** 3 * F /one_minus_Phi

        # Choose either forward or backward differencing for CA and CC
        # depending on the sign of U

        CA_grad_back = CA.apply_operator("grad_back", self.bc_CA)
        CA_grad_forw = CA.apply_operator("grad_forw", self.bc_CA)
        CA_grad = ScalarField(state.grid, np.where(U.data>0, CA_grad_back.data, \
            CA_grad_forw.data))

        CC_grad_back = CC.apply_operator("grad_back", self.bc_CC)
        CC_grad_forw = CC.apply_operator("grad_forw", self.bc_CC)
        CC_grad = ScalarField(state.grid, np.where(U.data>0, CC_grad_back.data, \
            CC_grad_forw.data))

        W = self.presum - self.rhorat * Phi ** 2 * F
        
        dCA_dt = - U * CA_grad - self.Da * ((1 - CA) \
                 * coA + self.lambda_ * CA * coC)

        dCC_dt = - U * CC_grad + self.Da * (self.lambda_ \
                 * (1 - CC) * coC + CC * coA)

        # Implementing equation 6 from l'Heureux.
        denominator = 1 - 2 * np.log(Phi)

        # Fiadeiro-Veronis scheme for equations 42 and 43
        # from l'Heureux. 
        common_Peclet  = W * self.delta_x / 2. 
        Peclet_cCa =  common_Peclet * denominator/ self.dCa       
        sigma_cCa_data = LMAHeureuxPorosityDiff.calculate_sigma(\
            Peclet_cCa.data, W.data, self.Peclet_min, self.Peclet_max)
        sigma_cCa = ScalarField(state.grid, sigma_cCa_data)

        Peclet_cCO3 = common_Peclet * denominator/ self.dCO3      
        sigma_cCO3_data = LMAHeureuxPorosityDiff.calculate_sigma(\
            Peclet_cCO3.data, W.data, self.Peclet_min, self.Peclet_max)
        sigma_cCO3 = ScalarField(state.grid, sigma_cCO3_data)

        # dPhi = self.auxcon * F * (Phi ** 3) / one_minus_Phi
        dPhi = self.dPhi_fixed

        Peclet_Phi = common_Peclet / dPhi
        sigma_Phi_data = LMAHeureuxPorosityDiff.calculate_sigma(\
            Peclet_Phi.data, W.data, self.Peclet_min, self.Peclet_max)
        sigma_Phi = ScalarField(state.grid, sigma_Phi_data)

        cCa_grad_back = cCa.apply_operator("grad_back", self.bc_cCa)
        cCa_grad_forw = cCa.apply_operator("grad_forw", self.bc_cCa)
        cCa_grad = 0.5 * ((1-sigma_cCa) * cCa_grad_forw +\
             (1+sigma_cCa) * cCa_grad_back)

        cCO3_grad_back = cCO3.apply_operator("grad_back", self.bc_cCO3)
        cCO3_grad_forw = cCO3.apply_operator("grad_forw", self.bc_cCO3)
        cCO3_grad = 0.5 * ((1-sigma_cCO3) * cCO3_grad_forw +\
             (1+sigma_cCO3) * cCO3_grad_back)

        Phi_grad_back = Phi.apply_operator("grad_back", self.bc_Phi)
        Phi_grad_forw = Phi.apply_operator("grad_forw", self.bc_Phi)
        Phi_grad = 0.5 * ((1-sigma_Phi) * Phi_grad_forw +\
             (1+sigma_Phi) * Phi_grad_back)

        Phi_denom = Phi/denominator
        grad_Phi_denom = Phi_grad * (denominator + 2) / denominator ** 2

        common_helper = coA - self.lambda_ * coC

        dcCa_dt = (cCa_grad * grad_Phi_denom + Phi_denom * cCa.laplace(self.bc_cCa)) \
                  * self.dCa /Phi -W * cCa_grad \
                  + self.Da * one_minus_Phi * (self.delta - cCa) * common_helper / Phi

        dcCO3_dt = (cCO3_grad * grad_Phi_denom + Phi_denom * cCO3.laplace(self.bc_cCO3)) \
                   * self.dCO3/Phi -W * cCO3_grad \
                   + self.Da * one_minus_Phi * (self.delta - cCO3) * common_helper / Phi

        dW_dx = -self.rhorat * Phi_grad * (2 * Phi * F + 10 * (F - 1))

        # This is closer to the original form of (43) from l' Heureux than
        # the Matlab implementation.
        dPhi_dt = - (Phi * dW_dx + W * Phi_grad) \
                  + dPhi * Phi.laplace(self.bc_Phi) \
                  + self.Da * one_minus_Phi * common_helper

        return FieldCollection([dCA_dt, dCC_dt, dcCa_dt, dcCO3_dt, dPhi_dt])

    def _make_pde_rhs_numba(self, state):
        """ the numba-accelerated evolution equation """
        # make attributes locally available
        KRat = self.KRat
        m1 = self.m1
        m2  = self.m2
        n1 = self.n1
        n2 = self.n2
        nu1 = self.nu1
        nu2 = self.nu2
        not_too_deep = self.not_too_deep.data
        not_too_shallow = self.not_too_shallow.data
        presum = self.presum
        rhorat= self.rhorat
        lambda_ = self.lambda_
        Da = self.Da
        dCa = self.dCa
        dCO3 = self.dCO3
        delta = self.delta
        # The following three numbers are also needed for Fiadeiro-Veronis.
        dPhi_fixed = self.dPhi_fixed
        Peclet_min = self.Peclet_min
        Peclet_max = self.Peclet_max
        delta_x = state.grid._axes_coords[0][1] - state.grid._axes_coords[0][0]
        grad_back_CA = state.grid.make_operator("grad_back", bc = self.bc_CA)
        grad_forw_CA = state.grid.make_operator("grad_forw", bc = self.bc_CA)
        grad_back_CC = state.grid.make_operator("grad_back", bc = self.bc_CC)
        grad_forw_CC = state.grid.make_operator("grad_forw", bc = self.bc_CC)
        grad_back_cCa = state.grid.make_operator("grad_back", bc = self.bc_cCa)
        grad_forw_cCa = state.grid.make_operator("grad_forw", bc = self.bc_cCa)
        laplace_cCa = state.grid.make_operator("laplace", bc = self.bc_cCa)
        grad_back_cCO3 = state.grid.make_operator("grad_back", bc = self.bc_cCO3)
        grad_forw_cCO3 = state.grid.make_operator("grad_forw", bc = self.bc_cCO3)
        laplace_cCO3 = state.grid.make_operator("laplace", bc = self.bc_cCO3)
        grad_back_Phi = state.grid.make_operator("grad_back", bc = self.bc_Phi)
        grad_forw_Phi = state.grid.make_operator("grad_forw", bc = self.bc_Phi)
        laplace_Phi = state.grid.make_operator("laplace", bc = self.bc_Phi)

        @njit
        def pde_rhs(state_data, t=0):
            """ compiled helper function evaluating right hand side """
            # Instead of the default central differenced gradient from py-pde
            # construct forward and backward differenced gradients and apply
            # either one of them, based on the sign of U.
            CA = state_data[0]
            CA_grad_back = grad_back_CA(CA)
            CA_grad_forw = grad_forw_CA(CA)

            CC = state_data[1]
            CC_grad_back = grad_back_CC(CC)
            CC_grad_forw = grad_forw_CC(CC)

            cCa = state_data[2]
            cCa_grad_back = grad_back_cCa(cCa)
            cCa_grad_forw = grad_forw_cCa(cCa)
            cCa_laplace = laplace_cCa(cCa)

            cCO3 = state_data[3]
            cCO3_grad_back = grad_back_cCO3(cCO3)
            cCO3_grad_forw = grad_forw_cCO3(cCO3)
            cCO3_laplace = laplace_cCO3(cCO3)

            Phi = state_data[4]
            Phi_grad_back = grad_back_Phi(Phi)
            Phi_grad_forw = grad_forw_Phi(Phi)
            Phi_laplace = laplace_Phi(Phi)

            rate = np.empty_like(state_data)

            # state_data.size should be the same as len(CA) or len(CC), check this.
            # So the number of depths, really.
            no_depths = state_data[0].size

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
            dPhi = np.empty(no_depths)
            dW_dx = np.empty(no_depths)
            one_minus_Phi = np.empty(no_depths)
            CA_grad = np.empty(no_depths)
            CC_grad = np.empty(no_depths)
            cCa_grad = np.empty(no_depths)
            cCO3_grad = np.empty(no_depths)
            Phi_grad = np.empty(no_depths)            

            for i in range(no_depths):
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
                     sigma_cCa = np.cosh(Peclet_cCa)/np.sinh(Peclet_cCa) - \
                        1/Peclet_cCa

                Peclet_cCO3 = W[i] * delta_x * denominator[i]/ (2. * dCO3)
                if np.abs(Peclet_cCO3) < Peclet_min:
                    sigma_cCO3 = 0
                elif np.abs(Peclet_cCO3) > Peclet_max:
                    sigma_cCO3 = np.sign(W[i])
                else:
                    sigma_cCO3 = np.cosh(Peclet_cCO3)/np.sinh(Peclet_cCO3) - \
                        1/Peclet_cCO3

                one_minus_Phi[i] = 1 - Phi[i]                 
                dPhi[i] = dPhi_fixed
                Peclet_Phi = W[i] * delta_x / (2. * dPhi[i])
                if np.abs(Peclet_Phi) < Peclet_min:
                    sigma_Phi = 0
                elif np.abs(Peclet_Phi) > Peclet_max:
                    sigma_Phi = np.sign(W[i])
                else:
                    sigma_Phi = np.cosh(Peclet_Phi)/np.sinh(Peclet_Phi) - \
                        1/Peclet_Phi

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
                rate[0][i] = - U[i] * CA_grad[i] - Da * ((1 - CA[i]) \
                             * coA[i] + lambda_ * CA[i] * coC[i])

                # This is dCC_dt
                rate[1][i] = - U[i] * CC_grad[i] + Da * (lambda_ * \
                             (1 - CC[i]) * coC[i] + CC[i] * coA[i])              

                # This is dcCa_dt
                rate[2][i] =  helper_cCa_grad[i]/Phi[i] - W[i] * \
                              cCa_grad[i] + Da * one_minus_Phi[i] * \
                              (delta - cCa[i]) * common_helper3[i] \
                              /Phi[i]                                 

                # This is dcCO3_dt
                rate[3][i] =  helper_cCO3_grad[i]/Phi[i] - W[i] * \
                              cCO3_grad[i] + Da * one_minus_Phi[i] * \
                              (delta - cCO3[i]) * common_helper3[i] \
                              /Phi[i]                       

                # This is dPhi_dt
                rate[4][i] = - (dW_dx[i] * Phi[i] + W[i] * Phi_grad[i]) \
                             + dPhi[i] * Phi_laplace[i] + Da * one_minus_Phi[i] \
                             * common_helper3[i] 

            return rate

        return pde_rhs 
