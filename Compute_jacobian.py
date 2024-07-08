""" This returns the 25 elements (from 5 fields) of the Jacobian, 
computed numerically, using the functional form from Derive_jacobian.
There is also the depth dimension. 
In the current implementations of LMAHeureuxPorosityDiff.fun and
LMAHeureuxPorosityDiff.fun_numba, the five fields with each, say, 400 depths,
are treated as 2000 fields, so n = 2000 voor scipy.integrate.solve_ivp.
The solve_ivp documention says that the Jacobian will have shape (n, n),
so (2000, 2000), so calling jac(t, y) should return 4 million numbers!
But most elements will be zero, because derivatives of the five rhs 
expressions (equations 40-43 from l'Heureux) wrt to the five fields will only
be non-zero for equal depths.

Say you want to compute Jacobian element [1,4]. This is 
d f_1 / d y_4. That is the derivate of the right hand side of the second 
equation (equation 41 from l'Heureux) wrt to field number 5, which is Phi. 
We have a functional form for that, from Derive_Jacobian.py, so we can 
compute it for all 400 depths. Now where do these 400 numbers end up in
the (2000, 2000) output matrix? For depth 0, it would be element [400, 1600].
For depth 1, it will be element [401, 1601] and so forth until element
[799, 1999]. So a diagonal within a (400, 400) cell. Ultimately we will fill
25*400 =10000 elements of the 4 million numbers with non-zero values. Actually,
a bit less, because two of the 25 elements of the Jacobian are always zero. 
These correspond with the derivatives of the rhs of equations 40 and 41 wrt
Phi. So we have 23 * 400 = 9200 non-zero elements out of 2000 * 2000 = 4e6
elements. Truly a sparse matrix!"""

import numpy as np
from numba import njit, vectorize
from numba import float32 as f32
from numba import int32 as i32
from numba.extending import overload

@vectorize
def _heaviside(x1, x2):
    """ vectorized implementation of the heaviside function """
    if np.isnan(x1):
        return np.nan
    elif x1 == 0:
        return x2
    elif x1 < 0:
        return 0.0
    else:
        return 1.0


@overload(np.heaviside)
def np_heaviside(x1, x2):

    def heaviside_impl(x1, x2):
        """ numba implementation of the heaviside function """
        return _heaviside(x1, x2)

    return heaviside_impl

@njit(cache=True)
def Jacobian(CA, CC, cCa, cCO3, Phi, KRat, m1, m2, \
            n1, n2, nu1, nu2, not_too_deep, \
            not_too_shallow, lambda_, Da, delta, no_depths, no_fields):
    """ Retun a sparse (n, n) matrix
    The Jacobian for five field and five right-hand sides, so 25 elements
    of which two are zero, so we need to evaluate 23 expressions. """
    selected_depths = not_too_deep * not_too_shallow
    two_factors = cCO3 * cCa
    two_f_m_one = two_factors -1
    one_m_two_f = -two_f_m_one
    max_two_f_m_one = np.fmax(0, two_f_m_one)
    max_one_m_two_f = np.fmax(0, one_m_two_f)
    hs_two_f_m_one = np.heaviside(two_f_m_one, 0)
    hs_one_m_two_f = np.heaviside(one_m_two_f, 0)

    three_factors = KRat * two_factors
    thr_f_m_one = three_factors - 1
    one_m_thr_f = - thr_f_m_one
    max_thr_f_m_one = np.fmax(0, thr_f_m_one)
    max_one_m_thr_f = np.fmax(0, one_m_thr_f)
    hs_thr_f_m_one = np.heaviside(thr_f_m_one, 0)
    hs_one_m_thr_f = np.heaviside(one_m_thr_f, 0)

    """ @njit(nogil=True)
    def jac00():        
        return  -Da * (-CA * (selected_depths * max_one_m_thr_f ** m2 \
                - nu1 * max_thr_f_m_one**m1) + CC * lambda_ * \
                (-nu2 * max_one_m_two_f**n2 + max_two_f_m_one**n1) + \
                (1 - CA) * (selected_depths * max_one_m_thr_f ** m2 \
                - nu1 * max_thr_f_m_one**m1))       
 
    @njit(nogil=True)
    def jac01():
        return  -CA * Da * lambda_ * (-nu2 * max_one_m_two_f**n2 + \
                 max_two_f_m_one**n1)

    @njit(nogil=True)
    def jac02():
        return  -Da * (CA *CC * lambda_ * (cCO3 * n1 * hs_two_f_m_one * \
                max_two_f_m_one ** (n1 - 1 ) + cCO3 * n2 * nu2 * hs_one_m_two_f *\
                max_one_m_two_f ** (n2 - 1)) + CA*(1 - CA) * (-KRat * cCO3 * m1 * nu1 *\
                hs_thr_f_m_one * max_thr_f_m_one ** (m1 - 1) - \
                KRat * cCO3 * m2 * selected_depths * hs_one_m_thr_f * \
                max_one_m_thr_f ** (m2 -  1)))

    @njit(nogil=True)
    def jac03():
        return -Da * (CA * CC * lambda_ * (cCa * n1 * hs_two_f_m_one * \
                max_two_f_m_one ** (n1 - 1) + cCa * n2 * nu2 * hs_one_m_two_f * \
                max_one_m_two_f ** (n2 - 1)) + CA * (1 - CA) * (-KRat * cCa * m1 * nu1 * \
                hs_thr_f_m_one * max_thr_f_m_one ** (m1 - 1) \
                - KRat * cCa * m2 * selected_depths * hs_one_m_thr_f * \
                max_one_m_thr_f ** (m2 - 1)))

    @njit(nogil=True)
    def jac04():
        return np.zeros(no_depths)

    @njit(nogil=True)
    def jac10():
        return  CC * Da * (selected_depths * max_one_m_thr_f ** m2 \
               - nu1 * max_thr_f_m_one ** m1)

    @njit(nogil=True)
    def jac11():
        return  Da * (CA * (selected_depths * max_one_m_thr_f ** m2 - \
                nu1 * max_thr_f_m_one ** m1) - CC * lambda_ * \
                (-nu2 * max_one_m_two_f ** n2 + max_two_f_m_one ** n1) + \
                lambda_*(1 - CC)*(-nu2 * max_one_m_two_f**n2 + max_two_f_m_one**n1))

    @njit(nogil=True)
    def jac12():        
        return  Da * (CA * CC * (-KRat * cCO3 * m1 * nu1 * hs_thr_f_m_one * \
                max_thr_f_m_one ** (m1- 1) - KRat * cCO3 * m2 * not_too_deep * \
                not_too_shallow * hs_one_m_thr_f * \
                max_one_m_thr_f ** (m2 - 1)) + CC* lambda_ * \
                (1 - CC) * (cCO3 * n1 * hs_two_f_m_one * \
                max_two_f_m_one ** (n1 - 1) + cCO3 * n2 * nu2 * \
                hs_one_m_two_f * max_one_m_two_f ** (n2 - 1)))


    @njit(nogil=True)
    def jac13():
        return  Da * (CA * CC * (-KRat * cCa * m1 * nu1 * hs_thr_f_m_one * \
                max_thr_f_m_one ** (m1 - 1) - KRat * cCa * m2 * \
                selected_depths * hs_one_m_thr_f * \
                max_one_m_thr_f ** (m2 -1)) + CC * lambda_ * \
                (1 - CC) * (cCa * n1 * hs_two_f_m_one * \
                max_two_f_m_one ** (n1- 1) + cCa * n2 * nu2 * \
                hs_one_m_two_f * max_one_m_two_f ** (n2 - 1)))

    @njit(nogil=True)
    def jac14():
        return np.zeros(no_depths)

    @njit(nogil=True)
    def jac20():
        return Da * (1 - Phi) * (-cCa + delta) * (selected_depths * \
               max_one_m_thr_f ** m2 - nu1 * \
               max_thr_f_m_one ** m1)/Phi

    @njit(nogil=True)
    def jac21():
        return -Da * lambda_ * (1 - Phi) * (-cCa + delta) * \
               (-nu2 * max_one_m_two_f ** n2 + \
               max_two_f_m_one ** n1)/Phi

    @njit(nogil=True)
    def jac22():
        return Da * (1 - Phi) * (-cCa + delta) * (CA* (-KRat * cCO3 * m1 * nu1 * \
               hs_thr_f_m_one * \
               max_thr_f_m_one ** (m1 - 1) - KRat * cCO3 * m2 * \
               not_too_deep  *not_too_shallow * hs_one_m_thr_f * \
               max_one_m_thr_f ** (m2 - 1)) - CC * lambda_ * \
               (cCO3*n1*hs_two_f_m_one * \
               max_two_f_m_one ** (n1 - 1) + cCO3 * n2 * nu2 * \
               hs_one_m_two_f * max_one_m_two_f ** (n2 - 1)))/Phi - \
               Da * (1 - Phi) * (CA * (selected_depths * \
               max_one_m_thr_f ** m2 - nu1 * \
               max_thr_f_m_one ** m1) - CC * lambda_ * \
               (-nu2 * max_one_m_two_f ** n2 + max_two_f_m_one**n1))/Phi

    @njit(nogil=True)
    def jac23():
        return Da * (1 - Phi) * (-cCa + delta) * (CA * (-KRat * cCa * m1 * nu1 * \
               hs_thr_f_m_one * max_thr_f_m_one ** (m1 - 1) - \
               KRat * cCa * m2 * selected_depths * hs_one_m_thr_f * \
               max_one_m_thr_f ** (m2 - 1)) - CC * lambda_ * (cCa * n1 * \
               hs_two_f_m_one * max_two_f_m_one ** (n1 - 1) + \
               cCa * n2 * nu2 * hs_one_m_two_f * \
               max_one_m_two_f ** (n2 - 1)))/Phi

    @njit(nogil=True)
    def jac24():
        return -Da * (-cCa + delta) * (CA * (selected_depths * \
               max_one_m_thr_f ** m2 - nu1 * \
               max_thr_f_m_one ** m1) - CC * lambda_ * (-nu2 * \
               max_one_m_two_f ** n2 + \
               max_two_f_m_one ** n1))/Phi - Da*(1 - Phi)*(-cCa + delta) * \
               (CA*(selected_depths * \
               max_one_m_thr_f ** m2 - nu1 * 
               max_thr_f_m_one ** m1) - CC * lambda_ * \
               (-nu2*max_one_m_two_f ** n2 + max_two_f_m_one ** n1))/Phi**2

    @njit(nogil=True)
    def jac30():
        return Da * (1 - Phi) * (-cCO3 + delta) * (selected_depths * \
               max_one_m_thr_f ** m2 - nu1 * \
               max_thr_f_m_one ** m1)/Phi
    
    @njit(nogil=True)
    def jac31():
        return -Da * lambda_ * (1 - Phi) * (-cCO3 + delta) * (-nu2 * \
               max_one_m_two_f ** n2 + max_two_f_m_one**n1)/Phi

    @njit(nogil=True)
    def jac32():
        return Da * (1 - Phi) * (-cCO3 + delta) * (CA*(-KRat * cCO3 * m1 * nu1 * \
               hs_thr_f_m_one * max_thr_f_m_one ** (m1- 1) - \
               KRat * cCO3 * m2 * selected_depths * \
               hs_one_m_thr_f * max_one_m_thr_f ** (m2 - 1)) - \
               CC * lambda_ * (cCO3 * n1 * hs_two_f_m_one * \
               max_two_f_m_one ** (n1 - 1) + cCO3 * n2 * nu2 * \
               hs_one_m_two_f * max_one_m_two_f ** (n2 - 1)))/Phi

    @njit(nogil=True)
    def jac33():
        return Da * (1 - Phi) * (-cCO3 + delta) * (CA* (-KRat * cCa* m1 * nu1 * \
               hs_thr_f_m_one * max_thr_f_m_one ** (m1 - 1) - \
               KRat * cCa * m2 * selected_depths * \
               hs_one_m_thr_f * max_one_m_thr_f ** (m2 - 1)) \
               - CC * lambda_ * (cCa * n1 * hs_two_f_m_one * \
               max_two_f_m_one ** (n1 - 1) + cCa * n2 * nu2 * \
               hs_one_m_two_f * max_one_m_two_f ** (n2 - 1)))/Phi - \
               Da * (1 - Phi) * (CA * (selected_depths * \
               max_one_m_thr_f ** m2 - nu1 * \
               max_thr_f_m_one ** m1) - CC * lambda_ * (-nu2 * \
               max_one_m_two_f ** n2 + max_two_f_m_one ** n1))/Phi

    @njit(nogil=True)
    def jac34():
        return -Da * (-cCO3 + delta) * (CA * (selected_depths * \
               max_one_m_thr_f ** m2 - nu1 * \
               max_thr_f_m_one ** m1) - CC*lambda_*(-nu2 * \
               max_one_m_two_f ** n2 + \
               max_two_f_m_one ** n1))/Phi - Da * (1 - Phi) * \
               (-cCO3 + delta) *(CA * (selected_depths * \
               max_one_m_thr_f ** m2 - nu1 * \
               max_thr_f_m_one ** m1) - CC * lambda_ * \
               (-nu2 * max_one_m_two_f ** n2 + \
               max_two_f_m_one ** n1))/Phi**2

    @njit(nogil=True)
    def jac40():
        return Da * (1 - Phi) * (selected_depths * \
               max_one_m_thr_f ** m2 - nu1 * \
               max_thr_f_m_one ** m1)
    
    @njit(nogil=True)
    def jac41():
        return -Da * lambda_ * (1 - Phi) * (-nu2 * max_one_m_two_f ** n2 + \
               max_two_f_m_one ** n1)
    
    @njit(nogil=True)
    def jac42():
        return Da * (1 - Phi) * (CA * (-KRat * cCO3 * m1 * nu1 * \
               hs_thr_f_m_one * \
               max_thr_f_m_one ** (m1 - 1) - KRat * cCO3 * \
               m2 * selected_depths * \
               hs_one_m_thr_f * \
               max_one_m_thr_f ** (m2 - 1)) - CC * lambda_ * \
               (cCO3 * n1 * hs_two_f_m_one * \
               max_two_f_m_one ** (n1 - 1) + \
               cCO3 * n2 * nu2 * hs_one_m_two_f * \
               max_one_m_two_f ** (n2 - 1)))

    @njit(nogil=True)
    def jac43():
        return Da * (1 - Phi) *(CA * (-KRat * cCa * m1 * nu1 * \
               hs_thr_f_m_one * \
               max_thr_f_m_one ** (m1 - 1) - KRat * cCa * m2 * \
               selected_depths * 
               hs_one_m_thr_f * \
               max_one_m_thr_f ** (m2 - 1)) - CC *lambda_ * \
               (cCa * n1 * hs_two_f_m_one * \
               max_two_f_m_one ** (n1 - 1) + cCa * n2 * nu2 * \
               hs_one_m_two_f * max_one_m_two_f ** (n2 - 1)))

    @njit(nogil=True)
    def jac44():
        return -Da * (CA * (selected_depths * \
               max_one_m_thr_f ** m2 - nu1 * \
               max_thr_f_m_one ** m1) - CC * lambda_ * \
               (-nu2 * max_one_m_two_f ** n2 + \
               max_two_f_m_one ** n1)) """

    # def compute_all_Jacobian_elements():
    row_indices = np.arange(no_depths)
    col_indices = np.arange(no_depths)
    n = no_fields * no_depths 
    all_jac_values = np.zeros((n, n))
    for i in range(no_fields):
        row = i * no_depths + row_indices
        for j in range(no_fields):
            col = j * no_depths + col_indices
            # row_and_col = (row, col)
            if i == 0 and j == 0:
                # Need this extra loop because of Numba limitations.
                jac_values = -Da*(-CA*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1) + CC*lambda_*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1) + (1 - CA)*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1))
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]
            elif i == 0 and j == 1:
                # Need this extra loop because of Numba limitations.
                jac_values = -CA*Da*lambda_*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1)
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i == 0 and j == 2:    
                # Need this extra loop because of Numba limitations.
                jac_values = -Da*(CA*CC*lambda_*(cCO3*n1*hs_two_f_m_one*max_two_f_m_one**(n1-1) + cCO3*n2*nu2*hs_one_m_two_f*max_one_m_two_f**(n2-1)) + CA*(1 - CA)*(-KRat*cCO3*m1*nu1*hs_thr_f_m_one*max_thr_f_m_one**(m1-1) - KRat*cCO3*m2*selected_depths*hs_one_m_thr_f*max_one_m_thr_f**(m2-1)))
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i == 0 and j == 3:
                # Need this extra loop because of Numba limitations.
                jac_values = -Da*(CA*CC*lambda_*(cCa*n1*hs_two_f_m_one*max_two_f_m_one**(n1-1) + cCa*n2*nu2*hs_one_m_two_f*max_one_m_two_f**(n2-1)) + CA*(1 - CA)*(-KRat*cCa*m1*nu1*hs_thr_f_m_one*max_thr_f_m_one**(m1-1) - KRat*cCa*m2*selected_depths*hs_one_m_thr_f*max_one_m_thr_f**(m2-1)))
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i == 0 and j == 4:
                # Need this extra loop because of Numba limitations.
                jac_values = np.zeros(no_depths)
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                     
            elif i==1 and j == 0:
                # Need this extra loop because of Numba limitations.
                jac_values = CC*Da*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1)
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==1 and j == 1:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(CA*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1) - CC*lambda_*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1) + lambda_*(1 - CC)*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1))
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==1 and j == 2:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(CA*CC*(-KRat*cCO3*m1*nu1*hs_thr_f_m_one*max_thr_f_m_one**(m1-1) - KRat*cCO3*m2*selected_depths*hs_one_m_thr_f*max_one_m_thr_f**(m2-1)) + CC*lambda_*(1 - CC)*(cCO3*n1*hs_two_f_m_one*max_two_f_m_one**(n1-1) + cCO3*n2*nu2*hs_one_m_two_f*max_one_m_two_f**(n2-1)))
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==1 and j == 3:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(CA*CC*(-KRat*cCa*m1*nu1*hs_thr_f_m_one*max_thr_f_m_one**(m1-1) - KRat*cCa*m2*selected_depths*hs_one_m_thr_f*max_one_m_thr_f**(m2-1)) + CC*lambda_*(1 - CC)*(cCa*n1*hs_two_f_m_one*max_two_f_m_one**(n1-1) + cCa*n2*nu2*hs_one_m_two_f*max_one_m_two_f**(n2-1)))
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==1 and j == 4:    
                # Need this extra loop because of Numba limitations.
                jac_values = np.zeros(no_depths)
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                          
            elif i==2 and j == 0:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(1 - Phi)*(-cCa + delta)*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1)/Phi
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==2 and j == 1:
                # Need this extra loop because of Numba limitations.
                jac_values = -Da*lambda_*(1 - Phi)*(-cCa + delta)*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1)/Phi
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==2 and j == 2:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(1 - Phi)*(-cCa + delta)*(CA*(-KRat*cCO3*m1*nu1*hs_thr_f_m_one*max_thr_f_m_one**(m1-1) - KRat*cCO3*m2*selected_depths*hs_one_m_thr_f*max_one_m_thr_f**(m2-1)) - CC*lambda_*(cCO3*n1*hs_two_f_m_one*max_two_f_m_one**(n1-1) + cCO3*n2*nu2*hs_one_m_two_f*max_one_m_two_f**(n2-1)))/Phi - Da*(1 - Phi)*(CA*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1) - CC*lambda_*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1))/Phi
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==2 and j == 3:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(1 - Phi)*(-cCa + delta)*(CA*(-KRat*cCa*m1*nu1*hs_thr_f_m_one*max_thr_f_m_one**(m1-1) - KRat*cCa*m2*selected_depths*hs_one_m_thr_f*max_one_m_thr_f**(m2-1)) - CC*lambda_*(cCa*n1*hs_two_f_m_one*max_two_f_m_one**(n1-1) + cCa*n2*nu2*hs_one_m_two_f*max_one_m_two_f**(n2-1)))/Phi
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==2 and j == 4:
                # Need this extra loop because of Numba limitations.
                jac_values = -Da*(-cCa + delta)*(CA*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1) - CC*lambda_*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1))/Phi - Da*(1 - Phi)*(-cCa + delta)*(CA*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1) - CC*lambda_*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1))/Phi**2
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==3 and j == 0:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(1 - Phi)*(-cCO3 + delta)*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1)/Phi
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                     
            elif i==3 and j == 1:
                # Need this extra loop because of Numba limitations.
                jac_values = -Da*lambda_*(1 - Phi)*(-cCO3 + delta)*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1)/Phi
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==3 and j == 2:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(1 - Phi)*(-cCO3 + delta)*(CA*(-KRat*cCO3*m1*nu1*hs_thr_f_m_one*max_thr_f_m_one**(m1-1) - KRat*cCO3*m2*selected_depths*hs_one_m_thr_f*max_one_m_thr_f**(m2-1)) - CC*lambda_*(cCO3*n1*hs_two_f_m_one*max_two_f_m_one**(n1-1) + cCO3*n2*nu2*hs_one_m_two_f*max_one_m_two_f**(n2-1)))/Phi
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==3 and j == 3:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(1 - Phi)*(-cCO3 + delta)*(CA*(-KRat*cCa*m1*nu1*hs_thr_f_m_one*max_thr_f_m_one**(m1-1) - KRat*cCa*m2*selected_depths*hs_one_m_thr_f*max_one_m_thr_f**(m2-1)) - CC*lambda_*(cCa*n1*hs_two_f_m_one*max_two_f_m_one**(n1-1) + cCa*n2*nu2*hs_one_m_two_f*max_one_m_two_f**(n2-1)))/Phi - Da*(1 - Phi)*(CA*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1) - CC*lambda_*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1))/Phi
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==3 and j == 4:
                # Need this extra loop because of Numba limitations.
                jac_values = -Da*(-cCO3 + delta)*(CA*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1) - CC*lambda_*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1))/Phi - Da*(1 - Phi)*(-cCO3 + delta)*(CA*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1) - CC*lambda_*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1))/Phi**2
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                      
            elif i==4 and j == 0:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(1 - Phi)*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1)
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==4 and j == 1:
                # Need this extra loop because of Numba limitations.
                jac_values = -Da*lambda_*(1 - Phi)*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1)
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                    
            elif i==4 and j == 2:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(1 - Phi)*(CA*(-KRat*cCO3*m1*nu1*hs_thr_f_m_one*max_thr_f_m_one**(m1-1) - KRat*cCO3*m2*selected_depths*hs_one_m_thr_f*max_one_m_thr_f**(m2-1)) - CC*lambda_*(cCO3*n1*hs_two_f_m_one*max_two_f_m_one**(n1-1) + cCO3*n2*nu2*hs_one_m_two_f*max_one_m_two_f**(n2-1)))
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                        
            elif i==4 and j == 3:
                # Need this extra loop because of Numba limitations.
                jac_values = Da*(1 - Phi)*(CA*(-KRat*cCa*m1*nu1*hs_thr_f_m_one*max_thr_f_m_one**(m1-1) - KRat*cCa*m2*selected_depths*hs_one_m_thr_f*max_one_m_thr_f**(m2-1)) - CC*lambda_*(cCa*n1*hs_two_f_m_one*max_two_f_m_one**(n1-1) + cCa*n2*nu2*hs_one_m_two_f*max_one_m_two_f**(n2-1)))
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                          
            elif i==4 and j == 4:
                # Need this extra loop because of Numba limitations.
                jac_values = -Da*(CA*(selected_depths*max_one_m_thr_f**m2 - nu1*max_thr_f_m_one**m1) - CC*lambda_*(-nu2*max_one_m_two_f**n2 + max_two_f_m_one**n1))
                for k in range(no_depths):
                    all_jac_values[row[k], col[k]] = jac_values[k]                                                                            
    return all_jac_values
