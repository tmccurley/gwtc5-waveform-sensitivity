"""Opt-in exact taper-integral reuse for the unchanged development mass model.

The original MassModel remains the independent numerical reference. This class
introduces neither interpolation nor a different probability distribution.
"""

import math

import numpy as np

from src.independent_mass_redshift import MassModel, checked_quad, log_taper, power_integral


class CachedTaperMassModel(MassModel):
    """Reuse the fully traversed secondary-mass taper at fixed hyperparameters.

    Substituting s=m*q makes its unnormalized integral independent of m.
    Partial tapers still use the original near-boundary, log-scaled routine.
    Primary-mass normalization and all density/support logic are inherited.
    """

    def __init__(self, parameters):
        super().__init__(parameters)
        self._cache_parameters = parameters
        self._log_taper_integral = None
        p = parameters
        if p.delta_2 > 0:
            # Dimensionless u avoids subtracting nearly equal masses when D
            # is small. Scaling bounds the quadrature integrand by one.
            scale = max(0., p.beta_q*math.log1p(p.delta_2/p.m_2_low))
            integral = checked_quad(
                lambda u: math.exp(p.beta_q*math.log1p(p.delta_2*u/p.m_2_low)
                                   + float(log_taper(u, 0., 1.))-scale),
                0., 1., epsabs=0.)
            if not math.isfinite(integral) or integral <= 0:
                raise ArithmeticError('Cached taper integral is not positive')
            self._log_taper_integral = (math.log(p.delta_2)
                + p.beta_q*math.log(p.m_2_low)+scale+math.log(integral))

    def q_log_normalizer(self, mass):
        if self.parameters is not self._cache_parameters:
            raise ValueError('Parameters changed; construct a fresh mass model')
        p = self.parameters
        if not math.isfinite(mass) or mass <= p.m_2_low:
            raise ValueError('Conditional q density requires m1 > m2_low')
        # Use the reference branch predicate, including its boundary convention.
        if p.delta_2 == 0 or mass-p.m_2_low < p.delta_2:
            return super().q_log_normalizer(mass)
        taper_end = min(1., (p.m_2_low+p.delta_2)/mass)
        tail = power_integral(taper_end, 1., p.beta_q)
        log_smooth = self._log_taper_integral-(p.beta_q+1)*math.log(mass)
        value = float(np.logaddexp(log_smooth, math.log(tail) if tail > 0 else -math.inf))
        if not math.isfinite(value):
            raise ArithmeticError('Conditional q normalization is not finite')
        return value
