"""Separately normalized cubic-tail spin variant for independent development.

This changes the statistical model. It does not clip importance weights,
repair release rows, or reproduce an official fit. The original v0.1 model
is imported without modification. See protocol/boundary_treatment_2026-09-12.md.
"""

from dataclasses import replace
import math

import numpy as np
from numpy.polynomial.legendre import leggauss

from src.independent_mass_redshift import checked_quad
from src.independent_population import PopulationModel
from src.independent_spin_model import SpinParameters, conditional_logpdf


MODEL_ID = 'independent_three_transitions_cubic_tail_v0.2'
_NODES, _WEIGHTS = leggauss(3)


def log_boundary_window(x, width):
    """Log quintic window in normalized spin x=chi_eff/A, width=delta/A."""
    values = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(values)) or not math.isfinite(width) or not 0 < width < 1:
        raise ValueError('Finite spin and a window width strictly between zero and one required')
    distance = 1-np.abs(values)
    result = np.full(values.shape, -np.inf)
    result[distance >= width] = 0
    inside = (distance > 0) & (distance < width)
    t = distance[inside]/width
    log_values = np.empty(t.shape)
    left = t <= 0.5
    log_values[left] = 3*np.log(t[left])+np.log(10-15*t[left]+6*t[left]*t[left])
    # w(t)=1-w(1-t) avoids cancellation and a spurious w>1 near the join.
    y = 1-t[~left]
    log_values[~left] = np.log1p(-y**3*(10-15*y+6*y*y))
    result[inside] = log_values
    return result


def uniform_window_normalizer(lower, upper, width):
    """Average window using exact piecewise polynomial integration.

    Three Gauss-Legendre nodes integrate degree-five polynomials exactly.
    Rescaling each interval avoids subtracting nearly equal antiderivatives.
    """
    if not (-1 <= lower < upper <= 1) or not all(math.isfinite(v) for v in (lower, upper)):
        raise ValueError('Uniform interval must lie inside [-1,1] with positive width')
    log_boundary_window(0, width)
    edges = sorted({lower, upper, *(p for p in (-1+width, 1-width) if lower < p < upper)})
    total = 0.
    for left, right in zip(edges[:-1], edges[1:]):
        mid, halfspan = left+(right-left)/2, (right-left)/2
        nodes = mid+halfspan*_NODES
        average = float(_WEIGHTS @ np.exp(log_boundary_window(nodes, width)))/2
        total += (right-left)/(upper-lower)*average
    if not math.isfinite(total) or total <= 0:
        raise ArithmeticError('Uniform window normalization is not positive')
    return total


class BoundaryRegularizedSpin:
    def __init__(self, parameters: SpinParameters, *, a_max, tail_width):
        if not isinstance(parameters, SpinParameters):
            raise TypeError('Validated spin parameters required')
        if not (math.isfinite(a_max) and 0 < a_max <= 1 and math.isfinite(tail_width) and 0 < tail_width < a_max):
            raise ValueError('Require 0 < physical tail_width < a_max <= 1')
        self.parameters, self.a_max, self.tail_width = parameters, a_max, tail_width
        self.width = tail_width/a_max
        self.gaussian_parameters = replace(parameters, xi_low=0, xi_high=0)
        sigma = 10**parameters.log10_sigma
        points = sorted({x for x in [-1+self.width, 1-self.width,
                                     *(parameters.mu+k*sigma for k in (-8, -3, -1, 0, 1, 3, 8))]
                         if -1 < x < 1})
        self.gaussian_norm = checked_quad(
            lambda x: float(np.exp(self.gaussian_logpdf(x)+log_boundary_window(x, self.width))),
            -1, 1, points=points, epsabs=0, epsrel=1e-10)
        if not math.isfinite(self.gaussian_norm) or self.gaussian_norm <= 0:
            raise ArithmeticError('Gaussian window normalization is not positive')
        self.uniform_norms = [uniform_window_normalizer(lower, upper, self.width)
                              for lower, upper in [(parameters.chi_min_low, parameters.chi_max_low),
                                                   (parameters.chi_min_high, parameters.chi_max_high)]]

    def gaussian_logpdf(self, x):
        """Unwindowed Gaussian in the original internal coordinate."""
        return conditional_logpdf(1., x, self.gaussian_parameters)

    def logpdf(self, mass, chi_eff):
        m, chi = np.broadcast_arrays(np.asarray(mass, dtype=float), np.asarray(chi_eff, dtype=float))
        if not np.all(np.isfinite(m)) or np.any(m <= 0) or not np.all(np.isfinite(chi)):
            raise ValueError('Finite positive masses and finite spins required')
        x, p = chi/self.a_max, self.parameters
        gaussian = self.gaussian_logpdf(x)-math.log(self.gaussian_norm)
        fractions = np.zeros(m.shape)
        uniform = np.full(m.shape, -np.inf)
        regions = [((m >= p.m_t_low) & (m < p.m_t_mid), p.xi_low, p.chi_min_low, p.chi_max_low),
                   (m >= p.m_t_high, p.xi_high, p.chi_min_high, p.chi_max_high)]
        for (region, fraction, lower, upper), normalizer in zip(regions, self.uniform_norms):
            fractions[region] = fraction
            uniform[region & (x >= lower) & (x <= upper)] = -math.log(upper-lower)-math.log(normalizer)
        with np.errstate(divide='ignore'):
            mixture = np.logaddexp(np.log1p(-fractions)+gaussian, np.log(fractions)+uniform)
        return mixture+log_boundary_window(x, self.width)-math.log(self.a_max)

    def boundary_ratio_limit(self, mass, q, *, side=1):
        """Analytic p_spin/p0 limit at chi=side*A; a diagnostic, not a floor."""
        if not (math.isfinite(mass) and mass > 0 and math.isfinite(q) and 0 < q <= 1 and side in (-1, 1)):
            raise ValueError('Require positive mass, 0<q<=1 and side=-1 or +1')
        p = self.parameters
        fraction, uniform = 0., 0.
        if p.m_t_low <= mass < p.m_t_mid:
            fraction, lower, upper, normalizer = p.xi_low, p.chi_min_low, p.chi_max_low, self.uniform_norms[0]
        elif mass >= p.m_t_high:
            fraction, lower, upper, normalizer = p.xi_high, p.chi_min_high, p.chi_max_high, self.uniform_norms[1]
        else:
            lower, upper, normalizer = -1., 1., 1.
        if (side == 1 and upper == 1) or (side == -1 and lower == -1):
            uniform = 1/((upper-lower)*normalizer*self.a_max)
        gaussian = math.exp(float(self.gaussian_logpdf(side)))/(self.gaussian_norm*self.a_max)
        coefficient = 10/self.tail_width**3*((1-fraction)*gaussian+fraction*uniform)
        prior_coefficient = (1+q)**4/(24*self.a_max**4*q*q)
        return coefficient/prior_coefficient


class BoundaryRegularizedPopulation(PopulationModel):
    """Same mass/redshift factors and prior coordinates, new spin-tail shape."""
    def __init__(self, parameters, *, a_max, tail_width, z_max, H0, omega_m):
        super().__init__(parameters, a_max=a_max, z_max=z_max, H0=H0, omega_m=omega_m)
        self.regularized_spin = BoundaryRegularizedSpin(parameters.spin, a_max=a_max, tail_width=tail_width)

    def spin_logpdf(self, mass, chi_eff):
        return self.regularized_spin.logpdf(mass, chi_eff)
