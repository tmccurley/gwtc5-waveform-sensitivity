"""Independent Three Transitions spin component, not an official-run clone.

The conditional density follows Eq. G4 of arXiv:2605.27226v1. It is a density
in chi_eff at fixed source-frame primary mass, NOT the joint population
density or a population-ready PE prior. See protocol/independent_route.md.

The endpoint prior is our explicit convention: two independent U(-1,1)
endpoints conditioned on lower < upper. The published table does not uniquely
specify this joint convention. No released posterior rows are transformed.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass
import math

import numpy as np
from scipy.special import ndtr


MODEL_ID = 'independent_three_transitions_spin_v0.1'
PARAMETER_NAMES = (
    'mu', 'log10_sigma', 'm_t_low', 'm_t_mid', 'm_t_high', 'xi_low', 'xi_high',
    'chi_min_low', 'chi_max_low', 'chi_min_high', 'chi_max_high',
)
ENDPOINT_PRIOR = 'independent_uniform_endpoints_conditioned_on_order'


@dataclass(frozen=True)
class SpinParameters:
    """Scalar hyperparameters; mass is in solar masses, spin is dimensionless.

    mu and sigma describe the underlying Gaussian before truncation. mu is
    generally not the mean of the resulting truncated distribution.
    """

    mu: float
    log10_sigma: float
    m_t_low: float
    m_t_mid: float
    m_t_high: float
    xi_low: float
    xi_high: float
    chi_min_low: float
    chi_max_low: float
    chi_min_high: float
    chi_max_high: float

    def __post_init__(self):
        if not all(np.ndim(value) == 0 and math.isfinite(value) for value in astuple(self)):
            raise ValueError('Spin hyperparameters must be finite scalars')
        if not (-1 <= self.mu <= 1 and -2 <= self.log10_sigma <= 1):
            raise ValueError('Gaussian hyperparameters are outside the declared domain')
        if not 0 < self.m_t_low < self.m_t_mid < self.m_t_high:
            raise ValueError('Transition masses must be positive and strictly ordered')
        if not (0 <= self.xi_low <= 1 and 0 <= self.xi_high <= 1):
            raise ValueError('Mixture fractions must be in [0,1]')
        for lower, upper in [(self.chi_min_low, self.chi_max_low),
                             (self.chi_min_high, self.chi_max_high)]:
            if not -1 <= lower < upper <= 1:
                raise ValueError('Uniform spin bounds must satisfy -1 <= lower < upper <= 1')


def conditional_logpdf(mass_1_source, chi_eff, parameters: SpinParameters):
    """Return log p(chi_eff | m1); preserve zero density as negative infinity.

    Inputs may broadcast. Nonfinite coordinates and nonpositive masses are
    rejected; finite spin values outside [-1,1] have zero density. Transitions
    are right-continuous, an explicit convention at measure-zero boundaries.
    No input rows are removed, clipped, sorted or reweighted here.
    """
    if not isinstance(parameters, SpinParameters):
        raise TypeError('parameters must be validated SpinParameters')
    mass, spin = np.broadcast_arrays(np.asarray(mass_1_source, dtype=float),
                                    np.asarray(chi_eff, dtype=float))
    if not (np.all(np.isfinite(mass)) and np.all(np.isfinite(spin))):
        raise ValueError('Mass and spin coordinates must be finite')
    if np.any(mass <= 0):
        raise ValueError('Source-frame primary masses must be positive')
    p = parameters
    sigma = 10.0 ** p.log10_sigma
    # With mu in [-1,1] and sigma in [0.01,10], this CDF difference is
    # positive and well resolved. It normalizes the Gaussian over [-1,1].
    normalizer = ndtr((1 - p.mu) / sigma) - ndtr((-1 - p.mu) / sigma)
    gaussian = np.full(spin.shape, -np.inf)
    physical = (spin >= -1) & (spin <= 1)
    gaussian[physical] = (-0.5 * ((spin[physical] - p.mu) / sigma) ** 2
                          - math.log(sigma) - 0.5 * math.log(2 * math.pi)
                          - math.log(normalizer))

    fraction = np.zeros(spin.shape)
    uniform = np.full(spin.shape, -np.inf)
    regions = [((mass >= p.m_t_low) & (mass < p.m_t_mid), p.xi_low,
                p.chi_min_low, p.chi_max_low),
               (mass >= p.m_t_high, p.xi_high, p.chi_min_high, p.chi_max_high)]
    for region, mixture, lower, upper in regions:
        fraction[region] = mixture
        uniform[region & (spin >= lower) & (spin <= upper)] = -math.log(upper - lower)
    # Fractions exactly 0 and 1 are legitimate limits. Their absent mixture
    # term has log weight -inf, not a small artificial probability floor.
    with np.errstate(divide='ignore'):
        return np.logaddexp(np.log1p(-fraction) + gaussian,
                            np.log(fraction) + uniform)


def conditional_pdf(mass_1_source, chi_eff, parameters: SpinParameters):
    """Linear-density wrapper; use conditional_logpdf for posterior recycling."""
    return np.exp(conditional_logpdf(mass_1_source, chi_eff, parameters))


def _ordered_endpoints(u_lower, u_upper):
    """Inverse conditional CDF for a constant density on the ordered triangle.

    The lower endpoint has density (1-lower)/2, not U(-1,1). Given it, the
    upper endpoint is uniform on (lower,1). The joint density is 1/2 and
    this transformation's Jacobian determinant is 2. A uniformly drawn lower
    endpoint followed by a conditional upper endpoint would be a DIFFERENT
    joint prior. This choice applies to new draws only, not reference data.
    """
    lower = 1 - 2 * math.sqrt(1 - u_lower)
    return lower, lower + (1 - lower) * u_upper


def spin_prior_transform(unit_cube):
    """Map one interior 11-dimensional unit-cube point to the declared prior.

    Scalar ranges and ordered transition-mass intervals follow Table 15.
    The endpoint joint prior is explicitly our independent reconstruction.
    Open-cube input avoids zero-width components and coincident transitions;
    a future sampler adapter must handle numerical boundary proposals without
    silently clipping them. This function uses no global or hidden RNG.
    """
    unit = np.asarray(unit_cube, dtype=float)
    if (unit.shape != (len(PARAMETER_NAMES),) or not np.all(np.isfinite(unit))
            or np.any(unit <= 0) or np.any(unit >= 1)):
        raise ValueError('Prior transform needs 11 finite coordinates strictly inside (0,1)')
    low_mass = 5 + 25 * unit[2]
    mid_mass = low_mass + 30 * unit[3]
    high_mass = mid_mass + 80 * unit[4]
    low_bounds = _ordered_endpoints(unit[7], unit[8])
    high_bounds = _ordered_endpoints(unit[9], unit[10])
    return SpinParameters(2 * unit[0] - 1, 3 * unit[1] - 2,
                          low_mass, mid_mass, high_mass, unit[5], unit[6],
                          *low_bounds, *high_bounds)


def spin_log_hyperprior(parameters: SpinParameters):
    """Normalized log prior in PARAMETER_NAMES coordinates, including log10 sigma.

    This is the hyperprior, not the PE prior in the event likelihood. It has
    constant density on its joint support. Converting it to a density in
    sigma instead of log10_sigma requires the corresponding Jacobian.
    """
    if not isinstance(parameters, SpinParameters):
        raise TypeError('parameters must be validated SpinParameters')
    p = parameters
    if not (5 <= p.m_t_low <= 30
            and p.m_t_mid <= p.m_t_low + 30
            and p.m_t_high <= p.m_t_mid + 80):
        return -math.inf
    return -math.log(2 * 3 * 25 * 30 * 80 * 2 * 2)
