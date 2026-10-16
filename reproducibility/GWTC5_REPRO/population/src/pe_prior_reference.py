"""Normalized PE-prior references, not a certified released-row denominator.

Masses use the paired Bilby UniformInComponents priors and explicit detector-
frame constraints. The distance reference includes a source-frame time factor.
Its optional grid follows the inspected Bilby v2.6.0 recipe without importing
downloaded source. Astropy/compiler versions are not a bitwise release clone.
Spin is the isotropic Uniform(0,A) induced prior at the supplied PE coordinates.
No assumption about spin evolution or released-row reweighting is hidden here.
"""

from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from scipy.optimize import brentq

from src.independent_mass_redshift import checked_quad
from src.spin_prior_reference import isotropic_uniform_spin_density


def interval(value, name, *, allow_zero=False):
    if len(value) != 2 or any(isinstance(v, (bool, np.bool_)) for v in value):
        raise ValueError(name+' needs two real bounds')
    low, high = value
    if not all(math.isfinite(v) for v in value) or not low < high or (low < 0 if allow_zero else low <= 0):
        raise ValueError(name+' needs finite positive ordered bounds')


def chirp_factor(q):
    return q**(3/5)/(1+q)**(1/5)


@dataclass(frozen=True)
class DetectorMassSupport:
    """All bounds are detector-frame solar masses, except dimensionless q."""
    chirp_mass: tuple
    mass_ratio: tuple
    mass_1: tuple
    mass_2: tuple
    total_mass: tuple | None = None

    def __post_init__(self):
        for name in ('chirp_mass', 'mass_ratio', 'mass_1', 'mass_2'):
            interval(getattr(self, name), name)
        if self.mass_ratio[1] > 1:
            raise ValueError('Ordered component masses require q<=1')
        if self.total_mass is not None:
            interval(self.total_mass, 'total_mass', allow_zero=True)


class FlatDetectorMassPrior:
    """Constant density in allowed (m1_detector,m2_detector) area.

Chirp-mass and q bounds do not generally make a rectangular component-mass
region. Crossings of the support curves split the normalization integral,
including thin allowed wedges that an unsplit quadrature could miss.
"""
    def __init__(self, support):
        if not isinstance(support, DetectorMassSupport):
            raise TypeError('Explicit detector-frame mass support required')
        self.support = support
        curves = self._curves(0)+self._curves(1)
        qlo, qhi = support.mass_ratio
        edges = {qlo, qhi}
        # Ratios of these four curve families are monotone on 0<q<=1.
        # Thus each pair has at most one interior crossing unless identical.
        for left, right in combinations(curves, 2):
            fun = lambda q: left(q)-right(q)
            a, b = fun(qlo), fun(qhi)
            if (a < 0 < b) or (b < 0 < a):
                edges.add(brentq(fun, qlo, qhi, xtol=1e-14, rtol=1e-14))
        self.q_edges = sorted(edges)
        self.area = sum(checked_quad(self.area_integrand, lo, hi, epsabs=0, epsrel=1e-10)
                        for lo, hi in zip(self.q_edges[:-1], self.q_edges[1:]))
        if not math.isfinite(self.area) or self.area <= 0:
            raise ValueError('Mass constraints have no positive resolved area')

    def _curves(self, side):
        s = self.support
        curves = [lambda q: s.mass_1[side], lambda q: s.mass_2[side]/q,
                  lambda q: s.chirp_mass[side]/chirp_factor(q)]
        if s.total_mass is not None:
            curves.append(lambda q: s.total_mass[side]/(1+q))
        return curves

    def m1_bounds(self, q):
        return max(f(q) for f in self._curves(0)), min(f(q) for f in self._curves(1))

    def area_integrand(self, q):
        low, high = self.m1_bounds(q)
        return 0.5*(high-low)*(high+low) if high > low else 0.

    def contains(self, mass_1, mass_ratio):
        m, q = np.broadcast_arrays(np.asarray(mass_1, float), np.asarray(mass_ratio, float))
        if not np.all(np.isfinite(m)) or not np.all(np.isfinite(q)):
            raise ValueError('Nonfinite mass coordinates')
        s = self.support
        result = (m >= s.mass_1[0]) & (m <= s.mass_1[1]) & (q >= s.mass_ratio[0]) & (q <= s.mass_ratio[1])
        result &= (m*q >= s.mass_2[0]) & (m*q <= s.mass_2[1])
        candidate = np.flatnonzero(result)
        good = np.zeros(result.size, bool)
        if candidate.size:
            mc = m.ravel()[candidate]*chirp_factor(q.ravel()[candidate])
            good[candidate] = (mc >= s.chirp_mass[0]) & (mc <= s.chirp_mass[1])
        result &= good.reshape(result.shape)
        if s.total_mass is not None:
            total = m*(1+q)
            result &= (total >= s.total_mass[0]) & (total <= s.total_mass[1])
        return result

    def logpdf_detector_mq(self, mass_1, mass_ratio):
        m, q = np.broadcast_arrays(np.asarray(mass_1, float), np.asarray(mass_ratio, float))
        good = self.contains(m, q)
        result = np.full(m.shape, -np.inf)
        result[good] = np.log(m[good])-math.log(self.area)
        return result


def planck15_lal():
    """The inspected Bilby v2.6.0 Planck15_LAL definition; no radiation."""
    return FlatLambdaCDM(H0=67.9, Om0=0.3065, Tcmb0=0, name='Planck15_LAL')


class SourceFrameDistancePrior:
    """Prior density in dL or z, with two explicitly distinct implementations.

continuous: p(z) proportional to (dVc/dz)/(1+z), normalized between the
distance-derived limits. bilby_2.6.0_grid_reference: 1000 padded z nodes,
finite-difference distance Jacobian, then 1000 uniform dL nodes with linear
interpolation and trapezoidal normalization, as in the archived source.
    This class handles Planck15_LAL only. Other PE/conversion cosmologies must
    not be silently mapped to it.

    The grid models Bilby's prob() path. Its interpolated inverse-CDF sampler
    and the release environment still require separate numerical comparison.
"""
    MODES = ('continuous', 'bilby_2.6.0_grid_reference')

    def __init__(self, distance_bounds, *, mode):
        interval(distance_bounds, 'luminosity_distance', allow_zero=True)
        if mode not in self.MODES:
            raise ValueError('Select an explicit supported distance implementation')
        self.distance_bounds, self.mode = tuple(distance_bounds), mode
        self.cosmology = planck15_lal()
        self.zmin, self.zmax = [self.redshift_at_distance(d) for d in distance_bounds]
        self.continuous_normalizer = checked_quad(self.raw_redshift_density, self.zmin, self.zmax,
                                                  epsabs=0, epsrel=1e-10)
        if mode == 'bilby_2.6.0_grid_reference':
            grid_limits = [0. if d == 0 else float(z_at_value(self.cosmology.luminosity_distance, d*u.Mpc).value)
                           for d in distance_bounds]
            z = np.linspace(0.99*grid_limits[0], 1.01*grid_limits[1], 1000)
            distance = self.cosmology.luminosity_distance(z).value
            density = self.raw_redshift_density(z)/np.gradient(distance, z)
            self.distance_nodes = np.linspace(*distance_bounds, 1000)
            values = np.interp(self.distance_nodes, distance, density, left=0., right=0.)
            self.grid_normalizer = float(np.trapezoid(values, self.distance_nodes))
            if self.grid_normalizer <= 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ArithmeticError('Invalid distance grid')
            self.distance_density_nodes = values/self.grid_normalizer
            self.source_recipe_redshift_limits = grid_limits

    def redshift_at_distance(self, distance):
        if not math.isfinite(distance) or distance < 0:
            raise ValueError('Finite nonnegative luminosity distance required')
        if distance == 0:
            return 0.
        upper = 1.
        while self.cosmology.luminosity_distance(upper).value < distance:
            upper *= 2
            if upper > 1024:
                raise ValueError('Distance inversion exceeded reference range')
        return brentq(lambda z: self.cosmology.luminosity_distance(z).value-distance,
                      0., upper, xtol=1e-14, rtol=1e-14)

    def raw_redshift_density(self, redshift):
        z = np.asarray(redshift, float)
        if not np.all(np.isfinite(z)) or np.any(z < 0):
            raise ValueError('Finite nonnegative redshift required')
        return self.cosmology.differential_comoving_volume(z).value/(1+z)

    def distance_derivative(self, redshift):
        z = np.asarray(redshift, float)
        return self.cosmology.comoving_distance(z).value+(1+z)*299792.458/self.cosmology.H(z).value

    def distance_pdf(self, distance):
        d = np.asarray(distance, float)
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError('Finite nonnegative luminosity distances required')
        if self.mode == 'bilby_2.6.0_grid_reference':
            return np.interp(d, self.distance_nodes, self.distance_density_nodes, left=0., right=0.)
        result = np.zeros(d.shape)
        good = (d >= self.distance_bounds[0]) & (d <= self.distance_bounds[1])
        for i in np.flatnonzero(good):
            z = self.redshift_at_distance(float(d.ravel()[i]))
            result.ravel()[i] = self.raw_redshift_density(z)/self.continuous_normalizer/self.distance_derivative(z)
        return result

    def redshift_logpdf(self, redshift):
        z = np.asarray(redshift, float)
        if not np.all(np.isfinite(z)) or np.any(z < 0):
            raise ValueError('Finite nonnegative redshift required')
        if self.mode == 'continuous':
            values = self.raw_redshift_density(z)/self.continuous_normalizer
        else:
            values = self.distance_pdf(self.cosmology.luminosity_distance(z).value)*self.distance_derivative(z)
        with np.errstate(divide='ignore'):
            result = np.log(values)
        return np.where((z >= self.zmin) & (z <= self.zmax), result, -np.inf)


class ReducedPEPriorReference:
    """Density in dm1_source dq dz dchi_eff for explicit reference assumptions.

The full mass transformation contributes m1_source*(1+z)^2. Marginalizing
spin nuisance coordinates supplies p0(chi_eff|q,A), not a flat chi_eff prior.
Using this marginal for posterior recycling additionally requires identical
population/PE conditional nuisance priors at a common spin reference. That
condition is not established by constructing this mathematical reference.
"""
    released_rows_validated = False

    def __init__(self, mass_prior, distance_prior, *, a_max):
        if not isinstance(mass_prior, FlatDetectorMassPrior) or not isinstance(distance_prior, SourceFrameDistancePrior):
            raise TypeError('Explicit mass and distance references required')
        if not math.isfinite(a_max) or not 0 < a_max <= 1:
            raise ValueError('Physical component-spin bound required')
        self.mass_prior, self.distance_prior, self.a_max = mass_prior, distance_prior, a_max

    def mass_redshift_logpdf(self, mass_source, q, z):
        m, q, z = np.broadcast_arrays(np.asarray(mass_source, float), np.asarray(q, float), np.asarray(z, float))
        if not np.all(np.isfinite(m)) or np.any(m <= 0):
            raise ValueError('Positive finite source masses required')
        redshift = self.distance_prior.redshift_logpdf(z)
        return self.mass_prior.logpdf_detector_mq(m*(1+z), q)+np.log1p(z)+redshift

    def logpdf(self, mass_source, q, z, chi_eff):
        m, q, z, chi = np.broadcast_arrays(np.asarray(mass_source, float), np.asarray(q, float),
                                         np.asarray(z, float), np.asarray(chi_eff, float))
        if not np.all(np.isfinite(chi)):
            raise ValueError('Finite effective spin required')
        result = np.asarray(self.mass_redshift_logpdf(m, q, z), float).copy()
        for i in np.flatnonzero(np.isfinite(result)):
            value, _ = isotropic_uniform_spin_density(float(chi.ravel()[i]), float(q.ravel()[i]), self.a_max)
            result.ravel()[i] += math.log(value) if value > 0 else -math.inf
        return result

    def require_released_rows_ready(self):
        raise RuntimeError('Released PE prior remains blocked: row provenance, support and common nuisance/spin coordinates not certified')
