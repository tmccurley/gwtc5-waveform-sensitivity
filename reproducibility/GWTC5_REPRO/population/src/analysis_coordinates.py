"""Explicit coordinate maps and a mass/distance prior factor, not a PE denominator.

The analysis cosmology and the cosmology defining the original distance prior
are distinct objects. Neither a spin transformation nor nuisance marginalization
is justified by this module. Original posterior columns are never overwritten.
"""

from dataclasses import dataclass
import math

import numpy as np
from astropy.cosmology import Planck15

from src.pe_prior_reference import FlatDetectorMassPrior, SourceFrameDistancePrior, planck15_lal


INVERSE_ATOL_MPC = 1e-8
INVERSE_RTOL = 1e-12
# Refine more tightly than the public round-trip screen, because a small
# distance residual can matter more to a steep density near its lower bound.
_REFINE_ATOL_MPC = 1e-11
_REFINE_RTOL = 1e-13


def real_array(value, name):
    """Reject complex, Boolean and textual inputs instead of coercing them."""
    array = np.asarray(value)
    if array.dtype.kind not in 'ifu' or not np.all(np.isfinite(array)):
        raise ValueError(name+' must contain finite real numbers')
    return np.asarray(array, dtype=float)


class AnalysisCosmology:
    """A named flat cosmology with safeguarded inversion, no extrapolation.

    The redshift ceiling bounds this numerical map only. It is not an event
    selection rule or a population prior. Initial grid guesses must pass the
    forward-distance residual test after Newton refinement.
    """

    def __init__(self, name, *, inversion_zmax):
        if name not in ('Planck15', 'Planck15_LAL'):
            raise ValueError('Select Planck15 or Planck15_LAL explicitly')
        if isinstance(inversion_zmax, (bool, np.bool_)) or not math.isfinite(inversion_zmax) or inversion_zmax <= 0:
            raise ValueError('Positive finite inversion ceiling required')
        self.name, self.zmax = name, float(inversion_zmax)
        self.cosmology = Planck15 if name == 'Planck15' else planck15_lal()
        self._z_nodes = np.expm1(np.linspace(0., math.log1p(self.zmax), 1025))
        self._z_nodes[-1] = self.zmax
        self._d_nodes = self.distance(self._z_nodes)
        if np.any(np.diff(self._d_nodes) <= 0):
            raise ArithmeticError('Cosmology distance map is not increasing')
        self.distance_limit = float(self._d_nodes[-1])

    def checked_redshift(self, redshift):
        z = real_array(redshift, 'redshift')
        if np.any(z < 0) or np.any(z > self.zmax):
            raise ValueError('Redshift outside numerical map; no clipping or extrapolation')
        return z

    def distance(self, redshift):
        z = self.checked_redshift(redshift)
        if z.size == 0: return np.empty(z.shape)
        return np.asarray(self.cosmology.luminosity_distance(z).value, float)

    def distance_derivative(self, redshift):
        """d[(1+z) D_comoving]/dz for the explicitly flat supported models."""
        z = self.checked_redshift(redshift)
        if z.size == 0: return np.empty(z.shape)
        value = (self.cosmology.comoving_distance(z).value
                 + (1+z)*299792.458/self.cosmology.H(z).value)
        if not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ArithmeticError('Invalid distance Jacobian')
        return np.asarray(value, float)

    def redshift_at_distance(self, luminosity_distance):
        distance = real_array(luminosity_distance, 'luminosity distance')
        if np.any(distance < 0) or np.any(distance > self.distance_limit):
            raise ValueError('Distance outside numerical map; no clipping or extrapolation')
        d = distance.ravel()
        if d.size == 0: return np.empty(distance.shape)
        # Endpoints are valid inputs. Index bounding selects an enclosing cell;
        # it does not clip the physical distance, checked above.
        upper_index = np.minimum(np.maximum(np.searchsorted(self._d_nodes, d), 1), 1024)
        lo = self._z_nodes[upper_index-1].copy()
        hi = self._z_nodes[upper_index].copy()
        z = np.interp(d, self._d_nodes, self._z_nodes)
        active = np.arange(d.size)
        for _ in range(16):
            residual = self.distance(z[active])-d[active]
            bad = np.abs(residual) > _REFINE_ATOL_MPC+_REFINE_RTOL*d[active]
            if not np.any(bad): return z.reshape(distance.shape)
            active, residual = active[bad], residual[bad]
            above = residual > 0
            hi[active[above]] = z[active[above]]
            lo[active[~above]] = z[active[~above]]
            proposal = z[active]-residual/self.distance_derivative(z[active])
            inside = np.isfinite(proposal) & (proposal > lo[active]) & (proposal < hi[active])
            z[active] = np.where(inside, proposal, .5*(lo[active]+hi[active]))
        residual = self.distance(z)-d
        if np.any(np.abs(residual) > _REFINE_ATOL_MPC+_REFINE_RTOL*d):
            raise ArithmeticError('Distance inversion failed its declared residual screen')
        return z.reshape(distance.shape)


def readonly_copy(value):
    result = np.array(value, dtype=float, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class AnalysisCoordinateView:
    """Derived arrays with no spin fields and no implicit prior or weight."""
    cosmology_name: str
    mass_1_source: np.ndarray
    mass_2_source: np.ndarray
    mass_ratio: np.ndarray
    redshift: np.ndarray
    log_detector_component_distance_jacobian: np.ndarray


def detector_to_analysis(mass_1, mass_2, luminosity_distance, *, cosmology):
    if not isinstance(cosmology, AnalysisCosmology):
        raise TypeError('Explicit analysis cosmology required')
    m1,m2,d = np.broadcast_arrays(real_array(mass_1,'mass 1'), real_array(mass_2,'mass 2'),
                                  real_array(luminosity_distance,'luminosity distance'))
    if np.any(m1 <= 0) or np.any(m2 <= 0) or np.any(m2 > m1):
        raise ValueError('Positive ordered detector-frame masses required')
    z = cosmology.redshift_at_distance(d)
    # The full component-mass/distance Jacobian includes both mass coordinates.
    log_jac = np.log(m1)+np.log1p(z)+np.log(cosmology.distance_derivative(z))
    return AnalysisCoordinateView(cosmology.name, *(readonly_copy(v) for v in
        (m1/(1+z),m2/(1+z),m2/m1,z,log_jac)))


class MassDistancePriorInAnalysisCoordinates:
    """Transform a specified PE mass/distance factor to dm_source dq dz.

    No spin or nuisance factor is included. Even a correct mathematical
    transformation does not certify the sampling density of released rows.
    The retained PE reference currently supports Planck15_LAL distance priors
    only; an analysis Planck15 map must not change that original prior.
    """

    released_rows_validated = False
    includes_spin_density = False
    includes_nuisance_density = False

    def __init__(self, mass_prior, distance_prior, *, analysis_cosmology, pe_cosmology):
        if not isinstance(mass_prior, FlatDetectorMassPrior) or not isinstance(distance_prior, SourceFrameDistancePrior):
            raise TypeError('Explicit mass and distance prior references required')
        if not isinstance(analysis_cosmology, AnalysisCosmology) or not isinstance(pe_cosmology, AnalysisCosmology):
            raise TypeError('Explicit coordinate and original-prior cosmology maps required')
        if pe_cosmology.name != 'Planck15_LAL':
            raise ValueError('Original PE reference is Planck15_LAL, not the analysis cosmology')
        if distance_prior.distance_bounds[1] > min(analysis_cosmology.distance_limit, pe_cosmology.distance_limit):
            raise ValueError('Numerical maps must cover the entire prior distance support')
        self.mass_prior, self.distance_prior = mass_prior, distance_prior
        self.analysis, self.pe = analysis_cosmology, pe_cosmology

    def distance_logpdf(self, luminosity_distance):
        d = real_array(luminosity_distance, 'luminosity distance')
        if np.any(d < 0): raise ValueError('Nonnegative distance required')
        out = np.full(d.shape, -np.inf)
        lo,hi = self.distance_prior.distance_bounds
        good = (d >= lo) & (d <= hi)
        if not np.any(good): return out
        if self.distance_prior.mode == 'continuous':
            z = self.pe.redshift_at_distance(d[good])
            values = (self.distance_prior.raw_redshift_density(z)
                      / self.distance_prior.continuous_normalizer / self.pe.distance_derivative(z))
        else:
            values = self.distance_prior.distance_pdf(d[good])
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ArithmeticError('Invalid reference distance density')
        with np.errstate(divide='ignore'):
            out[good] = np.log(values)
        return out

    def detector_component_logpdf(self, mass_1, mass_2, luminosity_distance):
        m1,m2,d = np.broadcast_arrays(real_array(mass_1,'mass 1'), real_array(mass_2,'mass 2'),
                                      real_array(luminosity_distance,'luminosity distance'))
        if np.any(m1 <= 0) or np.any(m2 <= 0) or np.any(m2 > m1):
            raise ValueError('Positive ordered detector-frame masses required')
        support = self.mass_prior.contains(m1,m2/m1)
        return np.where(support, self.distance_logpdf(d)-math.log(self.mass_prior.area), -np.inf)

    def source_mqz_logpdf(self, mass_source, mass_ratio, redshift):
        m,q,z = np.broadcast_arrays(real_array(mass_source,'source mass'), real_array(mass_ratio,'mass ratio'),
                                     self.analysis.checked_redshift(redshift))
        if np.any(m <= 0) or np.any(q <= 0) or np.any(q > 1):
            raise ValueError('Positive source mass and 0<q<=1 required')
        mdet = m*(1+z)
        original = self.detector_component_logpdf(mdet,mdet*q,self.analysis.distance(z))
        return original+np.log(m)+2*np.log1p(z)+np.log(self.analysis.distance_derivative(z))

    def require_released_rows_ready(self):
        raise RuntimeError('Mass/distance factor only: released prior, spin coordinates and nuisance treatment remain unvalidated')
