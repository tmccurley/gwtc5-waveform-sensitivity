"""Explicit, conditional event ratios for development, not production fits.

The population retains the PE reference's native spin nuisance conditional and
each product's other nuisance priors. Neither historical release processing
nor physical common-frequency equivalence is established by this module.
"""

from dataclasses import dataclass
import math
import numpy as np

from src.analysis_coordinates import (MassDistancePriorInAnalysisCoordinates,
    detector_to_analysis, real_array, readonly_copy)
from src.native_reference_stress import stressed_logpdf
from src.numerics import logmeanexp, importance_ess
from src.spin_prior_reference import isotropic_uniform_spin_density


def log_vector(value, name, *, finite=False):
    array = np.asarray(value)
    if array.dtype.kind not in 'ifu' or array.ndim != 1 or not array.size:
        raise ValueError(name+' must be a nonempty real vector')
    array = np.asarray(array, dtype=float)
    if np.isnan(array).any() or np.isposinf(array).any() or (finite and not np.isfinite(array).all()):
        raise ValueError(name+' has an invalid log density')
    return array


def factorized_log_ratio(log_mass_redshift, log_spin, log_pe_mass_distance, log_spin_reference):
    """Cancel equal conditional factors within a product, never between arms.

    Priors must be finite everywhere, even for a zero target. This prevents an
    invalid denominator being hidden by an out-of-population-support point.
    """
    arrays = [log_vector(value, name, finite=prior) for value,name,prior in (
        (log_mass_redshift, 'target mass/redshift', False), (log_spin, 'target spin', False),
        (log_pe_mass_distance, 'PE mass/distance', True), (log_spin_reference, 'spin reference', True))]
    if len({a.shape for a in arrays}) != 1:
        raise ValueError('Every factor must have the same row shape')
    with np.errstate(over='raise', invalid='raise'):
        result = arrays[0]-arrays[2]+arrays[1]-arrays[3]
    if np.isnan(result).any() or np.isposinf(result).any():
        raise ArithmeticError('Invalid event log ratio')
    return readonly_copy(result)


def summarize_log_ratios(values):
    """Evidence-normalized event factor; original count includes zero weights."""
    values = log_vector(values, 'event log ratios')
    finite = np.isfinite(values)
    return dict(rows=int(values.size), positive_weight_rows=int(finite.sum()),
        zero_weight_rows=int((~finite).sum()), log_mean_ratio=logmeanexp(values),
        kish_weight_ess=importance_ess(values),
        finite_log_ratio_range=[float(values[finite].min()),float(values[finite].max())] if finite.any() else None,
        is_absolute_event_likelihood=False, is_between_waveform_bayes_factor=False,
        sampler_ess_validated=False, integration_variance_validated=False)


@dataclass(frozen=True)
class PreparedEventReference:
    """Read-only native-coordinate values with an explicit declared PE factor."""
    mass_1_source: np.ndarray
    mass_ratio: np.ndarray
    redshift: np.ndarray
    chi_eff: np.ndarray
    log_pe_mass_distance: np.ndarray
    log_spin_reference: np.ndarray
    spin_reference_relative_quadrature_error: np.ndarray
    log_detector_component_distance_jacobian: np.ndarray
    distance_mode: str
    analysis_cosmology: str
    a_max: float

    def require_production_ready(self):
        raise RuntimeError('Declared-prior native-reference development only; production assumptions remain unresolved')


def prepare_event_reference(mass_1, mass_2, distance, chi_eff, *, prior, a_max):
    if not isinstance(prior, MassDistancePriorInAnalysisCoordinates):
        raise TypeError('Explicit normalized PE mass/distance reference required')
    if isinstance(a_max, (bool, np.bool_)) or not math.isfinite(a_max) or not 0 < a_max <= 1:
        raise ValueError('Explicit physical spin bound required')
    inputs = [real_array(value,name) for value,name in ((mass_1,'mass1'),(mass_2,'mass2'),
               (distance,'distance'),(chi_eff,'native effective spin'))]
    if any(a.ndim != 1 or not a.size for a in inputs) or len({a.shape for a in inputs}) != 1:
        raise ValueError('Nonempty matching one-dimensional event rows required')
    if np.any(abs(inputs[3]) >= a_max):
        raise ValueError('Native effective spin must be strictly inside the declared PE support')
    view = detector_to_analysis(*inputs[:3], cosmology=prior.analysis)
    # Evaluate at the original dL to avoid round-trip support ambiguity at an
    # exact distance boundary. The source-coordinate Jacobian is applied once.
    density = prior.detector_component_logpdf(*inputs[:3]) + view.log_detector_component_distance_jacobian
    if not np.isfinite(density).all():
        raise ValueError('At least one event row has invalid or zero PE prior support')
    spin = np.array([isotropic_uniform_spin_density(float(x),float(q),a_max,epsabs=1e-12,epsrel=1e-9)
                     for x,q in zip(inputs[3],view.mass_ratio)])
    if not np.isfinite(spin).all() or np.any(spin[:,0] <= 0) or np.any(spin[:,1] < 0):
        raise ArithmeticError('Invalid induced spin prior')
    relative = spin[:,1]/spin[:,0]
    if np.any(relative > 5e-6):
        raise ArithmeticError('Induced-spin quadrature error exceeds the fixed relative screen')
    return PreparedEventReference(*(readonly_copy(x) for x in (
        view.mass_1_source, view.mass_ratio, view.redshift, inputs[3], density,
        np.log(spin[:,0]), relative, view.log_detector_component_distance_jacobian)),
        distance_mode=prior.distance_prior.mode, analysis_cosmology=prior.analysis.name, a_max=float(a_max))


def evaluate_event_ratio(prepared, log_mass_redshift, spin_model, *, displacement):
    """Use original chi_eff in p0 even when the target marginal is stressed."""
    if not isinstance(prepared, PreparedEventReference):
        raise TypeError('Prepared event reference required')
    if spin_model.a_max != prepared.a_max:
        raise ValueError('Target and native PE reference spin bounds differ')
    spin = stressed_logpdf(spin_model, prepared.mass_1_source, prepared.chi_eff, displacement)
    return factorized_log_ratio(log_mass_redshift, spin, prepared.log_pe_mass_distance,
                                prepared.log_spin_reference)
