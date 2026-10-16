"""Validated numerical primitives; not the official GWTC-5 population likelihood.

All inputs here are synthetic in the checkpoint tests. Production integration
requires source-verified density measures, priors, selection and rate treatment.
"""

from __future__ import annotations

import math
import numpy as np
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance


def _log_vector(values, name="log weights"):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise ValueError(f"{name} contains NaN or positive infinity")
    return values


def _finite_vector(values, name="samples"):
    values = _log_vector(values, name)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def logmeanexp(log_weights):
    """Log mean retaining valid zero weights (-inf) in the original count."""
    values = _log_vector(log_weights)
    return float(logsumexp(values) - math.log(values.size))


def importance_ess(log_weights):
    """Kish weight ESS; this is not a chain ESS or independent-row count.

    Center before doubling the log weights to avoid arithmetic overflow when
    log values are huge. All-zero weights have no support and return zero.
    """
    values = _log_vector(log_weights)
    maximum = float(np.max(values))
    if maximum == -math.inf:
        return 0.0
    with np.errstate(over="ignore", under="ignore"):
        shifted = values - maximum
        value = float(np.exp(2 * logsumexp(shifted) - logsumexp(2 * shifted)))
    return min(float(values.size), max(1.0, value))


def event_log_likelihood(log_population, log_pe_prior):
    """Posterior recycling using densities in the same validated measure.

    A nonfinite PE prior for a retained row is an input error. A population
    density of zero is valid and contributes zero to the mean. The full
    original number of posterior rows remains the denominator.
    """
    pop = _log_vector(log_population, "log population density")
    prior = _finite_vector(log_pe_prior, "log PE prior density")
    if pop.shape != prior.shape:
        raise ValueError("Population and PE prior arrays have different shapes")
    with np.errstate(over="raise", invalid="raise"):
        log_weights = pop - prior
    return logmeanexp(log_weights), importance_ess(log_weights)


def selection_log_estimate(recovered_log_weights, total_generated):
    """Generic injection estimator, divided by ALL generated injections.

    The recovered array can be empty. This primitive omits observing-time,
    mixture-proposal and VT conventions; the official estimator must supply
    those correctly before production use.
    """
    if isinstance(total_generated, bool) or not isinstance(total_generated, (int, np.integer)) or total_generated <= 0:
        raise ValueError("total_generated must be a positive integer")
    raw = np.asarray(recovered_log_weights, dtype=float)
    if raw.ndim != 1 or raw.size > total_generated:
        raise ValueError("Invalid recovered injection count or shape")
    if raw.size == 0:
        return -math.inf, 0.0
    values = _log_vector(raw)
    return float(logsumexp(values) - math.log(total_generated)), importance_ess(values)


def shape_log_likelihood(event_log_factors, log_efficiency):
    """Shape likelihood for a verified log-uniform, marginalized rate ONLY."""
    factors = _log_vector(event_log_factors, "event log factors")
    if not math.isfinite(log_efficiency):
        raise ValueError("Selection estimate must be finite and positive")
    return float(np.sum(factors) - factors.size * log_efficiency)


def loo_shape_log_weights(event_log_factors, log_efficiencies):
    """Remove an event: multiply the original shape posterior by xi/L_i.

    Inputs are evaluated at hyperposterior draws of ONE arm. Apply separately
    to each arm and check overlap. Subtracting its waveform delta is not LOO.
    """
    event = _finite_vector(event_log_factors)
    selection = _finite_vector(log_efficiencies)
    if event.shape != selection.shape:
        raise ValueError("LOO arrays must have matching shapes")
    return selection - event


def hyper_nw1(x, y):
    """Empirical W1 divided by the mean central-90% posterior width."""
    x, y = _finite_vector(x), _finite_vector(y)
    width = float((np.diff(np.quantile(x, [0.05, 0.95]))[0] +
                   np.diff(np.quantile(y, [0.05, 0.95]))[0]) / 2)
    if width <= 0:
        raise ValueError("Normalized distance undefined for zero mean width")
    return float(wasserstein_distance(x, y) / width)


def interval_overlap90(x, y):
    """Length of intersection / length of union, not a significance test."""
    a, b = np.quantile(_finite_vector(x), [0.05, 0.95])
    c, d = np.quantile(_finite_vector(y), [0.05, 0.95])
    intersection = max(0.0, min(b, d) - max(a, c))
    union = (b - a) + (d - c) - intersection
    if union <= 0:
        raise ValueError("Overlap undefined for two zero-width intervals")
    return float(intersection / union)


def tilt_summary(g, delta_by_event, t):
    """Weighted mean and local event covariances along an exponential tilt.

    Rows are equal-weight baseline hyperposterior draws; columns are event
    perturbations. This function cannot detect missing baseline modes. Its
    covariances decompose the derivative of the mean, not a median or W1.
    """
    g = _finite_vector(g, "derived quantity")
    delta = np.asarray(delta_by_event, dtype=float)
    if delta.ndim != 2 or delta.shape[0] != g.size or not np.all(np.isfinite(delta)):
        raise ValueError("delta must be finite draws-by-events with matching rows")
    if not math.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("t must be between zero and one")
    total = delta.sum(axis=1)
    log_weights = t * total
    weights = np.exp(log_weights - logsumexp(log_weights))
    mean = float(weights @ g)
    delta_means = weights @ delta
    contributions = np.sum(weights[:, None] * (g - mean)[:, None] *
                           (delta - delta_means), axis=0)
    return {"mean": mean, "event_covariances": contributions,
            "importance_ess": importance_ess(log_weights)}
