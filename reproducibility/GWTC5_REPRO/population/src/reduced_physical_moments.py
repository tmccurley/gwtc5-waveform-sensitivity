"""Physical spin moments of the unchanged reduced windowed Gaussian.

This module postprocesses a hyperposterior. It neither evaluates a new
observed likelihood nor validates finite posterior or selection sampling.
"""

import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from src.boundary_regularized_population import log_boundary_window


def physical_moments(settings, *, order=64, a_max=.99, tail_width=.05,
                     batch_size=256, budget=None):
    """Integrate full spin support, resolving Gaussian features and window joins.

    Endpoints assigned to out-of-support landmarks only partition the
    integration domain. No Gaussian tail or hyperparameter node is removed.
    Centered variance avoids cancellation at narrow endpoint populations.
    """
    pars = np.asarray(settings, float)
    if pars.ndim != 2 or pars.shape[1] != 2 or not len(pars) or not np.isfinite(pars).all():
        raise ValueError('Finite nonempty (mu, log10_sigma) settings required')
    if np.any(abs(pars[:, 0]) > 1) or np.any(pars[:, 1] < -2) or np.any(pars[:, 1] > 1):
        raise ValueError('Settings outside unchanged broad prior')
    if not (math.isfinite(a_max) and math.isfinite(tail_width) and 0 < tail_width < a_max <= 1):
        raise ValueError('Require 0 < physical window width < a_max <= 1')
    for value in (order, batch_size):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('Positive integer integration and batch sizes required')
    if order < 8:
        raise ValueError('Spin quadrature order >=8 required')
    nodes, weights = leggauss(order)
    width = tail_width/a_max
    result = np.empty((len(pars), 3))
    for start in range(0, len(pars), batch_size):
        if budget:
            budget()
        subset = pars[start:start+batch_size]
        mu, sigma = subset[:, 0], 10.**subset[:, 1]
        features = mu[:, None]+sigma[:, None]*np.array([-8, -3, -1, 0, 1, 3, 8])
        features = np.minimum(1., np.maximum(-1., features))
        fixed = np.broadcast_to([-1., -1.+width, 1.-width, 1.], (len(subset), 4))
        edges = np.sort(np.column_stack((fixed, features)), axis=1)
        halfspan = np.diff(edges, axis=1)/2
        centers = edges[:, :-1]+halfspan
        x = centers[:, :, None]+halfspan[:, :, None]*nodes
        kernel = np.exp(log_boundary_window(x, width)-.5*((x-mu[:, None, None])/sigma[:, None, None])**2)
        weighted = kernel*halfspan[:, :, None]*weights
        j0 = weighted.sum(axis=(1, 2))
        if not np.isfinite(j0).all() or np.any(j0 <= 0):
            raise ArithmeticError('Spin normalization unavailable; nothing substituted')
        mean_x = (weighted*x).sum(axis=(1, 2))/j0
        variance_x = (weighted*(x-mean_x[:, None, None])**2).sum(axis=(1, 2))/j0
        if not np.isfinite(mean_x).all() or not np.isfinite(variance_x).all() or np.any(variance_x <= 0):
            raise ArithmeticError('Physical spin variance unavailable; nothing floored')
        result[start:start+len(subset)] = np.column_stack((j0, a_max*mean_x, a_max*np.sqrt(variance_x)))
    return result


def summarize_physical(values, masses):
    """Separate hyperparameter uncertainty, intrinsic width and predictive width."""
    v, p = np.asarray(values, float), np.asarray(masses, float)
    if v.ndim != 2 or v.shape[1] != 3 or p.shape != (len(v),) or not len(v):
        raise ValueError('Matching physical values and posterior masses required')
    if not np.isfinite(v).all() or np.any(v[:, 0] <= 0) or np.any(v[:, 2] <= 0):
        raise ValueError('Finite physical moments with positive normalizations/SDs required')
    if not np.isfinite(p).all() or np.any(p < 0) or abs(float(p.sum())-1) > 1e-10:
        raise ValueError('Normalized nonnegative posterior quadrature masses required')
    # Correct only representational normalization roundoff, not the prior,
    # likelihood, sampled support or any importance weight.
    p = p/p.sum()
    physical = v[:, 1:]
    mean = p @ physical
    variances = p @ ((physical-mean)**2)
    if np.any(variances <= 0) or not np.isfinite(variances).all():
        raise ArithmeticError('Posterior uncertainty of physical moments unavailable')
    average_intrinsic_variance = float(p @ (v[:, 2]**2))
    predictive_variance = average_intrinsic_variance+float(variances[0])
    return dict(physical_functionals=['population_mean', 'intrinsic_population_sd'],
                posterior_expectations=mean.tolist(), posterior_sds=np.sqrt(variances).tolist(),
                posterior_average_intrinsic_variance=average_intrinsic_variance,
                predictive_mean=float(mean[0]), predictive_sd=math.sqrt(predictive_variance),
                physical_quantiles_certified=False, density_curves_certified=False,
                finite_posterior_accuracy_validated=False, selection_variance_validated=False)


def physical_refinement(coarse, fine):
    """Apply the declared empirical representation screens to physical functionals."""
    ma, mb = np.asarray(coarse['posterior_expectations']), np.asarray(fine['posterior_expectations'])
    sa, sb = np.asarray(coarse['posterior_sds']), np.asarray(fine['posterior_sds'])
    if any(a.shape != (2,) or not np.isfinite(a).all() for a in (ma, mb, sa, sb)) or np.any(sa <= 0) or np.any(sb <= 0):
        raise ValueError('Two finite physical functionals with positive posterior SD required')
    changes = abs(ma-mb)/np.maximum(sa, sb)
    sd_changes = abs(sa-sb)/np.maximum(sa, sb)
    return dict(mean_change_in_posterior_sd=changes.tolist(), relative_posterior_sd_change=sd_changes.tolist(),
                empirical_physical_refinement_pass=bool(np.all(changes <= .01) and np.all(sd_changes <= .01)),
                predictive_mean_change=abs(coarse['predictive_mean']-fine['predictive_mean']),
                predictive_sd_change=abs(coarse['predictive_sd']-fine['predictive_sd']),
                scientific_result_validated=False)
