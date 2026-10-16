"""Exact empirical Gaussian-kernel sums and broad-prior tensor integration.

This does not repair posterior support or certify population inference.
Kernel factors omit a common spin normalization, which cancels only in
the matched, count-conditioned catalog likelihood.
"""

import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp


def prior_grid(order):
    """Nodes and normalized prior weights in (mu, log10_sigma), not sigma."""
    if isinstance(order, bool) or not isinstance(order, int) or order < 2:
        raise ValueError('Integer quadrature order >=2 required')
    nodes, weights = leggauss(order)
    mu, logsigma = np.meshgrid(nodes, 1.5*nodes-.5, indexing='ij')
    return np.column_stack((mu.ravel(), logsigma.ravel())), (np.outer(weights, weights)/4).ravel()


def kernel_log_factors(x, fixed_logs, settings, denominator, *, row_chunk=65536, batch_size=16):
    """Direct row sums, with bounded temporary memory and original denominators."""
    x, base, settings = np.asarray(x, float), np.asarray(fixed_logs, float), np.asarray(settings, float)
    if x.ndim != 1 or not x.size or base.shape != x.shape or not np.isfinite(x).all():
        raise ValueError('Finite spin coordinates and matching nonempty fixed factors required')
    if np.isnan(base).any() or np.isposinf(base).any():
        raise ValueError('Invalid fixed log factors')
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError('Original positive denominator required')
    if settings.ndim != 2 or settings.shape[1] != 2 or not len(settings) or not np.isfinite(settings).all():
        raise ValueError('Finite (mu, log10_sigma) settings required')
    if np.any(abs(settings[:, 0]) > 1) or np.any(settings[:, 1] < -2) or np.any(settings[:, 1] > 1):
        raise ValueError('Settings outside the unchanged broad domain')
    for size in (row_chunk, batch_size):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError('Positive integer chunk sizes required')
    result = np.full(len(settings), -np.inf)
    for start in range(0, len(settings), batch_size):
        pars = settings[start:start+batch_size]
        accumulated = np.full(len(pars), -np.inf)
        scales = 10.**pars[:, 1]
        for row in range(0, x.size, row_chunk):
            # No finite tail is floored; support zeros stay in the full count.
            d = (x[None, row:row+row_chunk]-pars[:, 0, None])/scales[:, None]
            logs = base[None, row:row+row_chunk]-.5*d*d
            accumulated = np.logaddexp(accumulated, logsumexp(logs, axis=1))
        result[start:start+batch_size] = accumulated-math.log(denominator)
    if not np.isfinite(result).all():
        raise ArithmeticError('At least one empirical factor has no positive support; nothing substituted')
    return result


def normalized_grid(settings, prior_weights, log_likelihood):
    """Quadrature masses and coordinate moments; no contrast coupling implied."""
    settings = np.asarray(settings, float)
    prior = np.asarray(prior_weights, float)
    logs = np.asarray(log_likelihood, float)
    if settings.ndim != 2 or settings.shape[1] != 2 or prior.shape != (len(settings),) or logs.shape != prior.shape:
        raise ValueError('Matching two-dimensional grid and weight vectors required')
    if not np.isfinite(settings).all() or not np.isfinite(prior).all() or np.any(prior <= 0):
        raise ValueError('Finite nodes and positive prior integration weights required')
    if abs(float(prior.sum())-1) > 1e-12 or np.isnan(logs).any() or np.isposinf(logs).any():
        raise ValueError('Normalized prior quadrature and valid log likelihood required')
    logz = float(logsumexp(logs+np.log(prior)))
    if not math.isfinite(logz):
        raise ArithmeticError('No positive finite posterior normalization')
    masses = np.exp(logs+np.log(prior)-logz)
    mean = masses @ settings
    variance = masses @ ((settings-mean)**2)
    if not np.isfinite(variance).all() or np.any(variance <= 0):
        raise ArithmeticError('Coordinate posterior SD unavailable')
    return dict(log_normalization=logz, mean=mean.tolist(), sd=np.sqrt(variance).tolist()), masses


def refinement_check(coarse, fine):
    """Predeclared empirical screens, separate from finite-sample precision."""
    a, b = np.asarray(coarse['mean']), np.asarray(fine['mean'])
    sa, sb = np.asarray(coarse['sd']), np.asarray(fine['sd'])
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in (a, b, sa, sb)) or np.any(sa <= 0) or np.any(sb <= 0):
        raise ValueError('Finite two-coordinate summaries with positive SDs required')
    logz_change = abs(coarse['log_normalization']-fine['log_normalization'])
    if not math.isfinite(logz_change):
        raise ValueError('Invalid normalization change')
    mean_change = abs(a-b)/np.maximum(sa, sb)
    sd_change = abs(sa-sb)/np.maximum(sa, sb)
    return dict(log_normalization_change=logz_change,
                mean_change_in_posterior_sd=mean_change.tolist(), relative_sd_change=sd_change.tolist(),
                empirical_refinement_pass=bool(logz_change <= .01 and np.all(mean_change <= .01) and np.all(sd_change <= .01)),
                sampling_accuracy_validated=False, all_modes_certified=False)
