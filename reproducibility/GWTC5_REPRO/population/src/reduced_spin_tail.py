"""Log-domain tail integrals for the unchanged reduced windowed Gaussian.

These are model probabilities, not certified Monte Carlo error bounds.
All integration intervals remain represented, including deeply positive
tails whose linear floating-point probabilities would underflow.
"""

import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from src.boundary_regularized_population import log_boundary_window
from src.independent_mass_redshift import checked_quad


def _validated(settings, threshold, a_max, tail_width):
    pars = np.asarray(settings, float)
    if pars.ndim != 2 or pars.shape[1] != 2 or not len(pars) or not np.isfinite(pars).all():
        raise ValueError('Nonempty finite (mu, log10_sigma) settings required')
    if np.any(abs(pars[:, 0]) > 1) or np.any(pars[:, 1] < -2) or np.any(pars[:, 1] > 1):
        raise ValueError('Settings outside unchanged prior')
    if not all(math.isfinite(v) for v in (threshold, a_max, tail_width)) or not 0 < tail_width < a_max <= 1 or not 0 <= threshold <= a_max:
        raise ValueError('Explicit valid support, window and tail threshold required')
    return pars


def log_tail_kernel(settings, *, order=128, threshold=.965, a_max=.99,
                    tail_width=.05, batch_size=256, budget=None):
    """Integrate the unnormalized spin kernel over both complete tails.

    Subtract log(J0) from this result to obtain log(tau). There is no 1/A
    left in this ratio, because both integrals use x=chi_eff/A. Clamping
    an out-of-interval integration landmark only partitions the interval;
    it does not alter a spin sample, weight, prior or integration domain.
    """
    pars = _validated(settings, threshold, a_max, tail_width)
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (order, batch_size)) or order < 8:
        raise ValueError('Order >=8 and positive integer batch size required')
    if threshold == a_max:
        return np.full(len(pars), -np.inf)  # Exactly empty integration region.
    nodes, weights = leggauss(order)
    width, cut = tail_width/a_max, threshold/a_max
    result = np.empty(len(pars))
    for start in range(0, len(pars), batch_size):
        if budget:
            budget()
        p = pars[start:start+batch_size]
        mu, sigma = p[:, 0], 10.**p[:, 1]
        landmarks = np.column_stack((mu[:, None]+sigma[:, None]*np.array([-8, -3, -1, 0, 1, 3, 8]),
                                     np.broadcast_to([-1+width, 1-width], (len(p), 2))))
        sides = []
        for low, high in ((-1., -cut), (cut, 1.)):
            edges = np.sort(np.column_stack((np.full(len(p), low), np.full(len(p), high),
                                             np.minimum(high, np.maximum(low, landmarks)))), axis=1)
            half = np.diff(edges, axis=1)/2
            x = edges[:, :-1, None]+half[:, :, None]*(1+nodes)
            with np.errstate(divide='ignore'):
                terms = (log_boundary_window(x, width)
                         -.5*((x-mu[:, None, None])/sigma[:, None, None])**2
                         +np.log(half[:, :, None])+np.log(weights))
            sides.append(logsumexp(terms, axis=(1, 2)))
        values = np.logaddexp(*sides)
        if not np.isfinite(values).all():
            raise ArithmeticError('Positive tail integral unavailable; no zero substituted')
        result[start:start+len(p)] = values
    return result


def adaptive_log_tail(setting, *, threshold=.965, a_max=.99, tail_width=.05):
    """Independent adaptive reference, rescaling each interval before quadrature."""
    p = _validated([setting], threshold, a_max, tail_width)[0]
    if threshold == a_max:
        return -math.inf
    mu, sigma, width = p[0], 10.**p[1], tail_width/a_max
    landmarks = [mu+k*sigma for k in (-8, -3, -1, 0, 1, 3, 8)] + [-1+width, 1-width]

    def integral(low, high):
        edges = sorted({low, high, *(v for v in landmarks if low < v < high)})
        logs = []
        for left, right in zip(edges[:-1], edges[1:]):
            peak_x = min(right, max(left, mu))
            peak = -.5*((peak_x-mu)/sigma)**2
            value = checked_quad(lambda x: math.exp(float(log_boundary_window(x, width))
                                                   -.5*((x-mu)/sigma)**2-peak),
                                 left, right, epsabs=0, epsrel=1e-11)
            if value <= 0:
                raise ArithmeticError('Positive scaled adaptive integral unavailable')
            logs.append(peak+math.log(value))
        return float(logsumexp(logs))

    cut = threshold/a_max
    return float(np.logaddexp(integral(-1., -cut), integral(cut, 1.))-integral(-1., 1.))


def tail_summary(log_probabilities, empirical_log_selection, posterior_mass):
    """Conditional plug-in sensitivity; the empirical factor is not exact alpha."""
    tail, sel, q = (np.asarray(v, float) for v in
                    (log_probabilities, empirical_log_selection, posterior_mass))
    if tail.ndim != 1 or not len(tail) or sel.shape != tail.shape or q.shape != tail.shape:
        raise ValueError('Matching complete one-dimensional arrays required')
    if not np.isfinite(tail).all() or np.any(tail > 1e-10) or not np.isfinite(sel).all() or not np.isfinite(q).all() or np.any(q < 0) or abs(float(q.sum())-1) > 1e-10:
        raise ValueError('Finite probabilities/factors and normalized posterior masses required')
    penalty = 104*np.logaddexp(0., tail-sel)
    with np.errstate(divide='ignore'):
        weighted_log_tail = float(logsumexp(np.log(q)+tail))
    average = float(q @ penalty)
    if not math.isfinite(average) or not math.isfinite(weighted_log_tail):
        raise ArithmeticError('Tail summary unavailable')
    return dict(posterior_log_mean_tail_probability=weighted_log_tail,
                posterior_mean_plugin_log_penalty=average,
                posterior_mass_above_plugin_penalty={str(c): float(q[penalty > c].sum()) for c in (.01, .1, 1.)},
                full_prior_max_log_tail_probability=float(tail.max()),
                full_prior_max_plugin_log_penalty=float(penalty.max()),
                empirical_selection_factor_above_one_nodes=int(np.count_nonzero(sel > 0)),
                certified_likelihood_error_bound=False,
                missing_support_recovery_validated=False), penalty
