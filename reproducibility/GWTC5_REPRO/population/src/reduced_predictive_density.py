"""Full-support conditional predictive mixtures, without a new fit or bands."""

import math
import numpy as np
from src.boundary_regularized_population import log_boundary_window


def predictive_curve(settings, j0, masses, chi, *, batch_size=256, budget=None):
    """Average normalized densities over every original hyperparameter node.

    Zero quadrature masses remain in the matrix. No fitted-region pruning,
    posterior renormalization or replacement with mean coordinates occurs.
    Display-density tail underflow does not establish mathematical zeros.
    """
    p, norm, q, x = (np.asarray(v, float) for v in (settings, j0, masses, chi))
    if p.ndim != 2 or p.shape[1] != 2 or not len(p) or norm.shape != (len(p),) or q.ndim != 2 or q.shape[0] != len(p):
        raise ValueError('Matching complete settings, normalizers and mass columns required')
    if not np.isfinite(p).all() or np.any(abs(p[:, 0]) > 1) or np.any(p[:, 1] < -2) or np.any(p[:, 1] > 1):
        raise ValueError('Settings outside unchanged broad prior')
    if not np.isfinite(norm).all() or np.any(norm <= 0) or not np.isfinite(q).all() or np.any(q < 0) or np.any(abs(q.sum(axis=0)-1) > 1e-10):
        raise ValueError('Positive normalizers and original normalized masses required')
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0) or x[0] != -.99 or x[-1] != .99:
        raise ValueError('Increasing full physical support grid required')
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('Positive integer batch size required')
    window = log_boundary_window(x/.99, .05/.99)
    result = np.zeros((len(x), q.shape[1]))
    for first in range(0, len(p), batch_size):
        if budget:
            budget()
        section = p[first:first+batch_size]
        density = np.exp(window[None, :]-.5*((x[None, :]/.99-section[:, 0, None])/10.**section[:, 1, None])**2
                         -np.log(.99*norm[first:first+batch_size, None]))
        result += density.T @ q[first:first+batch_size]
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ArithmeticError('Invalid predictive density; no substitution permitted')
    return result


def curve_moments(chi, curves):
    """Check unrenormalized curve integrals and centered predictive variance."""
    x, f = np.asarray(chi, float), np.asarray(curves, float)
    if x.ndim != 1 or len(x) < 3 or f.ndim != 2 or f.shape[0] != len(x) or not np.isfinite(x).all() or not np.isfinite(f).all() or np.any(np.diff(x) <= 0) or np.any(f < 0):
        raise ValueError('Finite ordered grid and nonnegative curve columns required')
    norm = np.trapz(f, x, axis=0)
    mean = np.trapz(x[:, None]*f, x, axis=0)
    variance = np.trapz((x[:, None]-mean)**2*f, x, axis=0)
    if np.any(norm <= 0) or np.any(variance <= 0) or not np.isfinite(variance).all():
        raise ArithmeticError('Predictive normalization/variance unavailable; no floor')
    return dict(integrals=norm.tolist(), predictive_means=mean.tolist(), predictive_sds=np.sqrt(variance).tolist(),
                curve_renormalized=False)


def curve_checks(moments, expected_means, expected_sds):
    errors = np.array([abs(np.asarray(moments['integrals'])-1),
                       abs(np.asarray(moments['predictive_means'])-expected_means),
                       abs(np.asarray(moments['predictive_sds'])-expected_sds)])
    return dict(normalization_errors=errors[0].tolist(), mean_errors=errors[1].tolist(), sd_errors=errors[2].tolist(),
                numerical_moment_pass=bool(np.isfinite(errors).all() and np.all(errors <= 1e-6)))


def curve_refinement(coarse_x, coarse, fine_x, fine):
    """Pointwise and integrated empirical representation screens, not calibration."""
    x, a, y, b = (np.asarray(v, float) for v in (coarse_x, coarse, fine_x, fine))
    if a.ndim != 2 or b.ndim != 2 or a.shape != (len(x), b.shape[1]) or b.shape[0] != len(y):
        raise ValueError('Matching curve columns required')
    if len(y) == len(x):
        np.testing.assert_array_equal(x, y)
        common, interpolated = b, a
    elif len(y) == 2*len(x)-1:
        np.testing.assert_array_equal(x, y[::2])
        common = b[::2]
        interpolated = np.column_stack([np.interp(y, x, a[:, c]) for c in range(a.shape[1])])
    else:
        raise ValueError('Equal or exactly nested curve grids required')
    peak = np.maximum(a.max(axis=0), b.max(axis=0))
    if not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0) or np.any(peak <= 0):
        raise ValueError('Finite nonnegative curves with positive peaks required')
    pointwise = abs(a-common).max(axis=0)/peak
    l1 = np.trapz(abs(interpolated-b), y, axis=0)
    return dict(max_pointwise_change_over_peak=pointwise.tolist(), integrated_absolute_curve_change=l1.tolist(),
                numerical_curve_refinement_pass=bool(np.all(pointwise <= .01) and np.all(l1 <= .01)))
