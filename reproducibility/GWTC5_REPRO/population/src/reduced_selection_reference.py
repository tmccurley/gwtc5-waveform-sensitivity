"""Independent arithmetic replay of existing conditional campaign references."""

import numpy as np


def replay_covariance_references(log_weights, strata, generated, accepted):
    """Max-anchored accepted-mark moments, keeping unrecorded zeros in counts.

    This replays the fixed/stopped reference algebra. It adds no validated
    real-campaign dependence model and never changes selection point estimates.
    """
    logs = np.asarray(log_weights, float)
    h, counts, successes = np.asarray(strata), np.asarray(generated), np.asarray(accepted)
    if logs.ndim != 2 or not logs.shape[0] or not logs.shape[1] or np.isnan(logs).any() or np.isposinf(logs).any():
        raise ValueError('Valid nonempty log-weight matrix required')
    if h.shape != (len(logs),) or not np.issubdtype(h.dtype, np.integer):
        raise ValueError('Integer row stratum IDs required')
    if counts.ndim != 1 or not counts.size or successes.shape != counts.shape or any(not np.issubdtype(a.dtype, np.integer) for a in (counts, successes)):
        raise ValueError('Actual integer generated/accepted reference counts required')
    if np.any(successes < 3) or np.any(counts < successes) or np.any(h < 0) or np.any(h >= len(counts)):
        raise ValueError('Invalid reference count ledger')
    anchors = logs.max(axis=0)
    if not np.isfinite(anchors).all():
        raise ArithmeticError('Every complete selection column needs positive support')
    weights = np.exp(logs-anchors)
    totals = weights.sum(axis=0)
    fixed = np.zeros((logs.shape[1], logs.shape[1]))
    stopped = np.zeros_like(fixed)
    corrected = np.zeros(logs.shape[1])
    for index, (n, r) in enumerate(zip(counts, successes)):
        n, r = int(n), int(r)
        subset = weights[h == index]
        if len(subset) > r:
            raise ValueError('Stored recovered rows exceed accepted reference count')
        sums, products = subset.sum(axis=0), subset.T @ subset
        fixed += n/(n-1)*(products-np.outer(sums, sums)/n)
        # Accepted marks Y=N*v restore zero marks through the full r.
        mean = n*sums/r
        sample_covariance = n*n*(products-np.outer(sums, sums)/r)/(r-1)
        a = (r-1)/(n-1)
        q = (r-1)*(r-2)/((n-1)*(n-2))
        delta = (r-1)*(n-r)/((n-1)**2*(n-2))
        corrected += a*mean
        stopped += delta*np.outer(mean, mean)+q*sample_covariance/r
    if np.any(corrected <= 0):
        raise ArithmeticError('Positive corrected reference sums required')
    return dict(fixed_relative_covariance=fixed/np.outer(totals, totals),
                stopped_relative_covariance=stopped/np.outer(corrected, corrected),
                point_multipliers=corrected/totals,
                log_uncorrected_sums=anchors+np.log(totals))
