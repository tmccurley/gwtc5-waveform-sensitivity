"""Exact shared Gaussian-kernel sums for deterministic row-sensitivity checks.

Every group has its own actual row count. Groups are not IID replicates,
and their factors cannot recover support absent from the released rows.
"""

import math
import numpy as np
from scipy.special import logsumexp

GROUP_NAMES = ('full', 'prefix_half', 'prefix_three_quarters',
               'contiguous_0', 'contiguous_1', 'contiguous_2', 'contiguous_3',
               'interleaved_0', 'interleaved_1', 'interleaved_2', 'interleaved_3')


def group_geometry(n, row_indices):
    """Fixed array_split-compatible quarters and ordered prefixes."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 4:
        raise ValueError('At least four original rows required')
    rows = np.asarray(row_indices)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer) or np.any(rows < 0) or np.any(rows >= n):
        raise ValueError('Original integer row indices required')
    counts = np.array([n//4+(k < n % 4) for k in range(4)], int)
    edges = np.r_[0, np.cumsum(counts)]
    masks = [np.ones(len(rows), bool), rows < n//2, rows < 3*n//4]
    masks += [(rows >= edges[k]) & (rows < edges[k+1]) for k in range(4)]
    masks += [rows % 4 == k for k in range(4)]
    sizes = np.array([n, n//2, 3*n//4, *counts, *counts], int)
    return masks, sizes


def grouped_kernel_log_factors(x, base, settings, *, row_chunk=65536, batch_size=16, budget=None):
    """Evaluate the Gaussian once per row/batch, retaining exact group sums."""
    x, base, settings = np.asarray(x, float), np.asarray(base, float), np.asarray(settings, float)
    if x.ndim != 1 or len(x) < 4 or base.shape != x.shape or not np.isfinite(x).all():
        raise ValueError('Matching original finite spin rows and fixed log factors required')
    if np.isnan(base).any() or np.isposinf(base).any():
        raise ValueError('Invalid fixed log factors')
    if settings.ndim != 2 or settings.shape[1] != 2 or not len(settings) or not np.isfinite(settings).all():
        raise ValueError('Finite nonempty (mu, log10_sigma) settings required')
    if np.any(abs(settings[:, 0]) > 1) or np.any(settings[:, 1] < -2) or np.any(settings[:, 1] > 1):
        raise ValueError('Settings outside unchanged broad domain')
    for size in (row_chunk, batch_size):
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError('Positive integer batch/chunk sizes required')
    _, sizes = group_geometry(len(x), np.array([], dtype=int))
    result = np.full((len(settings), 11), -np.inf)
    for first in range(0, len(settings), batch_size):
        pars = settings[first:first+batch_size]
        accumulated = np.full((len(pars), 11), -np.inf)
        for start in range(0, len(x), row_chunk):
            if budget:
                budget()
            end = min(start+row_chunk, len(x))
            masks, _ = group_geometry(len(x), np.arange(start, end))
            d = (x[None, start:end]-pars[:, 0, None])/10.**pars[:, 1, None]
            logs = base[None, start:end]-.5*d*d
            for group, mask in enumerate(masks):
                if np.any(mask):
                    accumulated[:, group] = np.logaddexp(accumulated[:, group], logsumexp(logs[:, mask], axis=1))
        result[first:first+len(pars)] = accumulated-np.log(sizes)[None, :]
    if not np.isfinite(result).all():
        missing = np.argwhere(~np.isfinite(result)).tolist()
        raise ArithmeticError('Group factor unavailable; no zero likelihood substituted: '+str(missing[:20]))
    return result, sizes
