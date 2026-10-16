"""Checked assembly for the approved deterministic posterior-row exercise."""

import numpy as np
from scipy.special import logsumexp
from src.reduced_row_group_kernels import GROUP_NAMES, group_geometry

ROUTES = ('continuous', 'bilby_2.6.0_grid_reference')
ARM_ROUTES = tuple((route, arm) for route in ROUTES for arm in ('ST', 'XPNR'))
COLUMNS = tuple((group, route, arm) for group in GROUP_NAMES for route, arm in ARM_ROUTES)


def quarter_residuals(factors, counts, rows):
    """Check actual-denominator quarter reconstruction for every node/route."""
    values = np.asarray(factors, float)
    counts = np.asarray(counts)
    expected = group_geometry(rows, np.array([], dtype=int))[1]
    if values.ndim != 3 or values.shape[1:] != (11, 2) or not np.isfinite(values).all():
        raise ValueError('Finite node/group/route event factors required')
    if counts.shape != (11,) or not np.array_equal(counts, expected):
        raise ValueError('Original actual group counts differ')
    residuals = []
    for groups in (slice(3, 7), slice(7, 11)):
        full = logsumexp(values[:, groups, :]+np.log(counts[groups]/rows)[None, :, None], axis=1)
        residuals.append(float(np.max(abs(full-values[:, 0, :]))))
    if max(residuals) > 2e-8:
        raise ArithmeticError('Quarter averages do not reconstruct original full factor')
    return residuals


def assemble_catalog(factors, arms, selection, *, events_per_arm=104):
    """Group-major columns, with the same full selection for every group."""
    values, sel = np.asarray(factors, float), np.asarray(selection, float)
    if values.ndim != 4 or values.shape[2:] != (11, 2):
        raise ValueError('Product/node/group/route factor tensor required')
    if len(arms) != len(values) or sel.shape != (values.shape[1],):
        raise ValueError('Matching product arms and full nodewise selection required')
    if not np.isfinite(values).all() or not np.isfinite(sel).all():
        raise ArithmeticError('Nonfinite complete factors; no likelihood substituted')
    if any(arm not in ('ST', 'XPNR') for arm in arms):
        raise ValueError('Unknown catalog arm')
    if any(arms.count(arm) != events_per_arm for arm in ('ST', 'XPNR')):
        raise ValueError('All declared events must remain in each arm')
    result = np.empty((values.shape[1], len(COLUMNS)))
    for column, (group, route, arm) in enumerate(COLUMNS):
        mask = np.array([a == arm for a in arms])
        result[:, column] = values[mask, :, GROUP_NAMES.index(group), ROUTES.index(route)].sum(axis=0)-events_per_arm*sel
    if not np.isfinite(result).all():
        raise ArithmeticError('Invalid combined catalog likelihood')
    return result
