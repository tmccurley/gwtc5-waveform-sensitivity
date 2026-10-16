"""Exact two-column reference driver, without a new sampling-dependence model."""

import numpy as np
from src.sample_selection_audit import fixed_stratum_reference, log_ratio_reference_sd
from src.stopped_campaign_reference import stopped_stratum_reference


def paired_selection_reference(x, base, settings, anchor_logs, strata, generated, accepted):
    """Return anchor/setting reference covariance and catalog-shape scales.

    Gaussian normalization cancels from relative covariance through column
    scaling invariance. These unnormalized sums are not physical efficiencies.
    Every original recovered row and monthly zero-count convention remains.
    """
    x, base, anchor = np.asarray(x, float), np.asarray(base, float), np.asarray(anchor_logs, float)
    pars = np.asarray(settings, float)
    if x.ndim != 1 or not x.size or base.shape != x.shape or anchor.shape != x.shape or not np.isfinite(x).all():
        raise ValueError('Matching original finite spin rows and fixed factors required')
    if np.isnan(base).any() or np.isposinf(base).any() or np.isnan(anchor).any() or np.isposinf(anchor).any():
        raise ValueError('Invalid fixed/anchor log weights')
    if pars.shape != (2,) or not np.isfinite(pars).all() or abs(pars[0]) > 1 or not -2 <= pars[1] <= 1:
        raise ValueError('One unchanged-domain Gaussian setting required')
    logs = np.column_stack((anchor, base-.5*((x-pars[0])/10.**pars[1])**2))
    fixed = fixed_stratum_reference(logs, strata, generated)
    stopped = stopped_stratum_reference(logs, strata, generated, accepted)
    return dict(fixed_relative_covariance=np.array(fixed['relative_covariance']),
        stopped_relative_covariance=np.array(stopped['relative_covariance']),
        fixed_catalog_shape_reference_sd=104*log_ratio_reference_sd(fixed['relative_covariance'], 1, 0),
        stopped_catalog_shape_reference_sd=104*log_ratio_reference_sd(stopped['relative_covariance'], 1, 0),
        actual_campaign_variance_validated=False)
