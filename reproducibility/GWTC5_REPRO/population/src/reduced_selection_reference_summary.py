"""Posterior-weighted pointwise reference scale, not parameter uncertainty."""

import numpy as np


def reference_rms(shape_sds, masses):
    values, q = np.asarray(shape_sds, float), np.asarray(masses, float)
    if values.ndim != 2 or values.shape[1] != 2 or not len(values) or q.shape != (len(values),):
        raise ValueError('Two full-node reference columns and matching posterior masses required')
    if not np.isfinite(values).all() or np.any(values < 0) or not np.isfinite(q).all() or np.any(q < 0) or abs(float(q.sum())-1) > 1e-10:
        raise ValueError('Nonnegative finite references and normalized posterior masses required')
    variances = np.sum(q[:, None]*values**2, axis=0)
    if not np.isfinite(variances).all() or np.any(variances < 0):
        raise ArithmeticError('Invalid reference RMS; no floor substituted')
    return np.sqrt(variances)


def reference_rms_refinement(coarse, fine):
    a, b = np.asarray(coarse, float), np.asarray(fine, float)
    if a.shape != (2,) or b.shape != (2,) or not np.isfinite(a).all() or not np.isfinite(b).all() or np.any(a < 0) or np.any(b < 0):
        raise ValueError('Two finite nonnegative reference summaries required')
    scale = np.maximum(a, b)
    changes = np.divide(abs(a-b), scale, out=np.zeros(2), where=scale > 0)
    return dict(relative_reference_rms_change=changes.tolist(), numerical_refinement_pass=bool(np.all(changes <= .01)), actual_campaign_variance_validated=False)
