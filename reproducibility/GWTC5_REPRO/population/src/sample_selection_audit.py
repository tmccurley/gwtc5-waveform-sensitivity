"""Event membership and fixed-stratum covariance references, not production gates.

The real injections used stopped sampling and structured time assignments.
Consequently, the independent fixed-N covariance implemented here is a
reference calculation, not certification of the actual campaign's variance.
"""

import math
import re
import numpy as np
from scipy.special import logsumexp

SEARCHES = ('CWB', 'gstlal', 'MBTA', 'pycbc')
YEAR_SECONDS = 31_557_600.
O4B_START, O4B_END = 1396969218., 1422118818.


def _unique_rows(rows):
    result = {}
    for row in rows:
        name = row['gw_name']
        if not isinstance(name, str) or not re.fullmatch(r'GW\d{6}_\d{6}', name):
            raise ValueError('Invalid candidate name')
        if name in result:
            raise ValueError('Duplicate candidate name')
        result[name] = row
    return result


def candidate_membership(summary, per_search):
    """Reconstruct the search union in Hz and retain every summary candidate.

    Missing search rows mean not reported, not FAR=0. A reported NaN is an
    error. No-data DQ labels are retained and never promoted to a DQ pass.
    """
    if set(per_search) != set(SEARCHES):
        raise ValueError('Require exactly the four documented searches')
    combined = _unique_rows(summary)
    searches = {name: _unique_rows(per_search[name]) for name in SEARCHES}
    for name, rows in searches.items():
        if set(rows) - set(combined):
            raise ValueError('Search candidate absent from summary: ' + name)
        for row in rows.values():
            value = float(row['combined_far'])
            if not math.isfinite(value) or value < 0:
                raise ValueError('Invalid reported FAR')
    result = []
    for name, row in sorted(combined.items()):
        far, gps = float(row['far']), float(row['gps_time'])
        if not math.isfinite(far) or far < 0 or not math.isfinite(gps):
            raise ValueError('Invalid summary significance or time')
        values = {s: float(rows[name]['combined_far']) if name in rows else None
                  for s, rows in searches.items()}
        reported = {s: x for s, x in values.items() if x is not None}
        if not reported or not math.isclose(min(reported.values()), far, rel_tol=1e-12, abs_tol=0):
            raise ValueError('Summary FAR differs from search union')
        if row['pipeline'] not in reported or not math.isclose(
                reported[row['pipeline']], far, rel_tol=1e-12, abs_tol=0):
            raise ValueError('Named best pipeline is inconsistent')
        if set(row['pipelines'].split(',')) != set(reported):
            raise ValueError('Reported pipeline membership differs')
        if row['dq_status'] not in ('no_data', 'DQ_veto'):
            raise ValueError('Unreviewed DQ label')
        in_time = O4B_START <= gps < O4B_END
        passes_far = far < 1 / YEAR_SECONDS
        veto = row['dq_status'] == 'DQ_veto'
        reasons = []
        if not in_time:
            reasons.append('outside_O4b_interval')
        if not passes_far:
            reasons.append('does_not_pass_strict_FAR')
        if veto:
            reasons.append('explicit_DQ_veto')
        result.append(dict(event=name, gps_time=gps, far_hz=far,
                           far_per_year=far * YEAR_SECONDS, per_search_far_hz=values,
                           best_pipeline=row['pipeline'], dq_status=row['dq_status'],
                           independently_verified_DQ=False, in_O4b=in_time,
                           passes_strict_FAR=passes_far, eligible=not reasons,
                           reasons=reasons))
    return result


def parse_months(readme):
    """Extract the explicitly published O4b generated-count table, not a fit."""
    rows = re.findall(r'\* (\w+ \d{4}) \((\d+) - (\d+)\) : (\d+)', readme)
    months = [dict(month=name, start=int(start), end=int(end), generated=int(n))
              for name, start, end, n in rows if int(start) >= 1393286656]
    if not months or len({m['month'] for m in months}) != len(months):
        raise ValueError('Missing or duplicate monthly count records')
    for index, month in enumerate(months):
        if month['start'] >= month['end'] or month['generated'] <= 1:
            raise ValueError('Invalid month interval or count')
        if index and months[index-1]['end'] != month['start']:
            raise ValueError('Monthly intervals must be contiguous and ordered')
    if months[0]['start'] > O4B_START or months[-1]['end'] < O4B_END:
        raise ValueError('Monthly coverage does not contain O4b')
    return months


def assign_months(times, months):
    """Assign by exact published half-open intervals; no calendar rounding."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError('Times must be a finite vector')
    result = np.full(len(times), -1, dtype=int)
    for index, month in enumerate(months):
        mask = (times >= month['start']) & (times < month['end'])
        if np.any(result[mask] != -1):
            raise ValueError('Overlapping monthly intervals')
        result[mask] = index
    if np.any(result < 0):
        raise ValueError('Unassigned injection time')
    return result


def fixed_stratum_reference(log_weights, strata, generated):
    """Relative covariance for a *hypothetical fixed-N independent* design.

    Unstored/unrecovered draws are zero contributions and remain in each N_h.
    Columns share the same injections; cross-covariance must be retained when
    comparing two target populations. We normalize columns first to avoid
    overflowing raw density ratios. The matrix is covariance divided by the
    empirical column means, also the first-order log-estimate covariance.
    """
    logs = np.asarray(log_weights, dtype=float)
    strata = np.asarray(strata)
    generated = np.asarray(generated)
    if (logs.ndim != 2 or logs.shape[1] == 0 or strata.shape != (logs.shape[0],)
            or not np.issubdtype(strata.dtype, np.integer)
            or generated.ndim != 1 or generated.size == 0
            or not np.issubdtype(generated.dtype, np.integer)
            or np.any(generated <= 1) or np.any(strata < 0)
            or np.any(strata >= generated.size)
            or np.isnan(logs).any() or np.isposinf(logs).any()):
        raise ValueError('Invalid weight, stratum or generated-count input')
    totals = logsumexp(logs, axis=0)
    if not np.isfinite(totals).all():
        raise ValueError('Every case needs a nonzero resolved sum')
    b = np.exp(logs - totals)
    covariance = np.zeros((logs.shape[1], logs.shape[1]))
    uncentered = np.zeros_like(covariance)
    monthly = []
    for h, count in enumerate(generated):
        subset = b[strata == h]
        if len(subset) > count:
            raise ValueError('Stored rows exceed generated trials')
        sums = subset.sum(axis=0)
        products = subset.T @ subset
        term = (products - np.outer(sums, sums) / count) * count / (count - 1)
        covariance += term
        uncentered += products
        monthly.append(dict(stratum=h, generated=int(count), stored=len(subset),
                            fraction_of_sum=sums.tolist(),
                            covariance_contribution=term.tolist()))
    covariance = (covariance + covariance.T) / 2
    if np.linalg.eigvalsh(covariance).min() < -1e-14:
        raise ValueError('Covariance is not positive semidefinite within precision')
    return dict(log_sums=totals.tolist(), relative_covariance=covariance.tolist(),
                uncentered_reference=uncentered.tolist(), monthly=monthly,
                actual_campaign_variance_validated=False, production_ready=False,
                assumptions='independent draws within fixed-size strata; zero unrecorded contributions')


def log_ratio_reference_sd(relative_covariance, index, baseline):
    """Delta-method reference SD with common-injection covariance included."""
    matrix = np.asarray(relative_covariance, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=0, atol=1e-14)):
        raise ValueError('Invalid covariance matrix')
    variance = float(matrix[index, index] + matrix[baseline, baseline] - 2 * matrix[index, baseline])
    if variance < -1e-14:
        raise ValueError('Negative reference variance')
    # Only roundoff in a cancellation of identical columns is set to zero.
    return math.sqrt(max(0., variance))
