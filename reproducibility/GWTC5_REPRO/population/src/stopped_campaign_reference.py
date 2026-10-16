"""Exact reference for independent inverse-binomial sampling, not actual-design certification."""

import numpy as np
from scipy.special import logsumexp


def stopped_stratum_reference(log_weights, strata, generated, accepted):
    """Reference covariance for the unbiased stopped estimator in each stratum.

    Inputs are the published-style per-row contributions proportional to Y/N,
    not Y/r. Unknown accepted zero marks are restored algebraically. Returned
    covariance is relative to the corrected reference estimate. No caller's
    weights or previous point estimates are changed.
    """
    logs=np.asarray(log_weights,float);h=np.asarray(strata);n=np.asarray(generated);r=np.asarray(accepted)
    if (logs.ndim!=2 or not logs.shape[1] or h.shape!=(logs.shape[0],)
            or not np.issubdtype(h.dtype,np.integer) or n.ndim!=1 or not n.size
            or r.shape!=n.shape or not np.issubdtype(n.dtype,np.integer)
            or not np.issubdtype(r.dtype,np.integer) or np.any(r<3) or np.any(n<r)
            or np.any(h<0) or np.any(h>=len(n)) or np.isnan(logs).any() or np.isposinf(logs).any()):
        raise ValueError('Invalid stopped-sampling inputs')
    totals=logsumexp(logs,axis=0)
    if not np.isfinite(totals).all():raise ValueError('Each case needs a nonzero contribution')
    b=np.exp(logs-totals);reference_sum=np.zeros(logs.shape[1]);covariance=np.zeros((logs.shape[1],logs.shape[1]))
    months=[]
    for index,(count,successes) in enumerate(zip(n,r)):
        count,successes=int(count),int(successes);v=b[h==index]
        if len(v)>successes:raise ValueError('Recovered rows exceed accepted count')
        s=v.sum(axis=0);products=v.T@v
        # For A=(r-1)/(N-1), E[A]=p and E[Q]=p^2. The positive
        # expression for A^2-Q avoids subtraction of nearly equal terms.
        A=(successes-1)/(count-1)
        Q=((successes-1)*(successes-2))/((count-1)*(count-2))
        delta=((successes-1)*(count-successes))/((count-1)**2*(count-2))
        multiplier=count*A/successes
        scatter=products-np.outer(s,s)/successes
        term=count**2*(delta*np.outer(s,s)/successes**2+Q*scatter/(successes*(successes-1)))
        covariance+=term;reference_sum+=multiplier*s
        months.append(dict(stratum=index,generated=count,accepted=successes,stored=len(v),
            accepted_count_is_caller_supplied=True,point_multiplier_vs_published=multiplier,
            uncorrected_fraction_of_sum=s.tolist(),unbiased_acceptance_estimate=A,
            unbiased_squared_acceptance_estimate=Q,positive_A_squared_minus_Q=delta,
            covariance_scaled_by_uncorrected_totals=term.tolist()))
    if np.any(reference_sum<=0):raise ValueError('Corrected reference sum is zero')
    relative=covariance/np.outer(reference_sum,reference_sum)
    relative=(relative+relative.T)/2
    if np.linalg.eigvalsh(relative).min()<-1e-14:raise ValueError('Covariance is not positive semidefinite')
    return dict(log_uncorrected_sums=totals.tolist(),point_multiplier_vs_published=reference_sum.tolist(),
        log_corrected_reference_sums=(totals+np.log(reference_sum)).tolist(),
        relative_covariance=relative.tolist(),monthly=months,
        actual_campaign_variance_validated=False,production_ready=False,
        original_point_estimates_modified=False,
        assumptions='independent accepted marks, independent inverse-binomial stopping count, independent strata')
