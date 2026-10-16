"""Slow independent reference for an induced effective-spin prior.

Assumptions: independent a_1,a_2 ~ U(0,A), independent isotropic tilt angles,
fixed q=m2/m1 in (0,1]. These assumptions are explicit, not inferred from a
posterior. This function is NOT a validated complete PE-prior evaluator and
must not be substituted for injection draw densities.
"""

import math
import warnings

from scipy.integrate import IntegrationWarning, quad


def isotropic_uniform_spin_density(chi_eff, q, a_max, *, epsabs=1e-10, epsrel=1e-8):
    """Return density and quadrature error estimate for one conditional prior.

    For z_i=a_i*cos(tilt_i), f(z_i)=log(A/abs(z_i))/(2*A) on (-A,A).
    chi_eff=(z_1+q*z_2)/(1+q); convolution gives the integral below. We
    integrate over u=z_2/A and split at logarithmic singularities. Symmetry
    handles negative chi_eff; the center has an exact closed form.

    The error estimate is numerical integration error, not sampling error or
    uncertainty in whether these assumptions describe a released PE product.
    The routine is intentionally slow and is for checking future optimized
    evaluators. No lookup table, KDE or density floor is used.
    """
    values = (chi_eff, q, a_max, epsabs, epsrel)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Inputs and integration tolerances must be finite scalars')
    if not (0 < q <= 1 and 0 < a_max <= 1 and epsabs > 0 and epsrel > 0):
        raise ValueError('Require 0<q<=1, 0<a_max<=1 and positive tolerances')
    if abs(chi_eff) >= a_max:
        return 0.0, 0.0
    if chi_eff == 0:
        return (1+q) * (2-math.log(q)) / (2*a_max), 0.0
    b = (1+q) * abs(chi_eff) / a_max
    lower, upper = max(-1.0, (b-1)/q), min(1.0, (b+1)/q)
    if not lower < upper:
        raise ArithmeticError('Interior spin support lost to finite-precision bounds')
    points = sorted({point for point in (0.0, b/q) if lower < point < upper})
    scale = (1+q)/(4*a_max)

    def integrand(u):
        return math.log(abs(u)) * math.log(abs(b-q*u))

    with warnings.catch_warnings():
        warnings.simplefilter('error', IntegrationWarning)
        value, error = quad(integrand, lower, upper, points=points,
                            epsabs=epsabs/scale, epsrel=epsrel, limit=400)
    density, absolute_error = scale*value, scale*error
    if not (math.isfinite(density) and math.isfinite(absolute_error)) or density < 0:
        raise ArithmeticError('Invalid quadrature result')
    if absolute_error > 5*max(epsabs, epsrel*abs(density)):
        raise ArithmeticError('Quadrature error exceeds its requested tolerance')
    return density, absolute_error
