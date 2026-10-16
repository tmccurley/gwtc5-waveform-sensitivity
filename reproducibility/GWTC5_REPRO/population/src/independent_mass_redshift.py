"""Normalized reference mass and redshift models for independent development.

The structure follows arXiv:2605.27226v1, Appendix B.4. All primary-mass
components are capped at m_high here, including the Gaussian peaks. This is
an explicit independent convention, not an exact clone of the paper's
left-truncated Gaussian prescription. Adaptive quadrature is a correctness
reference, not yet a production-speed evaluator for millions of samples.
"""

from dataclasses import astuple, dataclass
import math
import warnings

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import IntegrationWarning, quad
from scipy.special import logsumexp
from scipy.stats import truncnorm


def checked_quad(function, lower, upper, *, points=None, epsabs=1e-10, epsrel=1e-8):
    """Surface failed integration instead of treating it as zero support."""
    if lower == upper:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter('error', IntegrationWarning)
        value, error = quad(function, lower, upper, points=points,
                            epsabs=epsabs, epsrel=epsrel, limit=400)
    if not (math.isfinite(value) and math.isfinite(error)):
        raise ArithmeticError('Nonfinite normalization integral')
    if error > 5*max(epsabs, epsrel*abs(value)):
        raise ArithmeticError('Normalization integral exceeds error tolerance')
    return value


def taper(mass, minimum, width):
    """Planck low-mass turn-on; zero width is an explicit hard-cutoff limit."""
    return np.exp(log_taper(mass, minimum, width))


def log_taper(mass, minimum, width):
    """Log turn-on without converting tiny positive densities into false zeros."""
    values = np.asarray(mass, dtype=float)
    if not np.all(np.isfinite(values)) or not math.isfinite(minimum) or not math.isfinite(width) or width < 0:
        raise ValueError('Taper inputs must be finite with nonnegative width')
    if width == 0:
        return np.where(values > minimum, 0., -np.inf)
    result = np.full(values.shape, -np.inf)
    result[values >= minimum+width] = 0
    inside = (values > minimum) & (values < minimum+width)
    t = (values[inside]-minimum)/width
    with np.errstate(over='ignore', divide='ignore'):
        result[inside] = -np.logaddexp(0., 1/t + 1/(t-1))
    return result


def power_integral(lower, upper, exponent):
    """Integral of x**exponent, continuous through exponent=-1."""
    if not (0 < lower <= upper and all(math.isfinite(v) for v in (lower, upper, exponent))):
        raise ValueError('Power integral needs finite positive ordered limits')
    log_ratio = math.log(upper/lower)
    k = exponent+1
    if k == 0:
        return log_ratio
    return math.exp(k*math.log(lower))*math.expm1(k*log_ratio)/k


@dataclass(frozen=True)
class MassParameters:
    alpha_1: float
    alpha_2: float
    m_break: float
    mu_1: float
    sigma_1: float
    mu_2: float
    sigma_2: float
    m_1_low: float
    delta_1: float
    lambda_0: float
    lambda_1: float
    beta_q: float
    m_2_low: float
    delta_2: float
    m_high: float = 300.0

    def __post_init__(self):
        if not all(np.ndim(v) == 0 and math.isfinite(v) for v in astuple(self)):
            raise ValueError('Mass hyperparameters must be finite scalars')
        if not (0 < self.m_2_low <= self.m_1_low < self.m_break < self.m_high):
            raise ValueError('Invalid mass boundaries')
        if not (-4 <= self.alpha_1 <= 12 and -4 <= self.alpha_2 <= 12 and -2 <= self.beta_q <= 7):
            raise ValueError('Power-law index outside implemented prior domain')
        if min(self.sigma_1, self.sigma_2) <= 0 or min(self.delta_1, self.delta_2) < 0:
            raise ValueError('Peak widths must be positive and taper widths nonnegative')
        if min(self.lambda_0, self.lambda_1) < 0 or self.lambda_0+self.lambda_1 > 1:
            raise ValueError('Mixture fractions must lie in the simplex')


class MassModel:
    """p(m1,q)=p(m1)*p(q|m1), with separate conditional q normalization.

    Mixture weights specify the normalized components BEFORE the shared
    primary-mass taper. They need not equal integrated fractions after it.
    """

    def __init__(self, parameters: MassParameters):
        self.parameters = parameters
        p = parameters
        self.broken_norm = p.m_break*(
            power_integral(p.m_1_low/p.m_break, 1, -p.alpha_1)
            + power_integral(1, p.m_high/p.m_break, -p.alpha_2))
        self.peaks = [truncnorm((p.m_1_low-mu)/sigma, (p.m_high-mu)/sigma,
                                loc=mu, scale=sigma)
                      for mu, sigma in [(p.mu_1, p.sigma_1), (p.mu_2, p.sigma_2)]]
        # Integrating Gaussian taper expectations in CDF coordinates avoids
        # missing very narrow peaks during mass-space adaptive quadrature.
        power_expectation = checked_quad(
            lambda x: float(np.exp(self.broken_logpdf(x))*taper(x, p.m_1_low, p.delta_1)),
            p.m_1_low, p.m_high,
            points=sorted({x for x in (p.m_break, p.m_1_low+p.delta_1) if p.m_1_low < x < p.m_high}))
        peak_expectations = [checked_quad(
            lambda u, peak=peak: float(taper(peak.ppf(u), p.m_1_low, p.delta_1)), 0, 1)
            for peak in self.peaks]
        self.primary_norm = float(np.dot([p.lambda_0, p.lambda_1, 1-p.lambda_0-p.lambda_1],
                                          [power_expectation, *peak_expectations]))
        if not math.isfinite(self.primary_norm) or self.primary_norm <= 0:
            raise ArithmeticError('Primary-mass normalization is not positive')

    def broken_logpdf(self, mass):
        x = np.asarray(mass, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError('Mass coordinates must be finite')
        p = self.parameters
        result = np.full(x.shape, -np.inf)
        mask = (x >= p.m_1_low) & (x <= p.m_high)
        alpha = np.where(x[mask] < p.m_break, p.alpha_1, p.alpha_2)
        result[mask] = -alpha*np.log(x[mask]/p.m_break)-math.log(self.broken_norm)
        return result

    def primary_logpdf(self, mass):
        x = np.asarray(mass, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError('Mass coordinates must be finite')
        p = self.parameters
        component = np.stack([self.broken_logpdf(x), *(peak.logpdf(x) for peak in self.peaks)])
        weights = np.asarray([p.lambda_0, p.lambda_1, 1-p.lambda_0-p.lambda_1])
        with np.errstate(divide='ignore'):
            terms = component + np.log(weights).reshape((3,)+(1,)*x.ndim)
            return logsumexp(terms, axis=0) + log_taper(x, p.m_1_low, p.delta_1)-math.log(self.primary_norm)

    def q_normalizer(self, mass):
        """Linear convenience form; use q_log_normalizer for very small values."""
        value = math.exp(self.q_log_normalizer(mass))
        if value == 0:
            raise ArithmeticError('Linear q normalization underflows; use q_log_normalizer')
        return value

    def q_log_normalizer(self, mass):
        """Conditional integral with a rescaled near-boundary taper branch.

        When the entire q interval lies inside the taper, its unscaled
        integral can underflow. Factor out the largest taper, map the small
        interval to [0,1], and resolve its concentration near the upper end.
        This preserves finite log densities instead of declaring zero support.
        """
        p = self.parameters
        if not math.isfinite(mass) or mass <= p.m_2_low:
            raise ValueError('Conditional q density requires m1 > m2_low')
        minimum = p.m_2_low/mass
        if p.delta_2 == 0:
            return math.log(power_integral(minimum, 1, p.beta_q))
        gap = mass-p.m_2_low
        if gap < p.delta_2:
            log_scale = float(log_taper(gap, 0, p.delta_2))
            span = gap/mass
            width = min(1., gap/p.delta_2)
            points = sorted({1-multiple*width for multiple in (1, 4, 16, 64)
                             if 0 < 1-multiple*width < 1})
            def scaled_integrand(u):
                return math.exp(p.beta_q*math.log(minimum+span*u)
                                + float(log_taper(gap*u, 0, p.delta_2))-log_scale)
            scaled = checked_quad(scaled_integrand, 0, 1, points=points, epsabs=0)
            if scaled <= 0:
                raise ArithmeticError('Rescaled q normalization is not positive')
            return math.log(span)+log_scale+math.log(scaled)
        taper_end = min(1.0, (p.m_2_low+p.delta_2)/mass)
        smooth = checked_quad(lambda q: q**p.beta_q*float(taper(mass*q, p.m_2_low, p.delta_2)),
                              minimum, taper_end, epsabs=0)
        total = smooth+power_integral(taper_end, 1, p.beta_q)
        if not math.isfinite(total) or total <= 0:
            raise ArithmeticError('Conditional q normalization is not positive')
        return math.log(total)

    def conditional_q_logpdf(self, mass, mass_ratio):
        m, q = np.broadcast_arrays(np.asarray(mass, dtype=float), np.asarray(mass_ratio, dtype=float))
        if not (np.all(np.isfinite(m)) and np.all(np.isfinite(q))):
            raise ValueError('Mass and q coordinates must be finite')
        p = self.parameters
        result = np.full(m.shape, -np.inf)
        good = (m > p.m_2_low) & (q > 0) & (q <= 1) & (m*q > p.m_2_low)
        if np.any(good):
            # A slow exact reference; a validated interpolator can replace
            # repeated normalizations later, with its own accuracy checks.
            unique_mass, inverse = np.unique(m[good], return_inverse=True)
            log_norm = np.asarray([self.q_log_normalizer(float(value)) for value in unique_mass])[inverse]
            with np.errstate(divide='ignore'):
                result[good] = p.beta_q*np.log(q[good])+log_taper(m[good]*q[good], p.m_2_low, p.delta_2)-log_norm
        return result

    def logpdf(self, mass, mass_ratio):
        return self.primary_logpdf(mass)+self.conditional_q_logpdf(mass, mass_ratio)


MASS_PARAMETER_NAMES = ('alpha_1', 'alpha_2', 'm_break', 'mu_1', 'sigma_1', 'mu_2',
                        'sigma_2', 'm_1_low', 'delta_1', 'lambda_0', 'lambda_1',
                        'beta_q', 'm_2_low', 'delta_2')


def mass_prior_transform(unit_cube):
    """Interior unit-cube transform with a normalized Dirichlet(1,1,1) mixture."""
    u = np.asarray(unit_cube, dtype=float)
    if u.shape != (14,) or not np.all(np.isfinite(u)) or np.any(u <= 0) or np.any(u >= 1):
        raise ValueError('Mass prior needs 14 coordinates strictly inside (0,1)')
    minimum = 3+7*u[7]
    w0 = 1-math.sqrt(1-u[9])
    w1 = (1-w0)*u[10]
    return MassParameters(-4+16*u[0], -4+16*u[1], 20+30*u[2], 5+15*u[3], 10*u[4],
                           25+35*u[5], 10*u[6], minimum, 10*u[8], w0, w1,
                           -2+9*u[11], 3+(minimum-3)*u[12], 10*u[13])


def mass_log_hyperprior(p: MassParameters):
    """Joint prior in MASS_PARAMETER_NAMES coordinates; m_high is fixed."""
    if not (20 <= p.m_break <= 50 and 5 <= p.mu_1 <= 20 and 0 < p.sigma_1 <= 10
            and 25 <= p.mu_2 <= 60 and 0 < p.sigma_2 <= 10 and 3 < p.m_1_low <= 10
            and 0 <= p.delta_1 <= 10 and 3 <= p.m_2_low <= p.m_1_low
            and 0 <= p.delta_2 <= 10 and p.m_high == 300):
        return -math.inf
    widths = [16, 16, 30, 15, 10, 35, 10, 7, 10, 9, p.m_1_low-3, 10]
    return math.log(2)-sum(math.log(width) for width in widths)


class RedshiftModel:
    """Normalized detector-time density proportional to dVc/dz*(1+z)^(kappa-1).

    This flat matter+Lambda reference has no radiation (Tcmb0=0). H0 is in
    km/s/Mpc; differential comoving volume is integrated over the full sky.
    z_max is an explicit model bound, not estimated from observed samples.
    """

    def __init__(self, kappa, *, z_max, H0, omega_m):
        if not all(math.isfinite(v) for v in (kappa, z_max, H0, omega_m)):
            raise ValueError('Redshift parameters must be finite')
        if not (-10 <= kappa <= 10 and z_max > 0 and H0 > 0 and 0 <= omega_m <= 1):
            raise ValueError('Invalid redshift model parameters')
        self.kappa, self.z_max = kappa, z_max
        self.cosmology = FlatLambdaCDM(H0=H0, Om0=omega_m, Tcmb0=0)
        self.normalizer = checked_quad(lambda z: float(self.unnormalized(z)), 0, z_max,
                                       epsabs=1e-4, epsrel=1e-9)
        if not math.isfinite(self.normalizer) or self.normalizer <= 0:
            raise ArithmeticError('Redshift normalization is not positive')

    def unnormalized(self, redshift):
        z = np.asarray(redshift, dtype=float)
        if not np.all(np.isfinite(z)) or np.any(z < 0):
            raise ValueError('Redshifts must be finite and nonnegative')
        volume = 4*math.pi*self.cosmology.differential_comoving_volume(z).value
        return volume*np.power(1+z, self.kappa-1)

    def logpdf(self, redshift):
        z = np.asarray(redshift, dtype=float)
        values = self.unnormalized(z)
        with np.errstate(divide='ignore'):
            result = np.log(values)-math.log(self.normalizer)
        return np.where(z <= self.z_max, result, -np.inf)
