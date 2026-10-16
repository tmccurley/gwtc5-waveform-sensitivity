"""Independent joint population and full-spin extension for synthetic testing.

Reduced measure: dm1_source dq dz dchi_eff. A full Cartesian-spin density is
also provided for selection calculations. Neither supplies the PE prior of
a real event. Production data conventions remain separately gated.
"""

from dataclasses import dataclass
import math

import numpy as np

from src.independent_mass_redshift import MassModel, MassParameters, RedshiftModel, mass_log_hyperprior, mass_prior_transform
from src.independent_spin_model import SpinParameters, conditional_logpdf, spin_log_hyperprior, spin_prior_transform
from src.spin_prior_reference import isotropic_uniform_spin_density


MODEL_ID = 'independent_three_transitions_joint_v0.1'


@dataclass(frozen=True)
class PopulationParameters:
    mass: MassParameters
    spin: SpinParameters
    kappa: float

    def __post_init__(self):
        if not isinstance(self.mass, MassParameters) or not isinstance(self.spin, SpinParameters):
            raise TypeError('Validated mass and spin parameters required')
        if not math.isfinite(self.kappa) or not -10 <= self.kappa <= 10:
            raise ValueError('kappa must be in [-10,10]')


def population_prior_transform(unit_cube):
    """26 dimensions: 14 mass, 11 normalized-spin, then redshift evolution."""
    u = np.asarray(unit_cube, dtype=float)
    if u.shape != (26,) or not np.all(np.isfinite(u)) or np.any(u <= 0) or np.any(u >= 1):
        raise ValueError('Population prior needs 26 interior unit-cube coordinates')
    return PopulationParameters(mass_prior_transform(u[:14]), spin_prior_transform(u[14:25]), -10+20*u[25])


def population_log_hyperprior(parameters):
    """Normalized prior in the declared internal coordinates, not physical sigma."""
    return mass_log_hyperprior(parameters.mass)+spin_log_hyperprior(parameters.spin)-math.log(20)


class PopulationModel:
    def __init__(self, parameters, *, a_max, z_max, H0, omega_m):
        if not math.isfinite(a_max) or not 0 < a_max <= 1:
            raise ValueError('a_max must be explicit and in (0,1]')
        self.parameters, self.a_max = parameters, a_max
        self.mass = MassModel(parameters.mass)
        self.redshift = RedshiftModel(parameters.kappa, z_max=z_max, H0=H0, omega_m=omega_m)

    def spin_logpdf(self, mass, chi_eff):
        """Rescale the equation prototype into [-A,A], including 1/A Jacobian.

        Spin hyperparameters describe x=chi_eff/A. Physical Gaussian location,
        width and uniform endpoints are A times those internal parameters.
        This is an explicit prior/support choice, not reference-row editing.
        """
        return conditional_logpdf(mass, np.asarray(chi_eff)/self.a_max, self.parameters.spin)-math.log(self.a_max)

    def logpdf(self, mass_1_source, q, redshift, chi_eff):
        """Return normalized reduced-space population density in log form."""
        mass, ratio, z, spin = np.broadcast_arrays(*(np.asarray(x, dtype=float)
                                                   for x in (mass_1_source, q, redshift, chi_eff)))
        return self.mass.logpdf(mass, ratio)+self.redshift.logpdf(z)+self.spin_logpdf(mass, spin)

    def cartesian_logpdf(self, mass_1_source, mass_2_source, redshift, spin_1, spin_2):
        """Density in dm1 dm2 dz d3spin1 d3spin2 for injection weighting.

        The unmodeled spin coordinates follow the conditional distribution
        induced by independent Uniform(0,A) magnitudes and isotropic tilts,
        given q and chi_eff. Thus full spin density is r0(spins)*p(chi_eff)/
        p0(chi_eff|q), NOT merely the effective-spin marginal. Division by m1
        converts the mass density from (m1,q) to (m1,m2).

        This specifies an independent nuisance model. Whether it matches PE
        nuisance treatment and injection extrinsic proposals remains to be
        verified before real-data use. It can also yield divergent importance
        second moments near |chi_eff|=A for regular spin proposals. A finite
        evaluation or passing component test does not certify selection
        uncertainty; see protocol/joint_model_selection_2026-09-11.md.
        """
        s1, s2 = np.asarray(spin_1, dtype=float), np.asarray(spin_2, dtype=float)
        if s1.ndim < 1 or s2.ndim < 1 or s1.shape[-1] != 3 or s2.shape[-1] != 3:
            raise ValueError('Spin vectors must have final dimension 3')
        if not np.all(np.isfinite(s1)) or not np.all(np.isfinite(s2)):
            raise ValueError('Spin components must be finite')
        m1, m2, z, a1, a2, s1z, s2z = np.broadcast_arrays(
            np.asarray(mass_1_source, dtype=float), np.asarray(mass_2_source, dtype=float),
            np.asarray(redshift, dtype=float), np.linalg.norm(s1, axis=-1), np.linalg.norm(s2, axis=-1),
            s1[..., 2], s2[..., 2])
        if not (np.all(np.isfinite(m1)) and np.all(np.isfinite(m2))):
            raise ValueError('Component masses must be finite')
        if np.any(m2 <= 0) or np.any(m1 < m2) or np.any(a1 == 0) or np.any(a2 == 0):
            raise ValueError('Require ordered positive masses and nonzero spin magnitudes')
        q = m2/m1
        chi_eff = (s1z+q*s2z)/(1+q)
        reduced = self.logpdf(m1, q, z, chi_eff)
        result = np.full(reduced.shape, -np.inf)
        inside = np.isfinite(reduced) & (a1 <= self.a_max) & (a2 <= self.a_max)
        if np.any(inside):
            p0 = np.asarray([isotropic_uniform_spin_density(float(x), float(ratio), self.a_max)[0]
                             for x, ratio in zip(chi_eff[inside], q[inside])])
            if np.any(p0 <= 0) or not np.all(np.isfinite(p0)):
                raise ArithmeticError('Induced spin prior is not finite and positive at a retained point')
            reference = -2*math.log(4*math.pi*self.a_max)-2*np.log(a1[inside])-2*np.log(a2[inside])
            result[inside] = reduced[inside]-np.log(m1[inside])+reference-np.log(p0)
        return result
