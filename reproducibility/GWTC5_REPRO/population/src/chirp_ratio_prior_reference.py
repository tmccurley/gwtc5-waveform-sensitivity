"""Explicit bounded mass reference when no additional constraints are declared.

This does not infer absent historical constraints or silently change the old
prior recognizer. A caller must deliberately select the recorded-dictionary
interpretation and preserve its evidence separately.
"""

import math
import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1

from src.pe_prior_reference import interval, chirp_factor, DetectorMassSupport


def q_antiderivative(q):
    return -5 * np.asarray(q, dtype=float)**(-.2) * hyp2f1(-.4, -.2, .8, -np.asarray(q, dtype=float))


class ChirpRatioOnlyMassPrior:
    """Flat component-mass density on a bounded chirp-mass/ratio rectangle.

    The bounds are in detector-frame solar masses and dimensionless q. In
    (m1,q), the density is m1/area because dm1 dm2=m1 dm1 dq.
    """

    def __init__(self, chirp_mass, mass_ratio):
        self.chirp_mass, self.mass_ratio = tuple(chirp_mass), tuple(mass_ratio)
        interval(self.chirp_mass, 'chirp_mass')
        interval(self.mass_ratio, 'mass_ratio')
        if self.mass_ratio[1] > 1:
            raise ValueError('Require ordered component masses')
        lo, hi = self.mass_ratio
        integral, error = quad(lambda q: (1+q)**.4/q**1.2, lo, hi,
                               epsabs=0, epsrel=1e-11, limit=200)
        if not math.isfinite(integral) or integral <= 0 or error > integral*1e-9:
            raise ValueError('Unresolved mass-ratio normalization')
        self.q_integral = integral
        self.area = .5*(self.chirp_mass[1]**2-self.chirp_mass[0]**2)*integral
        self.q_special_integral = float(q_antiderivative(hi)-q_antiderivative(lo))
        if not math.isclose(integral,self.q_special_integral,rel_tol=2e-10):
            raise ValueError('Independent mass normalizations disagree')

    def contains(self, mass_1, mass_ratio):
        m,q=np.broadcast_arrays(np.asarray(mass_1,float),np.asarray(mass_ratio,float))
        if not np.isfinite(m).all() or not np.isfinite(q).all():
            raise ValueError('Mass coordinates must be finite')
        good=(m>0)&(q>=self.mass_ratio[0])&(q<=self.mass_ratio[1])
        result=np.zeros(m.shape,dtype=bool)
        mc=m[good]*chirp_factor(q[good])
        result[good]=(mc>=self.chirp_mass[0])&(mc<=self.chirp_mass[1])
        return result

    def logpdf_detector_mq(self, mass_1, mass_ratio):
        m,q=np.broadcast_arrays(np.asarray(mass_1,float),np.asarray(mass_ratio,float))
        good=self.contains(m,q);answer=np.full(m.shape,-np.inf)
        answer[good]=np.log(m[good])-math.log(self.area)
        return answer

    def logpdf_chirp_q(self, chirp_mass, mass_ratio):
        mc,q=np.broadcast_arrays(np.asarray(chirp_mass,float),np.asarray(mass_ratio,float))
        if not np.isfinite(mc).all() or not np.isfinite(q).all():
            raise ValueError('Mass coordinates must be finite')
        good=(mc>=self.chirp_mass[0])&(mc<=self.chirp_mass[1])&(q>=self.mass_ratio[0])&(q<=self.mass_ratio[1])
        answer=np.full(mc.shape,-np.inf)
        answer[good]=np.log(mc[good])-2*np.log(chirp_factor(q[good]))-math.log(self.area)
        return answer

    def probability_transforms(self, chirp_mass, mass_ratio):
        mc,q=np.broadcast_arrays(np.asarray(chirp_mass,float),np.asarray(mass_ratio,float))
        if np.any(~np.isfinite(self.logpdf_chirp_q(mc,q))):
            raise ValueError('Prior draws outside declared support')
        return ((mc**2-self.chirp_mass[0]**2)/(self.chirp_mass[1]**2-self.chirp_mass[0]**2),
                (q_antiderivative(q)-q_antiderivative(self.mass_ratio[0]))/self.q_special_integral)

    def equivalent_component_enclosure(self):
        """A derived enclosure for an independent integrator, not recorded priors.

        Outward rounding prevents an artificial floating-point truncation of
        the chirp-ratio rectangle at the component-mass extrema.
        """
        lo,hi=self.mass_ratio;a,b=self.chirp_mass
        def outward(low,high):
            return float(np.nextafter(low,-np.inf)),float(np.nextafter(high,np.inf))
        return DetectorMassSupport(self.chirp_mass,self.mass_ratio,
            outward(a/chirp_factor(hi),b/chirp_factor(lo)),
            outward(a*lo/chirp_factor(lo),b*hi/chirp_factor(hi)))

    def require_production_ready(self):
        raise RuntimeError('Recorded mass-reference family only; historical row density is not certified')


def declared_chirp_ratio_reference(analytic, *, recorded_dictionary_has_no_extra_mass_constraints=False):
    """Opt-in binding of a complete recorded dictionary, with no guessed bounds."""
    if recorded_dictionary_has_no_extra_mass_constraints is not True:
        raise ValueError('Explicit recorded-dictionary interpretation required')
    for name,item in analytic.items():
        if name in ('mass_1','mass_2','total_mass') or item.get('parsed',{}).get('constructor','').split('.')[-1]=='Constraint':
            raise ValueError('Additional declared mass/constraint entry requires another evaluator')
        if 'parsed' not in item:
            raise ValueError('Unparsed prior entry prevents a complete-dictionary claim')
    def bounds(name,constructor):
        p=analytic.get(name,{}).get('parsed',{})
        if p.get('constructor')!=constructor:
            raise ValueError('Missing or unsupported chirp-ratio prior')
        shape=p['shape_parameters']
        if shape.get('equal_mass',False) is not False or shape.get('boundary') is not None:
            raise ValueError('Unsupported sampling option')
        return shape['minimum'],shape['maximum']
    return ChirpRatioOnlyMassPrior(bounds('chirp_mass','bilby.gw.prior.UniformInComponentsChirpMass'),
                                  bounds('mass_ratio','bilby.gw.prior.UniformInComponentsMassRatio'))
