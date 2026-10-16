"""Normalized synthetic stresses, not physical spin transport or a PE prior.

The one-dimensional map is a declared sensitivity family. The optional
relative-azimuth factor changes the nuisance conditional while leaving the
effective-spin marginal unchanged. Neither is calibrated to real binaries.
"""

from dataclasses import dataclass
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import expit

from src.boundary_regularized_population import BoundaryRegularizedSpin
from src.independent_spin_model import SpinParameters

A = .99
DISPLACEMENTS = (-.01, -.005, -.001, 0., .001, .005, .01)
MASSES = (8., 15., 45., 70.)
CENTERS = (-.35, -.05, .2, .55)
WIDTHS = (.07, .12, .08, .18)


def map_and_logjac(chi, displacement, a_max=A):
    """Return T(x), log T'(x); signed displacement fixes sup |T(x)-x|.

    Restrict this diagnostic to |displacement|<A to stay well clear of
    numerically singular maps. Endpoints use their exact mathematical values,
    not clipping of invalid inputs. The inverse uses -displacement.
    """
    if (not math.isfinite(a_max) or not 0 < a_max <= 1
            or not math.isfinite(displacement) or abs(displacement) >= a_max):
        raise ValueError('Require 0<A<=1 and finite |displacement|<A')
    x = np.asarray(chi, dtype=float)
    if not np.isfinite(x).all() or np.any(abs(x) > a_max):
        raise ValueError('Map inputs must lie in the closed physical interval')
    if displacement == 0:
        return x.copy(), np.zeros(x.shape)
    b = 2*math.atanh(displacement/(2*a_max))
    t = math.tanh(b)
    z = x/a_max
    mapped = a_max*(z+t)/(1+z*t)
    mapped = np.where(abs(x) == a_max, x, mapped)
    logjac = math.log1p(-t*t)-2*np.log1p(z*t)
    return mapped, logjac


def stressed_logpdf(model, mass, chi, displacement):
    """g_b(x)=g(T_b(x))*T'_b(x), retaining native x in any r0/p0 factor."""
    if not isinstance(model, BoundaryRegularizedSpin):
        raise TypeError('Require the existing normalized boundary spin model')
    mass, chi = np.broadcast_arrays(np.asarray(mass, float), np.asarray(chi, float))
    if not np.isfinite(mass).all() or np.any(mass <= 0) or not np.isfinite(chi).all():
        raise ValueError('Require finite positive masses and finite spin values')
    # Validate displacement even for entirely out-of-support inputs.
    map_and_logjac(0., displacement, model.a_max)
    result = np.full(chi.shape, -np.inf)
    inside = abs(chi) <= model.a_max
    if np.any(inside):
        transformed, logjac = map_and_logjac(chi[inside], displacement, model.a_max)
        result[inside] = model.logpdf(mass[inside], transformed)+logjac
    return result


def azimuth_logfactor(phi, epsilon):
    """A positive conditional tilt with unit mean over uniform relative azimuth."""
    phi = np.asarray(phi, float)
    if not np.isfinite(phi).all() or not math.isfinite(epsilon) or abs(epsilon) >= 1:
        raise ValueError('Require finite angles and |epsilon|<1')
    return np.log1p(epsilon*np.cos(phi))


def shape_change_bounds(event_lower, event_upper, selection_lower, selection_upper):
    """Propagate assumed log-ratio bounds for the log-uniform-rate shape case.

    Inputs must already be valid bounds on the whole relevant support.
    Stored-row extrema alone do not supply that premise. This helper does
    not select the project's rate prior or establish a physical error bound.
    """
    lower, upper = np.asarray(event_lower, float), np.asarray(event_upper, float)
    if (lower.ndim != 1 or upper.shape != lower.shape or not lower.size
            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
            or np.any(lower > upper) or not math.isfinite(selection_lower)
            or not math.isfinite(selection_upper) or selection_lower > selection_upper):
        raise ValueError('Require nonempty ordered finite log-ratio bounds')
    return float(lower.sum()-lower.size*selection_upper), float(upper.sum()-lower.size*selection_lower)


@dataclass(frozen=True)
class Profile:
    name: str
    parameters: SpinParameters
    limiting_control: bool


def fixed_profiles():
    """Six declared test cases, not hyperposterior draws or fitted parameters."""
    def make(name, mu, log_sigma, xi, low=(-.5, .5), high=(-.4, .8), limiting=False):
        return Profile(name, SpinParameters(mu, log_sigma, 10., 30., 60., xi, xi,
                                            *low, *high), limiting)
    return [make('broad_gaussian', 0., -1., 0., limiting=True),
            make('narrow_gaussian', 0., -1.999, 0., limiting=True),
            make('mixed', .1, -.7, .5),
            make('narrow_uniform_mixture', 0., -1., .999, (-.005,.005), (-.005,.005)),
            make('pure_uniform_limit', 0., -1., 1., (-.005,.005), (-.005,.005), True),
            make('boundary_gaussian', .98, -1.999, 0., limiting=True)]


def spin_features(model):
    """All known component/window locations for deterministic integration splits."""
    p, a = model.parameters, model.a_max
    sigma = 10**p.log10_sigma
    features = [-a, a, -a+model.tail_width, a-model.tail_width, 0.,
                *(a*v for v in (p.chi_min_low,p.chi_max_low,p.chi_min_high,p.chi_max_high)),
                *(a*(p.mu+k*sigma) for k in (-8,-3,-1,0,1,3,8))]
    return sorted({float(x) for x in features if -a <= x <= a})


def integration_nodes(model, displacement, order):
    """Split at native and transformed features, then fixed-order quadrature."""
    if not isinstance(order, int) or order < 2:
        raise ValueError('Quadrature order must be an integer >=2')
    a = model.a_max
    base = spin_features(model)
    mapped = map_and_logjac(base, -displacement, a)[0]
    kernel = [center+k*width for center,width in zip(CENTERS,WIDTHS) for k in (-8,-3,-1,0,1,3,8)]
    edges = np.array(sorted({*base, *mapped, *(x for x in kernel if -a<x<a)}))
    left, right = edges[:-1], edges[1:]
    nodes, weights = leggauss(order)
    half = (right-left)/2
    x = (left[:,None]+half[:,None]*(1+nodes)).ravel()
    w = (half[:,None]*weights).ravel()
    return x, w


def toy_integrals(model, mass_index, displacement, order):
    """Four chosen event kernels and a detection function; no real data used."""
    if mass_index not in range(4):
        raise ValueError('Toy mass index must be 0,1,2,3')
    x, w = integration_nodes(model, displacement, order)
    density = np.exp(stressed_logpdf(model, MASSES[mass_index], x, displacement))
    weighted = w*density
    event = np.exp(-.5*((x-CENTERS[mass_index])/WIDTHS[mass_index])**2)
    return dict(normalization=float(weighted.sum()),
                event_factor=float(weighted@event),
                detection_integral=float(weighted@expit(4*x/model.a_max-1)),
                mean_effective_spin=float(weighted@x))


def density_grid(model, displacement):
    """Declared finite grid including both sides of native/moved hard edges."""
    features = spin_features(model)
    features += map_and_logjac(features, -displacement, model.a_max)[0].tolist()
    extra = [v+offset for v in features for offset in (-1e-8, 0., 1e-8)]
    return np.array(sorted({*np.linspace(-model.a_max, model.a_max, 1981),
                            *(v for v in extra if -model.a_max<=v<=model.a_max)}))
