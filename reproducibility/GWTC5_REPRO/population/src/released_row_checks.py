"""Read-only coordinate diagnostics, not a released-prior or convergence test."""

import hashlib
import json
import math
import numpy as np
from astropy.cosmology import Planck15

from src.pe_prior_reference import chirp_factor, planck15_lal


BASE_FIELDS = ('mass_1','mass_2','mass_1_source','mass_2_source','chirp_mass',
               'mass_ratio','redshift','luminosity_distance','chi_eff',
               'a_1','a_2','tilt_1','tilt_2','spin_1z','spin_2z')
SUFFIXES = ('_infinity','_infinity_only_prec_avg')
OPTIONAL_FIELDS = ('total_mass', 'reference_frequency') + tuple(
    name+suffix for suffix in SUFFIXES
    for name in ('chi_eff','tilt_1','tilt_2','spin_1z','spin_2z'))
ALG_RTOL, ALG_ATOL = 1e-10, 1e-12
DIST_RTOL, DIST_ATOL = 1e-6, 1e-6


def selected_value_digest(rows):
    """Hash field names/count and each ordered column encoded as little-endian f8."""
    names = list(rows.dtype.names)
    digest = hashlib.sha256(json.dumps({'fields':names,'rows':int(rows.size)},sort_keys=True).encode())
    for name in names:
        digest.update(np.asarray(rows[name],dtype='<f8').tobytes(order='C'))
    return digest.hexdigest()


def difference_summary(observed, expected, *, rtol=ALG_RTOL, atol=ALG_ATOL):
    a,b = np.asarray(observed,float),np.asarray(expected,float)
    valid = np.isfinite(a)&np.isfinite(b)
    residual = np.abs(a[valid]-b[valid])
    return {'compared_rows':int(valid.sum()),'nonfinite_rows':int((~valid).sum()),
            'maximum_absolute_residual':float(residual.max()) if residual.size else None,
            'maximum_relative_residual_with_unit_floor':float(np.max(residual/np.maximum(np.abs(b[valid]),1.))) if residual.size else None,
            'rows_above_screen':int(np.sum(residual>atol+rtol*np.abs(b[valid]))),
            'absolute_tolerance':atol,'relative_tolerance':rtol}


def distribution(values):
    v=np.asarray(values,float)
    good=np.isfinite(v)
    x=v[good]
    return {'finite_rows':int(good.sum()),'nonfinite_rows':int((~good).sum()),
            'quantiles_0_50_90_99_100':np.quantile(x,[0,.5,.9,.99,1]).tolist() if x.size else None}


def bound_check(values, bounds):
    low,high=bounds
    v=np.asarray(values,float); finite=np.isfinite(v)
    return {'bounds':list(bounds),'nonfinite_rows':int((~finite).sum()),
            'strict_below':int(np.sum(finite&(v<low))), 'strict_above':int(np.sum(finite&(v>high))),
            'below_beyond_screen':int(np.sum(finite&(v<low-ALG_ATOL-ALG_RTOL*abs(low)))),
            'above_beyond_screen':int(np.sum(finite&(v>high+ALG_ATOL+ALG_RTOL*abs(high))))}


def audit_coordinate_rows(rows, analytic, *, cosmology_name, distance_references=None, mass_reference=None):
    """Check all supplied rows; diagnostics never modify/filter their inputs.

Finite-only summaries report their omitted nonfinite count explicitly. The
distance comparison uses the product's recorded conversion cosmology, not an
automatic substitution of the candidate population cosmology.
"""
    if rows.ndim != 1 or rows.size==0 or rows.dtype.names is None:
        raise ValueError('Nonempty one-dimensional compound posterior data required')
    if any(k not in rows.dtype.names for k in BASE_FIELDS):
        raise ValueError('Missing required coordinate field')
    if any(rows.dtype[k].kind not in 'if' for k in rows.dtype.names):
        raise ValueError('Only real numeric coordinate fields accepted')
    d={k:np.asarray(rows[k],float) for k in rows.dtype.names}
    n=rows.size
    finite={k:int(np.sum(~np.isfinite(v))) for k,v in d.items()}
    physical={
        'nonpositive_detector_mass':int(np.sum((d['mass_1']<=0)|(d['mass_2']<=0))),
        'unordered_detector_mass':int(np.sum(d['mass_2']>d['mass_1'])),
        'nonpositive_source_mass':int(np.sum((d['mass_1_source']<=0)|(d['mass_2_source']<=0))),
        'negative_redshift':int(np.sum(d['redshift']<0)),
        'nonpositive_distance':int(np.sum(d['luminosity_distance']<=0)),
        'mass_ratio_outside_0_1':int(np.sum((d['mass_ratio']<=0)|(d['mass_ratio']>1))),
        'component_magnitude_outside_0_1':int(np.sum((d['a_1']<0)|(d['a_1']>1)|(d['a_2']<0)|(d['a_2']>1))),
        'native_tilt_outside_0_pi':int(np.sum((d['tilt_1']<0)|(d['tilt_1']>math.pi)|(d['tilt_2']<0)|(d['tilt_2']>math.pi))),
        'absolute_effective_spin_above_one':int(np.sum(np.abs(d['chi_eff'])>1))}
    with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
        q=d['mass_2']/d['mass_1']
        expected={'mass_ratio':q,'chirp_mass':d['mass_1']*chirp_factor(q),
                  'mass_1_source':d['mass_1']/(1+d['redshift']),
                  'mass_2_source':d['mass_2']/(1+d['redshift']),
                  'chi_eff':(d['mass_1']*d['spin_1z']+d['mass_2']*d['spin_2z'])/(d['mass_1']+d['mass_2']),
                  'spin_1z':d['a_1']*np.cos(d['tilt_1']), 'spin_2z':d['a_2']*np.cos(d['tilt_2'])}
        if 'total_mass' in d: expected['total_mass']=d['mass_1']+d['mass_2']
    checks={k:difference_summary(d[k],v) for k,v in expected.items()}
    named={'planck15_lal':planck15_lal,'planck15':lambda:Planck15}
    name=cosmology_name.lower() if isinstance(cosmology_name,str) else ''
    cosmology={'recorded_name':cosmology_name,'resolved':name in named}
    if name in named:
        z=d['redshift']; valid=np.isfinite(z)&(z>=0)
        predicted=np.full(n,np.nan)
        predicted[valid]=named[name]().luminosity_distance(z[valid]).value
        cosmology['distance_identity']=difference_summary(d['luminosity_distance'],predicted,rtol=DIST_RTOL,atol=DIST_ATOL)
    bounds={}; missing=[]
    for k in ('chirp_mass','mass_ratio','mass_1','mass_2','total_mass','luminosity_distance','a_1','a_2'):
        if k=='total_mass' and k not in analytic: continue
        p=analytic.get(k,{}).get('parsed',{}).get('shape_parameters',{})
        lo,hi=p.get('minimum'),p.get('maximum')
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in (lo,hi)) or not lo<hi:
            missing.append(k); continue
        values=d[k] if k in d else d['mass_1']+d['mass_2']
        bounds[k]=bound_check(values,(lo,hi))
    support={'complete_reference_available':mass_reference is not None}
    if mass_reference is not None:
        valid=np.isfinite(d['mass_1'])&np.isfinite(q)
        support['nonfinite_rows']=int((~valid).sum())
        support['strict_outside_reconstructed_mass_support']=int(np.sum(~mass_reference.contains(d['mass_1'][valid],q[valid])))
    grid_comparison={'available':False}
    if distance_references is not None and name=='planck15_lal':
        smooth,grid=distance_references
        z=d['redshift']; valid=np.isfinite(z)&(z>=0)
        delta=np.full(n,np.nan)
        with np.errstate(invalid='ignore'):
            delta[valid]=grid.redshift_logpdf(z[valid])-smooth.redshift_logpdf(z[valid])
        grid_comparison={'available':True,'log_grid_minus_log_continuous_same_z':distribution(delta),
                         'absolute_log_difference':distribution(np.abs(delta)),
                         'not_a_released_weight_error_bound':True}
    evolved={}
    for suffix in SUFFIXES:
        key='chi_eff'+suffix
        if key not in d: continue
        item={'effective_spin_changes_absolute':distribution(np.abs(d[key]-d['chi_eff'])), 'identity_checks':{}}
        delta=np.abs(d[key]-d['chi_eff']); valid=np.isfinite(delta)
        item['rows_above_absolute_change_0p001']=int(np.sum(delta[valid]>.001))
        item['rows_above_absolute_change_0p01']=int(np.sum(delta[valid]>.01))
        item['rows_outside_native_component_bound_0p99']=int(np.sum(np.isfinite(d[key])&(np.abs(d[key])>.99)))
        if all('tilt_'+str(i)+suffix in d for i in (1,2)):
            with np.errstate(invalid='ignore',divide='ignore'):
                s1=d['a_1']*np.cos(d['tilt_1'+suffix]); s2=d['a_2']*np.cos(d['tilt_2'+suffix])
                chi=(d['mass_1']*s1+d['mass_2']*s2)/(d['mass_1']+d['mass_2'])
            item['identity_checks']['from_evolved_tilts']=difference_summary(d[key],chi)
            for i,s in ((1,s1),(2,s2)):
                if 'spin_'+str(i)+'z'+suffix in d:
                    item['identity_checks']['spin_'+str(i)+'z']=difference_summary(d['spin_'+str(i)+'z'+suffix],s)
        evolved[suffix]=item
    return {'rows':int(n),'nonfinite_by_field':finite,'physical_violations':physical,
            'algebraic_identities':checks,'cosmology':cosmology,'recorded_bounds':bounds,
            'missing_required_bound_metadata':missing,'mass_support':support,
            'distance_prior_comparison':grid_comparison,'evolved_spin':evolved,
            'reference_frequency_column_present':'reference_frequency' in d,
            'released_prior_validated':False,'common_spin_coordinate_validated':False}
