"""Explicit bridge from a declared chirp-ratio rectangle to the old adapter.

The component-mass enclosure is derived geometry, not fabricated Constraint
metadata. The original reference and provenance are retained next to the
existing mass/distance implementation. No old family recognizer is relaxed.
"""

from dataclasses import dataclass
import math

from scripts.audit_population_prior_bindings import candidate_binding
from scripts.audit_prior_semantics import fingerprint
from src.chirp_ratio_prior_reference import declared_chirp_ratio_reference, ChirpRatioOnlyMassPrior
from src.pe_prior_reference import FlatDetectorMassPrior, SourceFrameDistancePrior
from src.analysis_coordinates import MassDistancePriorInAnalysisCoordinates


@dataclass(frozen=True)
class DeclaredChirpRatioEventBinding:
    original_mass_reference: ChirpRatioOnlyMassPrior
    analysis_prior: MassDistancePriorInAnalysisCoordinates
    analytic_metadata_sha256: str
    equivalent_area_relative_error: float
    component_enclosure_origin: str = 'derived_nonrestrictive_geometry_NOT_recorded_constraints'
    released_row_density_certified: bool = False

    def require_production_ready(self):
        raise RuntimeError('Declared-reference development adapter, not certified historical row density')


def bind_chirp_ratio_event(analytic, *, distance_mode, analysis_cosmology,
                          pe_cosmology, recorded_dictionary_has_no_extra_mass_constraints=False):
    """Construct a checked equivalent reference for the existing event API.

    Only the missing mass_1/mass_2 Constraint requirements may differ from
    the older supported family. All its other spin/distance checks still apply.
    """
    original = declared_chirp_ratio_reference(analytic,
        recorded_dictionary_has_no_extra_mass_constraints=recorded_dictionary_has_no_extra_mass_constraints)
    allowed = {kind+':'+name for name in ('mass_1', 'mass_2')
               for kind in ('missing_or_unexpected_prior', 'missing_or_invalid_bounds')}
    issues = set(candidate_binding(analytic)['recognition_issues'])
    if issues != allowed:
        raise ValueError('Additional or unexpected reference-binding issues: '+str(sorted(issues)))
    distance_shape = analytic['luminosity_distance']['parsed']['shape_parameters']
    geometry = FlatDetectorMassPrior(original.equivalent_component_enclosure())
    error = abs(geometry.area/original.area-1)
    if not math.isfinite(error) or error > 2e-10:
        raise ArithmeticError('Equivalent mass geometry failed normalization check')
    distance = SourceFrameDistancePrior(
        (distance_shape['minimum'], distance_shape['maximum']), mode=distance_mode)
    adapter = MassDistancePriorInAnalysisCoordinates(geometry, distance,
        analysis_cosmology=analysis_cosmology, pe_cosmology=pe_cosmology)
    return DeclaredChirpRatioEventBinding(original, adapter, fingerprint(analytic), error)
