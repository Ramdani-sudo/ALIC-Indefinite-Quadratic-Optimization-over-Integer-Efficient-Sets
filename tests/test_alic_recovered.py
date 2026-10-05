import numpy as np

from oqpes.instance import Instance
from oqpes.generator import GeneratorSettings, generate_instance
from oqpes.alic import solve_alic_revised
from oqpes.completion import certify_primary_uniqueness, certify_affine_on_primary_level


def reference_instance():
    return Instance(
        A=np.array([[1, -3], [16, 3]], dtype=int),
        b=np.array([0, 51], dtype=int),
        C=np.array([[5, -7], [-6, 5]], dtype=int),
        Q=np.array([[2, 0], [0, -8]], dtype=int),
        d=np.array([81, 158], dtype=int),
        alpha=40,
        u=np.array([3, 17], dtype=int),
    )


def test_reference_example_reproduces_revised_13_call_trace():
    r = solve_alic_revised(reference_instance(), time_limit=30)
    assert r.status == "OPTIMAL"
    assert r.objective == 820
    assert tuple(r.x) == (0, 10)
    assert r.master_calls == 7
    assert r.primary_pass_calls == 6
    assert r.secondary_pass_calls == 0
    assert r.uniqueness_checks == 6
    assert r.uniqueness_certificates == 6
    assert r.ilp_calls + r.milp_calls == 13


def test_reference_uniqueness_and_affine_certificates_are_exact():
    inst = reference_instance()
    ok, details = certify_primary_uniqueness(inst, np.array([1, 10]), 3)
    assert ok
    assert details["unique_x"] == [0, 9]
    cert = certify_affine_on_primary_level(inst, np.array([0, 9]), np.sum(inst.C, axis=0))
    assert cert.is_affine
    assert cert.gradient.tolist() == [81, 14]


def test_generated_secondary_milp_trace_matches_campaign():
    s = GeneratorSettings(mode="paper_range_complete_draw_rejection", min_unit_points=1)
    inst = generate_instance(5, 10, 10, 9, s)
    r = solve_alic_revised(inst, time_limit=30)
    assert r.status == "OPTIMAL"
    assert r.objective == 0
    assert r.master_calls == 2
    assert r.primary_pass_calls == 1
    assert r.secondary_pass_calls == 1
    assert r.secondary_milp_calls == 1
    assert r.ilp_calls + r.milp_calls == 4


def test_stall_regression_p_lt_n_returns_without_symbolic_rank_hang():
    s = GeneratorSettings(mode="paper_range_complete_draw_rejection", min_unit_points=1)
    inst = generate_instance(5, 10, 50, 11, s)
    r = solve_alic_revised(inst, time_limit=30)
    assert r.status == "OPTIMAL"
    assert r.objective == 0
    assert r.master_calls == 2
    assert r.secondary_pass_calls == 1
    assert r.uniqueness_certificates == 0


def test_certified_optimal_result_reports_closed_bound_and_zero_gap():
    r = solve_alic_revised(reference_instance(), time_limit=30)
    assert r.status == "OPTIMAL"
    assert r.dual_bound == r.objective
    assert r.gap == 0.0
