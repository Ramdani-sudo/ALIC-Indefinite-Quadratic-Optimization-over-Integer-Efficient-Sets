import numpy as np

from oqpes.generator import GeneratorSettings, generate_instance, benchmark_seed


def test_complete_draw_rejection_reproduces_published_campaign_hashes():
    s = GeneratorSettings(mode="paper_range_complete_draw_rejection", min_unit_points=1)
    expected = {
        1: ("c51606750f1e6ce4f9ed717f30ad86c6af6ce5f77843dd29a8088dc587c6e544", 1, 1),
        2: ("3a3018d8a4163d08a8daabc401a838641700abd18da0ff288ed71315cd064c99", 1, 5),
        4: ("461f90d7fbedd996c429317e2a9b366eb6afb87653b27854acb786699196aa26", 2, 2),
    }
    for rep, (sha, attempt, units) in expected.items():
        inst = generate_instance(5, 10, 10, rep, s)
        assert inst.canonical_hash() == sha
        assert inst.meta["accepted_attempt"] == attempt
        assert inst.meta["unit_feasible_count"] == units
        assert inst.meta["seed"] == benchmark_seed(5, 10, 10, rep)


def test_rejection_metadata_counts_all_failed_reasons():
    s = GeneratorSettings(mode="paper_range_complete_draw_rejection", min_unit_points=1)
    inst = generate_instance(5, 10, 10, 4, s)
    assert inst.meta["rejected_total"] == 1
    assert inst.meta["rejection_reasons"] == {
        "constant_preference_probe": 1,
        "no_nonzero_unit_feasible_point": 1,
    }


def test_generated_quadratic_matrix_is_strictly_indefinite():
    s = GeneratorSettings(mode="paper_range_complete_draw_rejection", min_unit_points=1)
    inst = generate_instance(20, 10, 20, 3, s)
    ev = np.linalg.eigvalsh(inst.Q.astype(float))
    assert ev[0] < 0 < ev[-1]
