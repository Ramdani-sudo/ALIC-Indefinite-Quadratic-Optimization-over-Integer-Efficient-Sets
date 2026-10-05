import numpy as np
from oqpes.generator import generate_instance, GeneratorSettings, benchmark_seed


def test_generator_reproducible_nontrivial():
    s = GeneratorSettings(mode="conditioned_nontrivial", min_unit_points=3)
    a = generate_instance(5, 10, 10, 1, s)
    b = generate_instance(5, 10, 10, 1, s)
    assert a.canonical_hash() == b.canonical_hash()
    assert a.meta["seed"] == benchmark_seed(5, 10, 10, 1)
    assert a.meta["unit_feasible_count"] >= 3
    assert np.any(a.u > 0)


def test_q_strictly_indefinite():
    inst = generate_instance(5, 10, 10, 2, GeneratorSettings())
    ev = np.linalg.eigvalsh(inst.Q.astype(float))
    assert ev[0] < 0 < ev[-1]
