import numpy as np

from oqpes.instance import Instance
from oqpes.validate import brute_force_optimum
from oqpes.alic import solve_alic
from oqpes.prerna_sharma_2024 import solve_prerna_sharma_2024


def reference_instance():
    return Instance(
        A=np.array([[1, -3], [16, 3]], dtype=int),
        b=np.array([0, 51], dtype=int),
        C=np.array([[5, -7], [-6, 5]], dtype=int),
        Q=np.array([[2, 0], [0, -8]], dtype=int),
        d=np.array([81, 158], dtype=int),
        alpha=40,
    )


def test_reference_example_bruteforce():
    inst = reference_instance()
    x, val, ne, nf = brute_force_optimum(inst)
    assert nf == 36
    assert ne == 20
    assert val == 820
    assert tuple(x) == (0, 10)


def test_alic_reference_example():
    inst = reference_instance()
    r = solve_alic(inst, time_limit=10, linearization_mode="auto")
    assert r.status == "OPTIMAL"
    assert r.objective == 820
    assert tuple(r.x) == (0, 10)


def test_prerna_reference_example():
    inst = reference_instance()
    r = solve_prerna_sharma_2024(inst, time_limit=10)
    assert r.status == "OPTIMAL"
    assert r.objective == 820
