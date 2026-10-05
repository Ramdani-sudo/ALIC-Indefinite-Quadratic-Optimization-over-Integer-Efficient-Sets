import numpy as np
from oqpes.instance import Instance
from oqpes.prerna_sharma_2024 import solve_prerna_sharma_2024


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


def test_reference_reconstruction_label_and_56_calls():
    r=solve_prerna_sharma_2024(reference_instance(), time_limit=30)
    assert r.method == "PRERNA_SHARMA_2024_RECONSTRUCTION"
    assert r.status == "OPTIMAL"
    assert r.objective == 820
    assert r.ilp_calls == 50
    assert r.milp_calls == 6
    assert r.preprocessing_calls == 2
    assert r.preprocessing_wall_time >= 0
    assert r.ilp_calls + r.milp_calls == 56
