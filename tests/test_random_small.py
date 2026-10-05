from oqpes.generator import generate_instance, GeneratorSettings
from oqpes.validate import brute_force_optimum
from oqpes.alic import solve_alic
from oqpes.prerna_sharma_2024 import solve_prerna_sharma_2024


def test_small_random_agreement():
    settings = GeneratorSettings(mode="conditioned_nontrivial", min_unit_points=2)
    for rep in range(1, 4):
        inst = generate_instance(3, 4, 3, rep, settings)
        _, val, _, _ = brute_force_optimum(inst, max_box_points=200000)
        a = solve_alic(inst, time_limit=10)
        p = solve_prerna_sharma_2024(inst, time_limit=10)
        assert a.status == "OPTIMAL"
        assert p.status == "OPTIMAL"
        assert a.objective == val
        assert p.objective == val
