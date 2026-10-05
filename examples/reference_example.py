from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oqpes.instance import Instance
from oqpes.validate import brute_force_optimum
from oqpes.alic import solve_alic
from oqpes.prerna_sharma_2024 import solve_prerna_sharma_2024

inst = Instance(
    A=np.array([[1, -3], [16, 3]], dtype=int),
    b=np.array([0, 51], dtype=int),
    C=np.array([[5, -7], [-6, 5]], dtype=int),
    Q=np.array([[2, 0], [0, -8]], dtype=int),
    d=np.array([81, 158], dtype=int),
    alpha=40,
)

print("Upper bounds:", inst.ensure_upper_bounds())
print("Brute force:", brute_force_optimum(inst))
print("ALIC:", solve_alic(inst, time_limit=30, linearization_mode="auto"))
print("Prerna-Sharma:", solve_prerna_sharma_2024(inst, time_limit=30))
