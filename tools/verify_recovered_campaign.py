from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from oqpes.generator import GeneratorSettings, generate_instance

# Known SHA-256 checkpoints from the archived final 1440-instance campaign.
CHECKPOINTS = {
    (5, 10, 10, 1): "c51606750f1e6ce4f9ed717f30ad86c6af6ce5f77843dd29a8088dc587c6e544",
    (5, 10, 10, 2): "3a3018d8a4163d08a8daabc401a838641700abd18da0ff288ed71315cd064c99",
    (5, 10, 10, 4): "461f90d7fbedd996c429317e2a9b366eb6afb87653b27854acb786699196aa26",
}

settings = GeneratorSettings(
    mode="paper_range_complete_draw_rejection",
    min_unit_points=1,
)

mismatches = []
for key, expected in CHECKPOINTS.items():
    inst = generate_instance(*key, settings=settings)
    got = inst.canonical_hash()
    print(f"{key}: {got}")
    if got != expected:
        mismatches.append((key, expected, got))

expected_count = 4 * 3 * 4 * 30
print(f"Extended protocol instance count: {expected_count}")

if mismatches:
    for key, expected, got in mismatches:
        print(f"Mismatch for {key}: expected {expected}, got {got}")
    raise SystemExit(1)

print("Generator checkpoints match the archived final-campaign hashes.")
print("The complete 1440/1440 hash verification performed during recovery is documented in VERIFICATION_REPORT.md.")
