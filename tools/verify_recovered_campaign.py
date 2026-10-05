from __future__ import annotations
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from oqpes.instance import Instance

RESULTS = ROOT / "results" / "extended_1440_300s" / "results.csv"
MANIFEST = ROOT / "instances" / "extended_1440" / "benchmark_manifest.csv"

expected = {}
with RESULTS.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        expected.setdefault(row["instance_id"], row["sha256"])

with MANIFEST.open(encoding="utf-8", newline="") as f:
    manifest = list(csv.DictReader(f))

generated = {r["instance_id"]: r for r in manifest}
missing = sorted(set(expected) - set(generated))
extra = sorted(set(generated) - set(expected))
mismatch = []
file_mismatch = []
for iid, sha in expected.items():
    row = generated.get(iid)
    if row is None:
        continue
    if row["sha256"] != sha:
        mismatch.append(iid)
    path = MANIFEST.parent / row["instance_file"]
    if path.exists() and Instance.load_npz(path).canonical_hash() != sha:
        file_mismatch.append(iid)

print(f"Archived result instances: {len(expected)}")
print(f"Generated manifest instances: {len(generated)}")
print(f"Missing: {len(missing)}")
print(f"Extra: {len(extra)}")
print(f"Manifest hash mismatches: {len(mismatch)}")
print(f"NPZ content hash mismatches: {len(file_mismatch)}")
if missing or extra or mismatch or file_mismatch:
    raise SystemExit(1)
print("Recovered campaign manifest and available NPZ files match the final-result hashes.")
