from oqpes.reporting import build_pair_checks
from oqpes.benchmark import RESULT_FIELDS


def test_pair_checks_use_revised_method_labels_and_multiple_optimum_semantics():
    rows=[
        {"instance_id":"I1","method":"ALIC_REVISED","status":"OPTIMAL","objective":"10","x":"[1,0]","wall_time":"1.0","lp_calls":"0","ilp_calls":"1","milp_calls":"1","sha256":"abc"},
        {"instance_id":"I1","method":"PRERNA_SHARMA_2024_RECONSTRUCTION","status":"OPTIMAL","objective":"10","x":"[0,1]","wall_time":"2.0","lp_calls":"0","ilp_calls":"5","milp_calls":"1","sha256":"abc"},
    ]
    out=build_pair_checks(rows)
    assert len(out)==1
    r=out[0]
    assert r["both_optimal"] is True
    assert r["objective_agreement"] is True
    assert r["decision_agreement"] is False
    assert r["outcome"] == "MULTIPLE_OPTIMAL_DECISIONS"
    assert r["sha_match"] is True


def test_result_fields_include_generation_and_revised_counters():
    required={
        "generation_mode","accepted_attempt","rejected_total","rejection_reasons","unit_feasible_count",
        "iterations","primary_pass_calls","secondary_pass_calls","secondary_ilp_calls","secondary_milp_calls",
        "uniqueness_checks","uniqueness_certificates","affine_checks","affine_certificates",
        "preprocessing_wall_time",
    }
    assert required.issubset(set(RESULT_FIELDS))
