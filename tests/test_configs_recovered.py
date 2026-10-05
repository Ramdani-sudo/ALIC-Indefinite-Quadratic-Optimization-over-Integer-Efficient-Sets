import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT/'config'/name).read_text(encoding='utf-8'))


def test_extended_config_matches_final_1440_campaign():
    c=load('extended_1440_300s.json')
    assert c['grid']['p'] == [5,10,20,50]
    assert c['grid']['n'] == [10,20,50]
    assert c['grid']['m'] == [10,20,30,50]
    assert c['grid']['replications'] == 30
    assert 4*3*4*c['grid']['replications'] == 1440
    assert c['time_limit_seconds'] == 300
    assert c['generation']['mode'] == 'paper_range_complete_draw_rejection'
    assert c['generation']['min_unit_points'] == 1


def test_paper_config_has_480_instances_and_300_seconds():
    c=load('paper_protocol_480_300s.json')
    assert c['grid']['replications'] == 10
    assert 4*3*4*c['grid']['replications'] == 480
    assert c['time_limit_seconds'] == 300


def test_launchers_reference_existing_config_paths():
    for bat, cfg in [
        ('RUN_PILOT_300S.bat','config/pilot_300s.json'),
        ('RUN_PAPER_PROTOCOL_480_300S.bat','config/paper_protocol_480_300S.json'),
        ('RUN_EXTENDED_1440_300S.bat','config/extended_1440_300s.json'),
    ]:
        text=(ROOT/bat).read_text(encoding='utf-8')
        assert cfg in text
        assert (ROOT/cfg).exists()
        assert '--root .' in text
