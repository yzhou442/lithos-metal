import json
from pathlib import Path

import pytest

from monolith.backends.metal.m5_max_40c import serving

ROOT = Path(serving.__file__).parent / 'recipes' / 'dspark'


def test_overlays_are_off_by_default(monkeypatch):
    monkeypatch.delenv(serving.OVERLAY_ENV, raising=False)
    sel = {'4096': {'target': {'attention': {'fuse': True}}}}
    assert serving.apply_overlays(json.loads(json.dumps(sel))) == sel


def test_named_overlay_merges_its_contexts_only(tmp_path):
    (tmp_path / 'numerics-overlays.json').write_text(json.dumps(
        {'x': {'_doc': 'd', '4096': {'target': {'attention': {'a': 1, 'fuse': None}}}, '9': {'target': {'z': 1}}}}))
    sel = {'4096': {'target': {'attention': {'fuse': True, 'b': 2}}}, '128': {'target': {}}}
    out = serving.apply_overlays(sel, 'x', root=tmp_path)
    assert out['4096']['target']['attention'] == {'b': 2, 'a': 1}
    assert out['128'] == {'target': {}} and '9' not in out


def test_env_names_overlays_and_unknown_names_fail(monkeypatch, tmp_path):
    (tmp_path / 'numerics-overlays.json').write_text(json.dumps({'x': {'128': {'target': {'k': 1}}}}))
    monkeypatch.setenv(serving.OVERLAY_ENV, ' x , ')
    assert serving.apply_overlays({'128': {'target': {}}}, root=tmp_path) == {'128': {'target': {'k': 1}}}
    with pytest.raises(ValueError, match='unknown numerics overlay'):
        serving.apply_overlays({'128': {}}, 'y', root=tmp_path)


def test_shipped_overlays_patch_measured_contexts_and_record_a_gate():
    overlays = json.loads((ROOT / 'numerics-overlays.json').read_text())
    contexts = set(json.loads((ROOT / 'selected-contexts.json').read_text())) | set(
        json.loads((ROOT / 'selected-nvfp4-endpoints.json').read_text()))
    for name, ov in overlays.items():
        if name.startswith('_'):
            continue
        assert '_gate' in ov, name
        assert {k for k in ov if not k.startswith('_')} <= contexts, name
