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


def test_rows8_recipe_patch_applies_to_the_eight_row_program_only():
    from monolith.backends.metal import load_configs
    from monolith.serving.setup import ServingAssets
    profile = load_configs()['apple-m5-max-40c']
    recipes = {'4096': {'target': {'attention': {'fuse': True, 'x': 1}},
                        'target_rows8': {'attention': {'fuse': False, 'big': 64, 'x': None}}}}

    def target(gamma, lookup):
        assets = ServingAssets(Path('m'), Path('p'), 32768, 32774, profile, Path('d'), Path('dp'), gamma, recipes,
                               lookup=lookup)
        return assets.options(5000)[1]['decoder_kernel_config']
    assert target(7, False) == {'attention': {'fuse': False, 'big': 64}}
    assert target(7, True) == recipes['4096']['target']          # the lookup extension compiles sixteen rows
    assert target(15, False) == recipes['4096']['target']        # blocks above 7 too
    assert recipes['4096']['target'] == {'attention': {'fuse': True, 'x': 1}}


def test_named_overlay_replaces_the_rows8_default(tmp_path):
    (tmp_path / 'numerics-overlays.json').write_text(json.dumps(
        {'x': {'4096': {'target': {'attention': {'a': 1}}}, '128': {'draft': {'d': 1}}}}))
    sel = {'4096': {'target': {'attention': {}}, 'target_rows8': {'attention': {'b': 2}}},
           '128': {'target': {}, 'draft': {}, 'target_rows8': {'attention': {'c': 3}}}}
    out = serving.apply_overlays(sel, 'x', root=tmp_path)
    assert 'target_rows8' not in out['4096'] and out['4096']['target'] == {'attention': {'a': 1}}
    assert out['128']['target_rows8'] == {'attention': {'c': 3}}     # an overlay that leaves the target alone keeps it


def test_shipped_rows8_defaults_are_the_byte_identical_big_tile():
    contexts = json.loads((ROOT / 'selected-contexts.json').read_text())
    exact = json.loads((ROOT / 'numerics-overlays.json').read_text())['attention_big_tile_exact']
    patched = {k for k, v in contexts.items() if 'target_rows8' in v}
    assert patched == {'4096', '16384'}
    for key in patched:
        assert contexts[key]['target_rows8'] == exact[key]['target']
        assert contexts[key]['target_rows8']['attention']['attention_big_tile_exact'] is True
