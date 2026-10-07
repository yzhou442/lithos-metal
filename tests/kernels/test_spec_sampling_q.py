"""Exact speculative sampling with sampled drafts (spec_sampling="q"; sglang's chain rule): drafts d_k ~ q_k =
softmax(corrected_k / T), accepted with min(1, p/q), the first rejection corrected from norm(max(p - q, 0)), the bonus
from p. The committed tokens must be distributed exactly as plain (non-speculative) sampling at the same temperature,
top-k and top-p: chi-square homogeneity of the first three generated tokens (marginals, and the joint of the first
two) between two independent seed sets; greedy (T = 0) is unchanged by the flag; the rule accepts more drafts than the
point-mass rule. The synthetic hybrid target (GDN + attention) and the random synthetic DSpark drafter of
test_spec_rollback."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "contract"))
from dspark_synth import build, write_checkpoint  # noqa: E402
from test_nn_lowering import _checkpoint  # noqa: E402

from monolith.formats import PackLayout  # noqa: E402
from monolith.generate import Session  # noqa: E402
from monolith.models.qwen3_5 import Qwen3_5Model  # noqa: E402
from monolith.nn.pack_plan import pack_model  # noqa: E402


@pytest.fixture(scope="module")
def packs(tmp_path_factory):
    root = tmp_path_factory.mktemp("specq")
    tdir, ddir = root / "target", root / "drafter"
    tdir.mkdir(); ddir.mkdir()
    _checkpoint(tdir)
    model = Qwen3_5Model.from_checkpoint(str(tdir), max_context=64)
    pack_model(model, str(tdir), str(tdir / "pack"), PackLayout(rows=16))
    write_checkpoint(ddir, seed=5, with_head=False, vocab_size=50, target_hidden_size=256, target_layer_ids=[-1, 1], block_size=3)
    drafter, _, _, _ = build(ddir, target_lm_head=model.lm_head)
    pack_model(drafter, str(ddir), str(ddir / "pack"), PackLayout(rows=16))
    return tdir, ddir


def _model(tdir):
    return Qwen3_5Model.from_checkpoint(str(tdir), max_context=64)


VERIFY = {
    "whole": dict(verify="threshold", verify_threshold=0.0),                       # every draft verified
    "cost": dict(verify="cost", verify_cost=[1.0, 1.0, 1.25, 1.9]),               # the cost rule often stops before the block's end
    "lookup": dict(verify="fixed", verify_length=3),                               # + context-lookup rows (point-mass proposals) after it
}


def _sessions(packs, mode="whole", **sampling):
    tdir, ddir = packs
    model = _model(tdir)
    drafter, _, _, _ = build(ddir, target_lm_head=model.lm_head)
    if mode == "lookup":
        drafter.lookup = {"nmin": 1}                                               # one-token matches: lookup rows in most rounds
    # one attention kernel for both programs (see test_spec_rollback); top-p within top-k for both (the q rule's default)
    plain = Session(_model(tdir), str(tdir / "pack"), eos=-1, autotune=False, attention="v2", topp_in_topk=True, **sampling)
    spec = Session(model, str(tdir / "pack"), eos=-1, autotune=False, attention="v2", drafter=drafter, drafter_pack=str(ddir / "pack"),
                   spec_sampling="q", **VERIFY[mode], **sampling)
    return plain, spec


def _chi2_homogeneity(a, b, min_expected=5.0):
    """Two-sample chi-square over categories pooled until each expected count reaches ``min_expected``; returns
    (statistic, degrees of freedom, a z-score by Wilson-Hilferty)."""
    keys = sorted(set(a) | set(b), key=lambda k: -(a.get(k, 0) + b.get(k, 0)))
    na, nb = sum(a.values()), sum(b.values())
    cells, ca, cb = [], 0, 0
    for k in keys:
        ca += a.get(k, 0); cb += b.get(k, 0)
        if (ca + cb) * min(na, nb) / (na + nb) >= min_expected:
            cells.append((ca, cb)); ca = cb = 0
    if ca + cb:
        if cells:
            x, y = cells.pop(); cells.append((x + ca, y + cb))
        else:
            cells.append((ca, cb))
    stat = 0.0
    for x, y in cells:
        tot = x + y
        ea, eb = tot * na / (na + nb), tot * nb / (na + nb)
        stat += (x - ea) ** 2 / ea + (y - eb) ** 2 / eb
    df = max(len(cells) - 1, 1)
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return stat, df, z


def _counts(seqs, key):
    c = {}
    for s in seqs:
        k = key(s)
        c[k] = c.get(k, 0) + 1
    return c


@pytest.mark.parametrize("mode,sampling", [("whole", dict(temperature=0.9, top_k=12, top_p=0.9)), ("whole", dict(temperature=1.6)),
                                           ("whole", dict(temperature=0.6, top_k=8)), ("cost", dict(temperature=1.0, top_k=20, top_p=0.95)),
                                           ("lookup", dict(temperature=1.0, top_k=20, top_p=0.95))],
                         ids=["t0.9-k12-p0.9", "t1.6", "t0.6-k8", "cost-t1-k20-p.95", "lookup-t1-k20-p.95"])
def test_q_rule_preserves_the_target_distribution(packs, mode, sampling):
    plain, spec = _sessions(packs, mode, **sampling)
    ids = [7, 23, 41, 3, 9, 7, 23, 41] if mode == "lookup" else [7, 23, 41, 3]
    n_seeds, n_new = 1500, 3
    plain_seqs, spec_seqs, accepted, vlen = [], [], [], []
    for seed in range(n_seeds):
        plain.seed, spec.seed = seed, seed + 100_000
        plain_seqs.append(tuple(plain.generate(ids, n_new).tokens))
        g = spec.generate(ids, n_new)
        spec_seqs.append(tuple(g.tokens))
        accepted += g.accepted or []
        vlen += g.verify_len or []
    zs = {}
    for name, key in (("t0", lambda s: s[0]), ("t1", lambda s: s[1]), ("t2", lambda s: s[2]), ("t0t1", lambda s: s[:2])):
        stat, df, z = _chi2_homogeneity(_counts(plain_seqs, key), _counts(spec_seqs, key))
        zs[name] = z
        print(f"\n{sampling} {name}: chi2 {stat:.1f} df {df} z {z:.2f}")
    acc_rate = sum(accepted) / max(1, len(accepted))
    hist = {L: vlen.count(L) for L in sorted(set(vlen))}
    print(f"{mode} {sampling}: mean accepted drafts per round {acc_rate:.3f} over {len(accepted)} rounds; verify lengths {hist}; "
          f"distinct sequences {len(set(spec_seqs))}")
    if mode == "cost":
        assert any(0 < L < 3 for L in vlen)                                 # the rule did stop inside the block
    if mode == "lookup":
        assert any(L > 3 for L in vlen)                                     # lookup rows were verified
    # a two-sample chi-square under the null: z ~ N(0, 1); 4 sigma over 4 statistics x 3 configurations
    assert all(z < 4.0 for z in zs.values()), zs
    assert len(set(spec_seqs)) > 10
    assert sum(accepted) > 0                                            # drafts do get accepted (the accept path is exercised)


def test_q_rule_accepts_more_than_the_point_mass_rule(packs):
    """At a high temperature the sampled-draft rule accepts at least as many drafts as the point-mass rule
    (sum min(p, q) >= p(argmax q) is not guaranteed per position, but on these flat distributions it holds clearly)."""
    tdir, ddir = packs
    rates = {}
    for rule in ("match", "q"):
        model = _model(tdir)
        drafter, _, _, _ = build(ddir, target_lm_head=model.lm_head)
        s = Session(model, str(tdir / "pack"), eos=-1, autotune=False, attention="v2", drafter=drafter, drafter_pack=str(ddir / "pack"),
                    verify="threshold", verify_threshold=0.0, spec_sampling=rule, temperature=2.5)
        acc = []
        for seed in range(300):
            s.seed = seed
            acc += s.generate([7, 23, 41, 3], 12).accepted or []
        rates[rule] = sum(acc) / max(1, len(acc))
    print(f"\nmean accepted per round at T=2.5: {rates}")
    assert rates["q"] > rates["match"]


def test_greedy_is_unchanged_by_the_flag(packs):
    tdir, ddir = packs
    plain = Session(_model(tdir), str(tdir / "pack"), eos=-1, autotune=False)
    model = _model(tdir)
    drafter, _, _, _ = build(ddir, target_lm_head=model.lm_head)
    spec = Session(model, str(tdir / "pack"), eos=-1, autotune=False, drafter=drafter, drafter_pack=str(ddir / "pack"), verify="threshold",
                   spec_sampling="q")
    assert drafter.sampling is None
    rng = np.random.default_rng(3)
    for n_prompt, n_new in ((5, 24), (11, 20), (1, 12)):
        ids = [int(x) for x in rng.integers(0, 50, n_prompt)]
        assert spec.generate(ids, n_new).tokens == plain.generate(ids, n_new).tokens
