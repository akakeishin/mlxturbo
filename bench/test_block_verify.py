"""Monte Carlo distribution-equivalence test for D4 Block Verification.

``SpecEngine._block_verify_tau`` (fastmlx/spec.py) is pure Python (no mlx
arrays), so this only needs ``random`` -- no Metal device required.

D4 acceptance criterion (a): with a synthetic small-vocab distribution,
confirm via Monte Carlo (1e4 samples, TV-distance threshold) that the
current sequential rejection sampler and the new block verification
sampler produce the *same* output distribution.
"""

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastmlx.spec import SpecEngine


def sequential_tau(p_l, u_l, n_avail):
    """The sequential rejection sampler this replaces (spec.py's previous
    ``while a < n_avail and u_l[a] < p_l[a]: a += 1``), reimplemented here
    standalone so the test has an independent "current method" reference
    that keeps working after D4 removes it from spec.py.
    """

    a = 0
    while a < n_avail and u_l[a] < p_l[a]:
        a += 1
    return a


def _residual_sample(row_probs, exclude_index, rng):
    """Sample from target row_probs, renormalized to exclude exclude_index
    (or the plain row when exclude_index is None) -- the same "mask one
    token out of the target distribution" step both samplers use for the
    resampled token.
    """

    weights = list(row_probs)
    if exclude_index is not None:
        weights[exclude_index] = 0.0
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


def _run_round(tau_fn, p_l, target_rows, draft_tokens, n_avail, rng):
    u_l = [rng.random() for _ in range(n_avail)]
    tau = tau_fn(p_l, u_l, n_avail)
    if tau == n_avail:
        y = _residual_sample(target_rows[tau], None, rng)
    else:
        exclude = draft_tokens[tau]  # the next drafted token, masked out
        y = _residual_sample(target_rows[tau], exclude, rng)
    return tau, y


def _row_dist(rng, vocab):
    weights = [rng.random() + 0.05 for _ in range(vocab)]
    total = sum(weights)
    return [w / total for w in weights]


def _tv_distance(counter_a, counter_b, n_a, n_b, support):
    total = 0.0
    for key in support:
        pa = counter_a.get(key, 0) / n_a
        pb = counter_b.get(key, 0) / n_b
        total += abs(pa - pb)
    return total / 2.0


def _monte_carlo_case(seed, vocab, n_avail, n_samples, tv_threshold):
    rng_setup = random.Random(seed)
    target_rows = [_row_dist(rng_setup, vocab) for _ in range(n_avail + 1)]
    # Deterministic drafted tokens: argmax of a *different* fixed
    # distribution, exactly like fastmlx's MTP/lookup draft is a fixed
    # proposal independent of the target model being verified against.
    draft_rows = [_row_dist(rng_setup, vocab) for _ in range(n_avail)]
    draft_tokens = [max(range(vocab), key=lambda i: row[i]) for row in draft_rows]
    p_l = [target_rows[i][draft_tokens[i]] for i in range(n_avail)]

    counts_seq = Counter()
    counts_block = Counter()
    rng_seq = random.Random(seed * 7 + 1)
    rng_block = random.Random(seed * 7 + 2)
    for _ in range(n_samples):
        counts_seq[_run_round(sequential_tau, p_l, target_rows, draft_tokens, n_avail, rng_seq)] += 1
        counts_block[
            _run_round(
                SpecEngine._block_verify_tau,
                p_l,
                target_rows,
                draft_tokens,
                n_avail,
                rng_block,
            )
        ] += 1

    support = {(tau, y) for tau in range(n_avail + 1) for y in range(vocab)}
    tv = _tv_distance(counts_seq, counts_block, n_samples, n_samples, support)
    assert tv < tv_threshold, (
        f"seed={seed}: TV distance {tv:.4f} >= threshold {tv_threshold} "
        f"between sequential and block verification output distributions"
    )
    return tv


def test_block_verification_matches_sequential_distribution():
    # 1e4 samples per method as specified by the D4 acceptance criterion;
    # several small-vocab configurations to avoid a single lucky seed.
    for seed in range(5):
        tv = _monte_carlo_case(
            seed=seed, vocab=5, n_avail=4, n_samples=10_000, tv_threshold=0.03
        )
        print(f"  seed={seed}: TV(sequential, block) = {tv:.4f}")


def test_block_verify_never_beats_sequential_in_tau_but_matches_in_law():
    """Sanity check on the acceptance-length claim (Theorem 2): expected
    accepted length should be statistically indistinguishable (not worse)
    between the two samplers for a deterministic draft -- fastmlx's draft
    proposal has no entropy for block verification to exploit, so unlike
    the paper's stochastic-drafter setting the expected gain here is ~0,
    not a regression. See docs/STATUS.md for the derivation.
    """

    rng_setup = random.Random(42)
    vocab, n_avail = 4, 3
    target_rows = [_row_dist(rng_setup, vocab) for _ in range(n_avail + 1)]
    draft_rows = [_row_dist(rng_setup, vocab) for _ in range(n_avail)]
    draft_tokens = [max(range(vocab), key=lambda i: row[i]) for row in draft_rows]
    p_l = [target_rows[i][draft_tokens[i]] for i in range(n_avail)]

    n = 20_000
    rng = random.Random(123)
    seq_sum = 0
    block_sum = 0
    for _ in range(n):
        u_l = [rng.random() for _ in range(n_avail)]
        seq_sum += sequential_tau(p_l, u_l, n_avail)
        block_sum += SpecEngine._block_verify_tau(p_l, u_l, n_avail)
    e_seq, e_block = seq_sum / n, block_sum / n
    assert e_block >= e_seq - 0.05, (
        f"block verification expected accepted length {e_block:.4f} regressed "
        f"below sequential's {e_seq:.4f}"
    )
    print(f"  E[tau_sequential]={e_seq:.4f} E[tau_block]={e_block:.4f}")


def main():
    tests = [
        test_block_verification_matches_sequential_distribution,
        test_block_verify_never_beats_sequential_in_tau_but_matches_in_law,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


if __name__ == "__main__":
    main()
