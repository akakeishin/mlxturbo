"""Metal-free unit tests for mlxturbo/sam.py: SAM matching statistics must
agree with a naive O(n^3) brute-force longest-repeated-suffix search on
known sequences (D3 acceptance criterion (a)).
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlxturbo.sam import SuffixAutomaton


def naive_longest_repeated_suffix(seq, n):
    """Longest L such that seq[n-L:n] also occurs ending at some position
    e < n-1 (i.e. a genuinely earlier occurrence, not the trivial tail
    itself). Returns (L, e) with e = None when L == 0.
    """

    for length in range(n, 0, -1):
        pattern = seq[n - length : n]
        for end in range(n - 2, length - 2, -1):
            if seq[end - length + 1 : end + 1] == pattern:
                return length, end
    return 0, None


def _check_against_naive(seq):
    sam = SuffixAutomaton()
    for n, token in enumerate(seq, start=1):
        sam.extend(token)
        assert len(sam) == n

        naive_len, naive_end = naive_longest_repeated_suffix(seq, n)
        sam_len, sam_end = sam.longest_match()

        assert sam_len == naive_len, (
            f"n={n} seq={seq[:n]}: sam matched {sam_len}, naive {naive_len}"
        )
        if naive_len == 0:
            assert sam_end is None
            continue
        # The automaton is free to report *a* different valid earlier
        # occurrence than naive's; verify it independently instead of
        # requiring the same end index.
        assert sam_end is not None and sam_end < n - 1
        assert seq[sam_end - sam_len + 1 : sam_end + 1] == seq[n - sam_len : n]


def test_known_sequence_matches_naive():
    # "abcabcabc"-style repeat with a distractor prefix, using small ints as
    # the "token ids" (SAM must not assume a bounded/contiguous alphabet).
    seq = [7, 1, 2, 3, 1, 2, 3, 1, 2, 3, 9, 1, 2, 4]
    _check_against_naive(seq)


def test_no_repeats_ever_matches():
    seq = list(range(30))  # every token distinct: no repeat is possible
    sam = SuffixAutomaton()
    for token in seq:
        sam.extend(token)
        assert sam.longest_match() == (0, None)


def test_single_token_alphabet():
    seq = [5] * 20
    _check_against_naive(seq)


def test_immediate_and_overlapping_repeats():
    seq = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 3, 1, 2, 3, 1, 2]
    _check_against_naive(seq)


def test_random_small_vocab_sequences():
    rng = random.Random(0)
    for _ in range(20):
        length = rng.randint(20, 120)
        vocab = rng.randint(2, 5)
        seq = [rng.randrange(vocab) for _ in range(length)]
        _check_against_naive(seq)


def test_draft_reads_a_genuine_continuation():
    # After the second "1,2,3,4", the longest repeated suffix is length 4,
    # matched against the earlier occurrence at indices 0..3; what follows
    # that earlier occurrence (index 4 onward: "9,9,1,...") is a genuine,
    # independently-checkable continuation.
    head = [1, 2, 3, 4, 9, 9, 1, 2, 3, 4]
    sam = SuffixAutomaton()
    sam.extend_all(head)
    match_len, end = sam.longest_match()
    assert match_len == 4
    assert end == 3
    assert sam.draft(max_len=3) == head[end + 1 : end + 4] == [9, 9, 1]
    assert sam.draft(max_len=100) == head[end + 1 :]
    assert sam.draft(max_len=100, min_len=5) is None  # match_len (4) < min_len


def test_draft_reads_most_recent_when_ngram_repeats_twice():
    # D-12 (docs/research/REVIEW-2026-09-02-INDEPENDENT.md): draft() must
    # follow the docstring's "most recent earlier occurrence", not the
    # first one. "1,2,3" occurs three times, with a different continuation
    # after each of the first two; once the third occurrence completes the
    # tail, the automaton must read the continuation from the *second*
    # (more recent) occurrence, not the first.
    seq = [1, 2, 3, 4, 4, 1, 2, 3, 5, 5, 1, 2, 3]
    sam = SuffixAutomaton()
    sam.extend_all(seq)
    match_len, end = sam.longest_match()
    assert match_len == 3
    assert end == 7  # the second occurrence (indices 5-7), not the first (0-2)
    assert sam.draft(max_len=2) == [5, 5]


def test_draft_none_when_nothing_repeats():
    sam = SuffixAutomaton()
    sam.extend_all([1, 2, 3, 4, 5])
    assert sam.draft(max_len=8) is None


def test_extend_o1_amortized_state_growth_is_linear():
    # Not a timing assertion (that belongs to the GPU-adjacent long-generation
    # gate), but the classic SAM invariant: state count stays O(n), never
    # blows up quadratically even under heavy repetition.
    seq = ([1, 2, 3] * 400) + [4, 5, 6]
    sam = SuffixAutomaton()
    sam.extend_all(seq)
    assert len(sam._states) <= 2 * len(seq) + 2


def test_peek_match_equals_real_extend_without_mutation():
    # peek_match(token) は extend(token) 後の longest_match() と一致し、
    # かつ状態を一切変えない。全 token でランダム列に対して照合する。
    import random

    rng = random.Random(7)
    for _ in range(30):
        seq = [rng.randrange(5) for _ in range(rng.randrange(2, 60))]
        sam = SuffixAutomaton()
        sam.extend_all(seq)
        before = (sam._match_state, sam._match_len, len(sam._states))
        for token in range(5):
            peek_len, peek_end = sam.peek_match(token)
            assert (sam._match_state, sam._match_len, len(sam._states)) == before
            twin = SuffixAutomaton()
            twin.extend_all(seq)
            twin.extend(token)
            real_len, _ = twin.longest_match()
            assert peek_len == real_len, (seq, token, peek_len, real_len)
            if peek_len:
                # endpos は「その一致がかつて終わった位置」なので、そこから
                # 遡った peek_len トークンが suffix+token と一致する。
                ext = seq + [token]
                occurrence = seq[peek_end - peek_len + 1 : peek_end + 1]
                assert occurrence == ext[-peek_len:], (seq, token)


def test_draft_after_reads_genuine_continuation():
    # [9, 1, 2, 3, 8, ... 1, 2] で token=3 の仮延長: 一致 [1,2,3] の続き 8 が返る。
    sam = SuffixAutomaton()
    sam.extend_all([9, 1, 2, 3, 8, 7, 6, 1, 2])
    match_len, cont = sam.draft_after(3, max_len=2, min_len=3)
    assert match_len == 3
    assert cont == [8, 7]
    # min_len に届かない場合は None
    match_len, cont = sam.draft_after(3, max_len=2, min_len=4)
    assert cont is None
    # 出現しない延長は (0, None)
    match_len, cont = sam.draft_after(5, max_len=2)
    assert match_len == 0 and cont is None


def main():
    tests = [
        test_known_sequence_matches_naive,
        test_no_repeats_ever_matches,
        test_single_token_alphabet,
        test_immediate_and_overlapping_repeats,
        test_random_small_vocab_sequences,
        test_draft_reads_a_genuine_continuation,
        test_draft_reads_most_recent_when_ngram_repeats_twice,
        test_draft_none_when_nothing_repeats,
        test_extend_o1_amortized_state_growth_is_linear,
        test_peek_match_equals_real_extend_without_mutation,
        test_draft_after_reads_genuine_continuation,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")


if __name__ == "__main__":
    main()
