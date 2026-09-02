"""Dynamic suffix automaton over the growing generated token sequence.

Online construction (Blumer, Blumer, Haussler, Ehrenfeucht, McConnell & Chen,
1985; see also the textbook incremental-extend presentation used by e.g.
cp-algorithms), generalized from characters to arbitrary token ids:
transitions are a ``dict`` keyed by token id instead of a fixed-alphabet
array, so vocabulary size never enters the automaton's size or the per-token
cost. Extending by one token is O(1) amortized (the standard SAM result:
the total number of states/transitions created over a length-n sequence is
O(n)).

This replaces ``SpecEngine._lookup_draft``'s O(n) backward scan for the most
recent exact repeat of the trailing ngram (RESEARCH.md / SAM-Decoding,
arXiv:2411.10666): the automaton is extended once per newly confirmed token
and already knows the longest repeated suffix in O(1) amortized time
(``longest_match`` / ``draft``), instead of rescanning the whole context on
every decode step.

Only the dynamic (self-generated-text) automaton is implemented here; a
static corpus-side automaton is out of scope for this pass (docs/PLAN.md,
Phase D3).
"""

from __future__ import annotations


class _State:
    __slots__ = ("length", "link", "trans", "endpos")

    def __init__(self, length: int, link: int):
        self.length = length
        self.link = link
        self.trans: dict[int, int] = {}
        # An ending index (0-based, into the automaton's own `_seq`) for this
        # state's substrings, from the (possibly large) endpos set.
        # `_advance_match` refreshes this to the most recent occurrence each
        # time the match exactly fills this state's own length (the state's
        # canonical representative, not merely a shorter member of its
        # class) -- that's the only case where "matching statistics" has
        # actually verified this state's own substring, so it's the only
        # case safe to overwrite. When a transition instead lands mid-state
        # (matched length short of the state's length), this field is left
        # alone and keeps whatever earlier-but-still-genuine occurrence it
        # already had, rather than risk recording a position that was only
        # ever verified for a shorter suffix.
        self.endpos: int | None = None


class SuffixAutomaton:
    """Suffix automaton over a token sequence, extended one token at a time.

    ``extend(token)`` both grows the automaton and advances an incremental
    "matching statistics" cursor that tracks the longest suffix of the
    sequence seen so far which also occurs *earlier* in it. The cursor is
    updated before the automaton is grown for the new token, so a
    successful transition can only have been created by an earlier
    occurrence -- this ordering is what makes ``longest_match()`` report a
    genuine repeat rather than the trivial match against the token that was
    just inserted.
    """

    __slots__ = ("_states", "_last", "_seq", "_match_state", "_match_len")

    def __init__(self):
        self._states = [_State(0, link=-1)]
        self._last = 0  # state representing the whole string seen so far
        self._seq: list[int] = []
        self._match_state = 0  # matching-statistics cursor
        self._match_len = 0

    def __len__(self) -> int:
        return len(self._seq)

    # ---------- build ----------

    def extend(self, token: int) -> None:
        """Append one token to the sequence and update the automaton."""

        self._advance_match(token)
        self._sam_extend(token)
        self._seq.append(token)

    def extend_all(self, tokens) -> None:
        for token in tokens:
            self.extend(token)

    def _advance_match(self, token: int) -> None:
        states = self._states
        state, length = self._match_state, self._match_len
        if length and length == states[state].length:
            states[state].endpos = len(self._seq) - 1
        while state != 0 and token not in states[state].trans:
            state = states[state].link
            length = states[state].length
            if length:
                states[state].endpos = len(self._seq) - 1
        nxt = states[state].trans.get(token)
        if nxt is not None:
            state, length = nxt, length + 1
        else:
            state, length = 0, 0
        self._match_state, self._match_len = state, length

    def _sam_extend(self, token: int) -> None:
        states = self._states
        cur = len(states)
        states.append(_State(states[self._last].length + 1, link=-1))
        # The new token will occupy this index once appended below.
        states[cur].endpos = len(self._seq)
        p = self._last
        while p != -1 and token not in states[p].trans:
            states[p].trans[token] = cur
            p = states[p].link
        if p == -1:
            states[cur].link = 0
        else:
            q = states[p].trans[token]
            if states[p].length + 1 == states[q].length:
                states[cur].link = q
            else:
                clone = len(states)
                states.append(_State(states[p].length + 1, link=states[q].link))
                states[clone].trans = dict(states[q].trans)
                states[clone].endpos = states[q].endpos
                while p != -1 and states[p].trans.get(token) == q:
                    states[p].trans[token] = clone
                    p = states[p].link
                states[q].link = clone
                states[cur].link = clone
        self._last = cur

    # ---------- query ----------

    def longest_match(self) -> tuple[int, int | None]:
        """(match_len, end_index): the longest suffix of the sequence seen
        so far that also occurred earlier, and the index in this
        automaton's own history where one such earlier occurrence ended.
        (0, None) if the current tail has never recurred.
        """

        if self._match_len == 0:
            return 0, None
        return self._match_len, self._states[self._match_state].endpos

    def draft(self, max_len: int, min_len: int = 1) -> list[int] | None:
        """Continuation following the most recent earlier occurrence of the
        current longest repeated suffix, capped at ``max_len`` tokens, or
        None if there is no repeat of at least ``min_len`` tokens.

        "Most recent" holds whenever the match exactly fills its automaton
        state (the common case); see ``_State.endpos`` for why a mid-state
        match instead reads an older, but still genuine, earlier occurrence.
        """

        if max_len <= 0:
            return None
        match_len, end = self.longest_match()
        if match_len < min_len or end is None:
            return None
        cont = self._seq[end + 1 : end + 1 + max_len]
        return cont or None

    def peek_match(self, token: int) -> tuple[int, int | None]:
        """Compute what ``longest_match()`` would return after ``extend(token)``,
        without changing any state (LogitSpec-style extended-key matching,
        docs/RESEARCH.md D7).

        Soundness: the match returned is limited to actual occurrences of
        "suffix of the current sequence + token". The current sequence does not
        yet end with token, so the trailing occurrence of the suffix (the end of
        the sequence) cannot be followed by token; the occurrence the transition
        points at therefore always completes before the end, which means the
        draft continuation can be read from real history.
        """

        states = self._states
        state, length = self._match_state, self._match_len
        while state != 0 and token not in states[state].trans:
            state = states[state].link
            length = states[state].length
        nxt = states[state].trans.get(token)
        if nxt is None:
            return 0, None
        return length + 1, states[nxt].endpos

    def draft_after(
        self, token: int, max_len: int, min_len: int = 1
    ) -> tuple[int, list[int] | None]:
        """(match length assuming token comes next, the draft continuing from it).

        Returns (match_len, None) if the match length is below ``min_len``, if
        there is no occurrence, or if the continuation is empty. The returned
        draft does not include token itself (the caller assembles [token] + cont).
        """

        match_len, end = self.peek_match(token)
        if max_len <= 0 or match_len < min_len or end is None:
            return match_len, None
        cont = self._seq[end + 1 : end + 1 + max_len]
        return match_len, (cont or None)
