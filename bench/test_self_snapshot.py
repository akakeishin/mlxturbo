from bench.self_snapshot import _output_token_count


class _Tokenizer:
    def encode(self, text):
        return text.split()


def test_output_token_count_does_not_treat_sse_chunks_as_tokens():
    assert _output_token_count(_Tokenizer(), "one two three four", 2) == 4


def test_output_token_count_keeps_chunk_fallback_for_empty_reply():
    assert _output_token_count(_Tokenizer(), "", 3) == 3
