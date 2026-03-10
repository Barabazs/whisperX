"""
Integration tests: full pipeline with whisper-tiny + wav2vec2-base-960h.

Run with:
    uv run pytest tests/ -m integration -v

Requires:
    - tests/fixtures/sample.wav  (8-second English speech)
    - Model downloads (~450 MB total, cached after first run)

The `first_segment_emission` fixture exposes the raw CTC emission matrix,
making it straightforward to add beam-tracking assertions in future commits
without re-running the acoustic model forward pass.
"""
import pytest
from whisperx.alignment import get_trellis, backtrack
from whisperx.audio import SAMPLE_RATE


@pytest.mark.integration
class TestTinyPipeline:
    def test_transcription_has_segments(self, whisper_result):
        assert len(whisper_result["segments"]) > 0

    def test_aligned_words_present(self, aligned_result):
        assert len(aligned_result["word_segments"]) > 0

    def test_word_timestamps_ordered(self, aligned_result):
        starts = [w["start"] for w in aligned_result["word_segments"] if "start" in w]
        assert len(starts) > 0, "No timed words found"
        assert starts == sorted(starts), f"Word timestamps not ordered: {starts}"

    def test_word_timestamps_within_audio_bounds(self, aligned_result, sample_audio):
        duration = len(sample_audio) / SAMPLE_RATE
        for word in aligned_result["word_segments"]:
            if "start" in word:
                assert word["start"] >= 0, f"Word starts before audio: {word}"
                assert word["end"] <= duration + 0.05, f"Word ends after audio: {word}"

    def test_no_negative_word_duration(self, aligned_result):
        for word in aligned_result["word_segments"]:
            if "start" in word and "end" in word:
                assert word["end"] >= word["start"], f"Negative duration: {word}"


@pytest.mark.integration
class TestBeamTracking:
    """Assertions on raw CTC alignment internals.

    Add beam-tracking comparisons here as the feature is developed.
    Each test receives the emission matrix + token list for the first
    segment, so different backtracking strategies can be compared on
    identical inputs.
    """

    def test_trellis_and_backtrack_succeed_on_real_emission(
        self, first_segment_emission
    ):
        emission, tokens, blank_id = first_segment_emission
        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack(trellis, emission, tokens, blank_id)
        assert path is not None, "backtrack failed on real audio emission"

    def test_path_time_indices_within_emission(self, first_segment_emission):
        emission, tokens, blank_id = first_segment_emission
        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack(trellis, emission, tokens, blank_id)
        T = emission.size(0)
        for point in path:
            assert 0 <= point.time_index < T, (
                f"time_index {point.time_index} outside emission range [0, {T})"
            )

    def test_path_covers_all_tokens(self, first_segment_emission):
        emission, tokens, blank_id = first_segment_emission
        trellis = get_trellis(emission, tokens, blank_id)
        path = backtrack(trellis, emission, tokens, blank_id)
        assigned = {p.token_index for p in path}
        expected = set(range(len(tokens)))
        assert assigned == expected, (
            f"Missing tokens in path: {expected - assigned}"
        )
