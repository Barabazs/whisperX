"""
Regression and unit tests for whisperx/alignment.py.

All tests use synthetic torch tensors — no model downloads, no ffmpeg required.
"""
import torch
import pytest

from whisperx.alignment import (
    get_trellis,
    backtrack,
    merge_repeats,
    merge_words,
    Point,
    Segment,
)


class TestGetTrellisBacktrack:
    def test_trellis_shape(self):
        T, V = 5, 3
        emission = torch.randn(T, V)
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens)
        assert trellis.shape == (T + 1, len(tokens) + 1)

    def test_backtrack_returns_path(self):
        """Clear token signal should produce a valid path covering both tokens."""
        T, V = 10, 3
        tokens = [1, 2]
        emission = torch.full((T, V), -10.0)
        emission[0:2, 0] = -0.01   # leading blank
        emission[2:6, 1] = -0.01   # token A region
        emission[6:10, 2] = -0.01  # token B region

        trellis = get_trellis(emission, tokens)
        path = backtrack(trellis, emission, tokens)

        assert path is not None
        assert len(path) > 0
        # both token indices (0-based) must appear in the path
        assert {p.token_index for p in path} == {0, 1}

    def test_backtrack_returns_none_on_impossible(self):
        """1 frame cannot accommodate 2 tokens — backtrack returns None."""
        T, V = 1, 3
        tokens = [1, 2]
        emission = torch.randn(T, V)
        trellis = get_trellis(emission, tokens)
        path = backtrack(trellis, emission, tokens)
        assert path is None


class TestMergeRepeats:
    def test_groups_consecutive_same_token(self):
        path = [
            Point(0, 0, 0.9),
            Point(0, 1, 0.8),
            Point(0, 2, 0.7),
            Point(1, 3, 0.6),
            Point(1, 4, 0.5),
        ]
        transcript = "ab"
        segments = merge_repeats(path, transcript)

        assert len(segments) == 2
        assert segments[0].label == "a"
        assert segments[0].start == 0
        assert segments[0].end == 3       # path[i2-1].time_index + 1 = 2 + 1
        assert segments[1].label == "b"
        assert segments[1].start == 3
        assert segments[1].end == 5       # 4 + 1


class TestMergeWords:
    def test_merges_chars_into_word(self):
        segments = [
            Segment("h", 0, 1, 0.9),
            Segment("i", 1, 2, 0.8),
            Segment("|", 2, 3, 0.0),
        ]
        words = merge_words(segments)

        assert len(words) == 1
        assert words[0].label == "hi"
        assert words[0].start == 0
        assert words[0].end == 2

    def test_merges_multiple_words(self):
        segments = [
            Segment("h", 0, 1, 0.9),
            Segment("i", 1, 2, 0.8),
            Segment("|", 2, 3, 0.0),
            Segment("y", 3, 4, 0.7),
            Segment("o", 4, 5, 0.6),
            Segment("|", 5, 6, 0.0),
        ]
        words = merge_words(segments)

        assert len(words) == 2
        assert words[0].label == "hi"
        assert words[1].label == "yo"


class TestAlignmentTimestampRegression:
    """Regression test for issue #1220 / PR #1367.

    The old ``backtrack_beam`` (introduced in #986, reverted in #1367) started
    the path search from the *last* emission frame rather than from the argmax
    of the last token column in the trellis.  For segments with padding silence
    this caused word timestamps to be spread across the silence regions.

    The correct ``backtrack`` sets ``t_start = argmax(trellis[:, j])`` so the
    path begins at the frame that best supports the last token and propagates
    only through the actual speech region.
    """

    def test_path_stays_within_speech_region(self):
        """Padded silence must not corrupt word timestamps.

        Layout:
            frames  0-4:   silence  (blank=0 dominates)
            frames  5-9:   token A  (token 1 peaks)
            frames 10-14:  token B  (token 2 peaks)
            frames 15-19:  silence  (blank=0 dominates)
            tokens = [1, 2]

        Correct backtrack: t_start ≈ 15 → max time_index ≤ 14 (speech only).
        Old backtrack_beam: t_start = 19 (last frame) → max time_index ≥ 15
          (reaches into trailing silence) → this assertion would fail.
        """
        T, V = 20, 3  # blank=0, tokenA=1, tokenB=2
        tokens = [1, 2]

        emission = torch.full((T, V), -10.0)
        emission[0:5, 0] = -0.01    # leading silence: blank high
        emission[5:10, 1] = -0.01   # token A region
        emission[10:15, 2] = -0.01  # token B region
        emission[15:20, 0] = -0.01  # trailing silence: blank high

        trellis = get_trellis(emission, tokens, blank_id=0)
        path = backtrack(trellis, emission, tokens, blank_id=0)

        assert path is not None, "Backtrack failed on a clearly aligned sequence"

        time_indices = [p.time_index for p in path]
        # The path must not reach into the trailing silence (frames 15-19).
        assert max(time_indices) < 15, (
            f"Path reached trailing silence: max time_index={max(time_indices)} "
            f"(expected < 15). This indicates the backtrack_beam regression "
            f"(issue #1220) has been reintroduced."
        )
