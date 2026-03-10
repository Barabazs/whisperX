"""
Session-scoped fixtures for integration tests.

Expensive resources (models, audio) are loaded once per pytest session and
shared across all integration tests.  This structure is designed to support
iterative beam-tracking additions: the `align_emission` fixture exposes the
raw log-softmax emission matrix so future tests can plug in alternative
backtracking strategies without re-running the forward pass.
"""
import pytest
import torch
import whisperx
from whisperx.audio import SAMPLE_RATE

SAMPLE_AUDIO = "tests/fixtures/test_audio.mp3"
DEVICE = "cpu"
LANGUAGE = "en"


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_audio():
    """Raw 1-D float32 numpy array at 16 kHz."""
    return whisperx.load_audio(SAMPLE_AUDIO)


# ---------------------------------------------------------------------------
# Transcription (whisper-tiny)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def whisper_result(sample_audio):
    model = whisperx.load_model("tiny", device=DEVICE, compute_type="int8")
    result = model.transcribe(sample_audio, batch_size=1)
    del model
    return result


# ---------------------------------------------------------------------------
# Alignment model (shared across fixtures that need it)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def alignment_model():
    model_a, metadata = whisperx.load_align_model(LANGUAGE, device=DEVICE)
    yield model_a, metadata
    del model_a


# ---------------------------------------------------------------------------
# Full aligned result
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def aligned_result(whisper_result, alignment_model, sample_audio):
    model_a, metadata = alignment_model
    return whisperx.align(
        whisper_result["segments"], model_a, metadata, sample_audio, device=DEVICE
    )


# ---------------------------------------------------------------------------
# Raw emission matrix for the first aligned segment
# (hook for future beam-tracking tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def first_segment_emission(whisper_result, alignment_model, sample_audio):
    """Log-softmax emission tensor for the first transcribed segment.

    Exposes the alignment model's raw output so beam-tracking tests can
    compare different backtracking strategies on the same forward pass.

    Returns:
        emission (Tensor): shape (T, vocab_size), log-softmax'd
        tokens (list[int]): token indices for the segment's characters
        blank_id (int): CTC blank token index
    """
    model_a, metadata = alignment_model
    model_dict = metadata["dictionary"]
    model_type = metadata["type"]

    segment = whisper_result["segments"][0]
    text = segment["text"].strip().lower()

    # Build token list (same logic as align())
    tokens = [model_dict[c] for c in text if c in model_dict]
    if not tokens:
        pytest.skip("First segment has no alignable characters")

    f1 = int(segment["start"] * SAMPLE_RATE)
    f2 = int(segment["end"] * SAMPLE_RATE)
    audio_tensor = torch.from_numpy(sample_audio).unsqueeze(0)
    waveform = audio_tensor[:, f1:f2]

    if waveform.shape[-1] < 400:
        waveform = torch.nn.functional.pad(waveform, (0, 400 - waveform.shape[-1]))
        lengths = torch.as_tensor([waveform.shape[-1]])
    else:
        lengths = None

    with torch.inference_mode():
        if model_type == "torchaudio":
            emissions, _ = model_a(waveform.to(DEVICE), lengths=lengths)
        else:
            emissions = model_a(waveform.to(DEVICE)).logits
        emissions = torch.log_softmax(emissions, dim=-1)

    emission = emissions[0].cpu().detach()

    blank_id = 0
    for char, code in model_dict.items():
        if char in ("[pad]", "<pad>"):
            blank_id = code

    return emission, tokens, blank_id
