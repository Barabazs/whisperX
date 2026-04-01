"""
Forced Alignment with Whisper
C. Max Bain
"""
from typing import Iterable, Optional, Union, List

import numpy as np
import torch
import torchaudio
from torchaudio.functional import forced_align
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from whisperx.audio import SAMPLE_RATE, load_audio
from whisperx.utils import PUNKT_LANGUAGES
from whisperx.schema import (
    AlignedTranscriptionResult,
    SingleSegment,
    SingleAlignedSegment,
    SingleWordSegment,
    SegmentData,
    ProgressCallback,
)
import nltk
from nltk.data import load as nltk_load
from whisperx.log_utils import get_logger

logger = get_logger(__name__)

LANGUAGES_WITHOUT_SPACES = ["ja", "zh"]

DEFAULT_ALIGN_MODELS_TORCH = {
    "en": "WAV2VEC2_ASR_BASE_960H",
    "fr": "VOXPOPULI_ASR_BASE_10K_FR",
    "de": "VOXPOPULI_ASR_BASE_10K_DE",
    "es": "VOXPOPULI_ASR_BASE_10K_ES",
    "it": "VOXPOPULI_ASR_BASE_10K_IT",
}

DEFAULT_ALIGN_MODELS_HF = {
    "ja": "jonatasgrosman/wav2vec2-large-xlsr-53-japanese",
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
    "nl": "jonatasgrosman/wav2vec2-large-xlsr-53-dutch",
    "uk": "Yehor/wav2vec2-xls-r-300m-uk-with-small-lm",
    "pt": "jonatasgrosman/wav2vec2-large-xlsr-53-portuguese",
    "ar": "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
    "cs": "comodoro/wav2vec2-xls-r-300m-cs-250",
    "ru": "jonatasgrosman/wav2vec2-large-xlsr-53-russian",
    "pl": "jonatasgrosman/wav2vec2-large-xlsr-53-polish",
    "hu": "jonatasgrosman/wav2vec2-large-xlsr-53-hungarian",
    "fi": "jonatasgrosman/wav2vec2-large-xlsr-53-finnish",
    "fa": "jonatasgrosman/wav2vec2-large-xlsr-53-persian",
    "el": "jonatasgrosman/wav2vec2-large-xlsr-53-greek",
    "tr": "mpoyraz/wav2vec2-xls-r-300m-cv7-turkish",
    "da": "saattrupdan/wav2vec2-xls-r-300m-ftspeech",
    "he": "imvladikon/wav2vec2-xls-r-300m-hebrew",
    "vi": 'nguyenvulebinh/wav2vec2-base-vi-vlsp2020',
    "ko": "kresnik/wav2vec2-large-xlsr-korean",
    "ur": "kingabzpro/wav2vec2-large-xls-r-300m-Urdu",
    "te": "anuragshas/wav2vec2-large-xlsr-53-telugu",
    "hi": "theainerd/Wav2Vec2-large-xlsr-hindi",
    "ca": "softcatala/wav2vec2-large-xlsr-catala",
    "ml": "gvs/wav2vec2-large-xlsr-malayalam",
    "no": "NbAiLab/nb-wav2vec2-1b-bokmaal-v2",
    "nn": "NbAiLab/nb-wav2vec2-1b-nynorsk",
    "sk": "comodoro/wav2vec2-xls-r-300m-sk-cv8",
    "sl": "anton-l/wav2vec2-large-xlsr-53-slovenian",
    "hr": "classla/wav2vec2-xls-r-parlaspeech-hr",
    "ro": "gigant/romanian-wav2vec2",
    "eu": "stefan-it/wav2vec2-large-xlsr-53-basque",
    "gl": "ifrz/wav2vec2-large-xlsr-galician",
    "ka": "xsway/wav2vec2-large-xlsr-georgian",
    "lv": "jimregan/wav2vec2-large-xlsr-latvian-cv",
    "tl": "Khalsuu/filipino-wav2vec2-l-xls-r-300m-official",
    "sv": "KBLab/wav2vec2-large-voxrex-swedish",
}


def load_align_model(language_code: str, device: str, model_name: Optional[str] = None, model_dir=None, model_cache_only: bool = False):
    if model_name is None:
        # use default model
        if language_code in DEFAULT_ALIGN_MODELS_TORCH:
            model_name = DEFAULT_ALIGN_MODELS_TORCH[language_code]
        elif language_code in DEFAULT_ALIGN_MODELS_HF:
            model_name = DEFAULT_ALIGN_MODELS_HF[language_code]
        else:
            logger.error(f"No default alignment model for language: {language_code}. "
                         f"Please find a wav2vec2.0 model finetuned on this language at https://huggingface.co/models, "
                         f"then pass the model name via --align_model [MODEL_NAME]")
            raise ValueError(f"No default align-model for language: {language_code}")

    if model_name in torchaudio.pipelines.__all__:
        pipeline_type = "torchaudio"
        bundle = torchaudio.pipelines.__dict__[model_name]
        align_model = bundle.get_model(dl_kwargs={"model_dir": model_dir}).to(device)
        labels = bundle.get_labels()
        align_dictionary = {c.lower(): i for i, c in enumerate(labels)}
    else:
        try:
            processor = Wav2Vec2Processor.from_pretrained(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
            align_model = Wav2Vec2ForCTC.from_pretrained(model_name, cache_dir=model_dir, local_files_only=model_cache_only)
        except Exception as e:
            print(e)
            print(f"Error loading model from huggingface, check https://huggingface.co/models for finetuned wav2vec2.0 models")
            raise ValueError(f'The chosen align_model "{model_name}" could not be found in huggingface (https://huggingface.co/models) or torchaudio (https://pytorch.org/audio/stable/pipelines.html#id14)')
        pipeline_type = "huggingface"
        align_model = align_model.to(device)
        labels = processor.tokenizer.get_vocab()
        align_dictionary = {char.lower(): code for char,code in processor.tokenizer.get_vocab().items()}

    align_metadata = {"language": language_code, "dictionary": align_dictionary, "type": pipeline_type}

    return align_model, align_metadata


def _interpolate_nearest(values):
    """Fill NaN values with nearest non-NaN neighbor."""
    arr = np.array(values, dtype=np.float64)
    nans = np.isnan(arr)
    if not nans.any() or nans.all():
        return arr
    known_idx = np.where(~nans)[0]
    nan_idx = np.where(nans)[0]
    insert_pos = np.searchsorted(known_idx, nan_idx)
    left = np.clip(insert_pos - 1, 0, len(known_idx) - 1)
    right = np.clip(insert_pos, 0, len(known_idx) - 1)
    left_dist = np.abs(nan_idx - known_idx[left])
    right_dist = np.abs(nan_idx - known_idx[right])
    nearest = np.where(left_dist <= right_dist, known_idx[left], known_idx[right])
    arr[nan_idx] = arr[nearest]
    return arr


def _merge_token_frames(aligned_tokens, scores, blank_id, tokens):
    """Group forced_align frame-level output into per-character segments.

    CTC monotonicity guarantees: same token after a blank = new target position,
    different token = new target position.

    Returns list of dicts with 'start' (inclusive frame), 'end' (exclusive frame),
    'score' (mean probability).
    """
    num_frames = aligned_tokens.size(0)
    if num_frames == 0 or len(tokens) == 0:
        return []

    segments = []
    target_pos = 0
    seg_start = None
    seg_scores = []
    saw_blank = False
    last_emit_frame = None

    for t in range(num_frames):
        tok = aligned_tokens[t].item()
        if tok == blank_id:
            if seg_start is not None:
                saw_blank = True
            continue

        score = scores[t].exp().item()
        last_emit_frame = t

        is_new = (seg_start is not None) and (tok != tokens[target_pos] or saw_blank)

        if is_new:
            segments.append({
                "start": seg_start,
                "end": t,
                "score": sum(seg_scores) / len(seg_scores),
            })
            target_pos += 1
            seg_start = t
            seg_scores = [score]
        elif seg_start is None:
            seg_start = t
            seg_scores = [score]
        else:
            seg_scores.append(score)
        saw_blank = False

    if seg_start is not None and last_emit_frame is not None:
        segments.append({
            "start": seg_start,
            "end": last_emit_frame + 1,
            "score": sum(seg_scores) / len(seg_scores),
        })

    return segments


def align(
    transcript: Iterable[SingleSegment],
    model: torch.nn.Module,
    align_model_metadata: dict,
    audio: Union[str, np.ndarray, torch.Tensor],
    device: str,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    print_progress: bool = False,
    combined_progress: bool = False,
    progress_callback: ProgressCallback = None,
) -> AlignedTranscriptionResult:
    """
    Align phoneme recognition predictions to known transcription.
    """

    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)

    MAX_DURATION = audio.shape[1] / SAMPLE_RATE

    model_dictionary = align_model_metadata["dictionary"]
    model_lang = align_model_metadata["language"]
    model_type = align_model_metadata["type"]

    # 1. Preprocess to keep only characters in dictionary
    total_segments = len(transcript)
    # Store temporary processing values
    segment_data: dict[int, SegmentData] = {}
    for sdx, segment in enumerate(transcript):
        # strip spaces at beginning / end, but keep track of the amount.
        if print_progress:
            base_progress = ((sdx + 1) / total_segments) * 100
            percent_complete = (50 + base_progress / 2) if combined_progress else base_progress
            print(f"Progress: {percent_complete:.2f}%...")

        num_leading = len(segment["text"]) - len(segment["text"].lstrip())
        num_trailing = len(segment["text"]) - len(segment["text"].rstrip())
        text = segment["text"]

        # split into words
        if model_lang not in LANGUAGES_WITHOUT_SPACES:
            per_word = text.split(" ")
        else:
            per_word = text

        clean_char, clean_cdx = [], []
        for cdx, char in enumerate(text):
            char_ = char.lower()
            # wav2vec2 models use "|" character to represent spaces
            if model_lang not in LANGUAGES_WITHOUT_SPACES:
                char_ = char_.replace(" ", "|")

            # ignore whitespace at beginning and end of transcript
            if cdx < num_leading:
                pass
            elif cdx > len(text) - num_trailing - 1:
                pass
            elif char_ in model_dictionary.keys():
                clean_char.append(char_)
                clean_cdx.append(cdx)
            elif char_ not in (" ", "|"):
                # unknown char (digit, symbol, foreign script) — use wildcard
                clean_char.append(char_)
                clean_cdx.append(cdx)

        clean_wdx = list(range(len(per_word)))

        # Use language-specific Punkt model if available otherwise we fallback to English.
        punkt_lang = PUNKT_LANGUAGES.get(model_lang, 'english')
        try:
            sentence_splitter = nltk_load(f'tokenizers/punkt_tab/{punkt_lang}.pickle')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)
            sentence_splitter = nltk_load(f'tokenizers/punkt_tab/{punkt_lang}.pickle')
        sentence_spans = list(sentence_splitter.span_tokenize(text))

        segment_data[sdx] = {
            "clean_char": clean_char,
            "clean_cdx": clean_cdx,
            "clean_wdx": clean_wdx,
            "sentence_spans": sentence_spans
        }

    aligned_segments: List[SingleAlignedSegment] = []

    # 2. Get prediction matrix from alignment model & align
    for sdx, segment in enumerate(transcript):

        t1 = segment["start"]
        t2 = segment["end"]
        text = segment["text"]
        avg_logprob = segment.get("avg_logprob")

        aligned_seg: SingleAlignedSegment = {
            "start": t1,
            "end": t2,
            "text": text,
            "words": [],
            "chars": None,
        }

        if avg_logprob is not None:
            aligned_seg["avg_logprob"] = avg_logprob

        if return_char_alignments:
            aligned_seg["chars"] = []

        # check we can align
        if len(segment_data[sdx]["clean_char"]) == 0:
            logger.warning(f'Failed to align segment ("{segment["text"]}"): no characters in this segment found in model dictionary, resorting to original')
            aligned_segments.append(aligned_seg)
            continue

        if t1 >= MAX_DURATION:
            logger.warning(f'Failed to align segment ("{segment["text"]}"): original start time longer than audio duration, skipping')
            aligned_segments.append(aligned_seg)
            continue

        text_clean = "".join(segment_data[sdx]["clean_char"])

        f1 = int(t1 * SAMPLE_RATE)
        f2 = int(t2 * SAMPLE_RATE)

        # TODO: Probably can get some speedup gain with batched inference here
        waveform_segment = audio[:, f1:f2]
        # Handle the minimum input length for wav2vec2 models
        if waveform_segment.shape[-1] < 400:
            lengths = torch.as_tensor([waveform_segment.shape[-1]]).to(device)
            waveform_segment = torch.nn.functional.pad(
                waveform_segment, (0, 400 - waveform_segment.shape[-1])
            )
        else:
            lengths = None

        with torch.inference_mode():
            if model_type == "torchaudio":
                emissions, _ = model(waveform_segment.to(device), lengths=lengths)
            elif model_type == "huggingface":
                emissions = model(waveform_segment.to(device)).logits
            else:
                raise NotImplementedError(f"Align model of type {model_type} not supported.")
            emissions = torch.log_softmax(emissions, dim=-1)

        emission = emissions[0].cpu().detach()

        blank_id = 0
        for char, code in model_dictionary.items():
            if char == '[pad]' or char == '<pad>':
                blank_id = code

        # Build tokens, mapping unknown chars to a wildcard column
        has_wildcard = any(c not in model_dictionary for c in text_clean)
        if has_wildcard:
            # Extend emission with a wildcard column: max non-blank score per frame
            non_blank_mask = torch.ones(emission.size(1), dtype=torch.bool)
            non_blank_mask[blank_id] = False
            wildcard_col = emission[:, non_blank_mask].max(dim=1).values
            emission = torch.cat([emission, wildcard_col.unsqueeze(1)], dim=1)
            wildcard_id = emission.size(1) - 1
            tokens = [model_dictionary.get(c, wildcard_id) for c in text_clean]
        else:
            tokens = [model_dictionary[c] for c in text_clean]

        try:
            aligned_tokens_out, align_scores = forced_align(
                emission.unsqueeze(0),
                torch.tensor([tokens], dtype=torch.long),
                blank=blank_id,
            )
            char_segments = _merge_token_frames(
                aligned_tokens_out[0], align_scores[0], blank_id, tokens,
            )
        except Exception:
            char_segments = []

        if not char_segments or len(char_segments) != len(text_clean):
            logger.warning(f'Failed to align segment ("{segment["text"]}"): forced alignment failed, resorting to original')
            aligned_segments.append(aligned_seg)
            continue

        duration = t2 - t1
        ratio = duration * waveform_segment.size(0) / emission.size(0)

        # assign timestamps to aligned characters
        char_segments_arr = []
        word_idx = 0
        for cdx, char in enumerate(text):
            start, end, score = np.nan, np.nan, np.nan
            if cdx in segment_data[sdx]["clean_cdx"]:
                char_seg = char_segments[segment_data[sdx]["clean_cdx"].index(cdx)]
                start = round(char_seg["start"] * ratio + t1, 3)
                end = round(char_seg["end"] * ratio + t1, 3)
                score = round(char_seg["score"], 3)

            char_segments_arr.append(
                {
                    "char": char,
                    "start": start,
                    "end": end,
                    "score": score,
                    "word-idx": word_idx,
                }
            )

            # increment word_idx, nltk word tokenization would probably be more robust here, but us space for now...
            if model_lang in LANGUAGES_WITHOUT_SPACES:
                word_idx += 1
            elif cdx == len(text) - 1 or text[cdx+1] == " ":
                word_idx += 1

        aligned_subsegments = []
        for sdx2, (sstart, send) in enumerate(segment_data[sdx]["sentence_spans"]):
            curr_chars = char_segments_arr[sstart:send + 1]

            sentence_text = text[sstart:send]
            starts = [c["start"] for c in curr_chars]
            sentence_start = float(np.nanmin(starts)) if starts else np.nan
            end_vals = [c["end"] for c in curr_chars if c["char"] != ' ']
            sentence_end = float(np.nanmax(end_vals)) if end_vals else np.nan
            sentence_words = []

            word_indices = sorted(set(c["word-idx"] for c in curr_chars))
            for word_idx in word_indices:
                word_chars = [c for c in curr_chars if c["word-idx"] == word_idx]
                word_text = "".join(c["char"] for c in word_chars).strip()
                if len(word_text) == 0:
                    continue

                # dont use space character for alignment
                word_chars = [c for c in word_chars if c["char"] != " "]

                wc_starts = [c["start"] for c in word_chars]
                wc_ends = [c["end"] for c in word_chars]
                wc_scores = [c["score"] for c in word_chars]
                valid_starts = [s for s in wc_starts if not np.isnan(s)]
                valid_ends = [s for s in wc_ends if not np.isnan(s)]
                valid_scores = [s for s in wc_scores if not np.isnan(s)]
                word_start = min(valid_starts) if valid_starts else np.nan
                word_end = max(valid_ends) if valid_ends else np.nan
                word_score = round(float(np.mean(valid_scores)), 3) if valid_scores else np.nan

                # -1 indicates unalignable
                word_segment = {"word": word_text}

                if not np.isnan(word_start):
                    word_segment["start"] = word_start
                if not np.isnan(word_end):
                    word_segment["end"] = word_end
                if not np.isnan(word_score):
                    word_segment["score"] = word_score

                sentence_words.append(word_segment)

            # Interpolate timestamps for words with no alignable characters
            if sentence_words:
                _starts = [w.get("start", np.nan) for w in sentence_words]
                _ends = [w.get("end", np.nan) for w in sentence_words]
                has_nan = any(np.isnan(s) for s in _starts)
                has_val = any(not np.isnan(s) for s in _starts)
                if has_nan and has_val:
                    _starts = _interpolate_nearest(_starts)
                    _ends = _interpolate_nearest(_ends)
                    for i, w in enumerate(sentence_words):
                        if "start" not in w and not np.isnan(_starts[i]):
                            w["start"] = float(_starts[i])
                        if "end" not in w and not np.isnan(_ends[i]):
                            w["end"] = float(_ends[i])

            subsegment = {
                "text": sentence_text,
                "start": sentence_start,
                "end": sentence_end,
                "words": sentence_words,
            }
            if avg_logprob is not None:
                subsegment["avg_logprob"] = avg_logprob
            aligned_subsegments.append(subsegment)

            if return_char_alignments:
                curr_chars_out = []
                for c in curr_chars:
                    d = {"char": c["char"]}
                    if not np.isnan(c["start"]):
                        d["start"] = c["start"]
                    if not np.isnan(c["end"]):
                        d["end"] = c["end"]
                    if not np.isnan(c["score"]):
                        d["score"] = c["score"]
                    curr_chars_out.append(d)
                aligned_subsegments[-1]["chars"] = curr_chars_out

        # Interpolate NaN timestamps across subsegments
        if aligned_subsegments:
            sub_starts = _interpolate_nearest([s["start"] for s in aligned_subsegments])
            sub_ends = _interpolate_nearest([s["end"] for s in aligned_subsegments])
            for i, sub in enumerate(aligned_subsegments):
                sub["start"] = float(sub_starts[i])
                sub["end"] = float(sub_ends[i])

        # Merge subsegments with identical timestamps
        text_join = "" if model_lang in LANGUAGES_WITHOUT_SPACES else " "
        merged = {}
        for sub in aligned_subsegments:
            key = (sub["start"], sub["end"])
            if key not in merged:
                merged[key] = dict(sub)
            else:
                merged[key]["text"] = text_join.join([merged[key]["text"], sub["text"]])
                merged[key]["words"] = merged[key]["words"] + sub["words"]
                if return_char_alignments and "chars" in sub:
                    merged[key]["chars"] = merged[key].get("chars", []) + sub.get("chars", [])
        aligned_subsegments = list(merged.values())

        if progress_callback is not None:
            progress_callback(((sdx + 1) / total_segments) * 100)

        aligned_segments += aligned_subsegments

    # create word_segments list
    word_segments: List[SingleWordSegment] = []
    for segment in aligned_segments:
        word_segments += segment["words"]

    return {"segments": aligned_segments, "word_segments": word_segments}
