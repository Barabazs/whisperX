#!/usr/bin/env python
"""A/B benchmark: alignment on main vs exp/torch_align.

Google Colab usage (paste each section as a cell):
    1. Run the SETUP cell
    2. Run the BENCHMARK cell (loops both branches automatically)
    3. Run the COMPARE cell

Local usage:
    python benchmarks/bench_alignment.py [--audio FILE] [--runs 20] [--device cuda]
"""

# %% [markdown]
# # WhisperX Alignment Benchmark
# **main** (hand-rolled CTC DP + pandas) vs **exp/torch_align** (`forced_align` + numpy)

# %% SETUP — run once
# fmt: off
REPO   = "https://github.com/Barabazs/whisperX.git"
BRANCHES = ["main", "exp/torch_align"]
RUNS   = 20     # alignment iterations per branch
DEVICE = None   # auto-detect; override with "cpu" or "cuda"
AUDIO  = None   # set to a local path to skip the download below
# fmt: on

# %% Install system deps (Colab only — skip locally if you have ffmpeg)
import subprocess, sys, os, shutil

if shutil.which("ffmpeg") is None:
    subprocess.check_call(["apt-get", "-qq", "install", "-y", "ffmpeg"],
                          stdout=subprocess.DEVNULL)

# %% Download sample audio if needed
if AUDIO is None:
    if not os.path.exists("/tmp/bench_audio.wav"):
        print("Downloading sample audio ...")
        import urllib.request, io
        # torchaudio tutorial asset — short English speech, ~3s
        _URL = "https://pytorch.org/audio/main/_static/Lab41-SRI-VOiCES-src-sp0307-ch127171-sg0042.wav"
        urllib.request.urlretrieve(_URL, "/tmp/bench_audio.wav")
        print("Saved to /tmp/bench_audio.wav")
    AUDIO = "/tmp/bench_audio.wav"

print(f"Audio: {AUDIO}")

# %% BENCHMARK — installs each branch and runs alignment in a subprocess

RUNNER = r'''
import sys, json, time, warnings, gc, os
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch, whisperx

audio_path = sys.argv[1]
n_runs     = int(sys.argv[2])
out_path   = sys.argv[3]
device     = sys.argv[4] if len(sys.argv) > 4 else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
compute    = "float16" if device == "cuda" else "int8"

# ---------- transcribe ----------
asr = whisperx.load_model("tiny.en", device, compute_type=compute)
audio = whisperx.load_audio(audio_path)
transcript = asr.transcribe(audio, batch_size=16)
segments = transcript["segments"]
del asr; gc.collect(); torch.cuda.empty_cache() if device == "cuda" else None

# ---------- load alignment model ----------
align_model, meta = whisperx.load_align_model("en", device)

# ---------- warmup ----------
_ = whisperx.align(segments, align_model, meta, audio, device)
if device == "cuda":
    torch.cuda.synchronize()

# ---------- timed runs ----------
times = []
for i in range(n_runs):
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = whisperx.align(segments, align_model, meta, audio, device)
    if device == "cuda":
        torch.cuda.synchronize()
    times.append(time.perf_counter() - t0)

# ---------- save ----------
words = []
for w in result["word_segments"]:
    words.append({
        "word":  w.get("word", ""),
        "start": w.get("start"),
        "end":   w.get("end"),
        "score": w.get("score"),
    })

out = {
    "branch":    os.environ.get("_BENCH_BRANCH", "unknown"),
    "device":    device,
    "n_segments": len(segments),
    "n_words":   len(words),
    "times":     times,
    "mean_s":    sum(times) / len(times),
    "median_s":  sorted(times)[len(times) // 2],
    "words":     words,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print(f"  {out['branch']:25s}  mean={out['mean_s']:.4f}s  "
      f"median={out['median_s']:.4f}s  words={out['n_words']}")
'''

import tempfile, pathlib, json

runner_path = pathlib.Path(tempfile.gettempdir()) / "_bench_runner.py"
runner_path.write_text(RUNNER)

results = {}
device_flag = DEVICE or ""

for branch in BRANCHES:
    print(f"\n{'='*60}")
    print(f"  Installing {branch} ...")
    print(f"{'='*60}")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--force-reinstall",
         f"whisperx @ git+{REPO}@{branch}"],
        stdout=subprocess.DEVNULL,
    )
    out_path = f"/tmp/bench_{branch.replace('/', '_')}.json"
    env = {**os.environ, "_BENCH_BRANCH": branch}
    cmd = [sys.executable, str(runner_path), AUDIO, str(RUNS), out_path]
    if device_flag:
        cmd.append(device_flag)
    print(f"  Benchmarking ({RUNS} runs) ...")
    subprocess.check_call(cmd, env=env)
    with open(out_path) as f:
        results[branch] = json.load(f)

# %% COMPARE — correctness & performance
import json

def _load(branch):
    path = f"/tmp/bench_{branch.replace('/', '_')}.json"
    with open(path) as f:
        return json.load(f)

old = _load(BRANCHES[0])
new = _load(BRANCHES[1])

print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)

# ---- timing ----
print(f"\n{'Metric':<20} {'main':>12} {'exp/torch_align':>16} {'speedup':>10}")
print("-" * 60)
for label, key in [("Mean (s)", "mean_s"), ("Median (s)", "median_s")]:
    o, n = old[key], new[key]
    speedup = o / n if n > 0 else float("inf")
    print(f"{label:<20} {o:>12.4f} {n:>16.4f} {speedup:>9.2f}x")
print(f"{'Runs':<20} {len(old['times']):>12d} {len(new['times']):>16d}")

# ---- word count ----
print(f"\n{'Words detected':<20} {old['n_words']:>12d} {new['n_words']:>16d}")
print(f"{'Segments':<20} {old['n_segments']:>12d} {new['n_segments']:>16d}")

# ---- word-level diff ----
old_words = old["words"]
new_words = new["words"]
n = min(len(old_words), len(new_words))

text_match = sum(1 for i in range(n) if old_words[i]["word"] == new_words[i]["word"])
print(f"\n{'Word text match':<20} {text_match}/{n}")

start_diffs, end_diffs, score_diffs = [], [], []
for i in range(n):
    ow, nw = old_words[i], new_words[i]
    if ow.get("start") is not None and nw.get("start") is not None:
        start_diffs.append(abs(ow["start"] - nw["start"]))
    if ow.get("end") is not None and nw.get("end") is not None:
        end_diffs.append(abs(ow["end"] - nw["end"]))
    if ow.get("score") is not None and nw.get("score") is not None:
        score_diffs.append(abs(ow["score"] - nw["score"]))

def _stats(diffs, label):
    if not diffs:
        print(f"  {label:<28} no comparable values")
        return
    import statistics
    mean = statistics.mean(diffs)
    med  = statistics.median(diffs)
    mx   = max(diffs)
    p95  = sorted(diffs)[int(len(diffs) * 0.95)]
    within_10ms = sum(1 for d in diffs if d <= 0.010)
    print(f"  {label:<28} mean={mean:.4f}  median={med:.4f}  "
          f"p95={p95:.4f}  max={mx:.4f}  <=10ms: {within_10ms}/{len(diffs)}")

print("\nTimestamp deltas (seconds):")
_stats(start_diffs, "start")
_stats(end_diffs, "end")

print("\nScore deltas:")
_stats(score_diffs, "score")

# ---- per-word detail (first 10 with diffs) ----
print(f"\nFirst words with timestamp diff > 10ms:")
shown = 0
for i in range(n):
    ow, nw = old_words[i], new_words[i]
    s_diff = abs((ow.get("start") or 0) - (nw.get("start") or 0))
    e_diff = abs((ow.get("end") or 0) - (nw.get("end") or 0))
    if s_diff > 0.01 or e_diff > 0.01:
        print(f"  [{i:3d}] {ow['word']:<15} "
              f"old=({ow.get('start','?'):>6}–{ow.get('end','?'):<6}) "
              f"new=({nw.get('start','?'):>6}–{nw.get('end','?'):<6}) "
              f"d_start={s_diff:.3f} d_end={e_diff:.3f}")
        shown += 1
        if shown >= 10:
            break
if shown == 0:
    print("  (none — all timestamps within 10ms)")

print("\n" + "=" * 70)


# %% CLI entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", help="Path to audio file")
    parser.add_argument("--runs", type=int, default=20, help="Alignment iterations")
    parser.add_argument("--device", help="cpu or cuda (auto-detect if omitted)")
    args = parser.parse_args()
    if args.audio:
        AUDIO = args.audio
    if args.runs:
        RUNS = args.runs
    if args.device:
        DEVICE = args.device
