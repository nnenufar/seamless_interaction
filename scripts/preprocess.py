# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Preprocessing script for Seamless Interaction dataset samples.

For each interaction pair (two mono WAVs from the same interaction):
  - Resample both channels to 16 kHz
  - Merge into a single stereo WAV (channel 0 = first participant sorted by file ID)
  - Extract and merge VAD segments into a nested list saved as JSON
  - Record the channel-to-speaker mapping in a CSV

Usage:
    python preprocess.py --input_dir <path> --output_dir <path> [--num_workers N]
"""

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TARGET_SR = 16_000
FILE_ID_RE = re.compile(r"(V\d+_S\d+_I\d+)_(P\d+)")


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_pairs(input_dir: str) -> dict[str, list[Path]]:
    """
    Recursively scan input_dir for .wav files and group them by interaction key.
    Only groups with exactly 2 files (one per participant) are returned.
    """
    groups: dict[str, list[Path]] = defaultdict(list)
    for wav_path in Path(input_dir).rglob("*.wav"):
        m = FILE_ID_RE.search(wav_path.stem)
        if m:
            interaction_key = m.group(1)
            groups[interaction_key].append(wav_path)

    pairs = {k: sorted(v) for k, v in groups.items() if len(v) == 2}
    skipped = len(groups) - len(pairs)
    if skipped:
        logger.warning(
            f"Skipped {skipped} interaction(s) that did not have exactly 2 WAV files."
        )
    return pairs


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D float32 array from orig_sr to target_sr."""
    if orig_sr == target_sr:
        return audio
    divisor = gcd(orig_sr, target_sr)
    up = target_sr // divisor
    down = orig_sr // divisor
    resampled = resample_poly(audio, up, down)
    return resampled.astype(np.float32)


def load_and_resample(wav_path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file and resample to TARGET_SR. Returns (audio, TARGET_SR)."""
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        # Already multi-channel: mix down to mono
        audio = audio.mean(axis=1)
    audio = _resample(audio, sr, TARGET_SR)
    return audio, TARGET_SR


# ---------------------------------------------------------------------------
# VAD helpers
# ---------------------------------------------------------------------------

def extract_vad_segments(json_path: Path) -> list[list[float]]:
    """
    Read the per-file JSON and return VAD segments as [[start, end], ...].
    Returns an empty list if the key is missing.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    raw = data.get("metadata:vad", [])
    return [[seg["start"], seg["end"]] for seg in raw]


# ---------------------------------------------------------------------------
# Per-pair processing (runs in a worker process)
# ---------------------------------------------------------------------------

def process_pair(
    interaction_key: str,
    wav_a: Path,
    wav_b: Path,
    wav_out_dir: Path,
    vad_out_dir: Path,
) -> dict:
    """
    Process one interaction pair:
      1. Resample both mono WAVs to 16 kHz
      2. Write stereo WAV  (channel 0 = wav_a, channel 1 = wav_b)
      3. Merge VAD into nested list and write JSON
      4. Return a dict with the CSV row data

    wav_a and wav_b are already sorted by file ID so channel assignment
    is deterministic.
    """
    m_a = FILE_ID_RE.search(wav_a.stem)
    m_b = FILE_ID_RE.search(wav_b.stem)
    participant_a = m_a.group(2) if m_a else wav_a.stem
    participant_b = m_b.group(2) if m_b else wav_b.stem

    # Load & resample
    audio_a, _ = load_and_resample(wav_a)
    audio_b, _ = load_and_resample(wav_b)

    # Pad to equal length so we can stack into stereo
    max_len = max(len(audio_a), len(audio_b))
    if len(audio_a) < max_len:
        audio_a = np.pad(audio_a, (0, max_len - len(audio_a)))
    if len(audio_b) < max_len:
        audio_b = np.pad(audio_b, (0, max_len - len(audio_b)))

    stereo = np.stack([audio_a, audio_b], axis=1)  # (samples, 2)

    out_stem = interaction_key
    stereo_path = wav_out_dir / f"{out_stem}.wav"
    sf.write(str(stereo_path), stereo, TARGET_SR, subtype="PCM_16")

    # VAD
    json_a = wav_a.with_suffix(".json")
    json_b = wav_b.with_suffix(".json")
    vad_a = extract_vad_segments(json_a) if json_a.exists() else []
    vad_b = extract_vad_segments(json_b) if json_b.exists() else []

    vad_merged = [vad_a, vad_b]
    vad_path = vad_out_dir / f"{out_stem}.json"
    with open(vad_path, "w") as f:
        json.dump(vad_merged, f)

    return {
        "interaction_key": interaction_key,
        "channel_0_file_id": wav_a.stem,
        "channel_0_participant": participant_a,
        "channel_1_file_id": wav_b.stem,
        "channel_1_participant": participant_b,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Seamless Interaction audio pairs."
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Root directory to search recursively for downloaded WAV + JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Root output directory. Stereo WAVs go to <output_dir>/wavs/, "
             "VAD JSONs go to <output_dir>/vad/.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Number of parallel worker processes (default: cpu_count - 1).",
    )
    args = parser.parse_args()

    wav_out_dir = Path(args.output_dir) / "wavs"
    vad_out_dir = Path(args.output_dir) / "vad"
    wav_out_dir.mkdir(parents=True, exist_ok=True)
    vad_out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.output_dir) / "channel_map.csv"

    logger.info(f"Scanning {args.input_dir} for interaction pairs …")
    pairs = discover_pairs(args.input_dir)
    logger.info(f"Found {len(pairs)} interaction pair(s). Processing with {args.num_workers} worker(s).")

    if not pairs:
        logger.error("No pairs found. Check that input_dir contains downloaded WAV files.")
        sys.exit(1)

    csv_rows: list[dict] = []
    csv_fieldnames = [
        "interaction_key",
        "channel_0_file_id",
        "channel_0_participant",
        "channel_1_file_id",
        "channel_1_participant",
    ]

    futures = {}
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        for interaction_key, (wav_a, wav_b) in pairs.items():
            future = executor.submit(
                process_pair,
                interaction_key,
                wav_a,
                wav_b,
                wav_out_dir,
                vad_out_dir,
            )
            futures[future] = interaction_key

        done = 0
        total = len(futures)
        for future in as_completed(futures):
            interaction_key = futures[future]
            try:
                row = future.result()
                csv_rows.append(row)
                done += 1
                if done % 100 == 0 or done == total:
                    percent = (done / total) * 100
                    logger.info(f"  {done}/{total} ({percent:.1f}%) pairs processed")
            except Exception as exc:
                logger.error(f"  Failed to process {interaction_key}: {exc}")

    # Write CSV sorted by interaction_key for deterministic output
    csv_rows.sort(key=lambda r: r["interaction_key"])
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info(
        f"Done. {len(csv_rows)}/{len(pairs)} pairs processed successfully.\n"
        f"  WAVs  → {wav_out_dir}\n"
        f"  VAD   → {vad_out_dir}\n"
        f"  CSV   → {csv_path}"
    )


if __name__ == "__main__":
    main()
