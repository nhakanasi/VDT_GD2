"""Export the input and output of every step in figures/overview.tex as separate images and text,
for laying the figure out by hand in an editor such as Canva.

    python figures/make_canva_assets.py --para-root ../Para

Writes figures/canva/: one SVG (vector) and one transparent PNG per waveform, named
<step>_<input|output>[_<part>], plus transcripts.txt with the text that goes with each step.
Uses the same pilot row, window and colours as make_waveforms.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_waveforms import (CANDIDATES, CM, EVENT, FOREIGN, ROW, SPEECH, WINDOW_S, load,
                            schematic, trim)

SLOT_SHADE = "#D9D9D9"
PNG_DPI = 600


def draw(stem: Path, x: np.ndarray, colors: np.ndarray, width_cm: float, height_cm: float,
         bins: int = 700, spans: list[tuple[float, float]] = ()) -> None:
    """Waveform envelope as SVG and PNG; `spans` are shaded (start, end) fractions of the width."""
    peak = np.max(np.abs(x)) + 1e-9
    edges = np.linspace(0, len(x), bins + 1).astype(int)
    lo = np.array([x[a:b].min() if b > a else 0 for a, b in zip(edges[:-1], edges[1:])]) / peak
    hi = np.array([x[a:b].max() if b > a else 0 for a, b in zip(edges[:-1], edges[1:])]) / peak
    col = colors[np.minimum(((edges[:-1] + edges[1:]) // 2), len(colors) - 1)]

    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
    ax = fig.add_axes((0, 0, 1, 1))
    for a, b in spans:
        ax.axvspan(a * bins - 0.5, b * bins - 0.5, color=SLOT_SHADE, zorder=0, linewidth=0)
    for c in np.unique(col):
        m = col == c
        ax.vlines(np.flatnonzero(m), lo[m], hi[m], colors=c, linewidth=0.55, zorder=1)
    ax.set_xlim(-0.5, bins - 0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    fig.savefig(stem.with_suffix(".svg"), transparent=True)
    fig.savefig(stem.with_suffix(".png"), transparent=True, dpi=PNG_DPI)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--para-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "canva")
    args = ap.parse_args()
    root, out = args.para_root, args.out
    out.mkdir(parents=True, exist_ok=True)

    slots = {
        json.loads(l)["id"]: json.loads(l)["candidates"]
        for l in (root / "data/work/stages/slots.jsonl").read_text(encoding="utf-8").splitlines()
    }[ROW]
    meta = {
        r["id"]: r
        for r in map(json.loads, (root / "data/output/default_run/metadata_synth.jsonl")
                     .read_text(encoding="utf-8").splitlines())
    }
    row = meta[ROW]
    t0, t1 = WINDOW_S
    shown = [s for s in slots if t0 < s["time_s"] < t1]

    # Source recording, the window shown in the figure.
    src, sr = load(root / f"data/raw/audio/{ROW}.wav")
    seg = src[int(t0 * sr): int(t1 * sr)]
    speech = np.full(len(seg), SPEECH, dtype=object)
    spans = [((s["time_s"] - s["pause_s"] / 2 - t0) / (t1 - t0),
              (s["time_s"] + s["pause_s"] / 2 - t0) / (t1 - t0)) for s in shown]

    # (1) Slot finding: the recording in; the recording with its pauses marked out.
    draw(out / "1_slot_finding_input", seg, speech, 12, 2.9)
    draw(out / "1_slot_finding_output", seg, speech, 12, 2.9, spans=spans)
    # (2) Tag selection: the same audio and slots in; a slot and a tag out (text only).
    # (3) Clip selection: the candidate clips of the chosen class in; the chosen one out.
    clips = []
    for i, rid in enumerate(CANDIDATES):
        clip, csr = load(root / f"data/work/vs_{rid}.wav")
        clip = trim(clip, csr)
        clips.append(clip)
        draw(out / f"3_clip_selection_input_candidate{i + 1}", clip,
             np.full(len(clip), FOREIGN, dtype=object), 4, 1.1, 250)
    draw(out / "3_clip_selection_output", clips[0], np.full(len(clips[0]), FOREIGN, dtype=object),
         4, 1.1, 250)
    # (4) Voice conversion: the chosen clip in another person's voice in; the host's voice out.
    draw(out / "4_voice_conversion_input", clips[0], np.full(len(clips[0]), FOREIGN, dtype=object),
         4, 1.1, 250)
    conv_name = row["converted_audio"].replace("\\", "/").rsplit("/", 1)[-1]
    conv, csr = load(root / "data/work/output_vc" / conv_name)
    conv = trim(conv, csr, top_db=20)
    draw(out / "4_voice_conversion_output", conv, np.full(len(conv), EVENT, dtype=object), 4, 1.1, 250)
    # (5) Splicing: the recording and the converted event in; the finished recording out.
    draw(out / "5_splicing_input_recording", seg, speech, 12, 2.9)
    draw(out / "5_splicing_input_event", conv, np.full(len(conv), EVENT, dtype=object), 4, 1.1, 250)
    para, psr = load(root / f"data/output/default_run/para_{ROW}.wav")
    added = len(para) / psr - len(src) / sr
    cut = row["splice_at_s"]
    pseg = para[int(t0 * psr): int((t1 + added) * psr)]
    colors = np.full(len(pseg), SPEECH, dtype=object)
    colors[int((cut - t0) * psr): int((cut - t0 + added) * psr)] = EVENT
    draw(out / "5_splicing_output", pseg, colors, 12, 2.9)
    grey = colors.copy()
    grey[grey == EVENT] = FOREIGN
    draw(out / "5_splicing_output_grey", pseg, grey, 12, 2.9)
    # (6) Filter: the finished recording in; the same recording out when it passes.
    draw(out / "6_filter_input", pseg, colors, 12, 2.9)
    draw(out / "6_filter_output", pseg, colors, 12, 2.9)

    # Panel (b): illustrative strips, not data.
    x, c = schematic(3, 2.2, None)
    draw(out / "b_inference_input_reference_voice", x, c, 4.5, 1.2, 300)
    x, c = schematic(7, 3.4, (1.25, 1.95))
    draw(out / "b_inference_output_speech", x, c, 7, 1.4, 400)

    # Transcripts. The figure shows the window around the two slots, numbered <1> and <2>.
    tagged = row["transcript"]
    plain = " ".join(tagged.replace(row["nv_tag"], "").split())
    lines = [
        f"Text for the Canva assets, from the pilot row {ROW} (window {t0}-{t1} s).",
        "",
        "Full transcript (untagged):",
        plain,
        "",
        "Full transcript (tagged):",
        tagged,
        "",
        "1 Slot finding",
        "  input : ...chứ thôi , để nó lành thôi chắc mai nó lành thôi , nhưng mà...",
        "  output: ...chứ thôi , <1> để nó lành thôi chắc mai nó lành thôi , <2> nhưng mà...",
        "  (grey bands in 1_slot_finding_output: left band = <1>, right band = <2>)",
        "",
        "2 Tag selection (Qwen3-Omni listens to the audio and reads the marked transcript)",
        "  input : audio + ...chứ thôi , <1> để nó lành thôi chắc mai nó lành thôi , <2> nhưng mà...",
        '  output: {"slot": 1, "tag": "laughter"}  ->  <1> + [laughter]',
        "",
        "3 Clip selection",
        "  input : [laughter] + 3 candidate laughter clips (other people's voices)",
        "  output: the clip drawn with probability p ∝ exp(-Σ λ d), shown as the first candidate",
        "",
        "4 Voice conversion (Seed-VC)",
        "  input : the chosen clip, in another person's voice",
        "  output: the same laugh in the host speaker's voice",
        "",
        "5 Splicing",
        "  input : the recording + the converted laugh",
        "  output: ...chứ thôi , [laughter] để nó lành thôi chắc mai nó lành thôi , nhưng mà...",
        "  (red span in 5_splicing_output = the inserted laugh with its silence)",
        "",
        "6 Filter (NISQA)",
        "  input : the finished recording",
        "  output: kept (quality and continuity did not drop past the thresholds)",
        "",
        "(b) Fine-tuning and inference (illustrative waveforms)",
        "  input : hôm nay vui quá [laughter] ...  + reference voice",
        "  output: speech with the laugh",
        "",
        "Colours: speech #2B2B2B, NVV event #C62828, other people's clips #8C8C8C, LLM #E07B1F,",
        "fine-tuned model #A61C1C, slots #D9D9D9.",
    ]
    (out / "transcripts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(list(out.glob('*.svg')))} SVG + PNG pairs and transcripts.txt to {out}")
    print(f"slots shown (i, time_s): {[(s['i'], s['time_s']) for s in shown]}")


if __name__ == "__main__":
    main()
