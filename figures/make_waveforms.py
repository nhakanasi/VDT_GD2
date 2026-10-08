"""Render the waveform strips used by figures/overview.tex from one real pilot row.

    python figures/make_waveforms.py --para-root ../Para

Reads the para-synth outputs for ROW and writes figures/assets/*.pdf plus
figures/assets/coords.tex, which tells the TikZ figure where the slots and the inserted
event fall along each strip (as fractions of its width), so the labels are set in the
paper's font rather than matplotlib's.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROW = "tXO46Ys-7Qc_00028"
# The two slots shown, as (time_s, pause_s); the first is the one the LLM chose.
WINDOW_S = (8.6, 13.2)
# Three laughter clips shown as selection candidates; the first is the one picked for ROW.
CANDIDATES = [ROW, "ZTdhpdp4Avk_00164", "YK4u38DDlJY_00066"]

SPEECH = "#3B6EA5"
EVENT = "#C55A11"
FOREIGN = "#8C8C8C"
CM = 1 / 2.54


def load(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(path, always_2d=True)
    return x.mean(axis=1), sr


def trim(x: np.ndarray, sr: int, top_db: float = 30.0) -> np.ndarray:
    """Cut leading/trailing silence, so a thumbnail shows the event and not its padding."""
    win = max(int(0.01 * sr), 1)
    env = np.sqrt(np.convolve(x**2, np.ones(win) / win, mode="same"))
    loud = np.flatnonzero(env > env.max() * 10 ** (-top_db / 20))
    return x[loud[0] : loud[-1] + 1] if len(loud) else x


def strip(path: Path, x: np.ndarray, colors: np.ndarray, width_cm: float, height_cm: float,
          bins: int = 700) -> None:
    """Min/max waveform envelope, one colour per bin, on a transparent background."""
    peak = np.max(np.abs(x)) + 1e-9
    edges = np.linspace(0, len(x), bins + 1).astype(int)
    lo = np.array([x[a:b].min() if b > a else 0 for a, b in zip(edges[:-1], edges[1:])]) / peak
    hi = np.array([x[a:b].max() if b > a else 0 for a, b in zip(edges[:-1], edges[1:])]) / peak
    col = colors[np.minimum(((edges[:-1] + edges[1:]) // 2), len(colors) - 1)]

    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
    ax = fig.add_axes((0, 0, 1, 1))
    for c in np.unique(col):
        m = col == c
        ax.vlines(np.flatnonzero(m), lo[m], hi[m], colors=c, linewidth=0.55)
    ax.set_xlim(-0.5, bins - 0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    fig.savefig(path, transparent=True)
    plt.close(fig)


def schematic(seed: int, dur_s: float, event: tuple[float, float] | None, sr: int = 8000):
    """Speech-like noise bursts for the illustrative (non-data) strips in panel (b)."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(dur_s * sr)) / sr
    env = np.zeros_like(t)
    s = 0.05
    while s < dur_s - 0.15:
        d = rng.uniform(0.12, 0.3)
        env += np.exp(-0.5 * ((t - s - d / 2) / (d / 3.2)) ** 2) * rng.uniform(0.5, 1.0)
        s += d + rng.uniform(0.0, 0.08)
    colors = np.full(len(t), SPEECH, dtype=object)
    if event:
        a, b = event
        m = (t >= a) & (t <= b)
        env[m] = 0
        for c in np.linspace(a + 0.05, b - 0.05, 5):
            env += np.exp(-0.5 * ((t - c) / 0.03) ** 2) * 0.9 * ((t >= a) & (t <= b))
        colors[m] = EVENT
    return rng.standard_normal(len(t)) * env, colors


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--para-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "assets")
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

    # (1) Source recording: speech, with the candidate pauses shaded.
    src, sr = load(root / f"data/raw/audio/{ROW}.wav")
    seg = src[int(t0 * sr) : int(t1 * sr)]
    strip(out / "source.pdf", seg, np.full(len(seg), SPEECH, dtype=object), 4.6, 1.1)

    # (3) Candidate VocalSound clips, in their original (foreign) voices.
    for i, rid in enumerate(CANDIDATES):
        clip, _ = load(root / f"data/work/vs_{rid}.wav")
        clip = trim(clip, 16000)
        strip(out / f"cand{i}.pdf", clip, np.full(len(clip), FOREIGN, dtype=object), 1.5, 0.42, 250)

    # (4) The chosen clip after Seed-VC, now in the host speaker's voice.
    # metadata holds the path on the machine that ran the pipeline; only the name is portable
    conv_name = row["converted_audio"].replace("\\", "/").rsplit("/", 1)[-1]
    conv, csr = load(root / "data/work/output_vc" / conv_name)
    conv = trim(conv, csr, top_db=20)  # 30 dB keeps a click Seed-VC leaves at the onset
    strip(out / "converted.pdf", conv, np.full(len(conv), EVENT, dtype=object), 1.5, 0.42, 250)

    # (5) The finished recording: the inserted span (gaps + event) is everything the splice
    # added, i.e. the length difference, starting at the cut.
    para, psr = load(root / f"data/output/default_run/para_{ROW}.wav")
    added = len(para) / psr - len(src) / sr
    cut = row["splice_at_s"]
    a, b = int(t0 * psr), int((t1 + added) * psr)
    pseg = para[a:b]
    colors = np.full(len(pseg), SPEECH, dtype=object)
    ia, ib = int((cut - t0) * psr), int((cut - t0 + added) * psr)
    colors[ia:ib] = EVENT
    strip(out / "spliced.pdf", pseg, colors, 4.6, 1.1)

    # Panel (b): illustrative strips, not data.
    x, c = schematic(3, 2.2, None)
    strip(out / "reference.pdf", x, c, 1.9, 0.5, 300)
    x, c = schematic(7, 3.4, (1.25, 1.95))
    strip(out / "generated.pdf", x, c, 3.0, 0.6, 400)

    def frac(t: float, lo: float, hi: float) -> float:
        return (t - lo) / (hi - lo)

    lines = [f"% Generated by make_waveforms.py from {ROW}; fractions of each strip's width."]
    for k, s in enumerate(shown, start=1):
        name = "ABCDEFG"[k - 1]
        lines.append(f"\\def\\slot{name}{{{frac(s['time_s'], t0, t1):.4f}}}")
        lines.append(f"\\def\\slot{name}w{{{s['pause_s'] / (t1 - t0):.4f}}}")
    lines.append(f"\\def\\insA{{{frac(cut, t0, t1 + added):.4f}}}")
    lines.append(f"\\def\\insB{{{frac(cut + added, t0, t1 + added):.4f}}}")
    (out / "coords.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(list(out.glob('*.pdf')))} strips + coords.tex to {out}")
    print(f"slots shown (i, time_s): {[(s['i'], s['time_s']) for s in shown]}")


if __name__ == "__main__":
    main()
