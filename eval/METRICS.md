# Evaluation metrics

How each metric for the NVV TTS evaluation is computed. The protocol follows
`VLSP/track_a.py` (VLSP 2026 NVV Track A, approximated with local models) and the
tag check in `VLSP/nvv_tags.py`, extended from 4 tags to the 14 tags of
`eval/nvv_test_vi.jsonl`.

Each metric is reported and judged on its own. No combined score is computed.

## Test items

- **Sentences:** `eval/nvv_test_vi.jsonl`, 420 Vietnamese sentences, 30 per tag, one tag
  per sentence.
- **Speakers:** every sentence is synthesised once for every held-out speaker, each
  prompted with a reference recording and its transcript. The item id is
  `<sentence id>__<speaker>`.
- **Synthesis:** OmniVoice `generate(text, language="vi", ref_audio, ref_text)` with a
  fixed seed, so every system gets the same seed for the same item.
- **Tokenisation:** upstream OmniVoice `generate()` tokenises `[tag]` separately from the
  text around it, while training tokenises the whole string. A fine-tuned model can then
  miss or garble its tags at inference. Every system must be run with whole-string
  tokenisation (the patch in `VLSP/OmniVoice/omnivoice/models/omnivoice.py`), including the
  base model, so all systems see the same input.

## Summary

| Metric | Measures | Model | Scale | Better |
|---|---|---|---|---|
| NVPA | the required vocalisation is present at the tag's position | FlexSED + Vietnamese wav2vec2 CTC aligner | 0–1 | higher |
| SN | naturalness | UTMOS22 strong | 1–5 | higher |
| Q | perceptual quality | NISQA v2.0, overall MOS | 1–5 | higher |
| WER | intelligibility | Zipformer (sherpa-onnx), greedy | 0–1 | lower |
| pMOS | predicted MOS | DNSMOS P.835, overall | 1–5 | higher |
| SS | similarity to the reference speaker | ECAPA-TDNN | −1–1, about 0–1 in practice | higher |

In the VLSP task, SN and Q are human ratings. UTMOS and NISQA are automatic stand-ins
for them. The listening test in the paper (MOS and NVV-MOS) is reported separately.

## NVPA: NV placement accuracy

For each item, NVPA asks whether a sound of the tag's class occurs where the tag sits in
the text.

1. **Word timing.** Remove the tag from the text and force-align the remaining words to
   the synthesised audio (16 kHz). The aligner is
   `nguyenvulebinh/wav2vec2-base-vietnamese-250h` with torchaudio `forced_align`.
   Tokens with no letter or digit, such as standalone punctuation, are not aligned. A
   word with no character in the model's vocabulary, such as a number, gets a
   zero-length span where the previous word ends. Every word boundary is a slot, and the
   tag sits at one slot.
   - Qwen3-ForcedAligner is not used here. On Vietnamese it stretches words over
     non-verbal sounds and collapses runs of words to one instant.
2. **Sound events.** Run FlexSED (`flexsed_as.pt`) over the whole recording, in 10 s
   windows with a half-window hop. Each window is peak-normalised, and each frame keeps
   its highest posterior across overlapping windows. A class's events are runs of frames
   above 0.2 that contain at least one frame at or above the class threshold. Runs closer
   than 0.12 s are merged, events shorter than 0.08 s are dropped, and events shorter
   than 0.25 s that touch the start or end of the clip are ignored as cut-off tails.
3. **Status of the tag:**
   - **ok:** an event of the tag's class lies within 3 s of the tag's slot, with no
     aligned word between them. A word that overlaps the sound by at most 0.05 s does not
     count as between.
   - **misplaced:** the nearest such event has exactly one word between it and the slot.
   - **other class:** no event of the tag's class, but another class's event is at the
     slot. The model produced a sound, just not the right one.
   - **unsupported:** no event at the slot at all.
4. **Score.** NVPA is the share of tags with status ok, reported overall and per tag.
   Misplaced, other class and unsupported counts are reported alongside it as
   diagnostics. They do not count towards NVPA.

FlexSED prompt and event threshold per tag:

| Tag | FlexSED prompt | Threshold |
|---|---|---|
| laughter | Laughter | 0.5 |
| sigh | Sigh | 0.5\* |
| cough | Cough | 0.5\* |
| throatclearing | Throat clearing | 0.4 |
| sneeze | Sneeze | 0.5\* |
| sniff | Sniff | 0.5 |
| breathing | Breathing | 0.7 |
| crying | Crying, sobbing | 0.5\* |
| gasp | Gasp | 0.5\* |
| moan | Wail, moan | 0.5\* |
| pant | Pant | 0.5\* |
| screaming | Screaming | 0.5\* |
| snoring | Snoring | 0.5\* |
| yawn | Yawn | 0.5\* |

The thresholds for laughter, throat clearing, sniff and breathing were set by listening
on vinv-tts. Breathing is stricter because FlexSED gives 0.5–0.7 to pauses where a
listener hears no breath. The other ten (\*) are placeholders and have not been checked
by listening. Calibrate them on real recordings of each class before reporting
per-tag NVPA for those classes. All 14 prompts have precomputed CLAP embeddings in
FlexSED, so no CLAP model is needed at scoring time.

Closely related classes are scored strictly: a sigh, pant or gasp also triggers
FlexSED's Breathing prompt, and a moan can trigger Crying. A tag counts as ok only when
its own class fires. When another class fires instead, the tag shows up as other class.

## SN: speech naturalness

UTMOS22 strong learner (`utmos22_strong_step7459_v1.pt` from `k2-fsa/TTS_eval_models`,
the model class in `omnivoice.eval.models.utmos`). The input is the synthesised
recording at 16 kHz, and the output is a predicted naturalness MOS from 1 to 5. Report
the mean over items per system.

## Q: quality

NISQA v2.0 overall MOS (torchmetrics `non_intrusive_speech_quality_assessment`). The
input is the synthesised recording, resampled to 48 kHz. NISQA also predicts noisiness,
coloration, discontinuity and loudness, which can be kept as diagnostics. Report the
mean overall MOS per system.

NISQA's weights are licensed CC BY-NC-SA 4.0, so they may be used for research only.

## WER: word error rate

- **ASR:** `sherpa-onnx-zipformer-vi-2025-04-20` (k2-fsa sherpa-onnx asr-models, the
  `epoch-12-avg-8` encoder, decoder and joiner), greedy search, 16 kHz input, 80-dim
  features.
- **Normalisation, applied to both reference and hypothesis:** remove `[tag]`s, apply
  Unicode NFC, lower-case, replace every character that is not a letter or digit with a
  space, and collapse spaces. An empty hypothesis is scored as `<empty>`.
- **Aggregation:** corpus-level WER (jiwer) over all items of a system, not the mean of
  per-item WERs.

Rows that came from FLEURS and MiniMax contain digits (e.g. 1889, 3,5%, 12:00). The
reference keeps the digits, but the ASR writes numbers out as words, so these rows raise
WER for every system. Before comparing WERs, either spell the numbers out in the test
text or also report WER on the digit-free rows.

## pMOS: predicted MOS

DNSMOS P.835 (torchmetrics `deep_noise_suppression_mean_opinion_score`,
`personalized=False`) on the recording at 16 kHz. pMOS is the OVRL score. SIG
(speech), BAK (background) and P.808 are kept as diagnostics. Report the mean per
system.

## SS: speaker similarity

ECAPA-TDNN (`speechbrain/spkrec-ecapa-voxceleb`). Both the synthesised recording and the
speaker's reference recording are embedded at 16 kHz, each embedding is L2-normalised,
and SS is the cosine between them. Report the mean per system.

## Reporting

Report one table per system with one column per metric, and a second table with NVPA per
tag (ok / tags) for every system. A per-tag breakdown of the other metrics is optional,
but useful when a tag's audio degrades only that tag, for example a scream that NISQA
scores poorly.
