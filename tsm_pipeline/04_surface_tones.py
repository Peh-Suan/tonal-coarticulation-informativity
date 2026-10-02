"""Derive surface (sandhi) tones for TAT-MOE syllables from citation tones.

Taiwanese Southern Min: every syllable that is not final in its tone group
takes its sandhi form; the tone-group-final syllable keeps its citation tone.
Tone-group edges aren't marked in the corpus, so each syllable gets a
juncture label (what follows it) and a certainty flag:

  juncture_after
    word_internal   next syllable is in the same word (joined by '-')     -> sandhi, certain
    pre_neutral     next syllable is a '--' neutral-tone syllable          -> citation, certain
    punct           punctuation follows                                   -> citation, certain
    utt_final       last syllable of the utterance                        -> citation, certain
    pause           word-final and followed by a pause >= --min-pause     -> citation, likely
    word_boundary   word-final, no punctuation or pause                    -> UNCERTAIN
                    (sandhi inside a phrase, citation at an XP edge; the best guess is
                     sandhi, flagged surface_certain = 0)
  Additional flags:
    neutral         the syllable is neutral tone ('--' and every later syllable
                    of that word, e.g. '--loh8-lai5') -> surface_tone 'N' (exclude)
    pre_a_suffix    next syllable is the suffix á (a2) in the same word: that
                    context has its own sandhi pattern -> surface_certain = 0
    tone 6 / 9 syllables -> surface_tone '' (exclude)

SANDHI below is the general ("通行腔", southern-type) pattern. PLEASE CHECK
IT, especially the checked tones and the T5 / T8-h regional variants.
--t5 north switches 5 -> 3 (northern / 泉-leaning); the default is 5 -> 7.

Per-speaker accent (2026-09-29): if <prep>/speaker_accents.csv exists (column
accent_group, built from the TAT-MOE 腔調 field), speakers in group 'zhang'
(mixed / general / southern / 漳-leaning / central / eastern) get 5 -> 7,
speakers in group 'quan' (泉-leaning) get 5 -> 3, and all other speakers
(不清楚, Hakka 四縣, 海口 / 老泉, 澎湖) are DROPPED from the output. The rest of
the SANDHI table is the same for both groups. --speaker-groups '' disables this.

Input : <prep>/syllable_alignments.csv (from 03)   Output: <prep>/syllables_surface.csv
  python 04_surface_tones.py --prep ~/tsm_mfa/all_condenser
"""
import argparse
import os
from pathlib import Path

import pandas as pd

# citation tone -> sandhi tone (Tâi-lô tone numbers)
SANDHI = {
    "1": "7",
    "2": "1",
    "3": "2",
    "5": "7",       # northern / 泉-leaning: "3" (use --t5 north)
    "7": "3",
    # checked tones: key = tone + coda type ('ptk' or 'h')
    "4ptk": "8",
    "4h": "2",      # glottal coda lost, becomes like T2 (high falling)
    "8ptk": "3",    # short low falling; CHECK
    "8h": "3",      # glottal coda lost; some southern speakers: "7"; CHECK
}


def sandhi_of(tone, base):
    if tone in ("4", "8"):
        key = tone + ("h" if base.endswith("h") else "ptk")
        return SANDHI[key]
    return SANDHI.get(tone, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--min-pause", type=float, default=0.12)
    ap.add_argument("--t5", choices=["south", "north"], default="south",
                    help="T5 rule for everyone when no speaker-groups file is used")
    ap.add_argument("--speaker-groups", default="speaker_accents.csv",
                    help="csv (relative to --prep) with speaker, accent_group in {zhang, quan, drop}; '' = off")
    a = ap.parse_args()
    if a.t5 == "north":
        SANDHI["5"] = "3"
    prep = Path(os.path.expanduser(a.prep))

    d = pd.read_csv(prep / "syllable_alignments.csv", dtype={"tone": str}, keep_default_na=False)
    grp_path = prep / a.speaker_groups if a.speaker_groups else None
    if grp_path and grp_path.exists():
        g = pd.read_csv(grp_path).set_index("speaker").accent_group
        d["accent_group"] = d.speaker.map(g).fillna("drop")
        n0, s0 = len(d), d.speaker.nunique()
        d = d[d.accent_group.isin(["zhang", "quan"])].copy()
        print(f"speaker groups from {grp_path.name}: kept {d.speaker.nunique()} of {s0} speakers "
              f"({d.groupby('accent_group').speaker.nunique().to_dict()}), {len(d)} of {n0} syllables")
    else:
        d["accent_group"] = "all"
        print(f"no speaker-groups file: T5 -> {SANDHI['5']} for every speaker")
    d = d.sort_values(["utt_id", "syl_idx"]).reset_index(drop=True)
    # '--' makes every following syllable of that word neutral (e.g. '--loh8-lai5'),
    # but 01 only flagged the first one
    d["neutral"] = d.groupby(["utt_id", "word_idx"]).neutral.cummax().astype(int)
    same_utt = d.utt_id.eq(d.utt_id.shift(-1))
    nxt_word = d.word_idx.shift(-1)
    nxt_neutral = d.neutral.shift(-1).fillna(0).astype(int).eq(1) & same_utt
    nxt_syl = d.syllable.shift(-1).fillna("")
    word_internal = same_utt & nxt_word.eq(d.word_idx) & ~nxt_neutral

    j = pd.Series("word_boundary", index=d.index)
    j[d.pause_after.astype(float) >= a.min_pause] = "pause"
    j[d.punct_after.astype(int).eq(1)] = "punct"
    j[~same_utt] = "utt_final"
    j[nxt_neutral] = "pre_neutral"
    j[word_internal] = "word_internal"
    d["juncture_after"] = j
    d["pre_a_suffix"] = (word_internal & nxt_syl.eq("a2")).astype(int)

    sandhi = [sandhi_of(t, b) for t, b in zip(d.tone, d.base)]
    # 泉-leaning speakers: T5 -> T3 instead of T7 (everything else as in SANDHI)
    quan = (d.accent_group.eq("quan") & d.tone.eq("5")).to_numpy()
    sandhi = ["3" if q else s for s, q in zip(sandhi, quan)]
    cit = d.tone.where(d.tone.isin(list("1234578")), "")
    d["sandhi_form"] = sandhi
    d["surface_tone"] = cit
    use_sandhi = j.isin(["word_internal", "word_boundary"])
    d.loc[use_sandhi, "surface_tone"] = pd.Series(sandhi, index=d.index)[use_sandhi]
    d["sandhi_applied"] = use_sandhi.astype(int)
    d["surface_certain"] = (j.isin(["word_internal", "pre_neutral", "punct", "utt_final"])
                            & d.pre_a_suffix.eq(0)).astype(int)
    d.loc[j.eq("pause"), "surface_certain"] = 1  # likely; filter on juncture_after if stricter
    neu = d.neutral.astype(int).eq(1)
    d.loc[neu, ["surface_tone", "surface_certain", "sandhi_applied"]] = ["N", 0, 0]
    d.loc[d.surface_tone.eq(""), "surface_certain"] = 0

    d.to_csv(prep / "syllables_surface.csv", index=False)
    print(f"{len(d)} syllables -> {prep / 'syllables_surface.csv'}")
    print(d.juncture_after.value_counts().to_string())
    print(f"surface tone certain: {d.surface_certain.mean():.1%}; neutral: {neu.sum()}; "
          f"before á: {d.pre_a_suffix.sum()}")


if __name__ == "__main__":
    main()
