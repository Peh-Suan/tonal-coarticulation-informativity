"""
Step 5: F0 at the syllable boundary of each within-word bigram:
the OFFSET of tone 1 and the ONSET of tone 2 (Parselmouth / Praat).

Method (as in the manuscript, sec. 2.1.4):
  * F0 is extracted for each syllable and the syllable is divided into 11
    equal portions; each portion's value is the mean F0 of the voiced pitch
    frames inside it.
  * Onset = mean of the first portion; offset = mean of the last portion.
    If that portion has no F0, the 2nd (2nd-to-last) portion is used, then
    the 3rd (3rd-to-last). If none of the three has F0, the value is missing
    (the token is discarded in the analysis).
  * The portion used is recorded as a portion index, 0 = closest to the
    syllable boundary, 2 = furthest (the paper's covariate; z-score it
    within tokens in the model as in the paper).
  * Onset / offset segment types (vowel, liquid, nasal, obstruent) are
    recorded for the random effects: the onset segment of syllable 2 and
    the offset (coda) segment of syllable 1.
  * Normalisation as in tc_mi/All_F0.csv: ln(Hz), z-scored within speaker
    over all 11 portion means of all syllables of that speaker (logz).

Pitch settings default to Praat's "To Pitch..." defaults (autocorrelation,
time step = 0.75 / floor, floor 75 Hz, ceiling 600 Hz); change them with
--floor / --ceiling / --time-step.

Measured interval (--interval, default "voiced"): the paper's stimuli had no
voiceless obstruents, so the whole syllable was voiced. In the corpus they are
everywhere, and Praat finds spurious "pitch" in frication noise (e.g. a
509 Hz onset on shi4). So by default the interval is the whole syllable when
the initial is a vowel, nasal or liquid, and starts at the final when the
initial is an obstruent. "syllable" = always whole syllable; "final" = always
from the final. Values more than 3 SD from the speaker mean (likely octave
errors) are flagged in *_suspect.

Run in macOS Terminal (Parselmouth is in your env):
    python 07_f0_boundary.py <path/to/COSPRO>
    python 07_f0_boundary.py <path/to/COSPRO> --jobs 8

Outputs (out/):
  syllables_f0.csv  per syllable: f0_p0..f0_p10 (Hz), onset/offset (Hz),
                    onset/offset portion index, segment types, logz values
  bigrams_f0.csv    bigrams.csv + offset1_* (tone 1) and onset2_* (tone 2)
"""
import argparse, math, os
from multiprocessing import Pool
import numpy as np
import pandas as pd

N_PORTIONS = 11
ARGS = None

def onset_segment(initial):
    if not initial: return "vowel"
    if initial in ("m", "n"): return "nasal"
    if initial in ("l", "Z`"): return "liquid"          # Z` = pinyin r
    return "obstruent"

def offset_segment(final):
    if final.endswith(("n", "N")): return "nasal"        # -n, -ng
    if final == "@`": return "liquid"                     # er
    return "vowel"

def portion_means(t, f, start, end):
    """Mean F0 (Hz) of the voiced frames in each of 11 equal portions."""
    edges = np.linspace(start, end, N_PORTIONS + 1)
    out = []
    for k in range(N_PORTIONS):
        lo, hi = edges[k], edges[k + 1]
        sel = (t >= lo) & ((t < hi) if k < N_PORTIONS - 1 else (t <= hi)) & (f > 0)
        out.append(float(f[sel].mean()) if sel.any() else math.nan)
    return out

def pick(vals, order):
    """First available value in `order`; returns (value, portion index 0-2)."""
    for idx, k in enumerate(order):
        if not math.isnan(vals[k]): return vals[k], idx
    return math.nan, math.nan

def process(item):
    rel, rows = item
    import parselmouth
    a = ARGS
    wav = os.path.join(a.root, rel.replace(".TextGrid", ".wav"))
    try:
        snd = parselmouth.Sound(wav)
        pitch = snd.to_pitch(time_step=a.time_step, pitch_floor=a.floor, pitch_ceiling=a.ceiling)
    except Exception as e:
        return [], f"{rel}: {e}"
    t = pitch.xs(); f = pitch.selected_array["frequency"]
    res = []
    for r in rows:
        if a.interval == "final": start = r["v_start"]
        elif a.interval == "syllable": start = r["start"]
        else: start = r["start"] if onset_segment(r["initial"]) != "obstruent" else r["v_start"]
        p = portion_means(t, f, start, r["end"])
        on, on_i = pick(p, [0, 1, 2])
        off, off_i = pick(p, [10, 9, 8])
        d = dict(file=rel, syl_idx=r["syl_idx"], speaker=r["speaker"],
                 onset_hz=on, onset_portion=on_i, offset_hz=off, offset_portion=off_i,
                 onset_segment=onset_segment(r["initial"]), offset_segment=offset_segment(r["final"]))
        d.update({f"f0_p{k}": v for k, v in enumerate(p)})
        res.append(d)
    return res, None

def init(args):
    global ARGS
    ARGS = args

def main():
    import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _freshness import require_fresh; require_fresh("out", [('syllables_mlm.csv', 'syllables_chars.csv'), ('bigrams.csv', 'syllables_mlm.csv')])
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("--out", default="out")
    ap.add_argument("--bigrams", default="bigrams.csv")
    ap.add_argument("--floor", type=float, default=75.0); ap.add_argument("--ceiling", type=float, default=600.0)
    ap.add_argument("--time-step", type=float, default=None, help="default: Praat's 0.75/floor")
    ap.add_argument("--interval", choices=["voiced", "syllable", "final"], default="voiced")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--renormalize", action="store_true",
                    help="skip pitch extraction: reload out/syllables_f0.csv and redo only the "
                         "speaker normalisation and the bigram table")
    a = ap.parse_args()

    if a.renormalize:
        big = pd.read_csv(f"{a.out}/{a.bigrams}", keep_default_na=False, low_memory=False)
        f0 = pd.read_csv(f"{a.out}/syllables_f0.csv", low_memory=False)
        f0 = f0.drop(columns=[c for c in f0 if c.endswith(("_logz", "_suspect"))])
        syl = pd.read_csv(f"{a.out}/syllables.csv", keep_default_na=False, low_memory=False,
                          usecols=["file", "syl_idx", "start"])
        print(f"renormalising {len(f0)} syllables from {a.out}/syllables_f0.csv")
        return finish(a, big, syl, f0)

    if not os.path.isdir(a.root):
        raise SystemExit(f"Corpus folder not found: {a.root}\nIs the drive plugged in? Check the name with: ls /Volumes")
    big = pd.read_csv(f"{a.out}/{a.bigrams}", keep_default_na=False, low_memory=False)
    files = list(dict.fromkeys(big.file))[: a.limit]
    syl = pd.read_csv(f"{a.out}/syllables.csv", keep_default_na=False, low_memory=False)
    syl = syl[syl.file.isin(set(files)) & (syl.pinyin != "")]
    items = [(f, g[["syl_idx", "speaker", "start", "v_start", "end", "initial", "final"]].to_dict("records"))
             for f, g in syl.groupby("file", sort=False)]
    print(f"{len(items)} recordings, {len(syl)} syllables, {a.jobs} processes, "
          f"pitch {a.floor:g}-{a.ceiling:g} Hz, interval={a.interval}")

    rows, errs = [], []
    with Pool(a.jobs, initializer=init, initargs=(a,)) as pool:
        for i, (res, err) in enumerate(pool.imap_unordered(process, items, chunksize=8), 1):
            rows += res
            if err: errs.append(err)
            if i >= 50 and len(errs) == i:
                pool.terminate()
                raise SystemExit(f"The first {i} recordings all failed to load, e.g.\n  {errs[0]}\n"
                                 "Is the drive plugged in and the path right? Nothing was written.")
            if i % 500 == 0 or i == len(items): print(f"  {i}/{len(items)} recordings")
    for e in errs[:10]: print("  ERROR", e)

    if not rows: raise SystemExit("No F0 values extracted; nothing written.")
    f0 = pd.DataFrame(rows)
    return finish(a, big, syl, f0)

def finish(a, big, syl, f0):
    # speaker normalisation: ln Hz, z-scored over all portion means of the speaker.
    # Speaker = subcorpus + speaker ID: COSPRO reuses IDs across subcorpora for
    # different people (e.g. F001 in corspro2 vs corspro3 have different pitch levels).
    f0["speaker_key"] = f0.file.str.split("/").str[0] + "_" + f0.speaker.astype(str)
    pcols = [f"f0_p{k}" for k in range(N_PORTIONS)]
    logs = np.log(f0[pcols])
    stats = pd.DataFrame({"speaker": f0.speaker_key.repeat(N_PORTIONS).values,
                          "v": logs.values.ravel()}).dropna().groupby("speaker").v.agg(["mean", "std"])
    mu, sd = f0.speaker_key.map(stats["mean"]), f0.speaker_key.map(stats["std"])
    for c in ("onset", "offset"):
        f0[f"{c}_logz"] = (np.log(f0[f"{c}_hz"]) - mu) / sd
        f0[f"{c}_suspect"] = f0[f"{c}_logz"].abs() > 3
    f0.to_csv(f"{a.out}/syllables_f0.csv", index=False)

    s1 = f0[["file", "syl_idx", "offset_hz", "offset_logz", "offset_portion", "offset_segment", "offset_suspect"]] \
        .rename(columns=lambda c: c if c in ("file", "syl_idx") else c + "1").rename(columns={"syl_idx": "syl_idx1"})
    s2 = f0[["file", "syl_idx", "onset_hz", "onset_logz", "onset_portion", "onset_segment", "onset_suspect"]] \
        .rename(columns=lambda c: c if c in ("file", "syl_idx") else c + "2").rename(columns={"syl_idx": "syl_idx2"})
    if "syl_idx1" not in big:     # older bigrams.csv: recover syllable indices from times
        big = big.merge(syl[["file", "syl_idx", "start"]].rename(columns={"syl_idx": "syl_idx1", "start": "start1"}),
                        on=["file", "start1"], how="left")
    big["syl_idx2"] = big.syl_idx1 + 1
    b = big.merge(s1, on=["file", "syl_idx1"], how="left").merge(s2, on=["file", "syl_idx2"], how="left")
    b.to_csv(f"{a.out}/bigrams_f0.csv", index=False)

    ok = b.offset_hz1.notna() & b.onset_hz2.notna()
    print(f"{len(b)} bigrams; offset1 found {b.offset_hz1.notna().mean():.1%}, "
          f"onset2 found {b.onset_hz2.notna().mean():.1%}, both {ok.mean():.1%} "
          f"(paper: 12.5% of tokens discarded)")
    print("offset1 portion used:", b.offset_portion1.value_counts(normalize=True).round(3).sort_index().to_dict())
    print("onset2 portion used: ", b.onset_portion2.value_counts(normalize=True).round(3).sort_index().to_dict())
    print(f"suspect (|z|>3): offset1 {b.offset_suspect1.eq(True).mean():.1%}, onset2 {b.onset_suspect2.eq(True).mean():.1%}")

if __name__ == "__main__":
    main()
