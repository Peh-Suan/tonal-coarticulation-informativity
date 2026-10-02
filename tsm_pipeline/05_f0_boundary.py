"""F0 at syllable boundaries for TAT-MOE (TSM), same method as
cospro_pipeline/07_f0_boundary.py (manuscript sec. 2.1.4):

  * each syllable's measured interval is split into 11 equal portions; each
    portion value = mean F0 of the voiced pitch frames inside it
  * onset = portion 0 (else 1, else 2); offset = portion 10 (else 9, else 8);
    onset_portion / offset_portion record which one (0 = nearest the boundary)
  * ln(Hz) z-scored within speaker over all portion means (logz);
    |logz| > 3 flagged *_suspect

Interval (--interval voiced, default): starts at the first vowel when the onset
is an obstruent (p ph b t th k kh g h s ts tsh j), otherwise at the syllable
start; ends at the last vowel when the coda is a stop (p_ t_ k_) or glottal
stop (q), otherwise at the syllable end. "syllable" = always the whole syllable.

Segment types (for random effects), as in COSPRO:
  onset_segment : vowel | nasal (m n ng) | liquid (l) | obstruent
  offset_segment: vowel | nasal (m n ng, syllabic m/ng) | obstruent (p_ t_ k_ q)

Input : <prep>/syllables_surface.csv (from 04) + <prep>/corpus/<spk>/<utt>.wav
Output: <prep>/syllables_f0.csv   one row per syllable
        <prep>/bigrams_f0.csv     adjacent syllable pairs in the same utterance:
          syllable 1 offset + syllable 2 onset, citation and surface tones,
          same_word, word_len, pause_between, juncture/certainty flags, durations

Run in macOS Terminal (needs parselmouth, numpy, pandas):
  python 05_f0_boundary.py --prep ~/tsm_mfa/all_condenser --jobs 8
"""
import argparse
import math
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

N_PORTIONS = 11
ARGS = None
OBSTRUENTS = {"p", "ph", "b", "t", "th", "k", "kh", "g", "h", "s", "ts", "tsh", "j"}


def onset_segment(first_phone):
    if first_phone in ("m", "n", "ng"):
        return "nasal"
    if first_phone == "l":
        return "liquid"
    if first_phone in OBSTRUENTS:
        return "obstruent"
    return "vowel"            # zero initial, incl. syllabic m= / ng=


def offset_segment(last_phone):
    if last_phone in ("p_", "t_", "k_", "q"):
        return "obstruent"
    if last_phone in ("m", "n", "ng", "m=", "ng="):
        return "nasal"
    return "vowel"


def portion_means(t, f, start, end):
    edges = np.linspace(start, end, N_PORTIONS + 1)
    out = []
    for k in range(N_PORTIONS):
        lo, hi = edges[k], edges[k + 1]
        sel = (t >= lo) & ((t < hi) if k < N_PORTIONS - 1 else (t <= hi)) & (f > 0)
        out.append(float(f[sel].mean()) if sel.any() else math.nan)
    return out


def pick(vals, order):
    for idx, k in enumerate(order):
        if not math.isnan(vals[k]):
            return vals[k], idx
    return math.nan, math.nan


def process(item):
    wav, rows = item
    import parselmouth
    a = ARGS
    try:
        snd = parselmouth.Sound(wav)
        pitch = snd.to_pitch(time_step=a.time_step, pitch_floor=a.floor, pitch_ceiling=a.ceiling)
    except Exception as e:
        return [], f"{wav}: {e}"
    t = pitch.xs()
    f = pitch.selected_array["frequency"]
    res = []
    for r in rows:
        ph = str(r["phones"]).split()
        onseg, offseg = onset_segment(ph[0]), offset_segment(ph[-1])
        start, end = float(r["start"]), float(r["end"])
        if a.interval == "voiced":
            if onseg == "obstruent" and r["vowel_start"] != "":
                start = float(r["vowel_start"])
            if offseg == "obstruent" and r["vowel_end"] != "":
                end = float(r["vowel_end"])
        p = portion_means(t, f, start, end) if end > start else [math.nan] * N_PORTIONS
        on, on_i = pick(p, [0, 1, 2])
        off, off_i = pick(p, [10, 9, 8])
        d = dict(utt_id=r["utt_id"], syl_idx=r["syl_idx"], onset_hz=on, onset_portion=on_i,
                 offset_hz=off, offset_portion=off_i, onset_segment=onseg,
                 offset_segment=offseg, meas_start=start, meas_end=end)
        d.update({f"f0_p{k}": v for k, v in enumerate(p)})
        res.append(d)
    return res, None


def init(args):
    global ARGS
    ARGS = args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--floor", type=float, default=75.0)
    ap.add_argument("--ceiling", type=float, default=600.0)
    ap.add_argument("--time-step", type=float, default=None, help="default: Praat's 0.75/floor")
    ap.add_argument("--interval", choices=["voiced", "syllable"], default="voiced")
    ap.add_argument("--min-dur", type=float, default=0.06,
                    help="syllables with a shorter measured interval are flagged too_short")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--limit", type=int, help="only the first N utterances (testing)")
    a = ap.parse_args()
    prep = Path(os.path.expanduser(a.prep))

    syl = pd.read_csv(prep / "syllables_surface.csv", keep_default_na=False, low_memory=False,
                      dtype={"tone": str, "surface_tone": str, "sandhi_form": str})
    utts = list(dict.fromkeys(syl.utt_id))[: a.limit]
    syl = syl[syl.utt_id.isin(set(utts))]
    cols = ["utt_id", "syl_idx", "start", "end", "vowel_start", "vowel_end", "phones"]
    items = [(str(prep / "corpus" / g.speaker.iat[0] / f"{u}.wav"), g[cols].to_dict("records"))
             for u, g in syl.groupby("utt_id", sort=False)]
    print(f"{len(items)} utterances, {len(syl)} syllables, {a.jobs} processes, "
          f"pitch {a.floor:g}-{a.ceiling:g} Hz, interval={a.interval}")

    rows, errs = [], []
    with Pool(a.jobs, initializer=init, initargs=(a,)) as pool:
        for i, (res, err) in enumerate(pool.imap_unordered(process, items, chunksize=16), 1):
            rows += res
            if err:
                errs.append(err)
            if i % 5000 == 0 or i == len(items):
                print(f"  {i}/{len(items)} utterances")
    for e in errs[:10]:
        print("  ERROR", e)

    f0 = pd.DataFrame(rows).merge(syl, on=["utt_id", "syl_idx"], how="left")
    pcols = [f"f0_p{k}" for k in range(N_PORTIONS)]
    logs = np.log(f0[pcols])
    stats = (pd.DataFrame({"speaker": f0.speaker.repeat(N_PORTIONS).values, "v": logs.values.ravel()})
             .dropna().groupby("speaker").v.agg(["mean", "std"]))
    mu, sd = f0.speaker.map(stats["mean"]), f0.speaker.map(stats["std"])
    for c in ("onset", "offset"):
        f0[f"{c}_logz"] = (np.log(f0[f"{c}_hz"]) - mu) / sd
        f0[f"{c}_suspect"] = f0[f"{c}_logz"].abs() > 3
    f0["meas_dur"] = f0.meas_end - f0.meas_start
    f0["too_short"] = f0.meas_dur < a.min_dur
    f0 = f0.sort_values(["utt_id", "syl_idx"]).reset_index(drop=True)
    f0.to_csv(prep / "syllables_f0.csv", index=False)

    # adjacent pairs within an utterance
    keep = ["utt_id", "speaker", "syl_idx", "syllable", "tone", "surface_tone", "surface_certain",
            "sandhi_applied", "juncture_after", "pre_a_suffix", "neutral", "word_idx",
            "dur", "meas_dur", "too_short", "pause_after", "pause_before",
            "onset_hz", "onset_logz", "onset_portion", "onset_segment", "onset_suspect",
            "offset_hz", "offset_logz", "offset_portion", "offset_segment", "offset_suspect"]
    s = f0[keep]
    s1 = s.add_suffix("1").rename(columns={"utt_id1": "utt_id", "speaker1": "speaker"})
    s2 = s.add_suffix("2").rename(columns={"utt_id2": "utt_id", "speaker2": "speaker"})
    s2["syl_idx1"] = s2.syl_idx2 - 1
    b = s1.merge(s2, on=["utt_id", "speaker", "syl_idx1"])
    b["same_word"] = (b.word_idx1 == b.word_idx2).astype(int)
    wl = f0.groupby(["utt_id", "word_idx"]).size().rename("word_len").reset_index()
    b = b.merge(wl.rename(columns={"word_idx": "word_idx1"}), on=["utt_id", "word_idx1"], how="left")
    b["pause_between"] = b.pause_after1
    b.to_csv(prep / "bigrams_f0.csv", index=False)

    ok = b.offset_hz1.notna() & b.onset_hz2.notna()
    wi = b.same_word.eq(1)
    print(f"{len(f0)} syllables; onset found {f0.onset_hz.notna().mean():.1%}, "
          f"offset found {f0.offset_hz.notna().mean():.1%}; too short {f0.too_short.mean():.1%}")
    print(f"{len(b)} bigrams ({wi.sum()} within-word); both F0 values {ok.mean():.1%}")
    print("offset1 portion used:", b.offset_portion1.value_counts(normalize=True).round(3).sort_index().to_dict())
    print("onset2 portion used: ", b.onset_portion2.value_counts(normalize=True).round(3).sort_index().to_dict())
    print(f"suspect (|z|>3): offset1 {b.offset_suspect1.eq(True).mean():.1%}, "
          f"onset2 {b.onset_suspect2.eq(True).mean():.1%}")


if __name__ == "__main__":
    main()
