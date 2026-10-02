"""Join MFA TextGrids back to syllables.csv -> syllable_alignments.csv.

The TextGrid 'words' tier has one interval per syllable (plus empty/silence
intervals). Non-empty intervals are matched, in order, to the rows of
syllables.csv for that utterance; utterances whose labels don't line up are
reported and dropped.

Adds: start, end, dur; sil_before/sil_after (raw MFA silence around the
syllable); pause_before/pause_after (the same, but 0 when shorter than
--min-pause: in the pilot most short "silences" were stop closures, e.g. 70-80%
of p/t/k/ph/th/kh-initial syllables had one, even word-internally); n_phones, vowel_start/vowel_end (first/last vowel phone, useful for
F0 onset/offset windows that skip the onset consonant).

Optionally copies N random wav+TextGrid pairs to <out>/handcheck for Praat.

  python 03_collect_alignments.py --prep ~/tsm_mfa/pilot \
      --textgrids ~/tsm_mfa/pilot/aligned --handcheck 50
"""
import argparse
import csv
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

VOWEL_RE = re.compile(r"^(a|i|u|e|o|oo|ir|er|ee)N?$|^(m|ng)=$")
SKIP = {"", "sil", "sp", "spn", "<eps>"}


def read_textgrid(path):
    """Minimal long-format TextGrid reader -> {tier_name: [(xmin, xmax, text)]}."""
    tiers, cur, iv = {}, None, {}
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if s.startswith("name ="):
            cur = s.split("=", 1)[1].strip().strip('"')
            tiers[cur] = []
        elif cur is None:
            continue
        elif s.startswith("xmin ="):
            iv = {"xmin": float(s.split("=")[1])}
        elif s.startswith("xmax =") and "xmin" in iv:
            iv["xmax"] = float(s.split("=")[1])
        elif s.startswith("text ="):
            txt = s.split("=", 1)[1].strip()
            if txt.startswith('"') and txt.endswith('"'):
                txt = txt[1:-1]
            tiers[cur].append((iv["xmin"], iv["xmax"], txt))
            iv = {}
    return tiers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True, help="output dir of 01_prepare_mfa_corpus.py")
    ap.add_argument("--textgrids", required=True, help="MFA output directory")
    ap.add_argument("--handcheck", type=int, default=0)
    ap.add_argument("--min-pause", type=float, default=0.12,
                    help="silences shorter than this (s) are treated as closures, not pauses")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    prep = Path(os.path.expanduser(a.prep))
    tgdir = Path(os.path.expanduser(a.textgrids))

    syl = defaultdict(list)
    with open(prep / "syllables.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            syl[r["utt_id"]].append(r)
    tgs = {p.stem: p for p in tgdir.rglob("*.TextGrid")}

    rows, bad, missing = [], [], 0
    for utt, srows in syl.items():
        if utt not in tgs:
            missing += 1
            continue
        t = read_textgrid(tgs[utt])
        words = t.get("words") or next(v for k, v in t.items() if "word" in k.lower())
        phones = t.get("phones") or next(v for k, v in t.items() if "phone" in k.lower())
        idx = [i for i, w in enumerate(words) if w[2] not in SKIP]
        if [words[i][2] for i in idx] != [r["syllable"] for r in srows]:
            bad.append(utt)
            continue
        for r, i in zip(srows, idx):
            xmin, xmax, _ = words[i]
            prev_sil = words[i - 1] if i > 0 and words[i - 1][2] in SKIP else None
            next_sil = words[i + 1] if i + 1 < len(words) and words[i + 1][2] in SKIP else None
            ph = [p for p in phones if p[0] >= xmin - 1e-6 and p[1] <= xmax + 1e-6
                  and p[2] not in SKIP]
            vw = [p for p in ph if VOWEL_RE.match(p[2])]
            gb = round(prev_sil[1] - prev_sil[0], 4) if prev_sil else 0.0
            ga = round(next_sil[1] - next_sil[0], 4) if next_sil else 0.0
            rows.append(dict(
                r, start=round(xmin, 4), end=round(xmax, 4), dur=round(xmax - xmin, 4),
                sil_before=gb, sil_after=ga,
                # short MFA "silences" are mostly stop/affricate closures or glottal onsets
                pause_before=gb if gb >= a.min_pause else 0.0,
                pause_after=ga if ga >= a.min_pause else 0.0,
                n_phones=len(ph), phones=" ".join(p[2] for p in ph),
                vowel_start=round(vw[0][0], 4) if vw else "",
                vowel_end=round(vw[-1][1], 4) if vw else ""))

    out = prep / "syllable_alignments.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} syllables from {len(syl) - missing - len(bad)} utterances -> {out}")
    print(f"no TextGrid (not aligned by MFA): {missing};  label mismatch: {len(bad)}")

    if a.handcheck:
        hc = prep / "handcheck"; hc.mkdir(exist_ok=True)
        done = sorted({r["utt_id"] for r in rows})
        pick = random.Random(a.seed).sample(done, min(a.handcheck, len(done)))
        for utt in pick:
            spk = next(iter(syl[utt]))["speaker"]
            wav = prep / "corpus" / spk / f"{utt}.wav"
            shutil.copy2(os.path.realpath(wav), hc / f"{utt}.wav")
            shutil.copy2(tgs[utt], hc / f"{utt}.TextGrid")
        print(f"copied {len(pick)} wav+TextGrid pairs to {hc}")


if __name__ == "__main__":
    main()
