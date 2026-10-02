"""Build an MFA training/alignment corpus from TAT-MOE (one microphone channel).

For each utterance JSON with a matching wav:
  * tokenises the 台羅數字調 field into syllables, keeping
      - word membership (syllables joined by a single '-')
      - neutral-tone marking ('--' before a syllable)
      - punctuation after a syllable (phrase-boundary cue, useful later for sandhi)
  * writes <out>/corpus/<speaker>/<utt>.lab   (one "word" per syllable, e.g. "tshia1")
    and links (or copies) the wav next to it
  * utterances with non-Tâi-lô tokens (English words, digits, unparseable
    syllables) are skipped and logged

Outputs in --out:
  corpus/<speaker>/*.wav + *.lab
  dictionary.txt          MFA pronunciation dictionary (syllable -> toneless phones)
  syllables.csv           one row per syllable (joins to the TextGrid word tier by order)
  utterances.csv          one row per kept utterance (speaker metadata, text)
  skipped.csv             utterances left out and why

Example (macOS):
  C="<path/to/TAT-MOE>/corpus-full"
  python 01_prepare_mfa_corpus.py \
    --json-root "$C/TAT-MOE-train-json" "$C/TAT-MOE-eval-json" "$C/TAT-MOE-test-json" \
    --wav-root  "$C/TAT-MOE-train" "$C/TAT-MOE-eval" "$C/TAT-MOE-test" \
    --channel condenser --out ~/tsm_mfa/all_condenser
"""
import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

from tailo_g2p import G2PError, split_syllable, syllable_to_phones

CHUNK_RE = re.compile(r"[A-Za-z0-9\-]+|[^A-Za-z0-9\-\s]+")
PUNCT_RE = re.compile(r"[^A-Za-z0-9\-\s]")


def tokenize(tailo):
    """Return list of dicts (syllable, word_idx, neutral, punct_after) or raise G2PError."""
    syls = []
    word_idx = -1
    for chunk in CHUNK_RE.findall(tailo):
        if PUNCT_RE.match(chunk):
            if syls:
                syls[-1]["punct_after"] = 1
            continue
        # a chunk starting with '--' attaches to the previous word (e.g. 'li2 --ah4')
        if not chunk.startswith("--") or word_idx < 0:
            word_idx += 1
        parts = re.split(r"(--|-)", chunk)
        neutral_next = False
        for p in parts:
            if p == "--":
                neutral_next = True
                continue
            if p in ("-", ""):
                continue
            base, tone = split_syllable(p)  # raises on English words etc.
            syllable_to_phones(p)           # raises if not parseable
            syls.append(dict(syllable=f"{base}{tone}", base=base, tone=tone,
                             word_idx=word_idx, neutral=int(neutral_next),
                             punct_after=0))
            neutral_next = False
    if not syls:
        raise G2PError("empty")
    syls[-1]["punct_after"] = 1  # utterance end
    return syls


def find_channel_root(wav_root, channel):
    """TAT-MOE unzips to <root>/<hash>-master/<channel>; accept either level."""
    wav_root = Path(wav_root)
    if (wav_root / channel).is_dir():
        return wav_root / channel
    hits = [p for p in wav_root.glob(f"*/{channel}") if p.is_dir()]
    if not hits:
        sys.exit(f"no '{channel}' directory under {wav_root}")
    return hits[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-root", required=True, nargs="+",
                    help="one or more JSON roots (e.g. train, eval, test), paired with --wav-root")
    ap.add_argument("--wav-root", required=True, nargs="+")
    ap.add_argument("--channel", default="condenser")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-speakers", type=int, default=0, help="0 = all")
    ap.add_argument("--speakers", nargs="*", help="explicit speaker dirs, e.g. KH_KHF1001")
    ap.add_argument("--copy", action="store_true", help="copy wavs instead of symlinking")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    out = Path(os.path.expanduser(a.out))
    corpus = out / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    if len(a.json_root) != len(a.wav_root):
        sys.exit("--json-root and --wav-root must be given in matching pairs")
    by_spk, chan_of = {}, {}
    for jr, wr in zip(a.json_root, a.wav_root):
        chan = find_channel_root(wr, a.channel)
        print(f"channel dir: {chan}")
        for p in Path(jr).rglob("*.json"):
            if p.name.startswith("._"):
                continue
            spk = p.parent.name
            if (chan / spk).is_dir():
                by_spk.setdefault(spk, []).append(p)
                chan_of[spk] = chan
    spks = sorted(by_spk)
    if a.speakers:
        spks = [s for s in spks if s in set(a.speakers)]
    elif a.n_speakers:
        random.Random(a.seed).shuffle(spks)
        spks = sorted(spks[:a.n_speakers])
    print(f"speakers ({len(spks)}): {' '.join(spks)}")

    lexicon = {}
    syl_rows, utt_rows, skipped = [], [], []
    why = Counter()
    for spk in spks:
        (corpus / spk).mkdir(exist_ok=True)
        for jp in sorted(by_spk[spk]):
            stem = jp.stem
            wav = chan_of[spk] / spk / f"{stem}.wav"
            utt = f"{spk}_{stem}"
            if not wav.exists():
                skipped.append((utt, "no wav")); why["no wav"] += 1
                continue
            d = json.load(open(jp, encoding="utf-8"))
            tl = d.get("台羅數字調", "")
            try:
                syls = tokenize(tl)
            except G2PError as e:
                skipped.append((utt, f"g2p: {e}")); why["non-Tâi-lô token"] += 1
                continue
            dst = corpus / spk / f"{utt}.wav"
            if not dst.exists():
                if a.copy:
                    shutil.copy2(wav, dst)
                else:
                    os.symlink(wav.resolve(), dst)
            (corpus / spk / f"{utt}.lab").write_text(
                " ".join(s["syllable"] for s in syls) + "\n", encoding="utf-8")
            for i, s in enumerate(syls):
                lexicon.setdefault(s["syllable"], " ".join(syllable_to_phones(s["syllable"])))
                syl_rows.append(dict(utt_id=utt, speaker=spk, syl_idx=i, n_syl=len(syls), **s))
            utt_rows.append(dict(
                utt_id=utt, speaker=spk, prompt=d.get("提示卡編號"), sent=d.get("句編號"),
                sex=d.get("性別"), age=d.get("年齡"), accent=d.get("腔調"),
                birthplace=d.get("出生地"), duration=d.get("音檔長度"),
                hanlo=d.get("漢羅台文"), tailo=tl, n_syl=len(syls)))

    with open(out / "dictionary.txt", "w", encoding="utf-8") as f:
        for w in sorted(lexicon):
            f.write(f"{w}\t{lexicon[w]}\n")
    for name, rows in (("syllables.csv", syl_rows), ("utterances.csv", utt_rows)):
        with open(out / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    with open(out / "skipped.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["utt_id", "reason"]); w.writerows(skipped)

    phones = Counter(p for v in lexicon.values() for p in v.split())
    print(f"kept {len(utt_rows)} utterances, {len(syl_rows)} syllables, "
          f"{len(lexicon)} dictionary entries, {len(phones)} phones")
    print("skipped:", dict(why))


if __name__ == "__main__":
    main()
