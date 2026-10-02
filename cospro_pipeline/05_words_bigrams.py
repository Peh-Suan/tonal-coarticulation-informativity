"""
Step 3: word segmentation -> within-word tone bigrams + informativity.
Run in macOS Terminal:
    python3 -m pip install ckip-transformers
    python3 05_words_bigrams.py            # uses out/syllables_mlm.csv

Segmentation: CKIP Transformers word segmenter (Traditional Chinese, trained
on the Sinica Corpus). Commas are inserted at B4/B5 prosodic breaks to help
it, then dropped. Each character is one syllable, so words map back to
syllables exactly.

Outputs (out/):
  words.csv       one row per word token: file, word_idx, word, n_syll,
                  syl_from, syl_to, context_before (preceding text of the
                  recording), quality flags
  bigrams.csv     one row per adjacent syllable pair INSIDE a word
                  (a 3-syllable word gives 2 pairs), with:
                  syllable1/2 (pinyin), char1/2, tone1/2, pair "t1-t2",
                  t3_sandhi (lexical 3-3), times, word info, and
                  informativity from tc_mi/informativity_TM.csv:
                    inf_carryover     = carryover row for pair (-> syllable 2)
                    inf_anticipatory  = anticipatory row for pair (-> syllable 1)
                  and the same from informativity_sandhi_TM.csv looked up with
                  the SURFACE pair (3-3 -> 2-3): inf_*_sandhi.
                  Neutral-tone pairs have no informativity (NaN).
"""
import argparse, re
import pandas as pd

def segment(texts, dry_run=False, ws_model="bert-base"):
    if dry_run:                      # plumbing test only: 2-character "words"
        return [[t[i:i+2] for i in range(0, len(t), 2)] for t in texts]
    from ckip_transformers.nlp import CkipWordSegmenter
    import torch
    dev = 0 if torch.cuda.is_available() else -1
    ws = CkipWordSegmenter(model=ws_model, device=dev)
    # CKIP's limit is strictly below 510; use_delim splits long recordings at the
    # commas inserted at B4/B5 breaks (the commas come back as tokens and are dropped)
    return ws(texts, use_delim=True, batch_size=32, max_length=500)

def main():
    import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _freshness import require_fresh; require_fresh("out", [('syllables_mlm.csv', 'syllables_chars.csv')])
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out"); ap.add_argument("--inp", default="syllables_mlm.csv")
    ap.add_argument("--inf-dir", default="../tables", help="folder with informativity_TM.csv")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ws-model", default="bert-base",
                    help="CKIP model: bert-base (default), albert-base, albert-tiny; use the one "
                         "that segmented the informativity lexicon (tc_mi segmented_texts)")
    a = ap.parse_args()

    syl = pd.read_csv(f"{a.out}/{a.inp}", keep_default_na=False)
    files, texts, layouts = [], [], []
    for f, g in syl.groupby("file", sort=False):
        chars, brk, pw = list(g.char_final), list(g.break_after), list(g.pw_idx)
        t = ""
        for k, c in enumerate(chars):
            t += c if c else "□"
            end_pw = k == len(chars) - 1 or pw[k + 1] != pw[k]
            if end_pw and brk[k] in ("B4", "B5") and k < len(chars) - 1: t += "，"
        files.append(f); texts.append(t)
    print(f"segmenting {len(texts)} recordings")
    segs = segment(texts, a.dry_run, a.ws_model)

    word_rows, assign = [], []
    n_syl = syl.groupby("file").size().to_dict()
    for f, words in zip(files, segs):
        joined = "".join(w.replace("，", "").strip() for w in words)
        if len(joined) != n_syl[f]:
            print(f"  WARNING {f}: segmenter returned {len(joined)} chars for {n_syl[f]} syllables; skipped")
            continue
        k, wi, so_far = 0, 0, ""
        for w in words:
            w = w.replace("，", "").strip()
            if not w: continue
            n = len(w)
            for j in range(n):
                assign.append((f, k + j, wi, w, j))
            word_rows.append(dict(file=f, word_idx=wi, word=w, n_syll=n, syl_from=k, syl_to=k + n - 1,
                                  context_before=so_far))
            so_far += w; k += n; wi += 1
    syl = syl.merge(pd.DataFrame(assign, columns=["file", "syl_idx", "word_idx", "word", "pos_in_word"]),
                    on=["file", "syl_idx"], how="left")
    syl["word_idx"] = syl.word_idx.fillna(-1).astype(int); syl["word"] = syl.word.fillna("")
    words = pd.DataFrame(word_rows)
    q = syl.groupby(["file", "word_idx"]).agg(
        n_filled=("source", lambda s: int((s == "filled").sum())),
        n_replaced=("source", lambda s: int((s == "replaced").sum())),
        n_unfilled=("source", lambda s: int((s == "unfilled").sum())),
        needs_retranscription=("needs_retranscription", "first")).reset_index()
    words = words.merge(q, on=["file", "word_idx"], how="left")
    words.to_csv(f"{a.out}/words.csv", index=False)

    # within-word bigrams
    s1 = syl.rename(columns=lambda c: c + "1")
    s2 = syl.rename(columns=lambda c: c + "2")
    s1["key"] = s1.syl_idx1 + 1
    b = s1.merge(s2, left_on=["file1", "key"], right_on=["file2", "syl_idx2"])
    b = b[(b.word_idx1 == b.word_idx2) & (b.word_idx1 >= 0)]
    b = pd.DataFrame(dict(
        file=b.file1, subcorpus=b.subcorpus1, speaker=b.speaker1, syl_idx1=b.syl_idx1,
        word_idx=b.word_idx1, word=b.word1,
        word_len=b.word1.str.len(), pos_in_word=b.pos_in_word1,
        syllable1=b.pinyin1, syllable2=b.pinyin2, char1=b.char_final1, char2=b.char_final2,
        tone1=b.tone1.astype(int), tone2=b.tone2.astype(int),
        start1=b.start1, end1=b.end1, v_start1=b.v_start1, start2=b.start2, end2=b.end2, v_start2=b.v_start2,
        break_within=b.break_after1.where(b.pw_idx1 != b.pw_idx2, ""),
        source1=b.source1, source2=b.source2))
    b["pair"] = b.tone1.astype(str) + "-" + b.tone2.astype(str)
    b["t3_sandhi"] = ((b.tone1 == 3) & (b.tone2 == 3)).astype(int)
    b["surface_pair"] = b.pair.where(b.t3_sandhi == 0, "2-3")

    for fname, suffix, key in [("informativity_TM.csv", "", "pair"),
                               ("informativity_sandhi_TM.csv", "_sandhi", "surface_pair")]:
        inf = pd.read_csv(f"{a.inf_dir}/{fname}")
        for d in ("carryover", "anticipatory"):
            m = inf[inf.direction == d].set_index("pair").informativity
            b[f"inf_{d}{suffix}"] = b[key].map(m)
    b.to_csv(f"{a.out}/bigrams.csv", index=False)
    print(f"{len(words)} words, {len(b)} within-word bigrams")
    print(pd.crosstab(b.tone1, b.tone2))

if __name__ == "__main__":
    main()
