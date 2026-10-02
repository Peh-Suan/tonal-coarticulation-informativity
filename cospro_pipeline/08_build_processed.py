"""
Step 6: build processed_data.csv for analysis (one row per bigram x position).

Run in macOS Terminal, after 07_f0_boundary.py:
    python 08_build_processed.py \
        --segmented "<path/to/segmented_texts>"
The word-frequency table is cached in out/word_freq.csv, so --segmented is
only needed the first time (or pass --rebuild-freq).

Long format, as in the manuscript's model: every within-word bigram gives two rows
  position = carryover     target = ONSET of syllable 2,  context = OFFSET of syllable 1,
                           informativity = carryover value for the pair
  position = anticipatory  target = OFFSET of syllable 1, context = ONSET of syllable 2,
                           informativity = anticipatory value for the pair
Main columns
  target_f0 / context_f0            ln-Hz z-scored within speaker (logz); *_hz raw
  target_portion / context_portion  portion used, 0 = at the boundary (paper covariate)
  onset_segment / offset_segment    segment type at the boundary (random effects)
  target_tone / context_tone        SURFACE tone (T3 before T3 -> 2) as five-level
                                    labels 55 / 35 / 21 / 51 (neutral = 0); *_lex = lexical
  informativity, informativity_sandhi   p(T | adjacent syllable, target position class) over all
                                    within-word pairs (informativity_positional.py);
                                    target_pos_class = initial / medial / final; *_allpairs =
                                    without position; *_tai = the TAI tables (disyllables only);
                                    informativity_full_context(_sandhi): context also includes the
                                    target's own segments; context_<measure> (every measure):
                                    the context tone's informativity (same pair, other
                                    direction, at context_pos_class)
  word_surprisal (bits, preceding context in the recording), target_syl_surprisal /
  context_syl_surprisal (the syllable's own surprisal),
  unigram_surprisal (bits, -log2 p(word); from the same GPT-2 as word_surprisal via
  06b_unigram_lm.py when out/unigram_lm.csv exists, otherwise from segmented_texts,
  add-one smoothed; see unigram_source; the corpus version is always kept as
  unigram_surprisal_corpus / context_gain_corpus) and
  context_gain = unigram_surprisal - word_surprisal (bits the preceding context adds
  to the word's predictability, i.e. PMI; word_surprisal = unigram - gain),
  word_oov (word absent from segmented_texts), log_word_freq (log10 per
  million, +1 smoothing, from the segmented_texts corpus used for informativity)
  word_idx_in_rec, word_repetition (earlier occurrences in the recording)
  phrase_time / phrase_prop / phrase_rate: position (s, proportion) and speech rate
                                    (syll/s) in the prosodic phrase (ends at B3+);
  bg_time / bg_prop                 position in the breath group (ends at B4+)
  break_after1 / break_after2       break after each syllable (B1 = inside a prosodic word,
                                    end = last syllable of the recording)
  dur1, dur2 (syllable), fdur1, fdur2 (final)
  speaker, sex, subcorpus, item_id (same text across speakers), token_id
  (item + word + occurrence: the same word token read by different speakers), word
  Quality: needs_retranscription, char_source1/2 (asr / replaced / filled),
  has_box, is_numeral (read-out numbers), target/context_suspect (|logz| > 3),
  keep (recommended subset)
"""
import argparse, glob, math, os, pickle, re
import numpy as np
import pandas as pd

FIVE = {1: "55", 2: "35", 3: "21", 4: "51", 5: "0"}
HAN_ONLY = re.compile(r"^[一-龥]+$")      # same filter as tc_mi/ptt.ipynb

def word_freq(seg_dir, cache, rebuild=False):
    if os.path.exists(cache) and not rebuild:
        return pd.read_csv(cache, keep_default_na=False)
    from collections import Counter
    if not seg_dir: raise SystemExit("word_freq.csv not cached yet: pass --segmented <folder>")
    cnt = Counter()
    fs = sorted(glob.glob(os.path.join(seg_dir, "*")))
    print(f"counting words in {len(fs)} segmented_texts files (first run only)...")
    for i, f in enumerate(fs, 1):
        try: texts = pickle.load(open(f, "rb"))
        except Exception as e:
            print("  skip", f, e); continue
        for text in texts:
            cnt.update(w for w in text if w != "^^" and HAN_ONLY.match(w))
        if i % 20 == 0: print(f"  {i}/{len(fs)} files, {sum(cnt.values()):,} tokens")
    df = pd.DataFrame(cnt.items(), columns=["word", "count"]).sort_values("count", ascending=False)
    df.to_csv(cache, index=False)
    return df

def phrase_features(syl):
    """COSPRO break hierarchy (Tseng): B2 prosodic word, B3 prosodic phrase,
    B4 breath group, B5 prosodic group. phrase_* = prosodic phrase (ends at
    B3 or higher); bg_* = breath group (ends at B4 or higher)."""
    syl = syl.sort_values(["file", "syl_idx"]).copy()
    nxt_pw = syl.groupby("file").pw_idx.shift(-1)
    last_in_pw = nxt_pw.isna() | (nxt_pw != syl.pw_idx)
    last_in_rec = nxt_pw.isna()
    lab = syl.break_after.where(syl.break_after != "", "B?")
    syl["break_after"] = np.where(last_in_rec, "end", np.where(last_in_pw, lab, "B1"))
    for name, levels in (("phrase", ["B3", "B4", "B5", "B6", "end"]), ("bg", ["B4", "B5", "B6", "end"])):
        ends = syl.break_after.isin(levels)
        syl[f"{name}_idx"] = ends.groupby(syl.file).cumsum() - ends.astype(int)
        g = syl.groupby(["file", f"{name}_idx"])
        syl[f"{name}_start"] = g.start.transform("min"); syl[f"{name}_end"] = g.end.transform("max")
        syl[f"{name}_n"] = g.start.transform("size")
    return syl

def main():
    import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _freshness import require_fresh; require_fresh("out", [('syllables_mlm.csv', 'syllables_chars.csv'), ('bigrams.csv', 'syllables_mlm.csv'), ('bigrams_f0.csv', 'bigrams.csv'), ('char_surprisal.csv', 'syllables_mlm.csv')])
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out"); ap.add_argument("--inp", default="bigrams_f0.csv")
    ap.add_argument("--segmented", default=None); ap.add_argument("--rebuild-freq", action="store_true")
    ap.add_argument("--dest", default="processed_data.csv")
    ap.add_argument("--inf-dir", default="../tables", help="folder with informativity_pos_TM.csv (informativity_positional.py)")
    a = ap.parse_args()
    rd = lambda n: pd.read_csv(f"{a.out}/{n}", keep_default_na=False, na_values=[""], low_memory=False)

    if not os.path.exists(f"{a.out}/{a.inp}"):
        raise SystemExit(f"{a.out}/{a.inp} not found: run 07_f0_boundary.py first")
    b = rd(a.inp); words = rd("words.csv")

    # --- positional informativity (tsm_pipeline/informativity_positional.py) ----------
    # p(T | adjacent syllable, position of the TARGET in the word) over all within-word pairs of
    # the segmented_texts lexicon (the TAI tables used by 05 counted disyllabic words only).
    # These replace inf_* ; pos_class "all" -> *_allpairs; the TAI values are kept as *_tai.
    # Neutral-tone pairs stay without informativity, as before.
    cls = lambda i, n: np.where(i == 0, "initial", np.where(i == n - 1, "final", "medial"))
    b["tpc_carryover"] = cls(b.pos_in_word + 1, b.word_len)       # target = syllable 2
    b["tpc_anticipatory"] = cls(b.pos_in_word, b.word_len)        # target = syllable 1
    neutral = b.pair.astype(str).str.contains("5") | b.inf_carryover.isna()
    for fname, suffix, key in [("informativity_pos_TM.csv", "", "pair"),
                               ("informativity_pos_sandhi_TM.csv", "_sandhi", "surface_pair")]:
        t = pd.read_csv(f"{a.inf_dir}/{fname}", dtype={"pair": str})
        for d in ("carryover", "anticipatory"):
            m = t[t.direction == d].set_index(["pair", "pos_class"]).informativity
            b[f"inf_{d}{suffix}_tai"] = b[f"inf_{d}{suffix}"]
            for pooled, suf in ((False, ""), (True, "_allpairs")):
                pc = pd.Series("all", index=b.index) if pooled else b[f"tpc_{d}"]
                v = m.reindex(pd.MultiIndex.from_arrays([b[key].astype(str), pc])).values
                b[f"inf_{d}{suffix}{suf}"] = pd.Series(v, index=b.index).mask(neutral)
    # full-context informativity (context also includes the target's own segments)
    for fname, suffix, key in [("informativity_full_context_TM.csv", "_fullctx", "pair"),
                               ("informativity_full_context_sandhi_TM.csv", "_sandhi_fullctx", "surface_pair")]:
        t = pd.read_csv(f"{a.inf_dir}/{fname}", dtype={"pair": str})
        for d in ("carryover", "anticipatory"):
            m = t[t.direction == d].set_index(["pair", "pos_class"]).informativity
            v = m.reindex(pd.MultiIndex.from_arrays([b[key].astype(str), b[f"tpc_{d}"]])).values
            b[f"inf_{d}{suffix}"] = pd.Series(v, index=b.index).mask(neutral)
    # functional load of the target tone (informativity_positional.py --fl-only): functional_load_pos_* /
    # functional_load_full_context_* mirror the informativity tables; functional_load = entropy lost
    # when the target tone merges with each competitor (summed), functional_load_max = strongest one
    FL_METRICS = {"functional_load": "fl", "functional_load_max": "flmax"}
    for metric, pre in FL_METRICS.items():
        for fname, suffix, key in [("functional_load_pos_TM.csv", "", "pair"),
                                   ("functional_load_pos_sandhi_TM.csv", "_sandhi", "surface_pair")]:
            t = pd.read_csv(f"{a.inf_dir}/{fname}", dtype={"pair": str})
            for d in ("carryover", "anticipatory"):
                m = t[t.direction == d].set_index(["pair", "pos_class"])[metric]
                for pooled, suf in ((False, ""), (True, "_allpairs")):
                    pc = pd.Series("all", index=b.index) if pooled else b[f"tpc_{d}"]
                    v = m.reindex(pd.MultiIndex.from_arrays([b[key].astype(str), pc])).values
                    b[f"{pre}_{d}{suffix}{suf}"] = pd.Series(v, index=b.index).mask(neutral)
        for fname, suffix, key in [("functional_load_full_context_TM.csv", "_fullctx", "pair"),
                                   ("functional_load_full_context_sandhi_TM.csv", "_sandhi_fullctx", "surface_pair")]:
            t = pd.read_csv(f"{a.inf_dir}/{fname}", dtype={"pair": str})
            for d in ("carryover", "anticipatory"):
                m = t[t.direction == d].set_index(["pair", "pos_class"])[metric]
                v = m.reindex(pd.MultiIndex.from_arrays([b[key].astype(str), b[f"tpc_{d}"]])).values
                b[f"{pre}_{d}{suffix}"] = pd.Series(v, index=b.index).mask(neutral)
    syl = rd("syllables.csv"); syl["break_after"] = syl.break_after.fillna("")
    syl = syl[syl.file.isin(set(b.file))]
    if "syl_idx1" not in b:
        b = b.merge(syl[["file", "syl_idx", "start"]].rename(columns={"syl_idx": "syl_idx1", "start": "start1"}),
                    on=["file", "start1"], how="left")
    b["syl_idx2"] = b.syl_idx1 + 1

    # --- prosodic position and speech rate -------------------------------------
    ph = phrase_features(syl)
    p = ph[["file", "syl_idx", "break_after", "phrase_idx", "phrase_start", "phrase_end", "phrase_n",
            "bg_idx", "bg_start", "bg_end", "v_start", "end", "start", "initial"]]
    b = b.merge(p[["file", "syl_idx", "break_after"]].rename(columns={"syl_idx": "syl_idx1", "break_after": "break_after1"}),
                on=["file", "syl_idx1"], how="left")
    b = b.merge(p.rename(columns={"syl_idx": "syl_idx2", "break_after": "break_after2", "v_start": "v_start2_",
                                  "end": "end2_", "start": "start2_", "initial": "initial2"}),
                on=["file", "syl_idx2"], how="left")
    b["phrase_time"] = b.start2 - b.phrase_start            # boundary time from phrase start (s)
    b["phrase_prop"] = b.phrase_time / (b.phrase_end - b.phrase_start)
    b["phrase_rate"] = b.phrase_n / (b.phrase_end - b.phrase_start)   # syllables / s
    b["bg_time"] = b.start2 - b.bg_start                     # boundary time from breath-group start
    b["bg_prop"] = b.bg_time / (b.bg_end - b.bg_start)
    b["dur1"], b["dur2"] = b.end1 - b.start1, b.end2 - b.start2
    b["fdur1"], b["fdur2"] = b.end1 - b.v_start1, b.end2 - b.v_start2

    # --- word-level: index, repetition, frequency, retranscription flag --------
    words = words.sort_values(["file", "word_idx"])
    words["word_repetition"] = words.groupby(["file", "word"]).cumcount()
    words["n_words_in_rec"] = words.groupby("file").word_idx.transform("size")
    fq = word_freq(a.segmented, f"{a.out}/word_freq.csv", a.rebuild_freq)
    N = fq["count"].sum()
    words["word_count"] = words.word.map(fq.set_index("word")["count"]).fillna(0).astype(int)
    words["log_word_freq"] = np.log10((words.word_count + 1) / N * 1e6)
    # unigram surprisal (bits) = -log2 p(word), add-one smoothed over the vocabulary;
    # a linear transform of log frequency, so use one or the other, never both
    V = len(fq)
    words["unigram_surprisal_corpus"] = -np.log2((words.word_count + 1) / (N + V))
    words["word_oov"] = words.word_count == 0
    # Preferred: unigram surprisal from the SAME language model as word_surprisal
    # (06b_unigram_lm.py: marginal p(w) over sampled contexts). Falls back to the
    # frequency-corpus estimate when out/unigram_lm.csv does not exist.
    lm_path = f"{a.out}/unigram_lm.csv"
    if os.path.exists(lm_path):
        lm = pd.read_csv(lm_path, keep_default_na=False).set_index("word")
        for c in ("unigram_lm_surprisal", "unigram_lm_se", "null_context_surprisal"):
            words[c] = words.word.map(lm[c])
        words["unigram_surprisal"] = words.unigram_lm_surprisal
        words["unigram_source"] = "lm"
        print(f"unigram surprisal: language model ({lm_path}); "
              f"{words.unigram_lm_surprisal.isna().mean():.1%} of word tokens missing")
    else:
        for c in ("unigram_lm_surprisal", "unigram_lm_se", "null_context_surprisal"): words[c] = np.nan
        words["unigram_surprisal"] = words.unigram_surprisal_corpus
        words["unigram_source"] = "corpus"
        print("unigram surprisal: frequency corpus (run 06b_unigram_lm.py for the LM version)")
    wcols = ["file", "word_idx", "word_repetition", "n_words_in_rec", "word_count", "log_word_freq",
             "unigram_surprisal", "unigram_source", "unigram_surprisal_corpus", "unigram_lm_se",
             "null_context_surprisal", "word_oov",
             "needs_retranscription"]
    b = b.drop(columns=[c for c in wcols[2:] if c in b]).merge(words[wcols], on=["file", "word_idx"], how="left")

    # contextual gain (bits) = unigram surprisal - contextual surprisal = log2 p(w|ctx)/p(w)
    # (pointwise mutual information of the word with its preceding context):
    # > 0 the context made the word MORE predictable than its frequency alone; < 0 less.
    # word_surprisal = unigram_surprisal - context_gain.
    b["context_gain"] = b.unigram_surprisal - b.word_surprisal
    b["context_gain_corpus"] = b.unigram_surprisal_corpus - b.word_surprisal

    # --- syllable surprisal ------------------------------------------------------
    cs_path = f"{a.out}/char_surprisal.csv"
    if os.path.exists(cs_path):
        cs = rd("char_surprisal.csv")[["file", "syl_idx", "char_surprisal"]]
        b = b.merge(cs.rename(columns={"syl_idx": "syl_idx1", "char_surprisal": "syl_surprisal1"}), on=["file", "syl_idx1"], how="left")
        b = b.merge(cs.rename(columns={"syl_idx": "syl_idx2", "char_surprisal": "syl_surprisal2"}), on=["file", "syl_idx2"], how="left")
    else:
        print("note: out/char_surprisal.csv missing (rerun 06_surprisal.py); syllable surprisal left empty")
        b["syl_surprisal1"] = b["syl_surprisal2"] = np.nan

    # --- ids -------------------------------------------------------------------
    base = b.file.str.rsplit("/", n=1).str[-1].str.replace(".TextGrid", "", regex=False)
    b["subcorpus"] = b.file.str.split("/").str[0]
    b["item_id"] = [bs.replace(sp, "") for bs, sp in zip(base, b.speaker)]
    # same word token across speakers: item + word + its occurrence number in the text
    # (robust to speakers whose syllable counts differ slightly)
    b["token_id"] = b.item_id + "#" + b.word + "#" + b.word_repetition.astype("Int64").astype(str)
    b["sex"] = b.speaker.str[0].map({"F": "F", "M": "M"})
    # COSPRO reuses speaker IDs across subcorpora for different people (F001 of corspro2
    # and of corspro3 differ in pitch level), so the analysis speaker is subcorpus-specific
    b["speaker_raw"] = b.speaker
    b["speaker"] = b.subcorpus + "_" + b.speaker_raw
    b["bigram_id"] = b.file + "#" + b.syl_idx1.astype("Int64").astype(str)
    b["surface_tone1"] = np.where(b.t3_sandhi == 1, 2, b.tone1)
    b["surface_tone2"] = b.tone2
    b["has_box"] = (b.char1.astype(str) == "□") | (b.char2.astype(str) == "□")
    # read-out numbers (百分之四點一三, 七億九千餘萬): CKIP makes them one "word",
    # but they are not lexical words and are absent from the frequency corpus
    NUM = set("零〇一二三四五六七八九十百千萬億兩點之分幾餘多")   # 第一 etc. stay lexical
    b["is_numeral"] = b.word.astype(str).map(lambda w: len(w) > 1 and all(c in NUM for c in w) and any(c in "零〇一二三四五六七八九十兩" for c in w))

    # --- long format -----------------------------------------------------------
    common = ["bigram_id", "file", "subcorpus", "speaker", "speaker_raw", "sex", "item_id", "token_id",
              "word", "word_len", "pos_in_word", "word_idx", "word_idx_in_rec", "n_words_in_rec",
              "word_repetition", "word_count", "log_word_freq", "word_oov", "word_surprisal",
              "unigram_surprisal", "context_gain", "unigram_source", "unigram_lm_se",
              "null_context_surprisal", "unigram_surprisal_corpus", "context_gain_corpus",
              "syllable1", "syllable2", "char1", "char2", "tone1", "tone2", "pair", "surface_pair",
              "t3_sandhi", "offset_segment1", "onset_segment2",
              "dur1", "dur2", "fdur1", "fdur2", "break_after1", "break_after2",
              "phrase_idx", "phrase_time", "phrase_prop", "phrase_rate", "bg_idx", "bg_time", "bg_prop",
              "source1", "source2", "has_box", "is_numeral", "needs_retranscription"]
    b["word_idx_in_rec"] = b.word_idx
    # context_*: the context tone's informativity = the same pair read in the
    # other direction (context syllable as target, predicted from the target syllable, at its own position)
    INF_OUT = {"informativity": "", "informativity_sandhi": "_sandhi",
               "informativity_allpairs": "_allpairs", "informativity_sandhi_allpairs": "_sandhi_allpairs",
               "informativity_tai": "_tai", "informativity_sandhi_tai": "_sandhi_tai",
               "informativity_full_context": "_fullctx", "informativity_full_context_sandhi": "_sandhi_fullctx"}
    rows = []
    for pos in ("carryover", "anticipatory"):
        d = b[[c for c in common if c in b]].copy()
        d.insert(1, "position", pos)
        if pos == "carryover":   # target: syllable 2 onset; context: syllable 1 offset
            T, C = ("onset", "2"), ("offset", "1")
            d["target_tone_lex"], d["context_tone_lex"] = b.tone2, b.tone1
            d["target_tone"], d["context_tone"] = b.surface_tone2.map(FIVE), b.surface_tone1.map(FIVE)
            d["target_pos_class"], d["context_pos_class"] = b.tpc_carryover, b.tpc_anticipatory
            for name, suf in INF_OUT.items():          # target informativity and context_* (context tone)
                d[name] = b[f"inf_carryover{suf}"]
                d[f"context_{name}"] = b[f"inf_anticipatory{suf}"]
            for name, suf in INF_OUT.items():          # functional load, same names (no TAI version)
                if suf.endswith("_tai"):
                    continue
                for metric, pre in FL_METRICS.items():
                    nm = name.replace("informativity", metric)
                    d[nm] = b[f"{pre}_carryover{suf}"]
                    d[f"context_{nm}"] = b[f"{pre}_anticipatory{suf}"]
            d["target_syl_surprisal"], d["context_syl_surprisal"] = b.syl_surprisal2, b.syl_surprisal1
        else:                    # target: syllable 1 offset; context: syllable 2 onset
            T, C = ("offset", "1"), ("onset", "2")
            d["target_tone_lex"], d["context_tone_lex"] = b.tone1, b.tone2
            d["target_tone"], d["context_tone"] = b.surface_tone1.map(FIVE), b.surface_tone2.map(FIVE)
            d["target_pos_class"], d["context_pos_class"] = b.tpc_anticipatory, b.tpc_carryover
            for name, suf in INF_OUT.items():          # target informativity and context_* (context tone)
                d[name] = b[f"inf_anticipatory{suf}"]
                d[f"context_{name}"] = b[f"inf_carryover{suf}"]
            for name, suf in INF_OUT.items():          # functional load, same names (no TAI version)
                if suf.endswith("_tai"):
                    continue
                for metric, pre in FL_METRICS.items():
                    nm = name.replace("informativity", metric)
                    d[nm] = b[f"{pre}_anticipatory{suf}"]
                    d[f"context_{nm}"] = b[f"{pre}_carryover{suf}"]
            d["target_syl_surprisal"], d["context_syl_surprisal"] = b.syl_surprisal1, b.syl_surprisal2
        for role, (kind, n) in (("target", T), ("context", C)):
            d[f"{role}_f0"] = b[f"{kind}_logz{n}"]
            d[f"{role}_hz"] = b[f"{kind}_hz{n}"]
            d[f"{role}_portion"] = b[f"{kind}_portion{n}"]
            d[f"{role}_suspect"] = b[f"{kind}_suspect{n}"].astype(str).eq("True")
        d = d.rename(columns={"offset_segment1": "offset_segment", "onset_segment2": "onset_segment"})
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)

    out["keep"] = (out.needs_retranscription.astype(str) != "True") & out.informativity.notna() \
        & out.target_f0.notna() & out.context_f0.notna() & ~out.target_suspect & ~out.context_suspect \
        & ~out.has_box & ~out.is_numeral & out.word_surprisal.notna()
    out.to_csv(f"{a.out}/{a.dest}", index=False)

    k = out[out.keep]
    print(f"\n{a.out}/{a.dest}: {len(out):,} rows ({len(b):,} bigrams x 2 positions); keep = {len(k):,}")
    print("dropped by: neutral tone", int(out.informativity.isna().sum()),
          "| missing F0", int((out.target_f0.isna() | out.context_f0.isna()).sum()),
          "| suspect F0", int((out.target_suspect | out.context_suspect).sum()),
          "| needs_retranscription", int((out.needs_retranscription.astype(str) == "True").sum()),
          "| □", int(out.has_box.sum()), "| numerals", int(out.is_numeral.sum()))
    print(f"speakers {k.speaker.nunique()}, items {k.item_id.nunique()}, word types {k.word.nunique()}")
    print("words not in frequency corpus:", f"{(k.word_count == 0).mean():.1%}")
    print(k.groupby("position")[["informativity", "word_surprisal", "log_word_freq", "target_f0", "context_f0"]]
          .mean().round(3).to_string())
    print("corr(word_surprisal, log_word_freq) =", round(k.word_surprisal.corr(k.log_word_freq), 3))
    print("corr(unigram_surprisal, context_gain) =", round(k.unigram_surprisal.corr(k.context_gain), 3),
          "| context_gain < 0:", f"{(k.context_gain < 0).mean():.1%}")

if __name__ == "__main__":
    main()
