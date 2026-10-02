"""Build processed_data.csv for the TSM analysis (one row per within-word bigram x
position), with the same column names as cospro_pipeline/08_build_processed.py so
the COSPRO analysis notebook carries over.

Inputs in <prep>: bigrams_f0.csv (05), syllable_surprisal.csv (06), unigram_lm.csv and
word_freq.csv (07, optional), syllables_surface.csv, utterances.csv; informativity
tables from the TAI study in --inf-dir (tc_mi/): informativity_TSM.csv (citation
tone pairs) and, per speaker accent group, informativity_sandhi_zhang_TSM.csv (T5->7) or
informativity_sandhi_quan_TSM.csv (T5->3) for surface tone pairs (column sandhi_table).

Long format, as for COSPRO:
  position = carryover     target = ONSET of syllable 2,  context = OFFSET of syllable 1
  position = anticipatory  target = OFFSET of syllable 1, context = ONSET of syllable 2
Main columns
  target_f0 / context_f0 (speaker logz), *_hz, *_portion, *_suspect, onset_segment /
  offset_segment (segment types at the boundary)
  target_tone / context_tone        SURFACE tone (Tâi-lô number; sandhi applied)
  target_tone_lex / context_tone_lex citation tone
  informativity (citation pair), informativity_sandhi (surface pair): from
      informativity_positional.py, i.e. p(T | adjacent syllable, target position class) over
      all within-word pairs of the lexicon; target_pos_class = initial / medial / final;
      *_allpairs = same without position; *_tai = the old TAI tables (disyllables only)
      informativity_full_context(_sandhi): the context also includes the target's own segments
      (informativity_full_context_*.csv; mostly ~0 bits, see informativity_positional.py)
      functional_load*, functional_load_max* (+ context_*): the target tone's functional load, with the
      same name pattern as each informativity column except *_tai (functional_load_pos_*.csv,
      functional_load_full_context_*.csv from informativity_positional.py --fl-only)
      context_<measure> (for every measure above): the CONTEXT tone's informativity = the same
      pair read in the other direction (context syllable as target, predicted from the target
      syllable) at the context's position class (context_pos_class)
  word_surprisal (bits, 06: preceding context in the sentence), target/context_syl_surprisal
  unigram_surprisal (07 LM marginal if unigram_lm.csv exists, else corpus; see
  unigram_source), context_gain = unigram_surprisal - word_surprisal, *_corpus versions,
  log_word_freq (log10 per million, +1, unique TAT sentences), word_count
  syl2_form   citation | sandhi | uncertain   (syllable 1 is always word-internal -> sandhi)
  dur1, dur2, pause_between, utt_prop (position of the bigram in the utterance, 0-1),
  utt_rate (syllables/s over the utterance)
  speaker, sex, age, accent, site, item_id (same sentence across speakers), token_id
  Quality: romanised (word has a romanised syllable; their surprisal is inflated, see 06),
  token_crosses_word, too_short1/2, pre_a_suffix1
  keep       recommended subset: F0 present and not suspect, not too short, citation informativity
             defined, no checked tones, both surface tones known (no neutral tones; unknown-accent speakers'
             T5 before another syllable of the word has no surface tone), syllable-2 surface tone certain AND citation
             (the disyllabic-word design of the TAI study), not before á, word surprisal
             present, not romanised, no token crossing the word boundary
  keep_any2  same, but syllable 2 may also be a (certain) sandhi form

  python 08_build_processed.py --prep ~/tsm_mfa/all_condenser
"""
import argparse
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from tsm_words import word_forms

CITATION_J = {"punct", "utt_final", "pre_neutral", "pause"}


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--inf-dir", default=str(here.parent / "tables"), help="folder with the informativity / functional-load tables")
    ap.add_argument("--dest", default="processed_data.csv")
    a = ap.parse_args()
    prep = Path(os.path.expanduser(a.prep))
    rd = lambda f, **kw: pd.read_csv(prep / f, keep_default_na=False, na_values=[""], low_memory=False, **kw)

    b = rd("bigrams_f0.csv", dtype={"tone1": str, "tone2": str, "surface_tone1": str, "surface_tone2": str})
    b = b[b.same_word == 1].copy()
    print(f"{len(b):,} within-word bigrams")
    # Surface tones come from 04's syllables_surface.csv, not from the copies carried
    # through 05: 04 now applies per-speaker T5 sandhi (漳 5->7, 泉 5->3) and drops
    # speakers, and F0 extraction does not depend on tones, so 05 need not be rerun.
    st = rd("syllables_surface.csv", usecols=["utt_id", "syl_idx", "surface_tone", "accent_group"],
            dtype={"surface_tone": str})
    n0 = b.speaker.nunique()
    b = b[b.utt_id.isin(set(st.utt_id))].copy()
    for n in ("1", "2"):
        b = b.drop(columns=f"surface_tone{n}").merge(
            st[["utt_id", "syl_idx", "surface_tone"]].rename(columns={"syl_idx": f"syl_idx{n}", "surface_tone": f"surface_tone{n}"}),
            on=["utt_id", f"syl_idx{n}"], how="left")
    b = b.merge(st.drop_duplicates("utt_id")[["utt_id", "accent_group"]], on="utt_id", how="left")
    print(f"  after speaker groups: {len(b):,} bigrams, {b.speaker.nunique()} of {n0} speakers "
          f"({b.groupby('accent_group').speaker.nunique().to_dict()})")

    # --- surprisal (06) ---------------------------------------------------------
    s = rd("syllable_surprisal.csv")
    sc = ["utt_id", "syl_idx", "unit", "syl_surprisal", "word_surprisal", "token_crosses_word"]
    for n in ("1", "2"):
        b = b.merge(s[sc].rename(columns={c: c + n for c in sc[1:]}), on=["utt_id", f"syl_idx{n}"], how="left")
    b["word_surprisal"] = b.word_surprisal1
    b["token_crosses_word"] = b.token_crosses_word1.fillna(0).astype(int)

    # --- word forms, frequency, unigram (07) --------------------------------------
    print("collecting word forms ...")
    wf = word_forms(prep, unique=False)[["utt_id", "word_idx", "hanlo_form", "tailo_form"]]
    b = b.merge(wf.rename(columns={"word_idx": "word_idx1"}), on=["utt_id", "word_idx1"], how="left")
    b["word"] = b.tailo_form
    b["romanised"] = b.hanlo_form.fillna("").str.contains(r"[A-Za-z]").astype(int)

    if (prep / "word_freq.csv").exists():
        fq = rd("word_freq.csv")
        N, V = fq["count"].sum(), len(fq)
        b = b.merge(fq.rename(columns={"tailo_form": "word", "count": "word_count"}), on="word", how="left")
        b["word_count"] = b.word_count.fillna(0)
        b["unigram_surprisal_corpus"] = -np.log2((b.word_count + 1) / (N + V))
        b["log_word_freq"] = np.log10((b.word_count + 1) / N * 1e6)
    else:
        print("note: word_freq.csv missing (run 07_unigram_lm.py)")
        b["word_count"] = b["unigram_surprisal_corpus"] = b["log_word_freq"] = np.nan
    if (prep / "unigram_lm.csv").exists():
        u = rd("unigram_lm.csv")[["hanlo_form", "unigram_lm_surprisal", "unigram_lm_se", "null_context_surprisal"]]
        b = b.merge(u, on="hanlo_form", how="left")
        b["unigram_surprisal"] = b.unigram_lm_surprisal
        b["unigram_source"] = "lm"
    else:
        print("note: unigram_lm.csv missing; unigram surprisal from corpus counts")
        b["unigram_lm_se"] = b["null_context_surprisal"] = np.nan
        b["unigram_surprisal"] = b.unigram_surprisal_corpus
        b["unigram_source"] = "corpus"
    b["context_gain"] = b.unigram_surprisal - b.word_surprisal
    b["context_gain_corpus"] = b.unigram_surprisal_corpus - b.word_surprisal

    # --- informativity (TAI tables) ---------------------------------------------
    b["pair"] = b.tone1 + "-" + b.tone2
    b["surface_pair"] = b.surface_tone1 + "-" + b.surface_tone2
    lookup = lambda fname, d: (pd.read_csv(Path(a.inf_dir) / fname)
                               .query("direction == @d").set_index("pair").informativity)
    # citation pairs: one table for everyone
    for d in ("carryover", "anticipatory"):
        b[f"inf_{d}"] = b.pair.map(lookup("informativity_TSM.csv", d))
    # surface pairs: the table must use the same T5 sandhi as the speaker's surface tones
    # (04: 漳 zhang 5->7, 泉 quan 5->3). The old informativity_sandhi_TSM.csv used 5->3 for
    # everyone; it is only used when no accent groups are defined (accent_group 'all').
    SANDHI_TABLE = {"zhang": "informativity_sandhi_zhang_TSM.csv",
                    "quan": "informativity_sandhi_quan_TSM.csv",
                    "all": "informativity_sandhi_TSM.csv"}
    for d in ("carryover", "anticipatory"):
        col = pd.Series(np.nan, index=b.index)
        for grp, fname in SANDHI_TABLE.items():
            sel = b.accent_group.eq(grp)
            if sel.any():
                col[sel] = b.loc[sel, "surface_pair"].map(lookup(fname, d))
        b[f"inf_{d}_sandhi"] = col
    b["sandhi_table"] = b.accent_group.map(SANDHI_TABLE)
    print("sandhi informativity tables:", b.drop_duplicates("speaker").sandhi_table.value_counts().to_dict(), "(speakers)")
    # the surface-pair table has no checked tones, but 4h->2 and 8->3 would otherwise match
    # a non-checked surface tone: as in TAI, checked syllables get no sandhi informativity
    chk = b.tone1.isin(["4", "8"]) | b.tone2.isin(["4", "8"])
    b.loc[chk, ["inf_carryover_sandhi", "inf_anticipatory_sandhi"]] = np.nan

    # --- context, position, ids -----------------------------------------------------
    b["syl2_form"] = np.where(b.juncture_after2.isin(CITATION_J), "citation",
                              np.where(b.juncture_after2 == "word_internal", "sandhi", "uncertain"))
    syl = rd("syllables_surface.csv", usecols=["utt_id", "syl_idx", "word_idx", "start", "end", "n_syl"])
    first = syl.groupby(["utt_id", "word_idx"]).syl_idx.min().rename("word_start").reset_index()
    b = b.merge(first.rename(columns={"word_idx": "word_idx1"}), on=["utt_id", "word_idx1"], how="left")
    b["pos_in_word"] = b.syl_idx1 - b.word_start
    ut = syl.groupby("utt_id").agg(t0=("start", "min"), t1=("end", "max"), n_syl=("n_syl", "first"))
    ut["utt_rate"] = ut.n_syl / (ut.t1 - ut.t0)
    b = b.merge(ut[["utt_rate", "n_syl"]].reset_index(), on="utt_id", how="left")
    b["utt_prop"] = (b.syl_idx1 + 0.5) / b.n_syl

    # --- positional informativity (informativity_positional.py) ---------------------
    # p(T | adjacent syllable, position of the TARGET in the word) over all within-word syllable
    # pairs of the lexicon (the TAI tables counted disyllabic words only). These become
    # informativity / informativity_sandhi; pos_class "all" (no position) -> *_allpairs; the TAI
    # values are kept as *_tai. Only within-word bigrams get a value.
    cls = lambda i, n: np.where(i == 0, "initial", np.where(i == n - 1, "final", "medial"))
    b["tpc_carryover"] = cls(b.pos_in_word + 1, b.word_len)       # target = syllable 2
    b["tpc_anticipatory"] = cls(b.pos_in_word, b.word_len)        # target = syllable 1
    # the lexicon cuts a word at '--' (neutral tone), so a syllable before a neutral syllable is final
    b.loc[b.juncture_after2.eq("pre_neutral"), "tpc_carryover"] = "final"
    within = b.juncture_after1.eq("word_internal")
    def pos_lookup(fname, d, keycol, pooled=False, col="informativity"):
        t = pd.read_csv(Path(a.inf_dir) / fname, dtype={"pair": str}).query("direction == @d")
        m = t.set_index(["pair", "pos_class"])[col]
        pc = pd.Series("all", index=b.index) if pooled else b[f"tpc_{d}"]
        v = pd.Series(m.reindex(pd.MultiIndex.from_arrays([b[keycol], pc])).values, index=b.index)
        return v.where(within)
    POS_SANDHI = {"zhang": "informativity_pos_sandhi_zhang_TSM.csv",
                  "quan": "informativity_pos_sandhi_quan_TSM.csv",
                  "all": "informativity_pos_sandhi_zhang_TSM.csv"}    # no accent groups: 04 default 5 -> 7
    for d in ("carryover", "anticipatory"):
        b[f"inf_{d}_tai"], b[f"inf_{d}_sandhi_tai"] = b[f"inf_{d}"], b[f"inf_{d}_sandhi"]
        for pooled, suf in ((False, ""), (True, "_allpairs")):
            b[f"inf_{d}{suf}"] = pos_lookup("informativity_pos_TSM.csv", d, "pair", pooled)
            col = pd.Series(np.nan, index=b.index)
            for grp, fname in POS_SANDHI.items():
                sel = b.accent_group.eq(grp)
                if sel.any():
                    col[sel] = pos_lookup(fname, d, "surface_pair", pooled)[sel]
            col[b.tone1.isin(["4", "8"]) | b.tone2.isin(["4", "8"])] = np.nan   # as above: no checked tones
            b[f"inf_{d}_sandhi{suf}"] = col
        # full-context informativity (context also includes the target's own segments)
        b[f"inf_{d}_fullctx"] = pos_lookup("informativity_full_context_TSM.csv", d, "pair")
        col = pd.Series(np.nan, index=b.index)
        for grp, fname in POS_SANDHI.items():
            sel = b.accent_group.eq(grp)
            if sel.any():
                col[sel] = pos_lookup(fname.replace("informativity_pos_", "informativity_full_context_"),
                                      d, "surface_pair")[sel]
        col[b.tone1.isin(["4", "8"]) | b.tone2.isin(["4", "8"])] = np.nan
        b[f"inf_{d}_sandhi_fullctx"] = col

    # --- functional load of the target tone (informativity_positional.py --fl-only) -----------
    # functional_load_pos_* / functional_load_full_context_* mirror the informativity tables (same
    # cells and estimators); functional_load = entropy lost when the target tone merges with each
    # competitor (summed), functional_load_max = strongest competitor only. Same lookups and the same
    # missing values (checked tones in the sandhi tables) as the informativity columns.
    FL_METRICS = {"functional_load": "fl", "functional_load_max": "flmax"}
    checked = b.tone1.isin(["4", "8"]) | b.tone2.isin(["4", "8"])
    for metric, pre in FL_METRICS.items():
        for d in ("carryover", "anticipatory"):
            for pooled, suf in ((False, ""), (True, "_allpairs")):
                b[f"{pre}_{d}{suf}"] = pos_lookup("functional_load_pos_TSM.csv", d, "pair", pooled, metric)
                col = pd.Series(np.nan, index=b.index)
                for grp, fname in POS_SANDHI.items():
                    sel = b.accent_group.eq(grp)
                    if sel.any():
                        col[sel] = pos_lookup(fname.replace("informativity_", "functional_load_"), d,
                                              "surface_pair", pooled, metric)[sel]
                col[checked] = np.nan
                b[f"{pre}_{d}_sandhi{suf}"] = col
            b[f"{pre}_{d}_fullctx"] = pos_lookup("functional_load_full_context_TSM.csv", d, "pair", col=metric)
            col = pd.Series(np.nan, index=b.index)
            for grp, fname in POS_SANDHI.items():
                sel = b.accent_group.eq(grp)
                if sel.any():
                    col[sel] = pos_lookup(fname.replace("informativity_pos_", "functional_load_full_context_"),
                                          d, "surface_pair", col=metric)[sel]
            col[checked] = np.nan
            b[f"{pre}_{d}_sandhi_fullctx"] = col

    meta = rd("utterances.csv", usecols=["utt_id", "sex", "age", "accent", "hanlo", "tailo"])
    meta["item_id"] = "s" + meta.groupby("tailo").ngroup().astype(str)
    b = b.merge(meta[["utt_id", "sex", "age", "accent", "item_id"]], on="utt_id", how="left")
    b["site"] = b.speaker.str.split("_").str[0]
    b["token_id"] = b.item_id + "#" + b.syl_idx1.astype(str)
    b["bigram_id"] = b.utt_id + "#" + b.syl_idx1.astype(str)

    # --- long format -----------------------------------------------------------
    common = ["bigram_id", "utt_id", "speaker", "sex", "age", "accent", "accent_group", "sandhi_table", "site", "item_id", "token_id",
              "word", "hanlo_form", "word_len", "pos_in_word", "word_count", "log_word_freq",
              "word_surprisal", "unigram_surprisal", "context_gain", "unigram_source", "unigram_lm_se",
              "null_context_surprisal", "unigram_surprisal_corpus", "context_gain_corpus",
              "syllable1", "syllable2", "unit1", "unit2", "tone1", "tone2", "surface_tone1", "surface_tone2",
              "pair", "surface_pair", "syl2_form", "juncture_after2", "offset_segment1", "onset_segment2",
              "dur1", "dur2", "pause_between", "utt_prop", "utt_rate",
              "romanised", "token_crosses_word", "too_short1", "too_short2", "pre_a_suffix1",
              "juncture_after1", "surface_certain2", "neutral1", "neutral2"]
    INF_OUT = {"informativity": "", "informativity_sandhi": "_sandhi",
               "informativity_allpairs": "_allpairs", "informativity_sandhi_allpairs": "_sandhi_allpairs",
               "informativity_tai": "_tai", "informativity_sandhi_tai": "_sandhi_tai",
               "informativity_full_context": "_fullctx", "informativity_full_context_sandhi": "_sandhi_fullctx"}
    rows = []
    for pos in ("carryover", "anticipatory"):
        d = b[common].copy()
        d.insert(1, "position", pos)
        if pos == "carryover":
            T, C, t, c = ("onset", "2"), ("offset", "1"), "2", "1"
        else:
            T, C, t, c = ("offset", "1"), ("onset", "2"), "1", "2"
        d["target_tone"], d["context_tone"] = b["surface_tone" + t], b["surface_tone" + c]
        d["target_tone_lex"], d["context_tone_lex"] = b["tone" + t], b["tone" + c]
        other = "anticipatory" if pos == "carryover" else "carryover"
        d["target_pos_class"], d["context_pos_class"] = b[f"tpc_{pos}"], b[f"tpc_{other}"]
        # each informativity measure for the target, and context_* = the context tone's informativity:
        # the context tone's informativity is the same pair read in the other direction (the context
        # syllable as target, predicted from the target syllable, at the context's own position class)
        for name, suf in INF_OUT.items():
            d[name] = b[f"inf_{pos}{suf}"]
            d[f"context_{name}"] = b[f"inf_{other}{suf}"]
        # functional load, named like the informativity columns (no TAI version)
        for name, suf in INF_OUT.items():
            if suf.endswith("_tai"):
                continue
            for metric, pre in FL_METRICS.items():
                nm = name.replace("informativity", metric)
                d[nm] = b[f"{pre}_{pos}{suf}"]
                d[f"context_{nm}"] = b[f"{pre}_{other}{suf}"]
        d["target_syl_surprisal"], d["context_syl_surprisal"] = b["syl_surprisal" + t], b["syl_surprisal" + c]
        for role, (kind, n) in (("target", T), ("context", C)):
            d[f"{role}_f0"] = b[f"{kind}_logz{n}"]
            d[f"{role}_hz"] = b[f"{kind}_hz{n}"]
            d[f"{role}_portion"] = b[f"{kind}_portion{n}"]
            d[f"{role}_suspect"] = b[f"{kind}_suspect{n}"].astype(str).eq("True")
        d = d.rename(columns={"offset_segment1": "offset_segment", "onset_segment2": "onset_segment"})
        rows.append(d)
    out = pd.concat(rows, ignore_index=True)

    tf = lambda col: out[col].astype(str).isin(["True", "1", "1.0"])
    base = (out.target_f0.notna() & out.context_f0.notna() & ~out.target_suspect & ~out.context_suspect
            & ~tf("too_short1") & ~tf("too_short2")
            # citation-tone informativity defined, no checked tones (4/8), both surface tones known
            # (was: informativity_sandhi defined, which also required a zhang/quan sandhi table, i.e. it
            # dropped unknown-accent speakers; the analysis uses citation-tone informativity)
            & out.informativity.notna() & ~out.tone1.isin(["4", "8"]) & ~out.tone2.isin(["4", "8"])
            & out.surface_tone1.fillna("").ne("") & out.surface_tone2.fillna("").ne("")
            & out.juncture_after1.eq("word_internal") & ~tf("pre_a_suffix1") & tf("surface_certain2")
            & ~tf("neutral1") & ~tf("neutral2") & out.word_surprisal.notna()
            & out.romanised.eq(0) & out.token_crosses_word.eq(0))
    out["keep_any2"] = base
    out["keep"] = base & out.syl2_form.eq("citation")
    out.to_csv(prep / a.dest, index=False)

    k = out[out.keep]
    print(f"\n{prep / a.dest}: {len(out):,} rows ({len(b):,} bigrams x 2); "
          f"keep = {len(k):,}; keep_any2 = {int(out.keep_any2.sum()):,}")
    print(f"speakers {k.speaker.nunique()}, items {k.item_id.nunique()}, word types {k.word.nunique()}")
    print(k.groupby("position")[["informativity", "informativity_sandhi", "word_surprisal",
                                 "unigram_surprisal", "context_gain", "target_f0", "context_f0"]]
          .mean().round(3).to_string())
    print("corr(unigram_surprisal, context_gain) =", round(k.unigram_surprisal.corr(k.context_gain), 3),
          "| corr(word_surprisal, log_word_freq) =", round(k.word_surprisal.corr(k.log_word_freq), 3))
    print("surface pairs in keep:", k[k.position == "carryover"].surface_pair.value_counts().to_dict())


if __name__ == "__main__":
    main()
