"""Informativity of a tone given its ADJACENT syllable and the target's POSITION IN THE WORD.

Replaces the TAI tables (informativity_TSM.csv etc.), which counted disyllabic words only.
Here every adjacent syllable pair inside a word of any length counts (token-weighted), and
the conditioning context is (adjacent syllable, position class of the target syllable):

    informativity(T, C, dir, pos) = sum_c  n(c, pos, T) * -log2 p(T | c, pos)  /  sum_c n(c, pos, T)
        c   = the adjacent syllable (segments + tone), over all c with tone C
        dir = carryover (target = 2nd syllable of the pair, context = 1st)
              anticipatory (target = 1st syllable, context = 2nd)
        pos = position class of the TARGET syllable in its word: initial / medial / final

This is the TAI formula (tc_mi/preprocess_text.ipynb, ptt.ipynb) with position added to the
context. Rows with pos_class == "all" are the same measure without position (all within-word
pairs pooled), for comparison / robustness.

Small-sample correction. Medial cells have ~10x fewer tokens than edge cells, and the plug-in
estimate of -log2 p(T | c, pos) is biased towards 0 when a context syllable is rare (seen once
-> p = 1). p(T | c, pos) is therefore smoothed towards the tone distribution of that position:
    p = (n(c, pos, T) + alpha * m(T | dir, pos)) / (n(c, pos) + alpha)
alpha is calibrated per language: the edge classes (carryover-final, anticipatory-initial) are
thinned (binomially) to the token count of the medial class, and alpha is the value at which the
thinned estimates recover the full-sample estimates without bias (see alpha_calibration_<lang>.csv);
if the plug-in bias is already < 0.01 bits, alpha = 0. Result (2026-09-29): TSM alpha = 4.1
(plug-in bias at medial size -0.059 bits); TM alpha = 0 (bias -0.004: the lexicon is 10x larger).
informativity_plugin = alpha 0 (the TAI estimator).

Sandhi (surface) tables
  TSM: every non-final syllable of a word takes its sandhi form, the final syllable keeps its
       citation tone; same sandhi function as the TAI table (T5 -> 7 for the zhang table,
       5 -> 3 for quan). Words with '--' (neutral tone) count up to the '--' (the syllable before
       it keeps its citation tone); neutral syllables are not counted.
  TM:  T3 before a (lexical) T3 in the same word -> T2, as in the TAI table and in
       cospro_pipeline/05_words_bigrams.py (pair-local rule).

Lexicons (the TAI ones)
  TSM: tc_mi/texts.txt (Tâi-lô); TM: the segmented_texts corpus, via its word counts cached in
  cospro_pipeline/out/word_freq.csv (same Han-only filter as ptt.ipynb), romanised with TAIWAN readings (moe_readings.TaiwanReadings: COSPRO-attested, else MOE); --tm-readings pypinyin = mainland readings as in the TAI tables.

  python informativity_positional.py --lang TSM
  python informativity_positional.py --lang TM        # needs: pip install pypinyin
Outputs (in --out-dir, default <repo>/tables/): informativity_pos_TSM.csv,
informativity_pos_sandhi_zhang_TSM.csv, informativity_pos_sandhi_quan_TSM.csv,
informativity_pos_TM.csv, informativity_pos_sandhi_TM.csv, alpha_calibration_<lang>.csv.
Also writes informativity_full_context_*.csv (same names with _full_context): the context also
includes the target's own segments (table_full_context; plug-in, no smoothing; extra column
share_determined).
Functional load (write_fl): functional_load_pos_*.csv and functional_load_full_context_*.csv mirror
the informativity tables (same cells and estimators); columns functional_load (entropy lost when the
target tone merges with each competitor, summed), functional_load_max (strongest competitor), *_plugin,
share_zero. --fl-only writes only these (alpha read from the existing informativity_pos table).
Columns: direction, pair (tone of syllable 1 - tone of syllable 2, as in the TAI tables),
pos_class, informativity, informativity_plugin, tokens, contexts.
"""
import argparse, math, re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SYL_TSM = re.compile(r"[a-z]+[1-9]")
ALPHAS = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
ANALYSIS_TONES = {"TSM": set("12357"), "TM": set("1234")}


# ------------------------------------------------------------------ lexicons
def tsm_words(path):
    """Counter of word tuples (syllables with tone digits), neutral part cut off."""
    cnt = Counter()
    for line in open(path, encoding="utf-8"):
        for tok in re.split(r"[^a-zA-Z0-9-]+", line):
            tok = tok.lower().split("--")[0]              # neutral tone: keep the part before '--'
            syl = [s for s in tok.split("-") if s]
            if syl and all(SYL_TSM.fullmatch(s) for s in syl):
                cnt[tuple(syl)] += 1
    return cnt


def tm_words(word_freq, readings="taiwan"):
    """readings: 'taiwan' (default; COSPRO-attested readings, else MOE with COSPRO-style
    adjustments, see moe_readings.py) or 'pypinyin' (mainland readings, as in the TAI tables)."""
    wf = pd.read_csv(word_freq, keep_default_na=False)
    if readings == "taiwan":
        from moe_readings import TaiwanReadings
        tw = TaiwanReadings(); rom = lambda w: tw.romanise(w)[0]
    else:
        from pypinyin import lazy_pinyin, Style
        rom = lambda w: lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)
    cnt = Counter()
    for w, n in zip(wf.word, wf["count"]):
        syl = rom(w)
        if len(syl) == len(w) and all(s[-1].isdigit() for s in syl):
            cnt[tuple(syl)] += int(n)
    return cnt


def tsm_sandhi_syl(s, t5):
    """TAI tone_sandhi() for one non-final syllable (tc_mi/preprocess_text.ipynb cell 12)."""
    t = s[-1]
    if t in "12357":
        return s[:-1] + {"1": "7", "2": "1", "3": "2", "5": t5, "7": "3"}[t]
    if t in "48":
        if s[-2] in "ptk":
            return s[:-1] + {"4": "8", "8": "4"}[t]
        if s[-2] == "h":
            return s[:-1] + {"4": "2", "8": "3"}[t]
    return None


def tsm_sandhi(cnt, t5):
    out = Counter()
    for w, n in cnt.items():
        sw = [tsm_sandhi_syl(s, t5) for s in w[:-1]] + [w[-1]]
        if None not in sw:
            out[tuple(sw)] += n
    return out


def tm_sandhi(cnt):
    out = Counter()
    for w, n in cnt.items():
        sw = [s[:-1] + "2" if (s[-1] == "3" and w[i + 1][-1] == "3") else s for i, s in enumerate(w[:-1])]
        out[tuple(sw + [w[-1]])] += n
    return out


# ------------------------------------------------------------------ counts
def pos_class(i, n):
    return "initial" if i == 0 else ("final" if i == n - 1 else "medial")


def events(cnt):
    """one row per (direction, context syllable, target tone, target position class), token counts"""
    ev = Counter()
    for w, n in cnt.items():
        L = len(w)
        for i in range(L - 1):
            a, b = w[i], w[i + 1]
            ev[("carryover", a, b[-1], pos_class(i + 1, L))] += n       # target = 2nd syllable
            ev[("anticipatory", b, a[-1], pos_class(i, L))] += n        # target = 1st syllable
    e = pd.DataFrame([(*k, v) for k, v in ev.items()], columns=["direction", "c", "T", "pos_class", "n"])
    e["C"] = e.c.str[-1]
    return e


def estimate(e, alpha):
    """informativity per direction x T x C x pos_class from event counts e (n may be thinned)"""
    e = e[e.n > 0]
    keys = ["direction", "pos_class"]
    n_cp = e.groupby(keys + ["c"]).n.transform("sum")
    m = e.groupby(keys + ["T"]).n.sum() / e.groupby(keys).n.sum()
    mT = pd.Series(m.reindex(pd.MultiIndex.from_frame(e[keys + ["T"]])).values, index=e.index)
    p = (e.n + alpha * mT) / (n_cp + alpha)
    e = e.assign(ws=e.n * -np.log2(p))
    g = e.groupby(keys + ["T", "C"])
    out = pd.DataFrame({"informativity": g.ws.sum() / g.n.sum(), "tokens": g.n.sum(),
                        "contexts": g.c.nunique()}).reset_index()
    return out


def table(cnt, alpha):
    e = events(cnt)
    pooled = e.groupby(["direction", "c", "T", "C"], as_index=False).n.sum().assign(pos_class="all")
    res = []
    for ev in (e, pooled):
        s = estimate(ev, alpha)
        s["informativity_plugin"] = estimate(ev, 0).informativity.values
        res.append(s)
    t = pd.concat(res, ignore_index=True)
    t["pair"] = np.where(t.direction == "carryover", t["C"] + "-" + t["T"], t["T"] + "-" + t["C"])
    return t[["direction", "pair", "pos_class", "informativity", "informativity_plugin", "tokens", "contexts", "T", "C"]]


def table_full_context(cnt):
    """Full-context informativity (plug-in, no smoothing): the context also includes the TARGET's
    segments,
        informativity(T | C, dir, pos) = sum_{c,s} n(c,s,T) * -log2 P(T | c, s, pos) / sum n(c,s,T)
    c = adjacent syllable (segments + tone) with tone C, s = the target syllable's segments.
    Once c and s are known the word is usually identified, so most tokens have surprisal 0
    (share_determined); values are small and depend on lexicon size (see the README note).
    pos_class 'all' = the same without position."""
    ctx, ct = Counter(), Counter()
    for w, n in cnt.items():
        L = len(w)
        for i in range(L - 1):
            a, b = w[i], w[i + 1]
            for d, t, c, p in (("carryover", b, a, pos_class(i + 1, L)), ("anticipatory", a, b, pos_class(i, L))):
                for pc in (p, "all"):
                    ctx[(d, pc, c, t[:-1])] += n
                    ct[(d, pc, c, t[:-1], t[-1])] += n
    acc = defaultdict(lambda: [0.0, 0, 0, set()])
    for (d, pc, c, sg, T), n in ct.items():
        tot = ctx[(d, pc, c, sg)]
        r = acc[(d, pc, c[-1], T)]
        r[0] += -n * math.log2(n / tot); r[1] += n; r[2] += n * (tot == n); r[3].add((c, sg))
    rows = []
    for (d, pc, C, T), (sw, n, det, ctxs) in acc.items():
        rows.append(dict(direction=d, pair=f"{C}-{T}" if d == "carryover" else f"{T}-{C}", pos_class=pc,
                         informativity=sw / n, tokens=n, contexts=len(ctxs), share_determined=det / n, T=T, C=C))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ functional load
# Entropy-based functional load (Hockett; Surendran & Niyogi) of the TARGET tone, in the same
# context cells as the informativity measures. In a cell c (everything fixed except the target
# tone) with tone distribution p_c, merging tones t and u loses
#     dH_c(t, u) = (p_t + p_u) log2(p_t + p_u) - p_t log2 p_t - p_u log2 p_u      (bits, >= 0)
# functional_load      FL_c(t) = sum_{u != t} dH_c(t, u)   (all competitors)
# functional_load_max  FL_c(t) = max_{u != t} dH_c(t, u)   (strongest competitor)
# Aggregated like informativity, over the contexts in which t occurs:
#     FL(t | C, dir, pos) = sum_c n(c, pos, t) FL_c(t) / sum_c n(c, pos, t)
# p_c uses the same estimator as the matching informativity table (smoothed with alpha for the
# positional tables, plug-in for the full-context tables); *_plugin = alpha 0. share_zero = share
# of the tokens whose plug-in FL_c(t) is 0 (no competitor attested in that context).
def _dh(a, b):
    """entropy lost (bits) when two tones with probabilities a and b in one context merge"""
    def xlx(x):
        x = np.asarray(x, dtype=float)
        return np.where(x > 0, x * np.log2(np.where(x > 0, x, 1.0)), 0.0)
    return xlx(a + b) - xlx(a) - xlx(b)


def _fl_cells(P):
    """P: cells x tones probabilities -> (sum, max) over competitors, each cells x tones"""
    k = P.shape[1]
    D = np.stack([_dh(P[:, [i]], P) for i in range(k)], axis=1)     # D[:, i, j] = dH(i, j)
    idx = np.arange(k)
    D[:, idx, idx] = 0.0
    return D.sum(axis=2), D.max(axis=2) if k > 1 else np.zeros_like(P)


def fl_estimate(e, alpha):
    """functional load per direction x T x C x pos_class from event counts e (as estimate())"""
    e = e[e.n > 0]
    out = []
    for (d, pc), g in e.groupby(["direction", "pos_class"]):
        N = g.pivot_table(index="c", columns="T", values="n", aggfunc="sum", fill_value=0)
        Nv = N.values.astype(float)
        m = Nv.sum(0) / Nv.sum()
        P = (Nv + alpha * m) / (Nv.sum(1, keepdims=True) + alpha)
        fs, fm = _fl_cells(P)
        P0 = Nv / Nv.sum(1, keepdims=True)
        fs0, _ = _fl_cells(P0)
        cells = pd.DataFrame({"c": np.repeat(N.index.values, N.shape[1]), "T": np.tile(N.columns.values, N.shape[0]),
                              "n": Nv.ravel(), "fs": fs.ravel(), "fm": fm.ravel(), "zero": fs0.ravel() < 1e-12})
        cells = cells[cells.n > 0]
        cells["C"] = cells.c.str[-1]
        cells = cells.assign(ws=cells.n * cells.fs, wm=cells.n * cells.fm, wz=cells.n * cells.zero)
        gg = cells.groupby(["T", "C"])
        r = pd.DataFrame({"functional_load": gg.ws.sum() / gg.n.sum(), "functional_load_max": gg.wm.sum() / gg.n.sum(),
                          "share_zero": gg.wz.sum() / gg.n.sum(), "tokens": gg.n.sum()}).reset_index()
        out.append(r.assign(direction=d, pos_class=pc))
    return pd.concat(out, ignore_index=True)


def table_fl(cnt, alpha):
    """functional load in the cells of table() (adjacent syllable x target position class)"""
    e = events(cnt)
    pooled = e.groupby(["direction", "c", "T", "C"], as_index=False).n.sum().assign(pos_class="all")
    keys = ["direction", "pos_class", "T", "C"]
    res = []
    for ev in (e, pooled):
        s = fl_estimate(ev, alpha)
        s0 = fl_estimate(ev, 0)[keys + ["functional_load", "functional_load_max"]].rename(
            columns={"functional_load": "functional_load_plugin", "functional_load_max": "functional_load_max_plugin"})
        res.append(s.merge(s0, on=keys, how="left"))
    t = pd.concat(res, ignore_index=True)
    t["pair"] = np.where(t.direction == "carryover", t["C"] + "-" + t["T"], t["T"] + "-" + t["C"])
    return t[["direction", "pair", "pos_class", "functional_load", "functional_load_max", "functional_load_plugin",
              "functional_load_max_plugin", "share_zero", "tokens", "T", "C"]]


def table_fl_full_context(cnt):
    """functional load in the cells of table_full_context() (adjacent syllable x target segments x
    position; plug-in, as the full-context informativity)"""
    ct = defaultdict(Counter)
    for w, n in cnt.items():
        L = len(w)
        for i in range(L - 1):
            a, b = w[i], w[i + 1]
            for d, t, c, p in (("carryover", b, a, pos_class(i + 1, L)), ("anticipatory", a, b, pos_class(i, L))):
                for pc in (p, "all"):
                    ct[(d, pc, c, t[:-1])][t[-1]] += n
    acc = defaultdict(lambda: [0.0, 0.0, 0, 0])
    for (d, pc, c, sg), tc in ct.items():
        tot = sum(tc.values())
        ps = {T: n / tot for T, n in tc.items()}
        for T, n in tc.items():
            dh = [float(_dh(ps[T], ps[u])) for u in ps if u != T]
            fs = sum(dh); fm = max(dh) if dh else 0.0
            r = acc[(d, pc, c[-1], T)]
            r[0] += n * fs; r[1] += n * fm; r[2] += n; r[3] += n * (len(dh) == 0)
    rows = []
    for (d, pc, C, T), (ws, wm, n, z) in acc.items():
        rows.append(dict(direction=d, pair=f"{C}-{T}" if d == "carryover" else f"{T}-{C}", pos_class=pc,
                         functional_load=ws / n, functional_load_max=wm / n, share_zero=z / n, tokens=n, T=T, C=C))
    return pd.DataFrame(rows)


def calibrate(cnt, lang, reps=10, seed=2026):
    """thin the edge classes to the medial class size; bias of the thinned estimate per alpha"""
    rng = np.random.default_rng(seed)
    e = events(cnt)
    tones = ANALYSIS_TONES[lang]
    rows = []
    for d, edge in (("carryover", "final"), ("anticipatory", "initial")):
        ed = e[(e.direction == d) & (e.pos_class == edge)]
        frac = e[(e.direction == d) & (e.pos_class == "medial")].n.sum() / ed.n.sum()
        full = estimate(ed, 0)
        full = full[full["T"].isin(tones) & full["C"].isin(tones)].set_index(["T", "C"]).informativity
        for r in range(reps):
            thin = ed.assign(n=rng.binomial(ed.n.values, frac))
            for a in ALPHAS:
                est = estimate(thin, a).set_index(["T", "C"]).informativity.reindex(full.index)
                diff = est - full
                rows.append(dict(direction=d, rep=r, alpha=a, bias=diff.mean(), mae=diff.abs().mean(),
                                 r=np.corrcoef(est.fillna(full), full)[0, 1], frac=frac))
    cal = pd.DataFrame(rows).groupby(["alpha"]).agg(bias=("bias", "mean"), mae=("mae", "mean"),
                                                     r=("r", "mean"), frac=("frac", "mean")).reset_index()
    # interpolate the alpha where the bias crosses 0
    b = cal.bias.values; a = cal.alpha.values
    if np.abs(b[0]) < 0.01:          # plug-in bias already < 0.01 bits: no smoothing needed
        return cal, 0.0
    k = np.where(np.sign(b[:-1]) != np.sign(b[1:]))[0]
    best = float(a[k[0]] + (a[k[0] + 1] - a[k[0]]) * (-b[k[0]]) / (b[k[0] + 1] - b[k[0]])) if len(k) else float(a[np.argmin(np.abs(b))])
    return cal, round(best, 2)


def check_tai(cnt, old_path, sandhi_fn=None):
    """disyllables only + plug-in should reproduce the TAI table"""
    if not Path(old_path).exists():
        return
    two = Counter({w: n for w, n in cnt.items() if len(w) == 2})
    if sandhi_fn: two = sandhi_fn(two)
    t = table(two, 0)
    t = t[t.pos_class == "all"]
    old = pd.read_csv(old_path, index_col=0)
    m = old.merge(t, on=["direction", "pair"], suffixes=("_tai", ""))
    print(f"  check vs {Path(old_path).name}: {len(m)}/{len(old)} pairs matched, "
          f"max |diff| {np.abs(m.informativity_tai - m.informativity_plugin).max():.3f}")


def write_fl(tabs, alpha, out, tones):
    """functional-load tables mirroring the informativity tables:
    informativity_pos_* -> functional_load_pos_*, informativity_full_context_* -> functional_load_full_context_*"""
    for fname, c in tabs.items():
        for kind, t in (("pos", table_fl(c, alpha).assign(alpha=alpha)), ("full_context", table_fl_full_context(c))):
            fl_name = fname.replace("informativity_pos_", f"functional_load_{kind}_")
            t.drop(columns=["T", "C"]).to_csv(out / fl_name, index=False)
            s = t[t["T"].isin(tones) & t["C"].isin(tones) & (t.pos_class != "all")]
            summ = s.groupby(["direction", "pos_class"]).apply(lambda g: pd.Series({
                "cells": len(g), "fl_mean": g.functional_load.mean(), "fl_max_mean": g.functional_load_max.mean(),
                "share_zero": np.average(g.share_zero, weights=g.tokens)}), include_groups=False)
            print(f"\n{fl_name}  (analysis tones only)\n{summ.round(3).to_string()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["TSM", "TM"], required=True)
    ap.add_argument("--texts", default=str(HERE.parent / "lexicon" / "texts.txt"))
    ap.add_argument("--word-freq", default=str(HERE.parent / "cospro_pipeline" / "out" / "word_freq.csv"))
    ap.add_argument("--out-dir", default=str(HERE.parent / "tables"))
    ap.add_argument("--alpha", type=float, default=None, help="skip the calibration and use this alpha")
    ap.add_argument("--tm-readings", choices=["taiwan", "pypinyin"], default="taiwan",
                    help="Mandarin readings: Taiwan (COSPRO-attested, else MOE; default) or pypinyin (mainland)")
    ap.add_argument("--fl-only", action="store_true",
                    help="only write the functional-load tables (alpha = the one in the existing informativity_pos table)")
    a = ap.parse_args()
    out = Path(a.out_dir)

    cnt = tsm_words(a.texts) if a.lang == "TSM" else tm_words(a.word_freq, a.tm_readings)
    L = Counter()
    for w, n in cnt.items(): L[len(w)] += n
    print(f"{a.lang}: {sum(cnt.values()):,} word tokens, {len(cnt):,} types; tokens by length:",
          dict(sorted(L.items())[:8]))

    if a.alpha is None and a.fl_only and (out / f"informativity_pos_{a.lang}.csv").exists():
        a.alpha = float(pd.read_csv(out / f"informativity_pos_{a.lang}.csv").alpha.iloc[0])
        print(f"  alpha from informativity_pos_{a.lang}.csv: {a.alpha}")
    if a.alpha is None:
        cal, alpha = calibrate(cnt, a.lang)
        cal.to_csv(out / f"alpha_calibration_{a.lang}.csv", index=False)
        print(cal.round(4).to_string(index=False))
        print(f"  -> alpha = {alpha}")
    else:
        alpha = a.alpha

    if a.lang == "TSM":
        if not a.fl_only: check_tai(cnt, out / "informativity_TSM.csv")
        if not a.fl_only: check_tai(cnt, out / "informativity_sandhi_zhang_TSM.csv", lambda c: tsm_sandhi(c, "7"))
        tabs = {"informativity_pos_TSM.csv": cnt,
                "informativity_pos_sandhi_zhang_TSM.csv": tsm_sandhi(cnt, "7"),
                "informativity_pos_sandhi_quan_TSM.csv": tsm_sandhi(cnt, "3")}
    else:
        if not a.fl_only:
            check_tai(cnt, out / "informativity_TM.csv")
            check_tai(cnt, out / "informativity_sandhi_TM.csv", tm_sandhi)
        tabs = {"informativity_pos_TM.csv": cnt, "informativity_pos_sandhi_TM.csv": tm_sandhi(cnt)}

    tones = ANALYSIS_TONES[a.lang]
    write_fl(tabs, alpha, out, tones)
    if a.fl_only:
        return
    for fname, c in tabs.items():
        t = table(c, alpha).assign(alpha=alpha)
        t.drop(columns=["T", "C"]).to_csv(out / fname, index=False)
        s = t[t["T"].isin(tones) & t["C"].isin(tones)]
        summ = s.groupby(["direction", "pos_class"]).agg(cells=("pair", "size"), mean=("informativity", "mean"),
                                                          min_tokens=("tokens", "min"))
        allv = s[s.pos_class == "all"].set_index(["direction", "pair"]).informativity
        summ["r_with_all"] = [np.corrcoef(g.informativity, allv.reindex(pd.MultiIndex.from_frame(g[["direction", "pair"]])))[0, 1]
                              for _, g in s.groupby(["direction", "pos_class"])]
        print(f"\n{fname}  (analysis tones only)\n{summ.round(3).to_string()}")

    # full-context versions (target segments in the context), same file names with _full_context
    for fname, c in tabs.items():
        fc_name = fname.replace("informativity_pos_", "informativity_full_context_")
        t = table_full_context(c)
        t.drop(columns=["T", "C"]).to_csv(out / fc_name, index=False)
        s = t[t["T"].isin(tones) & t["C"].isin(tones) & (t.pos_class != "all")]
        summ = s.groupby(["direction", "pos_class"]).apply(lambda g: pd.Series({
            "cells": len(g), "mean": g.informativity.mean(),
            "share_determined": np.average(g.share_determined, weights=g.tokens)}), include_groups=False)
        print(f"\n{fc_name}  (analysis tones only)\n{summ.round(3).to_string()}")


if __name__ == "__main__":
    main()
