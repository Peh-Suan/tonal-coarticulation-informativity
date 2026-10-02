"""
Step 1 of the COSPRO tone-bigram pipeline.

Parses every COSPRO TextGrid (tiers: "phon" = initials/finals in COSPRO's
SAMPA-like notation with the tone digit on the final; "break" = Tseng's
prosodic break indices B2-B5), rebuilds syllables, converts them to
pypinyin TONE3 style (e.g. zhong1, lv4, xue2, de5) so they match the
lexicon used for the informativity estimates in tc_mi/ptt.ipynb, and writes:

  syllables.csv  one row per syllable
  bigrams_pw.csv one row per adjacent-syllable pair inside the same
                 prosodic-word unit (break interval). PROVISIONAL: these are
                 prosodic words, not lexical words. Replace with lexical word
                 boundaries once we have the character text (step 2).

Usage: python3 01_extract_syllables.py <corpus_root> <out_dir>
"""
import sys, re, glob, os, csv

INITIALS = {"b":"b","p":"p","m":"m","f":"f","d":"d","t":"t","n":"n","l":"l",
            "g":"g","k":"k","h":"h","dj":"j","tj":"q","sj":"x",
            "dz`":"zh","ts`":"ch","s`":"sh","Z`":"r","dz":"z","ts":"c","s":"s"}
# final -> (spelling after a consonant, spelling with zero initial)
FINALS = {
 "a":("a","a"),"o":("o","o"),"@":("e","e"),"@`":("er","er"),"ai":("ai","ai"),
 "ei":("ei","ei"),"au":("ao","ao"),"ou":("ou","ou"),"an":("an","an"),
 "@n":("en","en"),"aN":("ang","ang"),"@N":("eng","eng"),"oN":("ong","weng"),
 "i":("i","yi"),"ia":("ia","ya"),"iE":("ie","ye"),"iau":("iao","yao"),
 "iou":("iu","you"),"iEn":("ian","yan"),"in":("in","yin"),"iaN":("iang","yang"),
 "iN":("ing","ying"),"yoN":("iong","yong"),
 "u":("u","wu"),"ua":("ua","wa"),"uo":("uo","wo"),"uai":("uai","wai"),
 "uei":("ui","wei"),"uan":("uan","wan"),"u@n":("un","wen"),"uaN":("uang","wang"),
 "y":("v","yu"),"yE":("ve","yue"),"yEn":("van","yuan"),"yn":("vn","yun"),
 "U`":("i","i"),"U":("i","i"),
}
PAUSES = {"sil","sp",""}

def read_short_textgrid(path):
    L = [l.rstrip("\r\n") for l in open(path, encoding="utf-8", errors="replace")]
    tiers, i = {}, 0
    while i < len(L):
        if L[i] == '"IntervalTier"':
            name, n = L[i+1].strip('"'), int(L[i+4])
            ivs = [(float(L[i+5+3*k]), float(L[i+6+3*k]), L[i+7+3*k][1:-1])
                   for k in range(n)]
            tiers[name] = ivs
            i += 5 + 3*n
        else:
            i += 1
    return tiers

def to_pinyin(ini, fin):
    base, tone = fin[:-1], fin[-1]
    if base not in FINALS:
        return None
    if ini is None:
        py = FINALS[base][1]
    else:
        py0 = INITIALS[ini]
        py = py0 + FINALS[base][0]
        if py0 in ("j","q","x"):                 # ju, que, xuan, jun
            py = py.replace("v", "u")
        if py0 in ("b","p","m","f") and base == "uo":   # bo, po, mo, fo
            py = py0 + "o"
    return py + tone

def syllabify(phon):
    """Group initial+final segments into syllables."""
    sylls, pending, problems = [], None, []
    for a, b, lab in phon:
        if lab in PAUSES:
            if pending: problems.append(("dangling_initial", pending[2]))
            pending = None
            continue
        if re.search(r"[1-5]$", lab):             # a final carries the tone
            ini = pending[2] if pending else None
            start = pending[0] if pending else a
            py = to_pinyin(ini, lab)
            if py is None: problems.append(("unknown_final", lab))
            sylls.append(dict(start=start, end=b, v_start=a, initial=ini or "",
                              final=lab[:-1], tone=int(lab[-1]), pinyin=py or ""))
            pending = None
        elif lab in INITIALS:
            if pending: problems.append(("double_initial", pending[2]+"+"+lab))
            pending = (a, b, lab)
        else:
            problems.append(("unknown_label", lab))
    return sylls, problems

def main(root, out):
    os.makedirs(out, exist_ok=True)
    # corspro4 has no tone labels on its finals, so it cannot be used
    files = sorted(f for f in glob.glob(os.path.join(root, "**", "*.TextGrid"), recursive=True)
                   if not os.path.basename(f).startswith("._") and "corspro4" not in f)
    syl_rows, bi_rows, prob_rows = [], [], []
    for f in files:
        rel = os.path.relpath(f, root)
        m = re.search(r"COSPRO\s*(\d+)_([FM]\d+)(\D+)(\d+)", os.path.basename(f))
        subcorpus, speaker, ptype, item = m.groups() if m else ("","","","")
        tg = read_short_textgrid(f)
        sylls, probs = syllabify(tg.get("phon", []))
        prob_rows += [dict(file=rel, problem=p, label=l) for p, l in probs]
        breaks = [iv for iv in tg.get("break", []) if re.fullmatch(r"B\d", iv[2])]
        for idx, s in enumerate(sylls):
            mid = (s["v_start"] + s["end"]) / 2
            pw = next((k for k, (a, b, _) in enumerate(breaks) if a <= mid < b), -1)
            s.update(file=rel, subcorpus=subcorpus, speaker=speaker, item=item,
                     syl_idx=idx, pw_idx=pw,
                     break_after=breaks[pw][2] if pw >= 0 else "")
            syl_rows.append(s)
        for s1, s2 in zip(sylls, sylls[1:]):
            if s1["pw_idx"] == s2["pw_idx"] and s1["pw_idx"] >= 0:
                bi_rows.append(dict(file=rel, speaker=speaker, item=item,
                    syl_idx1=s1["syl_idx"], syllable1=s1["pinyin"], syllable2=s2["pinyin"],
                    tone1=s1["tone"], tone2=s2["tone"],
                    t3_sandhi_context=int(s1["tone"] == 3 and s2["tone"] == 3),
                    start1=s1["start"], end1=s1["end"], start2=s2["start"], end2=s2["end"],
                    pw_idx=s1["pw_idx"]))
    cols = ["file","subcorpus","speaker","item","syl_idx","pinyin","initial","final",
            "tone","start","v_start","end","pw_idx","break_after"]
    for name, rows, c in [("syllables.csv", syl_rows, cols),
                          ("bigrams_pw.csv", bi_rows, list(bi_rows[0]) if bi_rows else []),
                          ("problems.csv", prob_rows, ["file","problem","label"])]:
        with open(os.path.join(out, name), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=c, extrasaction="ignore"); w.writeheader(); w.writerows(rows)
    print(f"{len(files)} TextGrids, {len(syl_rows)} syllables, "
          f"{len(bi_rows)} within-PW bigrams, {len(prob_rows)} problems")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
