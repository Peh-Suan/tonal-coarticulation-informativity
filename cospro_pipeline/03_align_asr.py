"""
Step 2b: align Whisper characters to the gold COSPRO syllables.

For each recording, Needleman-Wunsch alignment between the ASR character
string (punctuation removed) and the gold syllable sequence from
syllables.csv. A character "matches" a syllable if ANY of its dictionary
readings (pypinyin heteronyms, TONE3) equals the gold syllable (toned match),
or equals it ignoring tone (toneless match).

Writes out/syllables_chars.csv: syllables.csv + char, match
  match = toned | toneless | sub (different syllable) | gold_only (ASR dropped it)
and out/asr_insertions.csv for ASR characters with no gold syllable,
and prints a per-file summary to out/align_summary.csv.

NOTE: homophones with the same tone (油/遊, 宜/疑) always "match" here;
they are dealt with in the next (correction) step.
"""
import json, re, sys
import pandas as pd
from pypinyin import pinyin, Style
from opencc import OpenCC

cc = OpenCC("s2tw")          # character-level only (s2twp rewrites words, e.g. 高級->高階)
HAN = re.compile(r"[㐀-鿿]")

from functools import lru_cache
@lru_cache(maxsize=None)
def readings(ch):
    r = pinyin(ch, style=Style.TONE3, heteronym=True, neutral_tone_with_five=True)[0]
    r = {x.replace("ü", "v") for x in r}
    if ch == "一": r |= {"yi1", "yi2", "yi4"}     # 一/不 sandhi is labelled on the surface
    if ch == "不": r |= {"bu2", "bu4"}
    return r

@lru_cache(maxsize=None)
def cost(ch, syl):
    rs = readings(ch)
    if syl in rs: return 0.0, "toned"
    if syl[:-1] in {r[:-1] for r in rs}: return 0.35, "toneless"
    return 1.0, "sub"

GAP = 0.8
def align(chars, sylls):
    n, m = len(chars), len(sylls)
    D = [[0.0]*(m+1) for _ in range(n+1)]; B = [[None]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): D[i][0] = i*GAP; B[i][0] = "up"
    for j in range(1, m+1): D[0][j] = j*GAP; B[0][j] = "left"
    C = {}
    for i in range(1, n+1):
        for j in range(1, m+1):
            c, t = cost(chars[i-1], sylls[j-1]); C[i, j] = t
            opts = [(D[i-1][j-1]+c, "diag"), (D[i-1][j]+GAP, "up"), (D[i][j-1]+GAP, "left")]
            D[i][j], B[i][j] = min(opts)
    i, j, pairs = n, m, []
    while i or j:
        b = B[i][j]
        if b == "diag": pairs.append((i-1, j-1, C[i, j])); i, j = i-1, j-1
        elif b == "up": pairs.append((i-1, None, "asr_only")); i -= 1
        else: pairs.append((None, j-1, "gold_only")); j -= 1
    return pairs[::-1]

def main(out="out"):
    syl = pd.read_csv(f"{out}/syllables.csv", keep_default_na=False)
    syl = syl[syl.pinyin != ""]          # a few unparseable labels
    asr = {}
    import glob as _g
    lines = [l for fn in sorted(_g.glob(f"{out}/asr*.jsonl")) if "retrans" not in fn
             for l in open(fn, encoding="utf-8")]      # asr.jsonl + any shard files
    for l in lines:
        try: r = json.loads(l)          # last line may be half-written while Whisper runs
        except json.JSONDecodeError: continue
        asr[r["file"].replace(".wav", ".TextGrid")] = cc.convert(r["text_raw"])
    # chunked re-transcriptions (02b) replace failed whole-file transcriptions
    import os
    if os.path.exists(f"{out}/asr_retrans.jsonl"):
        for l in open(f"{out}/asr_retrans.jsonl", encoding="utf-8"):
            r = json.loads(l); asr[r["file"].replace(".wav", ".TextGrid")] = cc.convert(r["text_raw"])
    syl = syl[syl.file.isin(asr)].copy()
    syl["char"], syl["match"] = "", ""
    ins, summ = [], []
    for f, g in syl.groupby("file", sort=False):
        text = asr[f]; chars = HAN.findall(text)
        pairs = align(chars, list(g.pinyin))
        idx = list(g.index)
        for ci, sj, t in pairs:
            if sj is None: ins.append(dict(file=f, char=chars[ci])); continue
            syl.at[idx[sj], "char"] = chars[ci] if ci is not None else ""
            syl.at[idx[sj], "match"] = t
        vc = pd.Series([t for *_, t in pairs]).value_counts()
        summ.append(dict(file=f, n_syll=len(g), n_chars=len(chars),
                         **{k: int(vc.get(k, 0)) for k in ["toned","toneless","sub","gold_only","asr_only"]},
                         has_digits=bool(re.search(r"[0-9０-９A-Za-z]", text)), asr_text=text))
    syl.to_csv(f"{out}/syllables_chars.csv", index=False)
    pd.DataFrame(ins).to_csv(f"{out}/asr_insertions.csv", index=False)
    s = pd.DataFrame(summ); s.to_csv(f"{out}/align_summary.csv", index=False)
    tot = s[["n_syll","toned","toneless","sub","gold_only","asr_only"]].sum()
    print(f"{len(s)} files aligned"); print((tot/tot.n_syll).round(4).to_string())
    print("files with any non-toned syllable:", int(((s[["toneless","sub","gold_only","asr_only"]].sum(axis=1))>0).sum()))

if __name__ == "__main__":
    main(*sys.argv[1:])
