"""
Step 2c (v2): correct Whisper characters with a masked LM (BERT), constrained
by the gold syllables. Replaces the free-generation approach of
04_correct_llm_mlx.py, which could not keep the character count.

Run in macOS Terminal:
    python3 -m pip install torch transformers pypinyin opencc-python-reimplemented pandas
    python3 03_align_asr.py
    python3 04b_correct_mlm.py                       # all recordings
    python3 04b_correct_mlm.py --files 'F01phr00'    # a subset

Method (one recording at a time, whole recording as context, commas inserted
at B4/B5 breaks so the model sees phrase structure):
  1. FILL: positions where Whisper has no character, or a character whose
     reading does not fit the syllable, are masked together; the most confident
     one is filled with the best candidate, then the rest are re-predicted
     (iterative "easy-first" filling).
  2. CHECK: every Whisper character is masked in turn. The model's probability
     for Whisper's character is compared with the best candidate that fits the
     syllable. Whisper's character is replaced only if the candidate is at least
     --ratio times (default 30x) more probable.
  3. Steps 1-2 run twice so that fixes can help each other (油品 helps 儲油).
Candidates for a syllable = characters in the BERT vocabulary with a
pypinyin reading equal to the syllable (tone included). Characters that match
only without tone get a penalty (x --tone-penalty, default 0.05), which keeps
Taiwan readings (儲 chu2, 期 qi2) possible without letting tone-less matches
win easily. Whisper's own char is always a candidate if it fits (see fits()).

Outputs (out/):
  syllables_mlm.csv   syllables_chars.csv + char_final, source (asr / replaced /
                      filled / unfilled), logp_final, logp_asr
  mlm_changes.csv     every replaced/filled position with context, for review
"""
import argparse, json, math, os, re, collections
import pandas as pd
from pypinyin import pinyin, Style
from opencc import OpenCC

HAN = re.compile(r"[㐀-鿿]")
cc = OpenCC("s2tw")
ALLOW = {("she2", "什"), ("mo5", "麼"), ("ze3", "怎")}

# Standard Taiwan readings that differ from pypinyin's (mainland) readings
TW_STANDARD = {"期": "qi2", "危": "wei2", "企": "qi4", "息": "xi2", "惜": "xi2", "擊": "ji2",
    "質": "zhi2", "儲": "chu2", "驟": "zou4", "暫": "zhan4", "綜": "zong4", "寂": "ji2",
    "突": "tu2", "夕": "xi4", "垃": "le4", "圾": "se4", "髮": "fa3", "蝸": "gua1", "波": "po1",
    "播": "bo4", "誰": "shui2", "和": "han4", "亞": "ya3", "識": "shi4", "熟": "shu2",
    "研": "yan2", "伺": "si4", "側": "ce4", "淑": "shu2", "裔": "yi4", "暴": "bao4",
    "曝": "pu4", "蛤": "ge2", "骰": "tou2", "液": "yi4", "微": "wei2", "頗": "po3",
    "癌": "yan2", "迄": "qi4", "卡": "ka3", "框": "kuang1", "混": "hun4", "紡": "fang3"}

_ATTESTED = None
def _attested():
    """How often each reading of a character occurs inside real words in
    pypinyin's phrase dictionary (keyed by Simplified form)."""
    global _ATTESTED
    if _ATTESTED is None:
        from pypinyin import lazy_pinyin
        from pypinyin.phrases_dict import phrases_dict
        _ATTESTED = collections.defaultdict(collections.Counter)
        for ph in phrases_dict:
            pys = lazy_pinyin(ph, style=Style.TONE3, neutral_tone_with_five=True)
            if len(pys) == len(ph):
                for c, py in zip(ph, pys): _ATTESTED[c][py.replace("ü", "v")] += 1
    return _ATTESTED

_t2s = OpenCC("t2s")
_RCACHE = {}
def readings(ch):
    """COMMON readings only: pypinyin's primary reading, readings attested in
    >= 2 dictionary words, the Taiwan standard reading, and 一/不 sandhi forms.
    (pypinyin's full heteronym list includes rare classical readings such as
    以 si4, 費 fu2, 冷 ling2, which let wrong characters through.)"""
    if ch in _RCACHE: return _RCACHE[ch]
    from pypinyin import lazy_pinyin
    r = {lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)[0].replace("ü", "v")}
    att = _attested()
    for key in {ch, _t2s.convert(ch)}:
        r |= {py for py, n in att.get(key, {}).items() if n >= 2}
    if ch in TW_STANDARD: r.add(TW_STANDARD[ch])
    if ch == "一": r |= {"yi1", "yi2", "yi4"}
    if ch == "不": r |= {"bu2", "bu4"}
    _RCACHE[ch] = r
    return r

def norm(s):
    s = s[:-1] if s[-1:].isdigit() else s
    return s.replace("v", "u").replace("ng", "n")

TW_READINGS = set()
def learn_tw_readings(syl):
    x = syl[syl["match"].isin(["sub", "toneless"])]
    n = x.groupby(["pinyin", "char"]).speaker.nunique()
    TW_READINGS.update(n[n >= 3].index)

def asr_fits(ch, syl):
    """Is Whisper's own character acceptable for this syllable?"""
    if not ch or not HAN.fullmatch(ch): return False
    return ((syl, ch) in ALLOW or (syl, ch) in TW_READINGS or syl in readings(ch)
            or norm(syl) in {norm(r) for r in readings(ch)})

class Candidates:
    """syllable -> [(token_id, log_penalty)] from the model vocabulary."""
    def __init__(self, vocab, tone_penalty):
        self.toned, self.toneless = collections.defaultdict(set), collections.defaultdict(set)
        for tok, i in vocab.items():
            if len(tok) == 1 and HAN.fullmatch(tok):
                for r in readings(tok):
                    self.toned[r].add(i); self.toneless[norm(r)].add(i)
        self.lp = math.log(tone_penalty); self.cache = {}
    def get(self, syl):
        if syl not in self.cache:
            t = self.toned.get(syl, set())
            tl = self.toneless.get(norm(syl), set()) - t
            self.cache[syl] = [(i, 0.0) for i in t] + [(i, self.lp) for i in tl]
        return self.cache[syl]

def main():
    import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _freshness import require_fresh; require_fresh("out", [('syllables_chars.csv', 'syllables.csv')])
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out"); ap.add_argument("--files", default="")
    ap.add_argument("--model", default="ckiplab/bert-base-chinese")
    ap.add_argument("--ratio", type=float, default=30.0)
    ap.add_argument("--tone-penalty", type=float, default=0.05)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--restart", action="store_true", help="ignore saved progress and start over")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForMaskedLM.from_pretrained(a.model).to(dev).eval()
    vocab = tok.get_vocab(); MASK, UNK = tok.mask_token_id, tok.unk_token_id
    CLS, SEP, COMMA = tok.cls_token_id, tok.sep_token_id, vocab.get("，", vocab.get(",", UNK))
    cands = Candidates(vocab, a.tone_penalty); inv = {i: t for t, i in vocab.items()}
    print(f"device={dev}; vocabulary Han chars with readings: {sum(len(v) for v in cands.toned.values())}")

    syl = pd.read_csv(f"{a.out}/syllables_chars.csv", keep_default_na=False)
    learn_tw_readings(syl)
    files = [f for f in syl.file.unique() if re.search(a.files, f)][: a.limit]
    groups = dict(tuple(syl.groupby("file", sort=False)))

    # 3) resumable: results are appended to *.partial.csv every 200 recordings, and a
    #    rerun skips recordings already there. Finished output replaces the partial files.
    tag = "" if not a.files and not a.limit else "_subset"
    part_rows, part_ch = f"{a.out}/syllables_mlm{tag}.partial.csv", f"{a.out}/mlm_changes{tag}.partial.csv"
    if a.restart:
        for pth in (part_rows, part_ch):
            if os.path.exists(pth): os.remove(pth)
    done = set(pd.read_csv(part_rows, usecols=["file"]).file) if os.path.exists(part_rows) else set()
    files = [f for f in files if f not in done]
    print(f"{len(files)} recordings to do ({len(done)} already done in {part_rows})")

    def flush(rows, chs):
        if rows:
            pd.DataFrame(rows).to_csv(part_rows, mode="a", header=not os.path.exists(part_rows), index=False)
        if chs:
            pd.DataFrame(chs).to_csv(part_ch, mode="a", header=not os.path.exists(part_ch), index=False)

    # Score only the positions we need: run the encoder, then apply the MLM head to
    # the selected hidden states. (Computing logits for every position x 21k vocab
    # items takes GBs per batch on long recordings and got the process killed.)
    has_head = hasattr(model, "cls") and hasattr(model, "base_model")
    @torch.no_grad()
    def scores_at(x, rows, cols):
        if has_head:
            h = model.base_model(input_ids=x).last_hidden_state[rows, cols]
            return torch.log_softmax(model.cls(h).float(), -1).cpu()
        return torch.log_softmax(model(input_ids=x).logits[rows, cols].float(), -1).cpu()

    @torch.no_grad()
    def logprobs(seqs, mask_pos):
        """seqs: list of id lists (same length); returns log-softmax at mask_pos per seq."""
        x = torch.tensor(seqs, device=dev)
        return scores_at(x, torch.arange(len(seqs), device=dev), torch.tensor(mask_pos, device=dev))

    def best(lp, syl_, extra=None):
        opts = [(lp[i].item() + pen, i) for i, pen in cands.get(syl_)]
        if extra is not None: opts.append((lp[extra].item(), extra))
        return max(opts) if opts else (-1e9, UNK)

    rows_out, changes = [], []
    for fi, f in enumerate(files, 1):
        g = groups[f]
        sylls = list(g.pinyin); brk = list(g.break_after); pw = list(g.pw_idx)
        asr = list(g.char)
        asr_id = [vocab.get(c, UNK) if asr_fits(c, s) else None for c, s in zip(asr, sylls)]
        cur = [i if i is not None else MASK for i in asr_id]
        src = ["asr" if i is not None else "unfilled" for i in asr_id]
        lp_final = [float("nan")] * len(cur); lp_asr = [float("nan")] * len(cur)

        # token layout: [CLS] c1 c2 ... (， after B4/B5 phrase ends) [SEP]
        def layout(chars):
            ids, pos = [CLS], []
            for k, c in enumerate(chars):
                pos.append(len(ids)); ids.append(c)
                end_pw = k == len(chars) - 1 or pw[k + 1] != pw[k]
                if end_pw and brk[k] in ("B4", "B5"): ids.append(COMMA)
            return ids[:511] + [SEP], pos
        if len(layout(cur)[0]) > 512:
            print("  (long recording, truncated context)", f)

        for _ in range(a.passes):
            # 1. easy-first filling of masked positions
            while True:
                todo = [k for k, c in enumerate(cur) if c == MASK]
                if not todo: break
                ids, pos = layout(cur)
                todo = [k for k in todo if pos[k] < 511]
                if not todo: break
                x = torch.tensor([ids], device=dev)
                lg = scores_at(x, torch.zeros(len(todo), dtype=torch.long, device=dev),
                               torch.tensor([pos[k] for k in todo], device=dev))
                scored = [(best(lg[j], sylls[k]), k) for j, k in enumerate(todo)]
                (s, i), k = max(scored)
                cur[k] = i; src[k] = "filled"; lp_final[k] = s
            # 2. check every Whisper character, one mask per copy, batched
            ks = [k for k in range(len(cur)) if src[k] in ("asr", "replaced")]
            for b in range(0, len(ks), a.batch):
                kb = ks[b:b + a.batch]; seqs, mpos = [], []
                for k in kb:
                    c2 = cur.copy(); c2[k] = MASK
                    ids, pos = layout(c2); seqs.append(ids); mpos.append(min(pos[k], 511))
                L = logprobs(seqs, mpos)
                for j, k in enumerate(kb):
                    if asr_id[k] is None: continue
                    la = L[j][asr_id[k]].item(); lp_asr[k] = la
                    s, i = best(L[j], sylls[k])
                    if i != asr_id[k] and s - la > math.log(a.ratio):
                        cur[k] = i; src[k] = "replaced"; lp_final[k] = s
                    elif src[k] == "asr" or i == asr_id[k]:
                        cur[k] = asr_id[k]; src[k] = "asr"; lp_final[k] = la

        final = [inv.get(i, "□") if i not in (MASK, UNK) else "□" for i in cur]
        text = "".join(final)
        fill_rate = sum(x == "filled" for x in src) / len(src)
        # fills that are just spelled-out numbers (Whisper wrote digits) don't count
        NUM = set("零〇一二三四五六七八九十百千萬億兩點之分第")
        text_fill = sum(x == "filled" and f not in NUM for x, f in zip(src, final)) / len(src)
        needs_asr = text_fill > 0.2     # Whisper largely failed: re-transcribe in pieces
        for k, (idx, r) in enumerate(g.iterrows()):
            d = r.to_dict(); d.update(char_final=final[k], source=src[k] if final[k] != "□" else "unfilled",
                                    logp_final=lp_final[k], logp_asr=lp_asr[k],
                                    needs_retranscription=needs_asr)
            rows_out.append(d)
            if src[k] in ("replaced", "filled"):
                changes.append(dict(file=f, syl_idx=r.syl_idx, pinyin=r.pinyin, asr=asr[k],
                                    new=final[k], source=src[k],
                                    gain=None if math.isnan(lp_asr[k]) else round(lp_final[k] - lp_asr[k], 2),
                                    context=text[max(0, k - 8):k] + "[" + final[k] + "]" + text[k + 1:k + 9]))
        if fi % 50 == 0 or fi == len(files):
            print(f"{fi}/{len(files)}  {text[:40]}")
        if fi % 200 == 0 or fi == len(files):
            flush(rows_out, changes); rows_out, changes = [], []
            if dev == "mps": torch.mps.empty_cache()

    flush(rows_out, changes)
    allrows = pd.read_csv(part_rows, keep_default_na=False, low_memory=False)
    allrows.to_csv(f"{a.out}/syllables_mlm{tag}.csv", index=False)
    if os.path.exists(part_ch):
        pd.read_csv(part_ch, keep_default_na=False, low_memory=False).to_csv(f"{a.out}/mlm_changes{tag}.csv", index=False)
    for pth in (part_rows, part_ch):
        if os.path.exists(pth): os.remove(pth)
    print(allrows.file.nunique(), "recordings written")
    s = allrows.source.value_counts()
    print(s.to_string())
    bad = sorted(set(allrows.file[allrows.needs_retranscription.astype(str) == "True"]))
    print(f"{len(bad)} recordings need re-transcription (>30% filled):", *bad[:20], sep="\n  ")
    print(f"changes written to {a.out}/mlm_changes{tag}.csv")

if __name__ == "__main__":
    main()
