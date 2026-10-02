"""Taiwan Mandarin readings for the informativity lexicon (TaiwanReadings; MOE is a helper).

pypinyin uses mainland readings (垃圾 la1 ji1, 星期 xing1 qi1, 危險 wei1 xian3 ...), whereas the
COSPRO tone labels, and so the tone pairs in the analysis, follow the Taiwan standard. This module
romanises a Traditional-character word in pypinyin TONE3 style with MOE readings:

  1. the word is an entry of the MOE 重編國語辭典修訂本 (g0v moedict-data dict-revised.json):
     its (first) MOE reading;
  2. otherwise pypinyin, with MOE multi-character entries loaded as phrases (so sub-words such as
     垃圾 in 垃圾桶 get the MOE reading), and then per character: if pypinyin's reading is not one
     of the character's MOE readings (e.g. 期 qi1), it is replaced by the character's first MOE
     reading (qi2); readings MOE also lists are kept (pypinyin's choice among a polyphone's readings).

TaiwanReadings (what informativity_positional.py uses) takes, in order:
  a. COSPRO's own labels: the majority tone sequence of the word in COSPRO (Taiwan speakers;
     lexicon/cospro_word_readings.csv, built with `python moe_readings.py build-cospro`);
     covers 63% of the polysyllabic tokens of the segmented_texts lexicon;
  b. otherwise the MOE word reading, or pypinyin with per-character correction (COSPRO-attested
     character readings first, then MOE; see _char_fix), with two adjustments that match COSPRO's labelling:
     一/不 tone sandhi inside the word (COSPRO labels 一/不 with their surface tones), and MOE
     neutral tones replaced by the character's full tone except for neutral-tone suffixes and
     particles (們麼子的了著過個呢嗎吧啊裡頭); COSPRO labels e.g. 知道 zhi1 dao4, not dao5.
  Check against COSPRO's word readings (token-weighted, exact tone sequence): raw pypinyin 90.8%,
  MOE 90.7%, MOE + 一/不 + neutral adjustment 93.9% (b is only used for words COSPRO lacks).

Data: tc_mi/lexicon/dict-revised.json (https://github.com/g0v/moedict-data, dict-revised.json.xz;
教育部《重編國語辭典修訂本》, CC BY-ND 3.0 TW).
"""
import json, re
from pathlib import Path

from pypinyin import lazy_pinyin, load_phrases_dict, Style
from pypinyin.contrib.tone_convert import to_tone3

HAN = re.compile(r"^[一-鿿]+$")
DEFAULT_MOE = Path(__file__).resolve().parent.parent / "lexicon" / "dict-revised.json"


def _syllables(pinyin_str):
    """MOE pinyin string -> list of TONE3 syllables (drops annotations such as （語音）)."""
    s = re.sub(r"[（(][^）)]*[）)]", " ", pinyin_str)
    s = s.replace("-", " ").replace("，", " ").replace(",", " ")
    out = []
    for tok in s.split():
        tok = re.sub(r"[^a-zA-Zāáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜüńňǹḿ]", "", tok)
        if tok:
            out.append(to_tone3(tok.lower(), v_to_u=False, neutral_tone_with_five=True))
    return out


class MOE:
    def __init__(self, path=DEFAULT_MOE):
        data = json.load(open(path, encoding="utf-8"))
        self.word, self.char = {}, {}
        phrases = {}
        for e in data:
            t = e.get("title", "")
            if not HAN.match(t):
                continue
            readings = []
            for h in e.get("heteronyms", []):
                syl = _syllables(h.get("pinyin", "") or "")
                if len(syl) == len(t) and all(x[-1].isdigit() for x in syl):
                    readings.append(syl)
            if not readings:
                continue
            if len(t) == 1:
                self.char[t] = [r[0] for r in readings]
            else:
                self.word.setdefault(t, readings[0])
                phrases[t] = [[p] for p in lazy_pinyin_tone(readings[0])]
        load_phrases_dict(phrases)

    def romanise(self, w):
        if w in self.word:
            return list(self.word[w]), "moe_word"
        py = lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)
        if len(py) != len(w):
            return py, "pypinyin"
        out, changed = [], False
        for ch, p in zip(w, py):
            opts = self.char.get(ch)
            if opts and p not in opts:
                out.append(opts[0]); changed = True
            else:
                out.append(p)
        return out, ("moe_char" if changed else "pypinyin")


def lazy_pinyin_tone(tone3_list):
    """TONE3 syllables -> tone-mark pinyin, the format load_phrases_dict expects."""
    from pypinyin.contrib.tone_convert import to_tone
    return [to_tone(s.replace("5", "")) if s.endswith("5") else to_tone(s) for s in tone3_list]


# variant characters looked up under the form COSPRO and MOE file the reading under
VARIANTS = {"台": "臺", "着": "著", "裏": "裡", "綫": "線", "麪": "麵", "喫": "吃"}
NEUTRAL_OK = set("們麼子的了著過個呢嗎吧啊裡頭")
DEFAULT_COSPRO = Path(__file__).resolve().parent.parent / "lexicon" / "cospro_word_readings.csv"


def yibu_sandhi(r):
    """一/不 tone sandhi inside a word, as in COSPRO's labels."""
    r = list(r)
    for i in range(len(r) - 1):
        nt = r[i + 1][-1]
        if r[i] == "bu4" and nt == "4":
            r[i] = "bu2"
        elif r[i] == "yi1" and nt in "123":
            r[i] = "yi4"
        elif r[i] == "yi1" and nt == "4":
            r[i] = "yi2"
    return r


class TaiwanReadings:
    def __init__(self, moe_path=DEFAULT_MOE, cospro_path=DEFAULT_COSPRO):
        import pandas as pd
        self.moe = MOE(moe_path)
        self.cospro = {}
        if Path(cospro_path).exists():
            cr = pd.read_csv(cospro_path)
            self.cospro = {w: r.split() for w, r in zip(cr.word, cr.reading)}
        else:
            print(f"note: {cospro_path} missing; MOE readings only")
        self.char_att = {}
        chars_csv = Path(cospro_path).resolve().parent.parent / "cospro_pipeline" / "out" / "syllables_chars.csv"
        if Path(chars_csv).exists():
            from collections import Counter
            c = pd.read_csv(chars_csv, usecols=["char", "pinyin"], keep_default_na=False)
            for ch, py in zip(c.char, c.pinyin):
                if len(ch) == 1 and py and py[-1].isdigit():
                    self.char_att.setdefault(ch, Counter())[py] += 1

    def _full_neutral(self, w, r):
        out = []
        for ch, s in zip(w, r):
            if s.endswith("5") and ch not in NEUTRAL_OK:
                opts = [o for o in self.moe.char.get(ch, []) if not o.endswith("5")]
                p = lazy_pinyin(ch, style=Style.TONE3, neutral_tone_with_five=True)[0]
                s = p if (not p.endswith("5") and (not opts or p in opts)) else (opts[0] if opts else s)
            out.append(s)
        return out

    def _char_fix(self, w):
        """pypinyin (with MOE phrases), then per character: keep pypinyin's reading if COSPRO speakers
        use it (>= 10% of that character's labelled tokens) or, for characters COSPRO lacks, if MOE
        lists it; otherwise the character's COSPRO majority reading (>= 5 tokens), else its first MOE
        reading. (COSPRO first, because MOE files some characters under a variant: 台 tāi vs 臺 tái.)"""
        py = lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True)
        if len(py) != len(w):
            return py, "pypinyin"
        out, changed = [], False
        for ch, p in zip(w, py):
            key = VARIANTS.get(ch, ch)
            att = self.char_att.get(key)
            opts = self.moe.char.get(key)
            if att and sum(att.values()) >= 5:
                tot = sum(att.values())
                if att.get(p, 0) / tot >= 0.10:
                    out.append(p); continue
                q = att.most_common(1)[0][0]
            elif opts:
                if p in opts:
                    out.append(p); continue
                q = opts[0]
            else:
                out.append(p); continue
            out.append(q); changed = changed or q != p
        return out, ("char_fix" if changed else "pypinyin")

    def romanise(self, w):
        if w in self.cospro and len(self.cospro[w]) == len(w):
            return list(self.cospro[w]), "cospro"
        if w in self.moe.word:
            r, src = list(self.moe.word[w]), "moe_word"
        else:
            r, src = self._char_fix(w)
        if len(r) == len(w):
            r = yibu_sandhi(self._full_neutral(w, r))
        return r, src


def build_cospro(processed_csv, out_csv):
    """majority COSPRO-labelled tone sequence per (Han-only) word, from cospro processed_data.csv"""
    import pandas as pd
    from collections import Counter, defaultdict
    d = pd.read_csv(processed_csv, usecols=["position", "file", "word_idx", "word", "pos_in_word",
                                            "syllable1", "syllable2"], low_memory=False)
    d = d[(d.position == "carryover") & d.word.astype(str).str.match(r"^[\u4e00-\u9fff]+$")]
    tok = defaultdict(dict)
    for f, wi, w, p, s1, s2 in zip(d.file, d.word_idx, d.word, d.pos_in_word, d.syllable1, d.syllable2):
        t = tok[(f, wi, w)]; t[int(p)] = s1; t[int(p) + 1] = s2
    reads = defaultdict(Counter)
    for (f, wi, w), t in tok.items():
        if len(t) == len(w) and sorted(t) == list(range(len(w))):
            reads[w][tuple(t[i] for i in range(len(w)))] += 1
    rows = [(w, " ".join(c.most_common(1)[0][0]), c.most_common(1)[0][1], sum(c.values())) for w, c in reads.items()]
    pd.DataFrame(rows, columns=["word", "reading", "n_majority", "n_tokens"]).to_csv(out_csv, index=False)
    print(f"{out_csv}: {len(rows)} words")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "build-cospro":
        here = Path(__file__).resolve().parent.parent
        build_cospro(here / "cospro_pipeline" / "out" / "processed_data.csv", DEFAULT_COSPRO)
