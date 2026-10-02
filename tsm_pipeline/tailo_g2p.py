"""Tâi-lô (numeric tones) -> phone strings for Montreal Forced Aligner.

Phones are toneless: we only need syllable boundaries from MFA, and the
lexical (citation) tone labels in TAT-MOE do not match surface (sandhi)
tones anyway. Tone is kept in the word label (e.g. "tshia1") so the
word tier of the TextGrid is directly the syllable tier.

Phone set (ASCII, MFA-safe):
  onsets : p ph b m t th n l k kh g ng h ts tsh s j
  vowels : a i u e o oo ir er ee   (+ "N" suffix when nasalised: aN, iN ...)
  codas  : m n ng  p_ t_ k_ (unreleased stops)  q (glottal stop, Tâi-lô -h)
  syllabic nasals: m= ng=
"""
import re

INITIALS = ["tsh", "ts", "ph", "th", "kh", "ng", "p", "b", "m", "t", "n",
            "l", "k", "g", "h", "s", "j"]
VOWELS = ["oo", "ir", "er", "ee", "a", "i", "u", "e", "o"]
CODA_PHONE = {"m": "m", "n": "n", "ng": "ng", "p": "p_", "t": "t_",
              "k": "k_", "h": "q"}
SYL_RE = re.compile(r"^([a-z]+)([0-9])$")


class G2PError(ValueError):
    pass


def split_syllable(syl):
    """'tshiann1' -> ('tshiann', '1')."""
    m = SYL_RE.match(syl.lower())
    if not m:
        raise G2PError(f"not a numeric Tâi-lô syllable: {syl!r}")
    return m.group(1), m.group(2)


def _parse_rime(rest):
    """Return list of phones for the rime, or raise."""
    nasal = False
    coda = None
    core = rest
    if core.endswith("nnh"):
        nasal, coda, core = True, "h", core[:-3]
    elif core.endswith("hnn"):
        nasal, coda, core = True, "h", core[:-3]
    elif core.endswith("nn"):
        nasal, core = True, core[:-2]
    if coda is None:
        for c in ("ng", "m", "n", "p", "t", "k", "h"):
            if core.endswith(c) and len(core) > len(c):
                coda, core = c, core[:-len(c)]
                break
    # syllabic nasals: m, ng (optionally + glottal coda: hmh, ngh)
    if core in ("m", "ng"):
        phones = [core + "="]
        if coda:
            phones.append(CODA_PHONE[coda])
        return phones
    vowels = []
    i = 0
    while i < len(core):
        for v in VOWELS:
            if core.startswith(v, i):
                vowels.append(v)
                i += len(v)
                break
        else:
            raise G2PError(f"unparseable rime {rest!r}")
    if not vowels:
        raise G2PError(f"no vowel in rime {rest!r}")
    # Tâi-lô o before -ng/-k/-m/-p, and nasalised o (onn), is [ɔ]
    if (coda in ("ng", "k", "m", "p") or nasal) and vowels[-1] == "o":
        vowels[-1] = "oo"
    phones = [v + ("N" if nasal else "") for v in vowels]
    if coda:
        phones.append(CODA_PHONE[coda])
    return phones


def syllable_to_phones(syl):
    base, _tone = split_syllable(syl)
    for ini in INITIALS:
        if base.startswith(ini) and len(base) > len(ini):
            try:
                return [ini] + _parse_rime(base[len(ini):])
            except G2PError:
                pass  # e.g. try a shorter / zero initial
    return _parse_rime(base)  # zero initial (also 'm', 'ng', 'ai', ...)


if __name__ == "__main__":
    tests = ["tshia1", "sann1", "tshiann2", "ng5", "nng7", "hng5", "m7",
             "hmh4", "ong5", "kok4", "oo1", "hoo7", "sing5", "tsiunn2",
             "khuainn3", "ir1", "ker3", "ee5", "ngoo2", "gueh8", "hennh4",
             "iam7", "ai3", "ah4", "ooh4", "li2", "ji7", "tsap8"]
    for t in tests:
        print(f"{t:10s} {' '.join(syllable_to_phones(t))}")
