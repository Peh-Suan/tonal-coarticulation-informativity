"""Word forms of TAT-MOE utterances: 漢羅 substring + Tâi-lô hyphen group per word."""
import unicodedata

import pandas as pd

from hanlo_units import units


def word_forms(prep, unique=True):
    """-> DataFrame of word tokens (in unique sentences, or in every utterance):
    utt_id, word_idx, sentence text, char span, 漢羅 and Tâi-lô forms, n_syl."""
    syl = pd.read_csv(prep / "syllables_surface.csv",
                      usecols=["utt_id", "syl_idx", "word_idx", "syllable"], keep_default_na=False)
    utt = pd.read_csv(prep / "utterances.csv", usecols=["utt_id", "hanlo", "tailo", "n_syl"],
                      keep_default_na=False, low_memory=False)
    utt["hanlo"] = utt.hanlo.map(lambda s: unicodedata.normalize("NFC", str(s)).strip())
    if unique:
        utt = utt.drop_duplicates(["hanlo", "tailo"])
    syl = syl[syl.utt_id.isin(set(utt.utt_id))]
    by_utt = {u: g for u, g in syl.groupby("utt_id")}
    rows = []
    for u, text, n in zip(utt.utt_id, utt.hanlo, utt.n_syl):
        us = units(text)
        g = by_utt.get(u)
        if g is None or len(us) != n or len(g) != n:
            continue
        g = g.sort_values("syl_idx")
        for w, gw in g.groupby("word_idx", sort=True):
            k0, k1 = int(gw.syl_idx.min()), int(gw.syl_idx.max())
            a, b = us[k0][0], us[k1][1]
            rows.append((u, int(w), text, a, b, text[a:b], "-".join(gw.syllable), k1 - k0 + 1))
    return pd.DataFrame(rows, columns=["utt_id", "word_idx", "text", "start", "end", "hanlo_form",
                                       "tailo_form", "n_syl"])
