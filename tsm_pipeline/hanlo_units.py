"""Split a 漢羅台文 string into syllable units with character spans.

Each Han character is one syllable; each romanised syllable (Tâi-lô/POJ,
separated by '-' or spaces) is one syllable. Punctuation, spaces and hyphens
are skipped. If the unit count equals the number of Tâi-lô syllables of the
utterance, unit k <-> syllable k.
"""
import re

HAN = r"[㐀-䶿一-鿿豈-﫿\U00020000-\U0002ebef]"
ROM = r"[A-Za-zÀ-ɏ̀-ͯḀ-ỿⁿ͘ᴺ]+"
UNIT_RE = re.compile(f"{HAN}|{ROM}")


def units(text):
    """-> list of (start, end, unit_text)."""
    return [(m.start(), m.end(), m.group()) for m in UNIT_RE.finditer(text)]


if __name__ == "__main__":
    for s in ["蔣渭水高速公路佇二khòng khòng六年中通車，", "溫度二十九至三十三度，落雨機率六十pha",
              "好--矣", "Formosa 真súi"]:
        print(s, [u[2] for u in units(s)])
