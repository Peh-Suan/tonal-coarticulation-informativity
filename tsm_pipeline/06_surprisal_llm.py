"""Contextual surprisal for TAT-MOE syllables and words from a causal LM
(default: Bohanlu/Taigi-Llama-2-7B, a Taiwanese Hokkien Llama-2).

Each unique sentence is scored once from its 漢羅台文 text (Unicode NFC), with
only the sentence itself as context (like the COSPRO step).

Token -> syllable mapping:
  * hanlo_units.py splits the 漢羅 string into syllable units (one per Han
    character or romanised syllable). If the unit count differs from the
    number of Tâi-lô syllables (about 0.4% of sentences, mostly Mandarin words
    in 『』 quotes), the sentence is skipped.
  * each token's surprisal (-log2 p) goes to the units its character span
    overlaps. A token covering several units is split equally among them
    (multi_unit_token = 1); several tokens on one unit (romanised
    syllables, byte-fallback characters) are summed. Tokens on punctuation or
    whitespace only are dropped.
  * word_surprisal = sum of syllable surprisals over the Tâi-lô word
    (hyphen group, from syllables_surface.csv). token_crosses_word = 1 when a
    token spans a word boundary, i.e. the word's value includes an equal-split
    share of a token it only partly owns.

Output: <prep>/syllable_surprisal.csv  (utt_id, syl_idx, unit, syl_surprisal,
        word_surprisal, n_tokens, multi_unit_token, token_crosses_word, word_len)

Run in macOS Terminal (needs torch, transformers; ~14 GB download; MPS fp16):
  python 06_surprisal_llm.py --prep ~/tsm_mfa/all_condenser --limit 200   # quick check
  python 06_surprisal_llm.py --prep ~/tsm_mfa/all_condenser
"""
import argparse
import math
import os
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from hanlo_units import units

LN2 = math.log(2)


def pick_device(name):
    import torch
    if name != "auto":
        return name
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def score_batch(texts, tok, model, device):
    """-> list of (offsets, surprisal_bits) per text, special tokens removed."""
    import torch
    enc = tok(texts, return_offsets_mapping=True, add_special_tokens=True,
              padding=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    ids = enc["input_ids"].to(device)
    att = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=att).logits.float()
    logp = torch.log_softmax(logits[:, :-1], dim=-1)
    tgt = ids[:, 1:]
    lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).cpu().numpy()
    specials = set(tok.all_special_ids)
    out = []
    for b in range(len(texts)):
        res = []
        for i in range(1, ids.shape[1]):          # token i predicted from < i
            if att[b, i] == 0 or int(ids[b, i]) in specials:
                continue
            s, e = offsets[b, i].tolist()
            res.append((s, e, -lp[b, i - 1] / LN2))
        out.append(res)
    return out


def assign(text, n_syl, toks, word_of):
    us = units(text)
    if len(us) != n_syl:
        return None
    sur = np.zeros(n_syl)
    ntok = np.zeros(n_syl, int)
    multi = np.zeros(n_syl, int)
    cross = np.zeros(n_syl, int)
    for s, e, bits in toks:
        hit = [k for k, (a, b, _) in enumerate(us) if a < e and b > s]
        if not hit:
            continue
        for k in hit:
            sur[k] += bits / len(hit)
            ntok[k] += 1
        if len(hit) > 1:
            multi[hit] = 1
            if len({word_of[k] for k in hit}) > 1:
                cross[hit] = 1
    return us, sur, ntok, multi, cross


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--model", default="Bohanlu/Taigi-Llama-2-7B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, help="only the first N unique sentences (testing)")
    ap.add_argument("--out", default="syllable_surprisal.csv")
    a = ap.parse_args()
    prep = Path(os.path.expanduser(a.prep))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = pick_device(a.device)
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    if not tok.is_fast:
        raise SystemExit("need a fast tokenizer (offset mapping)")
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(device).eval()
    print(f"model {a.model} on {device} ({a.dtype})")

    syl = pd.read_csv(prep / "syllables_surface.csv", usecols=["utt_id", "syl_idx", "word_idx", "syllable"],
                      keep_default_na=False)
    utt = pd.read_csv(prep / "utterances.csv", usecols=["utt_id", "hanlo", "tailo", "n_syl"],
                      keep_default_na=False, low_memory=False)
    utt["hanlo"] = utt.hanlo.map(lambda s: unicodedata.normalize("NFC", str(s)).strip())
    utt["sent_id"] = utt.groupby(["hanlo", "tailo"]).ngroup()
    sents = utt.drop_duplicates("sent_id").set_index("sent_id")
    rep = sents.utt_id.to_dict()                  # a representative utterance per sentence
    words = syl.groupby("utt_id").word_idx.apply(list).to_dict()
    order = sorted(sents.index, key=lambda i: len(sents.at[i, "hanlo"]))
    if a.limit:
        order = order[: a.limit]
    print(f"{len(utt)} utterances, {len(sents)} unique sentences, scoring {len(order)}")

    rows, skipped, t0 = [], 0, time.time()
    for bstart in range(0, len(order), a.batch_size):
        batch = order[bstart: bstart + a.batch_size]
        texts = [sents.at[i, "hanlo"] for i in batch]
        scored = score_batch(texts, tok, model, device)
        for sid, text, toks in zip(batch, texts, scored):
            word_of = words.get(rep[sid])
            n = int(sents.at[sid, "n_syl"])
            r = assign(text, n, toks, word_of) if word_of and len(word_of) == n else None
            if r is None:
                skipped += 1
                continue
            us, sur, ntok, multi, cross = r
            for k in range(n):
                rows.append((sid, k, us[k][2], sur[k], ntok[k], multi[k], cross[k], word_of[k]))
        done = bstart + len(batch)
        if done % (a.batch_size * 50) < a.batch_size or done == len(order):
            el = time.time() - t0
            print(f"  {done}/{len(order)} sentences, {el / 60:.1f} min, "
                  f"~{el / done * (len(order) - done) / 60:.0f} min left")

    s = pd.DataFrame(rows, columns=["sent_id", "syl_idx", "unit", "syl_surprisal", "n_tokens",
                                    "multi_unit_token", "token_crosses_word", "word_idx"])
    g = s.groupby(["sent_id", "word_idx"])
    s["word_surprisal"] = g.syl_surprisal.transform("sum")
    s["word_len"] = g.syl_surprisal.transform("size")
    s["token_crosses_word"] = g.token_crosses_word.transform("max")
    out = utt[["utt_id", "sent_id"]].merge(s.drop(columns="word_idx"), on="sent_id")
    out.to_csv(prep / a.out, index=False)
    print(f"-> {prep / a.out}: {len(out)} syllable rows; {skipped} sentences skipped "
          f"(unit/syllable mismatch); multi-unit tokens {s.multi_unit_token.mean():.1%}, "
          f"tokens crossing words {s.token_crosses_word.mean():.1%} of syllables")


if __name__ == "__main__":
    main()
