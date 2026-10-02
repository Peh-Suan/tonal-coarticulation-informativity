"""Unigram (context-free) surprisal of every word type from the SAME LM as
06_surprisal_llm.py (Taigi-Llama-2-7B), mirroring cospro_pipeline/06b_unigram_lm.py:

    context_gain = unigram_surprisal - word_surprisal = log2 p(w | context) / p(w)

Word types are 漢羅 word forms: the substring of the sentence's 漢羅台文 text
that spans the word's syllable units (Tâi-lô word = hyphen group). Only words
of 2+ syllables are scored by default (--all-words scores monosyllables too).

Estimates (bits):
  unigram_lm_surprisal = -log2 of the MARGINAL p(w) = mean over K sampled contexts c of
      p(w | c). A context is the --ctx-len characters of 漢羅 text before a random
      word boundary in a random unique sentence. The same K contexts are used for
      every word type.
  unigram_lm_se        Monte-Carlo standard error of that estimate (bits)
  null_context_surprisal = -log2 p(w | start token only)
p(w | c) is computed as in 06: context + word are tokenised together and the word's value
is the sum of the surprisal of its tokens (tokens on punctuation/space only are dropped).

Speed: each context is run through the model ONCE, and many words are packed after it
in one sequence with a block attention mask (each word sees the context and itself
only, with its own position ids), so the context isn't recomputed for every word. At
start-up a self-test compares packed and one-by-one scores; if they differ (e.g. an
old transformers version ignores the custom mask) the script stops -- rerun with
--no-pack. Pairs whose tokenisation crosses the context/word boundary are scored
one by one as in the first version (tokens shared over syllable units).

Checkpointing: results are appended to <prep>/unigram_lm_ckpt.csv after every
context; rerunning the same command skips finished contexts (--fresh starts over).

Also writes the corpus frequency of each Tâi-lô word form in the unique TAT sentences
(word_freq.csv, for unigram_surprisal_corpus / log_word_freq in 08).

Output: <prep>/unigram_lm.csv, <prep>/word_freq.csv
Run in macOS Terminal (same env as 06):
  python 07_unigram_lm.py --prep ~/tsm_mfa/all_condenser --k 32 --limit 200   # timing check (--fresh after)
  caffeinate -i python 07_unigram_lm.py --prep ~/tsm_mfa/all_condenser --k 32 --fresh
"""
import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from hanlo_units import UNIT_RE, units
from tsm_words import word_forms

LN2 = math.log(2)
HAN_LO, HAN_HI = "㐀", "\U0002ebef"


def join(c, w):
    """context + separator: a space if both sides are romanised (else they'd merge)."""
    if c and re.match(UNIT_RE, c[-1]) and re.match(UNIT_RE, w[0]) \
            and not (HAN_LO <= c[-1] <= HAN_HI) and not (HAN_LO <= w[0] <= HAN_HI):
        return c + " "
    return c


def score_naive(texts, spans, tok, model, device):
    """bits of the word at char span (a, b) in each text, tokens shared over units as in 06."""
    import torch
    enc = tok(texts, return_offsets_mapping=True, add_special_tokens=True,
              padding=True, return_tensors="pt")
    offsets = enc.pop("offset_mapping")
    ids = enc["input_ids"].to(device)
    att = enc["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=att).logits.float()
    lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).cpu().numpy()
    specials = set(tok.all_special_ids)
    out = []
    for i, (text, (a, b)) in enumerate(zip(texts, spans)):
        us = units(text)
        inw = [a <= s and e <= b for s, e, _ in us]
        bits = 0.0
        for t in range(1, ids.shape[1]):
            if att[i, t] == 0 or int(ids[i, t]) in specials:
                continue
            s, e = offsets[i, t].tolist()
            hit = [k for k, (us_, ue, _) in enumerate(us) if us_ < e and ue > s]
            if hit:
                bits += (-lp[i, t - 1] / LN2) * sum(inw[k] for k in hit) / len(hit)
        out.append(bits)
    return out


def split_tokens(c, words, tok):
    """Tokenise context + each word jointly. -> (ctx_ids, {word: token ids of the word}, naive list).
    A word goes to the naive list when the joint tokenisation doesn't start with ctx_ids."""
    ctx_ids = tok(c, add_special_tokens=True)["input_ids"] if c else [tok.bos_token_id]
    texts = [join(c, w) + w for w in words]
    enc = tok(texts, add_special_tokens=True, return_offsets_mapping=True)
    good, naive = {}, []
    n = len(ctx_ids)
    for w, t, ids, offs in zip(words, texts, enc["input_ids"], enc["offset_mapping"]):
        if ids[:n] != ctx_ids or len(ids) == n:
            naive.append(w)
            continue
        a = len(t) - len(w)
        us = [(s, e) for s, e, _ in units(t) if s >= a]
        keep = [any(s < ue and e > us_ for us_, ue in us) for s, e in offs[n:]]
        good[w] = (ids[n:], keep)
    return ctx_ids, good, naive


def score_packed(ctx_ids, items, model, device, dtype, max_len):
    """items: list of (word, ids, keep). Packs words after the context. -> {word: bits}."""
    import torch
    Lc = len(ctx_ids)
    seqs, cur, cur_len = [], [], Lc
    for it in items:
        if cur and cur_len + len(it[1]) > max_len:
            seqs.append(cur)
            cur, cur_len = [], Lc
        cur.append(it)
        cur_len += len(it[1])
    if cur:
        seqs.append(cur)
    L = max(Lc + sum(len(it[1]) for it in s) for s in seqs)
    B = len(seqs)
    ids = torch.full((B, L), int(ctx_ids[0]), dtype=torch.long)
    pos = torch.zeros((B, L), dtype=torch.long)
    allow = torch.zeros((B, L, L), dtype=torch.bool)
    tri = torch.tril(torch.ones(Lc, Lc, dtype=torch.bool))
    pred = []                                         # (b, predictor idx, target idx, word, keep)
    for b, s in enumerate(seqs):
        ids[b, :Lc] = torch.tensor(ctx_ids)
        pos[b, :Lc] = torch.arange(Lc)
        allow[b, :Lc, :Lc] = tri
        p = Lc
        for w, wid, keep in s:
            n = len(wid)
            ids[b, p:p + n] = torch.tensor(wid)
            pos[b, p:p + n] = torch.arange(Lc, Lc + n)
            allow[b, p:p + n, :Lc] = True
            allow[b, p:p + n, p:p + n] = torch.tril(torch.ones(n, n, dtype=torch.bool))
            for j in range(n):
                pred.append((b, Lc - 1 if j == 0 else p + j - 1, p + j, w, keep[j]))
            p += n
        for q in range(p, L):                           # padding rows: attend to self only
            allow[b, q, q] = True
    mask = torch.zeros((B, 1, L, L), dtype=dtype)
    mask.masked_fill_(~allow.unsqueeze(1), torch.finfo(dtype).min)
    with torch.no_grad():
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device),
                       position_ids=pos.to(device)).logits
    bi = torch.tensor([x[0] for x in pred])
    pi = torch.tensor([x[1] for x in pred])
    ti = torch.tensor([x[2] for x in pred])
    rows = logits[bi.to(device), pi.to(device)].float()
    lp = torch.log_softmax(rows, -1).gather(-1, ids[bi, ti].to(device).unsqueeze(-1)).squeeze(-1).cpu().numpy()
    out = {}
    for (b, _, _, w, keep), v in zip(pred, lp):
        out[w] = out.get(w, 0.0) + (-v / LN2 if keep else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--model", default="Bohanlu/Taigi-Llama-2-7B")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--k", type=int, default=32, help="number of sampled contexts")
    ap.add_argument("--ctx-len", type=int, default=16, help="context length in characters")
    ap.add_argument("--pack-len", type=int, default=256, help="max tokens per packed sequence")
    ap.add_argument("--seqs-per-batch", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32, help="batch size for one-by-one scoring")
    ap.add_argument("--no-pack", action="store_true", help="score every pair one by one (slow)")
    ap.add_argument("--fresh", action="store_true", help="ignore the checkpoint")
    ap.add_argument("--all-words", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N word types (timing check)")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    prep = Path(os.path.expanduser(a.prep))

    wt = word_forms(prep, unique=True)
    freq = wt.tailo_form.value_counts().rename_axis("tailo_form").reset_index(name="count")
    freq.to_csv(prep / "word_freq.csv", index=False)
    print(f"{len(wt)} word tokens in unique sentences, {len(freq)} Tâi-lô word types -> word_freq.csv")

    types = wt if a.all_words else wt[wt.n_syl > 1]
    types = (types.groupby("hanlo_form").agg(tailo_form=("tailo_form", lambda s: s.mode().iat[0]),
                                             n_syl=("n_syl", "first"), tokens=("utt_id", "size"))
             .reset_index().sort_values(["tokens", "hanlo_form"], ascending=[False, True]))
    if a.limit:
        types = types.head(a.limit)
    words = list(types.hanlo_form)

    bounds = wt.sample(n=a.k, random_state=a.seed)
    contexts = [""] + [t[max(0, s - a.ctx_len): s] for t, s in zip(bounds.text, bounds.start)]
    # contexts[0] = null context (start token only); contexts[1:] = the K sampled contexts

    # --- checkpoint -------------------------------------------------------------
    ck, meta_f = prep / "unigram_lm_ckpt.csv", prep / "unigram_lm_ckpt.json"
    meta = dict(model=a.model, k=a.k, ctx_len=a.ctx_len, seed=a.seed, n_words=len(words),
                contexts=contexts, words_hash=hashlib.md5('\n'.join(words).encode()).hexdigest())
    done = {}
    if not a.fresh and ck.exists() and meta_f.exists():
        old = json.loads(meta_f.read_text(encoding="utf-8"))
        if all(old.get(k) == v for k, v in meta.items()):
            prev = pd.read_csv(ck, keep_default_na=False)
            for ci, g in prev.groupby("ctx_idx"):
                if len(g) == len(words):
                    done[int(ci)] = dict(zip(g.hanlo_form, g.bits))
            print(f"checkpoint: {len(done)}/{len(contexts)} contexts already done")
        else:
            print("checkpoint settings differ -> starting over")
    if not done:
        ck.unlink(missing_ok=True)
        meta_f.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(columns=["ctx_idx", "hanlo_form", "bits"]).to_csv(ck, index=False)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = a.device
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, a.dtype)
    tok = AutoTokenizer.from_pretrained(a.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.unk_token or tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(device).eval()
    print(f"{len(words)} word types x {len(contexts)} contexts (1 null + {a.k} sampled) on {device}")

    def naive(c, ws):
        res = {}
        for i in range(0, len(ws), a.batch_size):
            batch = ws[i: i + a.batch_size]
            cs = [join(c, w) for w in batch]
            bits = score_naive([x + w for x, w in zip(cs, batch)],
                               [(len(x), len(x) + len(w)) for x, w in zip(cs, batch)], tok, model, device)
            res.update(zip(batch, bits))
        return res

    def packed(c, ws):
        ctx_ids, good, nv = split_tokens(c, ws, tok)
        items = sorted(((w, *good[w]) for w in good), key=lambda it: len(it[1]))
        res = {}
        per = max(1, a.seqs_per_batch)
        # group items into packed sequences, then run seqs_per_batch sequences at a time
        chunk, chunks, cur_len = [], [], len(ctx_ids)
        for it in items:
            if chunk and cur_len + len(it[1]) > a.pack_len:
                chunks.append(chunk)
                chunk, cur_len = [], len(ctx_ids)
            chunk.append(it)
            cur_len += len(it[1])
        if chunk:
            chunks.append(chunk)
        for i in range(0, len(chunks), per):
            flat = [it for ch in chunks[i: i + per] for it in ch]
            res.update(score_packed(ctx_ids, flat, model, device, dtype,
                                    max(a.pack_len, len(ctx_ids) + max(len(it[1]) for it in flat))))
        if nv:
            res.update(naive(c, nv))
        return res, len(nv)

    if not a.no_pack:          # self-test: packed vs one-by-one on one context
        c = contexts[1] if len(contexts) > 1 else ""
        test = words[:48]
        p, _ = packed(c, test)
        n = naive(c, test)
        diff = max(abs(p[w] - n[w]) for w in test)
        print(f"self-test: max |packed - one-by-one| = {diff:.4f} bits over {len(test)} words")
        if diff > 0.05 * max(1.0, max(abs(n[w]) for w in test) / 10):
            raise SystemExit("packed scoring doesn't match (custom attention masks unsupported?); rerun with --no-pack")

    t0 = time.time()
    todo = [ci for ci in range(len(contexts)) if ci not in done]
    for j, ci in enumerate(todo, 1):
        c = contexts[ci]
        if a.no_pack:
            res, nnaive = naive(c, words), len(words)
        else:
            res, nnaive = packed(c, words)
        pd.DataFrame({"ctx_idx": ci, "hanlo_form": words, "bits": [res[w] for w in words]}) \
            .to_csv(ck, mode="a", header=False, index=False)
        done[ci] = res
        el = time.time() - t0
        print(f"  context {ci} done ({j}/{len(todo)}; {nnaive} words scored one by one), "
              f"{el / 60:.1f} min, ~{el / j * (len(todo) - j) / 60:.0f} min left", flush=True)

    out = []
    for w, tf, n, ntok in zip(types.hanlo_form, types.tailo_form, types.n_syl, types.tokens):
        bits = np.array([done[ci][w] for ci in range(1, len(contexts))])
        p = np.exp2(-bits)
        pm = p.mean()
        se = p.std(ddof=1) / math.sqrt(len(p)) / (pm * LN2) if pm > 0 and len(p) > 1 else np.nan
        out.append((w, tf, n, ntok, -math.log2(pm) if pm > 0 else np.inf, se, done[0][w], a.k))
    out = pd.DataFrame(out, columns=["hanlo_form", "tailo_form", "n_syl", "tokens_in_corpus",
                                     "unigram_lm_surprisal", "unigram_lm_se",
                                     "null_context_surprisal", "k"])
    out.to_csv(prep / "unigram_lm.csv", index=False)
    print(f"-> {prep / 'unigram_lm.csv'}; median unigram {out.unigram_lm_surprisal.median():.2f} bits, "
          f"median SE {out.unigram_lm_se.median():.2f}")


if __name__ == "__main__":
    main()
