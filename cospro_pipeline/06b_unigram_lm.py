"""
Step 4b: unigram (context-free) surprisal of every word type from the SAME
language model as the contextual surprisal (06_surprisal.py), so that

    context_gain = unigram_surprisal - word_surprisal
                 = log2 p(w | context) / p(w)          (pointwise mutual information)

is computed within one model.

Two estimates of p(w), both in bits:
  * unigram_lm_surprisal  (recommended) = -log2 of the MARGINAL probability
        p(w) = mean over K sampled contexts c of p(w | c).
    Contexts are the --ctx-len characters preceding a random word boundary in a
    random recording of the corpus (at a recording start: just the start token),
    so p(w) is the model's average probability of the word starting at a word
    boundary. The same K contexts are used for every word type.
  * null_context_surprisal = -log2 p(w | start-of-text token only). Cheaper, but it is
    the probability of the word at the START of a text, which favours text-initial
    words (但是, 我們). Kept for comparison.

Run in macOS Terminal after 06_surprisal.py (same --model as there):
    python 06b_unigram_lm.py                       # K = 200 contexts
    python 06b_unigram_lm.py --k 400               # more precise, twice as slow
Then rerun 08_build_processed.py: when out/unigram_lm{tag}.csv exists it replaces the
frequency-corpus unigram surprisal (kept as *_corpus columns).

Output: out/unigram_lm{tag}.csv  word, unigram_lm_surprisal, unigram_lm_se (Monte-Carlo
standard error of the estimate, bits), null_context_surprisal, n_tokens, k
"""
import argparse, math, random
import numpy as np
import pandas as pd

def load_lm(name, **kw):
    """Prefer safetensors weights: loading .bin weights needs torch >= 2.6 in
    recent transformers versions."""
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(name, use_safetensors=True, **kw)
    except Exception:
        try:
            return AutoModelForCausalLM.from_pretrained(name, **kw)
        except ValueError as e:
            if "torch.load" in str(e):
                raise SystemExit("This environment's PyTorch is too old to load this model's .bin weights. "
                                 "Run in the environment used before (conda activate default_env), "
                                 "or upgrade: pip install -U 'torch>=2.6'") from e
            raise

def main():
    import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from _freshness import require_fresh; require_fresh("out", [('syllables_mlm.csv', 'syllables_chars.csv'), ('words.csv', 'syllables_mlm.csv')])
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out"); ap.add_argument("--model", default="ckiplab/gpt2-base-chinese")
    ap.add_argument("--tokenizer", default=None); ap.add_argument("--tag", default="")
    ap.add_argument("--k", type=int, default=200, help="number of sampled contexts")
    ap.add_argument("--ctx-len", type=int, default=32, help="context length in characters")
    ap.add_argument("--batch", type=int, default=128); ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizerFast
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if a.tokenizer: tok = AutoTokenizer.from_pretrained(a.tokenizer)
    elif a.model.startswith("ckiplab/gpt2"): tok = BertTokenizerFast.from_pretrained("bert-base-chinese")
    else: tok = AutoTokenizer.from_pretrained(a.model)
    model = load_lm(a.model).to(dev).eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    if bos is None: bos = tok.eos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0

    words = pd.read_csv(f"{a.out}/words.csv", keep_default_na=False, low_memory=False)
    types = sorted({w for w in words.word if w and "□" not in w})
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
    type_ids = [enc(w) for w in types]
    print(f"model={a.model} device={dev}: {len(types)} word types, K={a.k} contexts of {a.ctx_len} chars")

    # --- sample K contexts at word boundaries of random recordings ---------------
    rng = random.Random(a.seed)
    recs = [(f, list(g.word)) for f, g in words.groupby("file", sort=False)]
    contexts = []
    for _ in range(a.k):
        f, ws = rng.choice(recs)
        cut = rng.randrange(len(ws))                       # boundary before word index `cut`
        prefix = "".join(ws[:cut]).replace("□", "")
        contexts.append(prefix[-a.ctx_len:])
    n_start = sum(c == "" for c in contexts)
    print(f"  {n_start} of the {a.k} contexts are recording starts")

    @torch.no_grad()
    def score(prefix_ids, word_ids_list):
        """log2 p(word | [bos] + prefix) for many words sharing one prefix."""
        out = np.empty(len(word_ids_list))
        P = [bos] + prefix_ids
        for s in range(0, len(word_ids_list), a.batch):
            chunk = word_ids_list[s:s + a.batch]
            L = len(P) + max(len(w) for w in chunk)
            x = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for i, w in enumerate(chunk):
                seq = P + w; x[i, :len(seq)] = torch.tensor(seq); att[i, :len(seq)] = 1
            # Keep only the logits that predict the word's tokens (positions len(P)-1 .. L-2)
            # and gather on the device: the full B x L x vocab log-softmax on the CPU
            # used several GB per batch and got the process killed.
            x, att = x.to(dev), att.to(dev)
            logits = model(input_ids=x, attention_mask=att).logits[:, len(P) - 1:L - 1, :]
            lp = torch.log_softmax(logits.float(), -1)                     # B x wmax x vocab
            tgt = x[:, len(P):L]                                           # B x wmax
            g = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) * att[:, len(P):L]
            out[s:s + len(chunk)] = (g.sum(1) / math.log(2)).cpu().numpy()
            del logits, lp, g
        if dev == "mps": torch.mps.empty_cache()
        return out

    null = score([], type_ids)                          # start-of-text only
    logs = np.empty((a.k, len(types)))                  # log2 p(w | c_k)
    try:
        from tqdm import tqdm
        ctx_iter = tqdm(enumerate(contexts), total=a.k, desc="contexts", unit="ctx")
    except ImportError:                                 # no tqdm: fall back to plain prints
        ctx_iter = enumerate(contexts)
    for k, c in ctx_iter:
        logs[k] = score(enc(c), type_ids)
        if not hasattr(ctx_iter, "update") and (k + 1) % 20 == 0: print(f"  {k + 1}/{a.k} contexts")
    # marginal: log2 mean_k 2^{logs}, computed stably
    mx = logs.max(0)
    p_rel = np.exp2(logs - mx)                          # K x types
    marg = mx + np.log2(p_rel.mean(0))
    # delta-method standard error of the Monte-Carlo estimate, in bits
    se = p_rel.std(0, ddof=1) / math.sqrt(a.k) / p_rel.mean(0) / math.log(2)

    res = pd.DataFrame(dict(word=types, unigram_lm_surprisal=-marg, unigram_lm_se=se,
                            null_context_surprisal=-null, n_tokens=[len(t) for t in type_ids], k=a.k))
    res.to_csv(f"{a.out}/unigram_lm{a.tag}.csv", index=False)
    print(res[["unigram_lm_surprisal", "unigram_lm_se", "null_context_surprisal"]].describe().round(2))
    print("corr(marginal, null-context) =", round(res.unigram_lm_surprisal.corr(res.null_context_surprisal), 3))
    try:
        fq = pd.read_csv(f"{a.out}/word_freq.csv", keep_default_na=False)
        N, V = fq["count"].sum(), len(fq)
        cnt = res.word.map(fq.set_index("word")["count"]).fillna(0)
        corpus = -np.log2((cnt + 1) / (N + V))
        print("corr(marginal LM, frequency-corpus unigram) =", round(res.unigram_lm_surprisal.corr(corpus), 3))
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    main()
