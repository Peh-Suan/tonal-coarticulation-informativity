"""
Step 4: contextual surprisal of each word, given the preceding text of the
same recording, from a causal (left-to-right) language model.
Run in macOS Terminal after 05:
    python3 06_surprisal.py                                   # default model
    python3 06_surprisal.py --model Qwen/Qwen2.5-7B --tag _qwen7b

surprisal(word) = -sum_i log2 p(token_i | everything before it), over the
tokens that START inside the word (a token straddling a word boundary is
counted for the word it starts in; with the default character-level model
this never happens). Context is truncated to the last --max-context tokens.
Words containing □ (unrecoverable characters) get NaN.

Default model: ckiplab/gpt2-base-chinese — GPT-2 trained on Traditional
Chinese, one token per character.

Outputs: out/words{tag}.csv with surprisal, n_tokens; out/bigrams{tag}.csv
with word_surprisal merged on; out/char_surprisal{tag}.csv with the surprisal
of every syllable (character) given everything before it (syl_idx = position
in the recording, since every syllable is one character).
"""
import argparse, math
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
    ap.add_argument("--tokenizer", default=None, help="defaults to --model (ckiplab gpt2 needs bert-base-chinese)")
    ap.add_argument("--max-context", type=int, default=512); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BertTokenizerFast
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    if a.tokenizer: tok = AutoTokenizer.from_pretrained(a.tokenizer)
    elif a.model.startswith("ckiplab/gpt2"): tok = BertTokenizerFast.from_pretrained("bert-base-chinese")
    else: tok = AutoTokenizer.from_pretrained(a.model)
    model = load_lm(a.model, torch_dtype=torch.float16 if dev != "cpu" else torch.float32).to(dev).eval()
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    if bos is None: bos = tok.eos_token_id
    print(f"model={a.model} device={dev} bos={tok.convert_ids_to_tokens(bos)}")

    words = pd.read_csv(f"{a.out}/words.csv", keep_default_na=False, low_memory=False)
    words = words.drop(columns=[c for c in ("surprisal", "n_tokens") if c in words])   # rerun-safe
    out, char_rows = [], []
    for fi, (f, g) in enumerate(words.groupby("file", sort=False), 1):
        ws = list(g.word); text = "".join(ws)
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        # char -> word index
        c2w = [wi for wi, w in enumerate(ws) for _ in w]
        # token surprisals (bits). Blocks of targets; each block sees up to
        # --max-context preceding tokens (+ a start-of-text token).
        n, W = len(ids), a.max_context
        surp, b0 = [], 0
        with torch.no_grad():
            while b0 < n:
                b1 = min(n, W if b0 == 0 else b0 + W // 2)
                c0 = max(0, b1 - W)
                seq = [bos] + ids[c0:b1]
                lp = torch.log_softmax(model(torch.tensor([seq], device=dev)).logits[0].float(), -1)
                for t in range(b0, b1):
                    surp.append(-lp[t - c0, ids[t]].item() / math.log(2))
                b0 = b1
        # per-character (= per-syllable) surprisal: each token counted on the char it starts at
        per_c = [0.0] * len(text); ctok = [0] * len(text)
        for (st, en), sv in zip(offs, surp):
            if st < len(text): per_c[st] += sv; ctok[st] += 1
        for k, c in enumerate(text):
            char_rows.append(dict(file=f, syl_idx=k, char=c,
                                  char_surprisal=per_c[k] if ctok[k] and c != "□" else float("nan")))
        per_w = [0.0] * len(ws); ntok = [0] * len(ws)
        for (st, en), sv in zip(offs, surp):
            if st < len(c2w): per_w[c2w[st]] += sv; ntok[c2w[st]] += 1
        for wi, w in enumerate(ws):
            out.append(dict(file=f, word_idx=int(g.word_idx.iloc[wi]),
                            surprisal=float("nan") if "□" in w or ntok[wi] == 0 else per_w[wi],
                            n_tokens=ntok[wi]))
        if fi % 200 == 0: print(f"{fi} recordings")
    pd.DataFrame(char_rows).to_csv(f"{a.out}/char_surprisal{a.tag}.csv", index=False)
    sp = pd.DataFrame(out)
    words = words.merge(sp, on=["file", "word_idx"], how="left")
    words.to_csv(f"{a.out}/words{a.tag}.csv", index=False)
    b = pd.read_csv(f"{a.out}/bigrams.csv", keep_default_na=False, low_memory=False)
    b = b.drop(columns=[c for c in ("word_surprisal", "word_n_tokens") if c in b])   # rerun-safe
    b = b.merge(sp.rename(columns={"surprisal": "word_surprisal", "n_tokens": "word_n_tokens"}),
                on=["file", "word_idx"], how="left")
    b.to_csv(f"{a.out}/bigrams{a.tag}.csv", index=False)
    print(words.surprisal.describe())

if __name__ == "__main__":
    main()
