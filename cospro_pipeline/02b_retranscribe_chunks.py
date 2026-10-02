"""
Step 2a-bis: re-transcribe recordings where Whisper failed (hallucinated,
stopped early), by cutting the audio at the pauses in the gold TextGrid
("sil" intervals >= 0.25 s) and transcribing each piece separately.
Run in macOS Terminal after 03_align_asr.py:

    python3 02b_retranscribe_chunks.py <path/to/COSPRO>

Which recordings: those whose alignment is poor, i.e. more than 25% of gold
syllables are unmatched (dropped or substituted), ignoring recordings where
the mismatch is just Arabic digits. Override with --files REGEX.
Output: out/asr_retrans.jsonl (same fields as asr.jsonl); 03_align_asr.py
uses it in place of asr.jsonl for those recordings.
"""
import argparse, json, os, re, sys
import numpy as np
import pandas as pd
import mlx_whisper
from mlx_whisper.audio import load_audio, SAMPLE_RATE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
ex = import_module("01_extract_syllables")

MODEL = "mlx-community/whisper-large-v3-mlx"
PROMPT = "以下是台灣的中文新聞與文章朗讀，請使用繁體中文與標點符號。"

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("--out", default="out")
ap.add_argument("--files", default=None); ap.add_argument("--min-pause", type=float, default=0.25)
a = ap.parse_args()

s = pd.read_csv(f"{a.out}/align_summary.csv")
if a.files:
    todo = s[s.file.str.contains(a.files, regex=True)].file.tolist()
else:
    bad = (s.gold_only + s["sub"]) / s.n_syll
    todo = s[(bad > 0.25) & ~s.has_digits].file.tolist()
res = f"{a.out}/asr_retrans.jsonl"
done = {json.loads(l)["file"] for l in open(res, encoding="utf-8")} if os.path.exists(res) else set()
todo = [f for f in todo if f.replace(".TextGrid", ".wav") not in done]
print(f"{len(todo)} recordings to re-transcribe")

with open(res, "a", encoding="utf-8") as fh:
    for i, tg in enumerate(todo, 1):
        phon = ex.read_short_textgrid(os.path.join(a.root, tg))["phon"]
        # speech stretches between long pauses
        cuts, start = [], None
        for t0, t1, lab in phon:
            pause = lab in ("sil", "") or (lab == "sp" and t1 - t0 >= a.min_pause)
            if pause and t1 - t0 >= a.min_pause:
                if start is not None: cuts.append((start, t0)); start = None
            elif start is None:
                start = t0
        if start is not None: cuts.append((start, phon[-1][1]))
        # load_audio returns an MLX array; convert so it can be sliced like numpy
        audio = np.array(load_audio(os.path.join(a.root, tg.replace(".TextGrid", ".wav"))), dtype=np.float32)
        segs = []
        for t0, t1 in cuts:
            x = audio[int(max(0, t0 - 0.1) * SAMPLE_RATE): int((t1 + 0.1) * SAMPLE_RATE)]
            if len(x) < 0.3 * SAMPLE_RATE: continue
            r = mlx_whisper.transcribe(x.astype(np.float32), path_or_hf_repo=MODEL, language="zh",
                                       initial_prompt=PROMPT, temperature=0.0,
                                       condition_on_previous_text=False)
            segs.append({"start": t0, "end": t1, "text": r["text"].strip()})
        text = "，".join(s_["text"] for s_ in segs if s_["text"])
        fh.write(json.dumps({"file": tg.replace(".TextGrid", ".wav"), "text_raw": text,
                             "text": text, "segments": segs, "chunked": True},
                            ensure_ascii=False) + "\n"); fh.flush()
        print(f"{i}/{len(todo)} {tg}: {text[:50]}")
