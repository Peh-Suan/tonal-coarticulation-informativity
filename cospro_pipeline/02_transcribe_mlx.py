"""
Step 2a: transcribe COSPRO wavs with Whisper on Apple Silicon (MLX).
Run in macOS Terminal, NOT in Claude's VM:

    pip install mlx-whisper opencc-python-reimplemented
    python3 02_transcribe_mlx.py <path/to/COSPRO> out/asr.jsonl

Resumable: files already in the output are skipped, so you can stop and rerun.
Output: one JSON line per wav {"file", "text", "text_raw", "segments"}.
Text is converted to Taiwan Traditional (OpenCC s2tw) because Whisper
sometimes drifts into Simplified.
"""
import sys, os, glob, json
import mlx_whisper
from opencc import OpenCC

MODEL = "mlx-community/whisper-large-v3-mlx"
PROMPT = "以下是台灣的中文新聞與文章朗讀，請使用繁體中文與標點符號。"
cc = OpenCC("s2tw")   # char-level only; s2twp rewrites words (高級->高階)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("out")
ap.add_argument("--shard", default="1/1",
                help="k/n: this process does every n-th file starting at k, e.g. 1/2 and 2/2 "
                     "in two Terminal windows, each with its OWN output file")
a = ap.parse_args()
root, out = a.root, a.out
k_shard, n_shard = map(int, a.shard.split("/"))
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
done = set()
# skip anything already transcribed in ANY asr*.jsonl next to the output (all shards)
for other in glob.glob(os.path.join(os.path.dirname(out) or ".", "asr*.jsonl")):
    if "retrans" in other: continue
    for l in open(other, encoding="utf-8"):
        try: done.add(json.loads(l)["file"])
        except json.JSONDecodeError: pass      # half-written line from an interrupted run
if os.path.exists(out):
    # make sure the next record starts on a fresh line
    with open(out, "rb+") as fh:
        fh.seek(0, 2)
        if fh.tell() > 0:
            fh.seek(-1, 2)
            if fh.read(1) != b"\n": fh.write(b"\n")

# corspro4 has no tone labels, so it is not used downstream
wavs = sorted(f for f in glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)
              if not os.path.basename(f).startswith("._") and "corspro4" not in f)
wavs = [w for i, w in enumerate(wavs) if i % n_shard == k_shard - 1]
todo = [w for w in wavs if os.path.relpath(w, root) not in done]
print(f"{len(wavs)} wavs, {len(todo)} to do")

with open(out, "a", encoding="utf-8") as fh:
    for i, w in enumerate(todo, 1):
        r = mlx_whisper.transcribe(w, path_or_hf_repo=MODEL, language="zh",
                                   initial_prompt=PROMPT, temperature=0.0,
                                   condition_on_previous_text=False)
        rec = {"file": os.path.relpath(w, root), "text_raw": r["text"],
               "text": cc.convert(r["text"]),
               "segments": [{"start": s["start"], "end": s["end"], "text": cc.convert(s["text"])}
                            for s in r["segments"]]}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        if i % 20 == 0 or i == len(todo):
            print(f"{i}/{len(todo)}  {rec['file']}: {rec['text'][:40]}")
