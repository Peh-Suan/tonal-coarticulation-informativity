# Information-theoretic conditioning of tonal coarticulation in Taiwan Mandarin and Taiwan Southern Min

Code for the ICPhS 2027 submission by Po-Hsuan Huang. This repository is a review version: it contains the full
processing pipelines, the informativity / functional-load tables, and the analysis notebook, but **no corpus data**
(see [Data](#data)).

**Question.** Does a tone's informativity (how much it contributes to identifying words, Cohen Priva 2015) and its
functional load predict how much it coarticulates with a neighbouring tone, in Taiwan Mandarin (TM, COSPRO) and
Taiwan Southern Min (TSM, TAT-MOE)? Is the effect of word predictability (word surprisal, contextual gain) on
coarticulation conditioned by the tone system?

**Measure of coarticulation.** For every within-word syllable pair, the target syllable's F0 at the shared boundary
(`target_f0`) is regressed on the neighbouring syllable's boundary F0 (`context_f0`), in both directions
(carryover: target = syllable 2; anticipatory: target = syllable 1). Effects on coarticulation are interactions with
`context_f0`, i.e. changes in this slope.

## Repository layout

```
tsm_pipeline/          Taiwan Southern Min (TAT-MOE): alignment, sandhi, F0, surprisal → processed_data.csv
  informativity_positional.py   informativity + functional-load tables for BOTH languages
  moe_readings.py                Taiwan Mandarin readings for the Mandarin lexicon
cospro_pipeline/       Taiwan Mandarin (COSPRO): characters, words, F0, surprisal → processed_data.csv
analysis/
  paper_model.ipynb              the model reported in the paper: lmer fit, contrasts, Figure 1 (R / IRkernel)
  paper_contrasts.R              every beta / CI / p in the Results (called from paper_model.ipynb)
  paper_figure.R                 Figure 1 (called from paper_model.ipynb)
  cross_language_models.ipynb    earlier exploratory models (tone-pair intercepts, brms); not used for the paper
tables/                informativity / functional-load tables used in the paper (see tables/README.md)
lexicon/               lexicon inputs for the tables (not distributed; see lexicon/README.md)
environment/           Python requirements and R package list
```

## Data

| corpus | used for | how to obtain |
|---|---|---|
| COSPRO (Academia Sinica) | Mandarin speech, syllable segmentation, tone labels, prosodic breaks | licence from the Institute of Linguistics, Academia Sinica |
| TAT-MOE (Taiwanese Across Taiwan, MOE) | Southern Min speech, Tâi-lô / 漢羅 transcripts, speaker metadata | from the corpus distributor |
| MOE 重編國語辭典修訂本 (g0v moedict-data `dict-revised.json`) | Taiwan Mandarin readings | https://github.com/g0v/moedict-data |
| Mandarin text corpus (`segmented_texts`) | Mandarin word frequencies → lexicon for informativity | see `lexicon/README.md` |
| Southern Min text (`texts.txt`, Tâi-lô) | Southern Min lexicon for informativity | see `lexicon/README.md` |

The corpora cannot be redistributed, and the two processed analysis tables (≈ 0.6–0.8 GB each) are derived from
them, so they are not included. Everything that does not depend on the audio is included: the informativity and
functional-load tables (`tables/`) and the TSM speaker accent grouping. The processed tables can be shared with
reviewers who hold the corpus licences; please contact the author.

## Reproducing the results

Paths below are examples. `P` is the TSM working directory; the COSPRO pipeline writes to `cospro_pipeline/out/`.
Hardware: the ASR, language-model and MFA steps were run on an Apple-Silicon Mac (macOS); the remaining steps run on
any machine with Python ≥ 3.10 and R ≥ 4.4.

### 0. Environment

```bash
pip install -r environment/requirements.txt          # pipeline steps (see the file for which step needs what)
conda create -n aligner -c conda-forge montreal-forced-aligner   # TSM step 02 only
Rscript environment/install_R_packages.R             # analysis notebook
```

### 1. Taiwan Southern Min (TAT-MOE)

```bash
cd tsm_pipeline
C="<path/to/TAT-MOE>/corpus-full";  P=~/tsm_mfa/all_condenser
# 1  Tâi-lô → syllable .lab files, pronunciation dictionary, syllables.csv / utterances.csv
python 01_prepare_mfa_corpus.py --json-root "$C/TAT-MOE-train-json" "$C/TAT-MOE-eval-json" "$C/TAT-MOE-test-json" \
    --wav-root "$C/TAT-MOE-train" "$C/TAT-MOE-eval" "$C/TAT-MOE-test" --channel condenser --out $P
# 2  forced alignment (MFA). We trained the acoustic model on the test split (pilot) and aligned all splits with it:
bash 02_run_mfa.sh <pilot_prep_dir> 8                 # trains tsm_acoustic_model.zip on a prepared split
conda run -n aligner mfa align $P/corpus $P/dictionary.txt <pilot_prep_dir>/tsm_acoustic_model.zip $P/aligned_pilotmodel
# 3  TextGrids → syllable times and pauses
python 03_collect_alignments.py --prep $P --textgrids $P/aligned_pilotmodel
# 4  surface (sandhi) tones, juncture labels; per-speaker T5 sandhi from the accent grouping
cp ../tables/speaker_accents_TSM.csv $P/speaker_accents.csv
python 04_surface_tones.py --prep $P
# 5  F0 at syllable boundaries (Parselmouth; 11 portions, ln Hz z-scored per speaker)
python 05_f0_boundary.py --prep $P --jobs 8
# 6  contextual surprisal (Bohanlu/Taigi-Llama-2-7B, 漢羅 text)
python 06_surprisal_llm.py --prep $P
# 7  context-free (unigram) word surprisal from the same LM, corpus word counts
python 07_unigram_lm.py --prep $P --k 32
```

### 2. Taiwan Mandarin (COSPRO)

```bash
cd cospro_pipeline
python 01_extract_syllables.py <path/to/COSPRO> out        # TextGrids → syllables, tones, prosodic breaks
python 02_transcribe_mlx.py <path/to/COSPRO> out/asr.jsonl  # Whisper (MLX) characters
python 03_align_asr.py                                      # align ASR characters to the gold syllables
python 02b_retranscribe_chunks.py <path/to/COSPRO>          # re-transcribe poorly aligned recordings
python 03_align_asr.py                                      # re-align with the re-transcriptions
python 04b_correct_mlm.py                                   # BERT correction constrained by the gold syllables
python 05_words_bigrams.py                                  # CKIP word segmentation → within-word bigrams
python 06_surprisal.py                                      # contextual word surprisal (ckiplab/gpt2-base-chinese)
python 06b_unigram_lm.py                                    # context-free word surprisal, same LM
python 07_f0_boundary.py <path/to/COSPRO> --jobs 8          # F0 at syllable boundaries
```

### 3. Informativity and functional-load tables (both languages)

The tables in `tables/` are the ones used in the paper; this step regenerates them. It needs the two lexicons in
`lexicon/` and the Mandarin word counts, which `cospro_pipeline/08_build_processed.py --segmented <dir>` writes to
`cospro_pipeline/out/word_freq.csv` on its first run.

```bash
cd tsm_pipeline
python moe_readings.py build-cospro           # COSPRO-attested Mandarin word readings (needs a first COSPRO processed_data.csv;
                                              # the result is included as lexicon/cospro_word_readings.csv)
python informativity_positional.py --lang TSM # calibrates the smoothing (alpha), writes tables/*_TSM.csv
python informativity_positional.py --lang TM  # Taiwan readings by default (--tm-readings pypinyin = mainland readings)
```

`informativity_positional.py` writes, per language and tone representation (citation / sandhi):

* `informativity_pos_*`: expected surprisal of the target tone given the adjacent syllable (segments + tone) and the
  target's position in the word (initial / medial / final), over all within-word syllable pairs; additive smoothing
  calibrated by thinning (TSM α = 4.09, TM α = 0).
* `informativity_full_context_*`: the same, with the target's own segments added to the context (plug-in).
* `functional_load_pos_*`, `functional_load_full_context_*`: in the same cells, the entropy lost when the target tone
  merges with each competing tone (`functional_load`, summed over competitors; `functional_load_max`, strongest one).

### 4. Analysis tables

```bash
python tsm_pipeline/08_build_processed.py --prep $P                               # → $P/processed_data.csv
(cd cospro_pipeline && python 08_build_processed.py --segmented <path/to/segmented_texts>)   # → out/processed_data.csv
```

Both write the same column names: one row per within-word bigram × direction, the analysis subsets (`keep_any2` for TSM, `keep` for TM; see below),
every informativity / functional-load measure for the target (`informativity…`, `functional_load…`) and for the
context tone (`context_…`), word surprisal (`unigram_surprisal`), contextual surprisal and gain (`context_gain`),
durations, F0 portions, segment types, position classes and speaker / item ids. Column definitions are in the
docstrings of the two `08_build_processed.py` scripts.

### 5. Model, contrasts and figure (paper)

Open `analysis/paper_model.ipynb` (R kernel), set the data paths if needed

```bash
export TSM_DATA=$P/processed_data.csv
export COSPRO_DATA=$PWD/cospro_pipeline/out/processed_data.csv
```

and run all cells. The lmer fit takes about 65–80 min on an Apple M-series laptop (≈ 553k rows: TSM 181,988 from
381 speakers, TM 371,174 from 96 speakers) and is saved to `analysis/fits/paper_model/`; later runs reuse it as long
as the formula and data are unchanged. The last two cells write `paper_contrasts.csv` (every statistic in the Results)
and `fig1.png/pdf`.

Analysis subsets: TSM `keep_any2` (within-word pairs whose second syllable has a certain surface tone, citation or
sandhi; checked tones excluded; speakers of unknown accent kept, minus tones whose sandhi form depends on the accent),
TM `keep`.

`INF_COL` in the Settings cell selects the informativity measure (`informativity` = positional informativity, used
in the paper). `informativity_full_context` and `functional_load_pos` can be run the same way as robustness checks;
outputs then go to `fits/paper_model_<INF_COL>/`.

## Notes for reviewers

* Effects on coarticulation are contrasts on the `context_f0` slope, per 1 within-language SD of each predictor, with
  every predictor not shown at its mean (0). `pos` (anticipatory −0.5 / carryover +0.5) and `lang` (TM −0.5 / TSM +0.5)
  are effects-coded; target tone is sum-coded within language. p-values use Satterthwaite df (lmerTest).
* Random effects: tone-pair random **slopes** for `context_f0`, `context_f0:unigram` and `context_f0:gain` (no tone-pair
  intercepts) and target-tone (surface tone × language × position) intercepts and `context_f0` slopes. The slopes make
  the effects on coarticulation be tested against between-pair variation (without them, df ≈ 500k:
  pseudo-replication). Dropping tone-pair intercepts while conditioning on the target tone keeps the carryover and
  anticipatory slopes from being two readings of the same covariance between the two boundary F0 values.
* Contrasts on the overall slope (language and position differences) have df ≈ 11–15, as they are tested against the
  target-tone groups.
* lme4 may print a gradient convergence warning (max |grad| ≈ 0.005); refitting from the converged estimates leaves
  REML and all fixed effects unchanged (see the convergence-check cell).
