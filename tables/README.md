# Informativity and functional-load tables

Written by `tsm_pipeline/informativity_positional.py` (and, for the `informativity_TSM/TM` and
`informativity_sandhi_*` files, by the earlier lab scripts; they are read only for the legacy `*_tai` columns).
`TSM` = Taiwan Southern Min, `TM` = Taiwan Mandarin. Sandhi tables use surface tones: TSM `zhang` (T5 → T7) and
`quan` (T5 → T3) speaker groups; TM T3 → T2 before T3.

| file | content |
|---|---|
| `informativity_pos_{TSM,sandhi_zhang_TSM,sandhi_quan_TSM,TM,sandhi_TM}.csv` | positional informativity (adjacent syllable + target position class); used as `informativity`, `informativity_sandhi`, `*_allpairs` (`pos_class == "all"`) |
| `informativity_full_context_*.csv` | full-context informativity (adds the target's own segments); `share_determined` = share of tokens whose tone is fully predictable |
| `functional_load_pos_*.csv`, `functional_load_full_context_*.csv` | functional load in the same cells: `functional_load` (sum over competitors), `functional_load_max` (strongest competitor), `*_plugin` (no smoothing), `share_zero` |
| `alpha_calibration_{TSM,TM}.csv` | smoothing calibration (bias of thinned edge cells by alpha) |
| `informativity_{TSM,TM}.csv`, `informativity_sandhi_*.csv` | legacy tables (disyllabic words only) → `*_tai` columns |
| `speaker_accents_TSM.csv` | TAT-MOE speaker → accent group (`zhang`, `quan`, `drop`), from the corpus 腔調 field; copy to `<prep>/speaker_accents.csv` before TSM step 4 |

Common columns: `direction` (`carryover`: target = syllable 2; `anticipatory`: target = syllable 1), `pair`
(tone of syllable 1 – tone of syllable 2), `pos_class` (target position in the word: `initial` / `medial` / `final`,
or `all`), `tokens` (lexicon tokens in the cell), values in bits.

Definitions (c = context cell, t = target tone):

* informativity(t | C, dir, pos) = Σ_c n(c, t) · −log2 p(t | c) / Σ_c n(c, t), c = adjacent syllable (segments + tone C)
  [+ target segments for `full_context`] at the target's position class.
* functional load: merging tones t and u in cell c loses
  ΔH_c(t, u) = (p_t + p_u) log2(p_t + p_u) − p_t log2 p_t − p_u log2 p_u bits;
  FL_c(t) = Σ_{u≠t} ΔH_c(t, u) (`functional_load`) or max_u (`functional_load_max`), averaged over cells as above.
* The context tone's value (`context_*` columns in the processed data) is the same pair read in the other direction,
  at the context syllable's position class.
