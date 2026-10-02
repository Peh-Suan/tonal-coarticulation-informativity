# Lexicon inputs (not distributed)

`tsm_pipeline/informativity_positional.py` and `moe_readings.py` read:

| file | content | source |
|---|---|---|
| `texts.txt` | Southern Min text in Tâi-lô (hyphenated words, tone digits) → TSM lexicon (word token counts) | TODO: state the source / licence |
| `../cospro_pipeline/out/word_freq.csv` | Mandarin word counts (Han-only words) from the `segmented_texts` corpus, written by `cospro_pipeline/08_build_processed.py --segmented <dir>` | TODO: state the source / licence of `segmented_texts` |
| `dict-revised.json` | MOE 重編國語辭典修訂本 | `dict-revised.json.xz` from https://github.com/g0v/moedict-data (unpack here) |
| `cospro_word_readings.csv` | COSPRO-attested majority reading of each word (included) | `python tsm_pipeline/moe_readings.py build-cospro` |

The resulting tables are included in `../tables/`, so the processed data and the models can be rebuilt without
these files.
