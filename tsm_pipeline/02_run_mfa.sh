#!/usr/bin/env bash
# Train a Taiwanese Southern Min MFA model on the prepared corpus and align it.
# Run in macOS Terminal (MFA needs conda; it can't be installed in Claude's sandbox).
#
# One-time setup:
#   conda create -n aligner -c conda-forge montreal-forced-aligner
#   conda activate aligner
#
# Usage:  bash 02_run_mfa.sh ~/tsm_mfa/pilot [n_jobs]
set -euo pipefail
PREP="${1:?usage: 02_run_mfa.sh <prep_dir> [n_jobs]}"
JOBS="${2:-8}"
PREP="$(cd "$PREP" && pwd)"

# 1. sanity check: every .lab word is in the dictionary, audio readable
mfa validate "$PREP/corpus" "$PREP/dictionary.txt" --clean -j "$JOBS" \
  | tee "$PREP/mfa_validate.log"

# 2. train acoustic model (monophone -> triphone -> LDA/MLLT -> SAT) and
#    write TextGrids for the training data in the same pass
mfa train "$PREP/corpus" "$PREP/dictionary.txt" "$PREP/tsm_acoustic_model.zip" \
  --output_directory "$PREP/aligned" --clean -j "$JOBS" \
  2>&1 | tee "$PREP/mfa_train.log"

echo "Done. TextGrids in $PREP/aligned ; model in $PREP/tsm_acoustic_model.zip"
echo "Next: python 03_collect_alignments.py --prep $PREP --textgrids $PREP/aligned --handcheck 50"
