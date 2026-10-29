#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs/archive
if [[ -f logs/train_5090.log && ! -f logs/archive/train_5090_before_20261029.log ]]; then
    cp -p logs/train_5090.log logs/archive/train_5090_before_20261029.log
fi
exec >>logs/train_5090.log 2>&1
echo '[Launch] Waiting for verified corpus and instruction preparation to finish.'
while ! grep -q CORPUS_PREPARATION_COMPLETE logs/corpus_20261029.log 2>/dev/null; do
    if ! tmux has-session -t uttt_prepare_20261029 2>/dev/null; then
        echo '[Launch] Corpus preparation exited without completion; training was not started.'
        exit 1
    fi
    sleep 5
done
test -f data/pretrain_20261029/tokenized/manifest.json
test -f data/instruction_20261029/train.jsonl
grep -q INSTRUCTION_PREPARATION_COMPLETE logs/instruction_20261029.log
# Recheck the completed corpus before releasing the GPU from the chat service.
.venv/bin/python -u deploy/prepare_pretrain_corpus.py --root data/pretrain_20261029
if tmux has-session -t uttt_serve 2>/dev/null; then
    echo '[Launch] Stopping chat service to release GPU memory for training.'
    tmux send-keys -t uttt_serve C-c
    for _ in {1..10}; do
        tmux has-session -t uttt_serve 2>/dev/null || break
        sleep 1
    done
    tmux kill-session -t uttt_serve 2>/dev/null || true
fi
export OMP_NUM_THREADS=2 RAYON_NUM_THREADS=4 PYTHONUNBUFFERED=1
echo '[Launch] Starting 1.584B pretraining and instruction-tuning pipeline.'
exec .venv/bin/python -u runs/20261029/deploy/run_training_pipeline.py --run-id 20261029 --pretrain-steps 100000 --sft-steps 6000
