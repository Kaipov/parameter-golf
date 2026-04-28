# Activation Compare Runbook

Goal: reproduce the baseline once, then change only the MLP activation and compare final `val_bpb`.

## Setup

Use the official Runpod Parameter Golf template, then run:

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

For full challenge-like runs, omit `--train-shards 10` so the script downloads all 80 training shards.

## Baseline

```bash
RUN_ID=baseline_relu2 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

This uses the default `MLP_ACTIVATION=relu2`, matching the original baseline activation.

## Experiment

```bash
RUN_ID=experiment_silu2 \
MLP_ACTIVATION=silu2 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Optional second activation:

```bash
RUN_ID=experiment_gelu2 \
MLP_ACTIVATION=gelu2 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
VAL_LOSS_EVERY=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Compare

For each run, record these final log lines:

```text
final_int8_zlib_roundtrip_exact val_loss:... val_bpb:...
Total submission size int8+zlib: ... bytes
```

Lower `val_bpb` is better. Treat a single run as directional only; if an activation looks promising, repeat baseline and the experiment with the same settings before trusting the difference.
