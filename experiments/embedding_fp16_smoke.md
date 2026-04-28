# FP16 Embedding Smoke Run

Goal: compare the normal int8 embedding artifact against an fp16-stored embedding artifact, without changing training.

The baseline quantizer stores large tensors as int8 with per-row fp16 scales. Setting `EMBEDDING_STORE_DTYPE=fp16` keeps `tok_emb.weight` as fp16 in the compressed artifact, then restores it to the original dtype after loading.

Run from `/workspace/parameter-golf-fork` after pulling the latest branch:

```bash
git pull
```

Use the existing `WARMDOWN_ITERS=100` smoke run as the reference:

```text
smoke_relu2_warmdown100_1000 final val_bpb: 1.36562538
smoke_relu2_warmdown100_1000 total int8+zlib bytes: 14553178
```

Now run the fp16 embedding variant with the same training settings:

```bash
RUN_ID=smoke_relu2_warmdown100_embfp16_1000 \
EMBEDDING_STORE_DTYPE=fp16 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=1000 \
WARMDOWN_ITERS=100 \
VAL_LOSS_EVERY=0 \
TRAIN_LOG_EVERY=100 \
MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Compare:

```text
final_int8_zlib_roundtrip_exact val_loss:... val_bpb:...
Total submission size int8+zlib: ... bytes
```

This isolates the serialization/evaluation effect of fp16 embeddings. Training should be essentially the same as the int8 reference because the change happens only when writing the final compressed artifact.
