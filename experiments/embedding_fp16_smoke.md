# FP16 Embedding Smoke Run

Goal: compare the normal int8 embedding artifact against an fp16-stored embedding artifact, without changing training.

The baseline quantizer stores large tensors as int8 with per-row fp16 scales. Setting `EVAL_EMBEDDING_STORE_DTYPES=int8,fp16` trains once, then serializes and evaluates the same trained weights twice:

- `int8`: baseline artifact format
- `fp16`: keep `tok_emb.weight` as fp16 in the compressed artifact

Run from `/workspace/parameter-golf-fork` after pulling the latest branch:

```bash
git pull
```

Use the existing `WARMDOWN_ITERS=100` smoke run as the reference:

```text
smoke_relu2_warmdown100_1000 final val_bpb: 1.36562538
smoke_relu2_warmdown100_1000 total int8+zlib bytes: 14553178
```

Now run the paired artifact comparison with the same training settings:

```bash
RUN_ID=smoke_relu2_warmdown100_embcompare_1000 \
EVAL_EMBEDDING_STORE_DTYPES=int8,fp16 \
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
final_int8_zlib_roundtrip_exact embedding_store_dtype:int8 val_loss:... val_bpb:...
final_int8_zlib_roundtrip_exact embedding_store_dtype:fp16 val_loss:... val_bpb:...
embedding_store_compare baseline:int8 candidate:fp16 delta_val_loss:... delta_val_bpb:... delta_bytes:...
```

This isolates the serialization/evaluation effect of fp16 embeddings because both artifact variants come from the same trained weights.
