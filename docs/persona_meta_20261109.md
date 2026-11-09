# Cross-request TTT training

The new stage trains later answer CE through earlier differentiable writes.
It starts from the corrected foundation checkpoint at step 1000, SHA256
`f289fa44b107129f6463668b9ad66310a06a130f47c7cb123995d43482c05718`.
This is genuine meta training, rather than relabeling SFT weights as meta.
Quality is assessed after training; a successful capacity probe is not memory recall.

## Changes

- An episode consists of one or two complete support turns followed by a separate
  query-only forward. Positions and attention KV start afresh at every request.
  Only W/S/LN state crosses requests. Returned states stay differentiable only
  when explicitly requested; the old detached API remains the default.
- Train only per-layer K/V/eta/alpha: 445,818,948 parameters across 204 tensors.
  Freeze the other 1,138,060,800 parameters, including Q, output projections and W0.
  Thus language behavior with writes disabled is retained by construction.
  The weak underlying language model still limits real conversation quality.
- Require non-reentrant checkpointing. Freezing input embeddings otherwise
  removes the gradient source expected by reentrant checkpointing.
- Use 256 train / 32 dev / 32 test episodes with distinct entities, random codes
  and prompt families. Actual BPE lengths fit 128 tokens per request; cropping
  and packing are rejected. Recall, replacement and a narrow reply-prefix
  preference are included. No test examples are trained on.
- Preserve FP32 meta parameters and FP32 persona W/S masters during serving;
  CUDA GEMMs use BF16 autocast. Previously converting W0 to BF16 before cloning
  a new user irreversibly rounded the supposedly exact masters.
- Expose explicit `server.app --enable-ttt` / `--disable-ttt` and a health flag.
  Enabling writes on an incompatible pretrain/SFT checkpoint raises an error.

## Objective and architecture

Surprise is the inner reconstruction error used to compute writes. Later query
CE is the outer objective; separately minimizing the learned Surprise target can
collapse the writer. Auxiliary Surprise weight remains zero. The reconstruction
gradient still participates fully through the differentiable inner update.

The existing `chunk_shared_fp32_exact_v2` protocol is retained. It follows the
project prompt's sequential shared SwiGLU pool and MSE reconstruction rule.
The [uTTT paper](https://arxiv.org/html/2610.05484v1) describes distinct outer-task
and inner associative objectives and aggregated shared-memory updates at chunk
boundaries. This project is not an exact reproduction of that paper's update
schedule or inner objective.

## Validation and resource limits

The isolated 1.584B probe completed two full FP32-Adam steps on real correction
episodes, with all 204 writer tensors receiving finite nonzero query gradients.
Peak allocated memory was 24,886,702,592 bytes (23.18 GiB); peak reserved was
25,990,004,736 bytes (24.21 GiB). Frozen parameter SHA256 and no-write logits
remained bitwise identical. The initial CPU run passed 611 tests (22 skipped).
See the accompanying measurement JSON for provenance and exact budgets.

The initial training budget is 512 steps, with latest resumable and best
weights-only checkpoints. There is no `.prev` copy. Tensor-only steady storage
is about 16.24 GB and atomic replacement can temporarily reach 26.14 GB;
launch requires another 100 GB free on C after that peak. Old models are retained.

Exact full W/S/LN persona state costs 424,693,760 bytes (405.02 MiB), plus
106,168,320 bytes for optional BF16 read caches. Alias files share storage when
hardlinks are available; snapshots and copy fallbacks add storage. This exceeds
the original prompt's tens-of-MB aspiration. Low-rank/compressed state would
require a new validated protocol and is not silently introduced here.

## Quality checks

`deploy/evaluate_persona_memory.py` generates answers without teacher forcing.
It compares W0, support read without writes, another user's state, correct
writes, and real disk save/eviction/reload. Reads receive no earlier text/KV.
Raw analytical and production-guarded writes must be reported separately.
The optional `--production-turns 2` also uses actual generated support replies,
then an empty-history question through `uTTTPersonalizer`.

For an explicitly chosen compatible meta checkpoint:

```sh
python -m server.app --config configs/5090_persona_meta_20261109.yaml \
  --base-ckpt /path/to/validated-meta.best.pt --enable-ttt
```

Use a separate persona directory for each base model. Personal states carry the
base fingerprint and incompatible states are not reused.
