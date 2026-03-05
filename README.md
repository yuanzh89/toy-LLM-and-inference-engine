# ToyLLM Inference Engine

A from-scratch prototype of a modern LLM inference engine written in PyTorch, built to demonstrate and study the core systems and architectural techniques that power production inference stacks like [vLLM](https://github.com/vllm-project/vllm).

The engine implements a decoder-only transformer with a full suite of inference optimisations: paged KV-cache management, prefix KV caching, chunked prefill, and prefill–decode disaggregation, all wired together with a multi-threaded scheduler.

---

## Features

### Model Architecture

| Feature | File | Notes |
|---|---|---|
| Decoder-only transformer | `toy_llm_model.py` | Embedding → N × TransformerBlock → RMSNorm → LM head |
| Weight-tied embedding & LM head | `toy_llm_model.py` | Embedding matrix is shared with the output projection, halving parameters at the vocabulary boundary (Press & Wolf, 2017) |
| Pre-RMSNorm | `gqa_with_kv_cache.py`, `swiglu_ffn.py` | Norm applied *before* each sublayer rather than after, matching LLaMA / Mistral and stabilising training at scale |
| Grouped Query Attention (GQA) | `gqa_with_kv_cache.py` | Configurable `num_query_heads` / `num_kv_heads`; KV heads expanded via `repeat_interleave` before attention |
| Rotary Position Embedding (RoPE) | `rope.py` | In-place rotation of head dimensions; supports per-sequence `start_pos` offsets for correct absolute positions during batched decode |
| SwiGLU feed-forward network | `swiglu_ffn.py` | `FFN(x) = W₃(SiLU(W₁x) ⊙ W₂x)`; also includes a fused `W₁₂` variant that halves GEMM kernel launches |
| Flash Attention (prefill) | `gqa_with_kv_cache.py` | Calls a Triton `flash_attention` kernel during prefill to avoid materialising the O(seq_len²) attention matrix |

### Inference Engine

| Feature | Files | Notes |
|---|---|---|
| Paged KV-cache management | `kv_cache_block.py`, `block_manager.py` | Fixed-size physical blocks with reference counting; pool-level eviction frees zero-refcount blocks |
| Depth-aware eviction policy | `block_manager.py` | Evicts deepest trie nodes first (longest, most request-specific suffixes), preserving common shallow prefixes that benefit future requests |
| Prefix KV caching | `prefix_caching.py`, `block_manager.py` | `BlockTrieTree` enables O(depth) prefix lookup; matched blocks are pinned and reused across requests without recomputation |
| Chunked prefill | `sequence.py`, `gqa_with_kv_cache.py`, `model_runner.py` | Prompt is split into fixed-size query chunks processed sequentially; only the uncached KV suffix is computed per chunk |
| Batched decode with variable-length sequences | `gqa_with_kv_cache.py` | KV tensors padded to `max_kv_len` with a boolean attention mask; enables continuous batching across sequences of different lengths |
| Greedy decoding | `toy_llm_model.py` | Argmax over softmax logits at each step; structured for easy swap-in of temperature / top-k / nucleus sampling |
| Prefill–decode disaggregation | `model_runner.py`, `launch.py` | Prefill and decode run in separate threads with independent block managers and model weights, communicating through typed queues |

---

## Architecture Overview

### Transformer Block

Each of the `N` transformer blocks follows the Pre-Norm pattern:

```
x  ──► RMSNorm ──► GQA (with KV cache) ──► + ──► RMSNorm ──► SwiGLU FFN ──► + ──►
│                                           ▲                                  ▲
└───────────────────────────────────────────┘  ────────────────────────────────┘
              residual                                     residual
```

### Paged KV Cache

Token IDs are chunked into fixed-size **blocks**. Each block stores pre-computed K and V tensors for every transformer layer. Blocks are managed by a pool with reference counting:

```
Sequence token IDs:  [ t0  t1  t2  t3 | t4  t5  t6  t7 | t8  t9 ]
                     └────── Block 0 ──┘ └────── Block 1 ──┘ └─ Block 2 ─┘
                      (full, in trie)     (full, in trie)     (partial, appendable)
```

Full blocks are sealed into the prefix-caching trie after decode. Partial last blocks stay out of the trie (`trie_tree_depth == 0`) and remain appendable until they fill up.

### Prefix KV Caching

A `BlockTrieTree` maps sequences of token-ID chunks to their cached `Block` objects. On each new request, `BlockManager.allocate_blocks` walks the trie to find the longest matching cached prefix and only computes KV for the uncached suffix:

```
Trie root
 └── [t0..t15]  ── Block A  (cached, ref_count += 1)
      └── [t16..t31] ── Block B  (cached, ref_count += 1)
           └── [t32..t47] ── Block C  (new, must compute)
```

Eviction prefers the deepest nodes first, keeping the most-reused shallow prefix blocks alive longest.

### Prefill–Decode Disaggregation

```
                  ┌─────────────────────────────────────────┐
                  │             Prefill Node                │
                  │                                         │
   prefill_queue ─►  PrefillChunker  ──►  PrefillForward    ├──► decode_schedule_queue
                  │      Thread              Thread         │
                  └─────────────────────────────────────────┘
                                                                       │
                  ┌─────────────────────────────────────────┐          ▼
                  │             Decode Node                 │      Scheduler
                  │                                         │     (main thread)
   decode_worker_queue ◄────────────────────────────────────────────── │
                  │        │                                │
                  │        ▼                                │
                  │  DecodeWorker Thread  ──────────────────┼──► decode_schedule_queue
                  └─────────────────────────────────────────┘
```

The three inter-node queues (`prefill_queue`, `decode_schedule_queue`, `decode_worker_queue`) are `queue.Queue` objects in this prototype. In a real deployment they would be replaced by gRPC / ZMQ channels, with KV tensors transferred via RDMA or NCCL.

**Shutdown** propagates as a strict cascade so no thread exits while another is still writing to a queue it reads: `Scheduler → PrefillChunker → PrefillForward → decode_schedule_queue → Scheduler → DecodeWorker`.

---

## File Structure

```
├── config.py                          # ToyLLMConfig dataclass
│
├── engine/
│   ├── sequence.py                    # Sequence lifecycle, chunked activations, KV block references
│   ├── block_manager.py               # Block pool, prefix lookup, eviction, sealing
│   ├── kv_cache_block.py              # Physical KV-cache block with per-layer tensors
│   └── prefix_caching.py             # BlockTrieTree and BlockTrieNode
│
├── layer/
│   ├── gqa_with_kv_cache.py           # Grouped Query Attention with paged KV cache + Flash Attention
│   ├── rope.py                        # Rotary Position Embedding (in-place, batched start_pos)
│   ├── swiglu_ffn.py                  # SwiGLU FFN (standard + fused variants)
│   ├── transformer_block_with_kv_cache.py   # Single transformer block (GQA + FFN)
│   └── toy_llm_model.py               # Full decoder-only model (embedding, N blocks, LM head)
│
├── model_runner.py                    # PrefillRunner and DecodeRunner (threaded)
├── scheduler.py                       # Scheduler and Tokenizer
└── launch.py                          # Entry point — wires all components and starts threads
```

---

## Getting Started

### Requirements

```
python >= 3.11
torch >= 2.0
triton >= 2.0   # for the Flash Attention kernel
```

### Running the engine

```python
from config import ToyLLMConfig
from launch import launch

config = ToyLLMConfig(
    vocab_size=32000,
    d_model=512,
    d_ff=1024,
    num_transformer_layers=4,
    num_query_heads=8,
    num_kv_heads=2,            # 4 query heads share each KV head (GQA)
    dropout=0.0,
    max_block_size=64,         # max blocks in the KV-cache pool
    max_token_size_per_kv_cache_block=16,
    max_sequence_length=256,
    query_chunk_size=16,       # tokens per prefill chunk
    decode_max_batch_size=8,
)

finished_sequences = launch(config, prompts=[
    "The capital of France is",
    "Once upon a time in a land far away",
])

for seq in finished_sequences:
    print(seq.seq_id, seq.token_ids)
```

---

## Key Design Decisions

**Only full blocks enter the trie.** Partial last blocks (`len(token_ids) < max_token_size_per_kv_cache_block`) are kept out of the prefix-caching trie with `trie_tree_depth == 0` so they remain appendable during decode. They are sealed into the trie by `BlockManager.seal_full_decode_blocks` once all transformer layers have written their KV tensors for that block.

**Depth-first eviction over LRU.** When the block pool is full, eviction sorts by `(ref_count ASC, trie_tree_depth DESC)` and removes zero-refcount blocks from the deepest trie levels first. Deep blocks represent long, request-specific suffixes with low reuse probability; shallow blocks represent common prompt prefixes with high reuse probability. This gives better cache hit rates than naive LRU on workloads with shared system prompts.

**`update_token_ids` flag on decode append.** Because all transformer layers call `decode_append_token_ids_and_kv_cache` for the same logical new token, the token ID must only be appended to `block.token_ids` once (on `layer_id == 0`). All subsequent layer calls pass `update_token_ids=False` to extend only the KV tensors.

**Threading over multiprocessing.** `torch.Tensor` KV caches are not pickle-serialisable without custom shared-memory wrappers, so `threading.Thread` is used instead of `multiprocessing.Process`. The queue interfaces are intentionally kept identical to what a real network-transport-backed implementation would expose.

---

## References

- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017)
- Press & Wolf, [*Using the Output Embedding to Improve Language Models*](https://arxiv.org/abs/1608.05859) (2017) — weight tying
- Su et al., [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864) (2021) — RoPE
- Shazeer, [*GLU Variants Improve Transformer*](https://arxiv.org/abs/2002.05202) (2020) — SwiGLU
- Ainslie et al., [*GQA: Training Generalized Multi-Query Transformer Models*](https://arxiv.org/abs/2305.13245) (2023) — GQA
- Dao et al., [*FlashAttention: Fast and Memory-Efficient Exact Attention*](https://arxiv.org/abs/2205.14135) (2022) — Flash Attention
- Kwon et al., [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180) (2023) — paged KV cache
- Zheng et al., [*SGLang: Efficient Execution of Structured Language Model Programs*](https://arxiv.org/abs/2312.07104) (2024) — prefix caching trie
