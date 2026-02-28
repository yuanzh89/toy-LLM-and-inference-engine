from dataclasses import dataclass

@dataclass
class ToyLLMConfig:
    # Model configs
    pretrained_model_name_or_path: str
    max_num_batched_tokens: int
    vocab_size: int
    d_model: int
    num_query_heads: int
    num_kv_heads: int
    d_ff: int
    num_transformer_layers: int
    dropout: float

    # KV cache configs
    max_block_size: int
    max_token_size_per_kv_cache_block: int
    max_sequence_length: int

    # Decode config
    decode_max_batch_size: int

    def __post_init__(self):
        assert self.pretrained_model_name_or_path is not None
        assert self.d_ff >= self.d_model
        assert self.d_ff % self.d_model == 0
        assert self.num_query_heads >= self.num_kv_heads
        assert self.num_query_heads % self.num_kv_heads == 0
        assert self.num_transformer_layers >= 0
        assert self.dropout >= 0.0
        assert self.max_block_size >= 0
        assert self.max_token_size_per_kv_cache_block >= 0
        assert self.max_sequence_length >= 0
