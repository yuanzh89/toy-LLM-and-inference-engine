import torch
from transformers import AutoTokenizer, AutoConfig

from layer.toy_llm_model import ToyLLMModel


class ToyLLM:
    def __init__(self, tokenizer_name: str | None = None, num_query_heads: int = 4, num_kv_heads: int = 2,
                 dropout: float = 0.1):
        self.tokenizer_name = 'bert-base-uncased' if tokenizer_name is None else tokenizer_name

        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.vocab_size = len(self.tokenizer)

        config = AutoConfig.from_pretrained(self.tokenizer_name)
        self.d_model = config.hidden_size
        self.d_ff = self.d_model * 4

        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.dropout = dropout

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ToyLLMModel(self.d_model, self.d_ff, self.num_query_heads, self.num_kv_heads, self.dropout)
        self.model = self.model.to(self.device)
        self.model = self.model.compile()

    def step(self, prompt: str) -> str:
        """
        Performs a single forward pass to generate the next token from a text prompt.

        This method handles the tokenization of the input string, transfers the
        tensors to the configured device (GPU/CPU), executes the layer forward
        pass, and decodes the resulting token ID back into a human-readable string.

        Args:
            prompt (str): The input text sequence to continue.

        Returns:
            str: The single decoded string representation of the next generated token.

        Note:
            The conversion from GPU tensor to Python integer is handled by
            `.item()`, which implicitly synchronizes the CUDA device with the CPU.
        """
        token_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        token_ids = token_ids.to(self.device)

        next_token_id = self.model.forward(token_ids)

        # 1. Copy the value from GPU to CPU
        # 2. Converts the tensor to a native Python scalar
        next_token_id = next_token_id.item()
        next_token: str = self.tokenizer.decode(next_token_id)

        return next_token
