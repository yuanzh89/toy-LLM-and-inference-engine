import queue
from multiprocessing import Queue

from block_manager import BlockManager
from config import ToyLLMConfig
from engine.sequence import SequenceStatus
from layer.toy_llm_model import ToyLLMModel
from sequence import Sequence


# class ModelRunMode(Enum):
#     PREFILL = auto()
#     DECODE = auto()

class ToyLLMModelRunner:
    """
    This class wrap a model and its config, and provide an API to run the model in prefill or decode mode.
    """
    SENTINEL = object()

    def __init__(self, llm_config: ToyLLMConfig, queue: Queue):
        self.llm_config = llm_config
        # The queue to receive incoming prefill / decode sequences
        self.queue = queue

        self.model = ToyLLMModel(llm_config.vocab_size, llm_config.d_model, llm_config.num_query_heads,
                                 llm_config.num_kv_heads,
                                 llm_config.d_ff, llm_config.num_transformer_layers, llm_config.dropout)

        self.block_manager = BlockManager(llm_config.max_block_size, llm_config.max_token_size_per_kv_cache_block,
                                          llm_config.num_transformer_layers)

    def run_in_prefill_mode(self):
        """
        This function is the entry point to run the model in prefill mode.

        When the model is running in prefill mode.
        The model computes and writes KV caches via block manager.
        """
        while True:
            sequence = self.queue.get()
            if sequence is ToyLLMModelRunner.SENTINEL:
                break

            # Run prefill for sequence
            pass

    def prefill_sequence(self, seq: Sequence):
        """
        This function is responsible for prefilling a sequence, update sequence status and populate its KV cache.
        """

        if not self.block_manager.allocate_blocks(seq):
            # block manager currently doesn't have enough KV cache blocks to serve this sequence
            # We could either:
            #  1. Put this sequence back to the end of the queue for a retry later, with a max_retry count.
            #  2. Drop this sequence and return failure.
            # In this demo code, for simplicity, we could just drop the sequence.
            seq.status = SequenceStatus.FAILED
            seq.release()  # Release resources related the resources allocated for this sequence
            return

        # At this point, the KV cache blocks are properly allocated for the input sequence and reference are built between these resources.
        seq.status = SequenceStatus.RUNNING
        # TODO: Double check the correctness here.
        self.model.prefill(seq)

        return

    def run_in_decode_mode(self):
        """
        This function is the entry point to run the model in decode mode.

        When the model reads KV caches via block manager.
        Assuming KV cache is already transferred from prefill node to decode node and properly populated in block manager.
        """
        while True:
            batch: list[Sequence] = []

            stop_listening = False
            for _ in range(self.llm_config.decode_max_batch_size):
                try:
                    sequence = self.queue.get_nowait()
                    if sequence is ToyLLMModelRunner.SENTINEL:
                        stop_listening = True
                        break
                    batch.append(sequence)
                except queue.Empty:
                    break

            if batch:
                # Run batched decode
                pass

            if stop_listening:
                break
