from multiprocessing import Queue

from config import ToyLLMConfig
from sequence import Sequence, SequenceStatus


class Tokenizer:
    """Stub tokenizer.  Replace ``tokenize`` with a real implementation."""

    # Dummy EOS token ID used to detect end-of-sequence during generation.
    EOS_TOKEN_ID = 10000

    def __init__(self, config: ToyLLMConfig):
        self.config = config

    def tokenize(self, prompt: str) -> list[int]:
        """
        Convert *prompt* to a list of token IDs.

        Parameters
        ----------
        prompt : str
            Raw text prompt.

        Returns
        -------
        list[int]
            Sequence of integer token IDs.
        """
        # TODO: implement with a real tokeniser (e.g. tiktoken, sentencepiece).
        pass


class Scheduler:
    """
    Coordinates the flow of sequences between the tokeniser, prefill runner,
    and decode runner.

    Queue ownership
    ---------------
    * ``prefill_queue`` – scheduler **produces**; prefill runner **consumes**.
    * ``decode_schedule_queue`` – scheduler **consumes**; both runners **produce**.
      Carries sequences that have just completed a prefill chunk or a decode step
      and need the scheduler to decide what happens next.
    * ``decode_worker_queue`` – scheduler **produces**; decode runner **consumes**.
      Carries sequences that are ready for the next decode step.

    Parameters
    ----------
    llm_config : ToyLLMConfig
        Unified model and inference configuration.
    tokenizer : Tokenizer
        Used to convert raw prompt strings to token ID lists.
    prompts : list[str]
        The initial set of prompts to process.
    decode_schedule_queue : Queue[Sequence]
        Inbound queue: receives sequences from the runners after each step.
    prefill_queue : Queue[Sequence]
        Outbound queue: newly tokenised sequences waiting for prefill.
    decoder_queue : Queue[Sequence]
        Outbound queue: sequences ready for the next decode step.

    Attributes
    ----------
    finished_sequences : list[Sequence]
        Accumulates sequences that have reached EOS or max length.
    """

    SENTINEL = None

    def __init__(
            self,
            llm_config: ToyLLMConfig,
            tokenizer: Tokenizer,
            prompts: list[str],
            decode_schedule_queue: Queue[Sequence],
            prefill_queue: Queue[Sequence],
            decoder_queue: Queue[Sequence],
    ):
        self.llm_config = llm_config
        self.tokenizer = tokenizer
        self.prompts = prompts

        # Inbound: receives sequences after each prefill chunk or decode step.
        self.decode_schedule_queue = decode_schedule_queue
        # Outbound: newly tokenised sequences waiting for prefill.
        self.prefill_queue = prefill_queue
        # Outbound: sequences dispatched for the next decode step.
        self.decode_worker_queue = decoder_queue

        self.eos_token_id = Tokenizer.EOS_TOKEN_ID
        self.finished_sequences: list[Sequence] = []

    def step(self) -> None:
        """Single scheduler tick (stub — extend as needed)."""
        pass

    def run(self) -> None:
        """
        Main scheduler loop.

        1. Tokenises all input prompts and pushes them onto ``prefill_queue``.
        2. Reads completed-step sequences from ``decode_schedule_queue``.
        3. Sequences that generated EOS are marked ``FINISHED`` and collected.
        4. All other sequences are re-dispatched to ``decode_worker_queue`` for
           their next decode step.

        Exits when it receives the sentinel value from ``decode_schedule_queue``.
        """
        # Send all prompts to the prefill runner.
        for prompt in self.prompts:
            self.prefill_queue.put(
                Sequence(
                    self.tokenizer.tokenize(prompt),
                    self.llm_config.max_token_size_per_kv_cache_block,
                    self.llm_config.max_sequence_length,
                    self.llm_config.query_chunk_size,
                )
            )

        # Process sequences as they complete each step.
        while True:
            # Read from the inbound queue: sequences returned by the runners.
            sequence = self.decode_schedule_queue.get()
            if sequence is Scheduler.SENTINEL:
                break

            if sequence.get_last_token_id() == self.eos_token_id:
                sequence.status = SequenceStatus.FINISHED
                self.finished_sequences.append(sequence)
                continue

            # Not finished — dispatch for the next decode step.
            self.decode_worker_queue.put(sequence)