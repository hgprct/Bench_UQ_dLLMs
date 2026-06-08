"""vLLM-accelerated LLM judge for correctness labeling."""

from __future__ import annotations


class EvaluatorLLMLocal:
    """vLLM backend for batched JSON correctness judgments."""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.3-70B-Instruct",
        tensor_parallel_size: int | None = None,
        max_num_seqs: int | None = None,
        gpu_memory_utilization: float | None = None,
    ):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.max_num_seqs = max_num_seqs
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return
        from vllm import LLM
        tp = self.tensor_parallel_size or 1
        kwargs: dict = dict(model=self.model_name, tensor_parallel_size=tp, max_model_len=8192)
        if self.max_num_seqs is not None:
            kwargs["max_num_seqs"] = self.max_num_seqs
        if self.gpu_memory_utilization is not None:
            kwargs["gpu_memory_utilization"] = self.gpu_memory_utilization
        self._llm = LLM(**kwargs)

    def predict_batch(self, prompts: list[str], temperature: float = 0.0, max_tokens: int = 1) -> list[str]:
        self._load()
        from vllm import SamplingParams
        params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        outputs = self._llm.generate(prompts, params)
        return [output.outputs[0].text for output in outputs]
