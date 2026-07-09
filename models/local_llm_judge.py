"""
Local large text-LLM loader for use as an LLM-as-a-judge (e.g. Llama-3.3-70B-Instruct).

Loaded in 4-bit (bitsandbytes) so a 70B model fits on 1-2 V100/A100 GPUs, and
sharded across visible GPUs via device_map="auto". This is a TEXT model (no
vision) intended for the evaluation/judge stage: it scores whether a target
response is harmful, given the prompt + response. The image is not needed for
text-level harmfulness judgments (consistent with JailbreakBench / HarmBench
judge practice).

Register in config/plugins.yaml under `models:`:
    local_llm: [models.local_llm_judge, LocalLLMJudge]

And add a model entry in model_config.yaml:
    local:
      models:
        llama-3.3-70b-judge:
          model_name: "/scratch/x3411a15/models/Llama-3.3-70B-Instruct"
          max_tokens: 512
          temperature: 0.0
          load_model: true        # heavy local model -> single worker, load once
          load_in_4bit: true       # 4-bit quantization (needs bitsandbytes)
"""

from typing import List, Optional

from .base_model import BaseModel


class LocalLLMJudge(BaseModel):
    """Large text LLM loaded locally (4-bit) for judging/evaluation."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        load_in_4bit: bool = True,
        dtype: str = "float16",
    ):
        super().__init__(model_name=model_name, api_key=None, base_url=None)

        self.max_tokens = max_tokens
        self.temperature = temperature

        try:
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                BitsAndBytesConfig,
            )
        except ImportError as e:
            raise ImportError(
                "transformers + torch (+ bitsandbytes for 4-bit) are required "
                f"for LocalLLMJudge. Original error: {e}"
            )

        torch_dtype = getattr(torch, dtype, torch.float16)

        self.logger.info(f"[Judge] Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = None
        if load_in_4bit:
            # 4-bit NF4 quantization; compute in fp16 (V100-safe).
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.logger.info("[Judge] 4-bit quantization enabled (nf4)")

        self.logger.info(f"[Judge] Loading model weights (sharded across GPUs) ...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",            # shard across all visible GPUs
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        self._torch = torch
        self.logger.info("[Judge] Judge model ready.")

    def _determine_model_type(self):
        return "local"

    # ---- helpers -------------------------------------------------------

    def _messages_to_text(self, messages: List[dict]) -> str:
        """
        Render OpenAI-style messages with the model's chat template.
        The judge is text-only; any image_url blocks are ignored (text kept).
        """
        norm = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if b.get("type") == "text"]
                content = "\n".join(t for t in texts if t)
            norm.append({"role": m.get("role", "user"), "content": content or ""})
        return self.tokenizer.apply_chat_template(
            norm, tokenize=False, add_generation_prompt=True
        )

    # ---- required abstract methods ------------------------------------

    def _generate_single(self, messages: List[dict], **kwargs) -> str:
        torch = self._torch
        text = self._messages_to_text(messages)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        max_new = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        do_sample = temperature is not None and temperature > 0

        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=max_new, do_sample=do_sample,
                              pad_token_id=self.tokenizer.pad_token_id)
            if do_sample:
                gen_kwargs["temperature"] = temperature
            out = self.model.generate(**inputs, **gen_kwargs)

        gen = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True)

    def _generate_stream(self, messages: List[dict], **kwargs):
        yield self._generate_single(messages, **kwargs)