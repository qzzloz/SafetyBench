"""
Local Qwen2.5-VL target model (transformers, no vLLM server needed).

Loads the weights directly with `from_pretrained` from a local path or HF repo id,
so it works without a running vLLM/OpenAI-compatible server. Implements only
`_generate_single`; the BaseModel parent handles TestCase -> messages (image is
passed as a base64 data URL), text extraction, and ModelResponse construction.

Register in config/plugins.yaml under `models:`:
    qwen_local: [models.qwen_local_model, QwenLocalModel]

And point a model entry at it, e.g. in model_config.yaml:
    providers:
      local:
        models:
          qwen2.5-vl-7b-local:
            provider: local            # -> uses this class via registry
            model_name: "/home/minji/models/Qwen2.5-VL-7B-Instruct"
            max_tokens: 1000
            temperature: 0.0
"""

import base64
import re
from io import BytesIO
from typing import List, Optional

from PIL import Image

from .base_model import BaseModel


class QwenLocalModel(BaseModel):
    """Qwen2.5-VL loaded locally via transformers."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        device: str = "cuda",
        dtype: str = "bfloat16",
    ):
        # No api_key/base_url -> parent marks this as a "local" model type.
        super().__init__(model_name=model_name, api_key=None, base_url=None)

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = device

        try:
            import torch
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
            )
        except ImportError as e:
            raise ImportError(
                "transformers + torch are required for QwenLocalModel. "
                f"Original error: {e}"
            )

        torch_dtype = getattr(torch, dtype, torch.bfloat16)

        self.logger.info(f"[QwenLocal] Loading processor from: {model_name}")
        # use_fast processor; trust_remote_code not needed for Qwen2.5-VL on recent TF
        # Qwen-VL 비전 어텐션은 이미지 패치 수의 제곱으로 메모리를 쓴다.
        # 파일이 작아도 해상도가 높으면(패치多) 추론 중 GPU 메모리가 폭증해 OOM.
        # max_pixels 로 비전 토큰 상한을 두면 큰 이미지를 자동 축소해 폭증을 막는다.
        # 1280*28*28 ≈ 100만 픽셀. 안전 평가에는 충분한 해상도.
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            max_pixels=1280 * 28 * 28,
        )

        self.logger.info(f"[QwenLocal] Loading model weights ({dtype}) ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()
        self._torch = torch
        self.logger.info("[QwenLocal] Model ready.")

    def _determine_model_type(self):
        # Always local regardless of api_key/base_url heuristics.
        return "local"

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _data_url_to_image(data_url: str) -> Image.Image:
        """Decode a 'data:image/...;base64,XXXX' URL into a PIL image."""
        match = re.match(r"data:image/[^;]+;base64,(.*)", data_url, re.DOTALL)
        b64 = match.group(1) if match else data_url
        raw = base64.b64decode(b64)
        img = Image.open(BytesIO(raw)).convert("RGB")
        # 큰 이미지는 메모리를 폭증시켜 OOM 을 유발한다(특정 케이스에서 1.7GB+ 추가 할당).
        # 최대 변을 1280px 로 제한해 픽셀 수를 줄인다(종횡비 유지). 안전 평가에는
        # 이 해상도면 충분하다.
        max_side = 1280
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return img

    def _messages_to_qwen(self, messages: List[dict]):
        """
        Convert BaseModel's OpenAI-style messages into Qwen chat format
        and collect PIL images.

        BaseModel builds, for an image case:
            {"role": "user", "content": [
                {"type": "text", "text": ...},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            ]}
        and for text-only:
            {"role": "user", "content": "..."}
        """
        qwen_messages = []
        images = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")

            if isinstance(content, str):
                qwen_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
                continue

            parts = []
            for block in content or []:
                if block.get("type") == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    img = self._data_url_to_image(url)
                    images.append(img)
                    parts.append({"type": "image"})
            qwen_messages.append({"role": role, "content": parts})

        return qwen_messages, images

    # ---- required abstract methods ------------------------------------

    def _generate_single(self, messages: List[dict], **kwargs) -> str:
        torch = self._torch
        qwen_messages, images = self._messages_to_qwen(messages)

        # Build the text prompt with the chat template.
        text = self.processor.apply_chat_template(
            qwen_messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=images if images else None,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        max_new = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        do_sample = temperature is not None and temperature > 0

        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=max_new, do_sample=do_sample,
                              use_cache=True)
            if do_sample:
                gen_kwargs["temperature"] = temperature
            generated = self.model.generate(**inputs, **gen_kwargs)

        # 생성 결과를 CPU 로 즉시 옮겨 GPU 텐서 참조를 끊는다.
        input_len = inputs.input_ids.shape[1]
        generated_cpu = generated[:, input_len:].to("cpu")

        output_text = self.processor.batch_decode(
            generated_cpu, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result = output_text[0] if output_text else ""

        # GPU 메모리 누적 방지: 모든 GPU 텐서를 명시적으로 해제하고 캐시를 비운다.
        # (Qwen-VL 은 케이스마다 이미지 크기가 달라 메모리가 누적되기 쉽다.
        #  CPU 로 결과를 뺀 뒤 입력/출력 GPU 텐서를 지우고 empty_cache 로 회수.)
        del generated, generated_cpu
        for v in list(inputs.values()):
            del v
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _generate_stream(self, messages: List[dict], **kwargs):
        # Streaming not needed for the benchmark; yield the full result once.
        yield self._generate_single(messages, **kwargs)