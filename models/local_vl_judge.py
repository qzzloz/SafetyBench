"""
Local vision-LLM loader for use as a multimodal LLM-as-a-judge
(Qwen2.5-VL-32B-Instruct).

TEXT judge (local_llm_judge.py) 와 달리, 이 judge 는 IMAGE 를 본다.
SIUO 처럼 이미지+텍스트가 결합돼야 위험이 드러나는 cross-modality 케이스를
제대로 채점하려면 judge 도 이미지를 봐야 한다(텍스트 judge 는 이미지 맥락을 놓침).

4-bit(bitsandbytes) 로딩 + device_map="auto" 로 32B 를 V100 2장/A100 에 분산.
max_pixels 로 비전 토큰을 제한해 이미지가 큰 케이스의 메모리 폭증을 막는다.

인터페이스는 텍스트 judge 와 동일(BaseModel, _generate_single):
  판정 stage 에서 (프롬프트 + [이미지] + 타깃응답) 을 보고 유해성 점수를 낸다.

Register in config/plugins.yaml under `models:`:
    local_vl: [models.local_vl_judge, LocalVLJudge]

And add a model entry in model_config.yaml (별도 provider 블록 local_vl):
    local_vl:
      models:
        qwen2.5-vl-32b-judge:
          model_name: "/scratch/x3411a15/models/Qwen2.5-VL-32B-Instruct"
          max_tokens: 512
          temperature: 0.0
          load_model: true
          load_in_4bit: true
          provider: local_vl
"""

import base64
import re
from io import BytesIO
from typing import List, Optional

from PIL import Image

from .base_model import BaseModel


class LocalVLJudge(BaseModel):
    """Qwen2.5-VL loaded locally (4-bit) for multimodal judging/evaluation."""

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
                AutoProcessor,
                BitsAndBytesConfig,
            )
            # Qwen2.5-VL 전용 클래스 (없으면 범용 ImageTextToText 로 폴백)
            try:
                from transformers import Qwen2_5_VLForConditionalGeneration as _VLModel
            except ImportError:
                try:
                    from transformers import AutoModelForImageTextToText as _VLModel
                except ImportError:
                    from transformers import AutoModelForVision2Seq as _VLModel
            self._VLModel = _VLModel
        except ImportError as e:
            raise ImportError(
                "transformers + torch (+ bitsandbytes for 4-bit) are required "
                f"for LocalVLJudge. Original error: {e}"
            )

        torch_dtype = getattr(torch, dtype, torch.float16)

        self.logger.info(f"[VLJudge] Loading processor: {model_name}")
        # 비전 토큰 상한: 큰 이미지가 메모리를 폭증시키는 것을 막는다.
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            max_pixels=1280 * 28 * 28,
        )

        quant_config = None
        if load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
            self.logger.info("[VLJudge] 4-bit quantization enabled (nf4)")

        self.logger.info("[VLJudge] Loading model weights (sharded across GPUs) ...")
        self.model = self._VLModel.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",            # shard across all visible GPUs
            torch_dtype=torch_dtype,
        )
        self.model.eval()
        self._torch = torch
        self.logger.info("[VLJudge] Judge model ready.")

    def _determine_model_type(self):
        return "local"

    # ---- helpers -------------------------------------------------------

    @staticmethod
    def _data_url_to_image(data_url: str) -> Image.Image:
        """Decode a 'data:image/...;base64,XXXX' URL into a PIL image."""
        match = re.match(r"data:image/[^;]+;base64,(.*)", data_url, re.DOTALL)
        b64 = match.group(1) if match else data_url
        raw = base64.b64decode(b64)
        img = Image.open(BytesIO(raw)).convert("RGB")
        # 큰 이미지 다운스케일 (belt-and-suspenders; processor max_pixels 와 별개 안전장치)
        max_side = 1280
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
        return img

    def _messages_to_qwen(self, messages: List[dict]):
        """
        OpenAI-style messages -> Qwen chat format + PIL 이미지 수집.
        텍스트 judge 와 달리 image_url 블록을 버리지 않고 이미지로 살린다.
        """
        qwen_messages = []
        images = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if isinstance(content, str):
                qwen_messages.append(
                    {"role": role, "content": [{"type": "text", "text": content}]}
                )
                continue
            parts = []
            for block in content or []:
                if block.get("type") == "text":
                    parts.append({"type": "text", "text": block.get("text", "")})
                elif block.get("type") == "image_url":
                    url = block.get("image_url", {}).get("url", "")
                    try:
                        img = self._data_url_to_image(url)
                        images.append(img)
                        parts.append({"type": "image"})
                    except Exception as e:
                        self.logger.debug(f"[VLJudge] image decode skipped: {e}")
            qwen_messages.append({"role": role, "content": parts})
        return qwen_messages, images

    # ---- required abstract methods ------------------------------------

    def _generate_single(self, messages: List[dict], **kwargs) -> str:
        torch = self._torch
        qwen_messages, images = self._messages_to_qwen(messages)

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
            gen_kwargs = dict(max_new_tokens=max_new, do_sample=do_sample)
            if do_sample:
                gen_kwargs["temperature"] = temperature
            generated = self.model.generate(**inputs, **gen_kwargs)

        input_len = inputs.input_ids.shape[1]
        gen_cpu = generated[:, input_len:].to("cpu")
        output_text = self.processor.batch_decode(
            gen_cpu, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        result = output_text[0] if output_text else ""

        # 메모리 정리
        del generated, gen_cpu
        for v in list(inputs.values()):
            del v
        del inputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _generate_stream(self, messages: List[dict], **kwargs):
        yield self._generate_single(messages, **kwargs)