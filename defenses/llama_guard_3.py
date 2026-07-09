from core.base_classes import BaseDefense, BaseGuard
from core.data_formats import TestCase
from core.unified_registry import UNIFIED_REGISTRY
from .utils import generate_output
from config.config_loader import get_model_config
from PIL import Image

try:
    from transformers import AutoModelForImageTextToText as _AutoVisionModel
except ImportError:  # older transformers
    from transformers import AutoModelForVision2Seq as _AutoVisionModel
from transformers import AutoProcessor
import torch
import threading


class LlamaGuard3Defense(BaseGuard):
    """Llama-Guard-3 defense method - GPU inference thread-safe version"""

    # Class variables: singleton instance and locks
    _instance = None
    _instance_lock = threading.Lock()
    _model_loaded = False
    _shared_target_model = None   # 타깃 모델을 한 번만 만들어 모든 케이스가 재사용

    # Model loading lock (protects model loading process)
    _model_init_lock = threading.Lock()

    # Inference lock (protects model inference process)
    _inference_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton pattern: ensure only one instance"""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config):
        """Initialize model (execute only once)"""
        # Double-check locking to ensure model is loaded only once
        if not self._model_loaded:
            with self._model_init_lock:
                if not self._model_loaded:
                    # Call parent class initialization
                    super().__init__(config)

                    # Log initialization start
                    self.logger.info(
                        "Starting Llama-Guard-3 model initialization (GPU version)"
                    )

                    # Get model path
                    llama_guard_path = self.config["Llama-Guard-3"]
                    self.logger.info(f"Model path: {llama_guard_path}")

                    # Check GPU availability
                    if not torch.cuda.is_available():
                        self.logger.warning("CUDA not available, will use CPU")
                        self.device = torch.device("cpu")
                    else:
                        # 가드는 타깃과 다른 GPU 에 올려 경쟁을 피한다.
                        # 기본 cuda:1 (타깃은 cuda:0). GPU 가 1장뿐이면 cuda:0 로 폴백.
                        # 환경변수 GUARD_DEVICE 로 덮어쓸 수 있음 (예: "cuda:0").
                        import os
                        n_gpu = torch.cuda.device_count()
                        guard_dev = os.environ.get("GUARD_DEVICE")
                        if guard_dev is None:
                            guard_dev = "cuda:1" if n_gpu >= 2 else "cuda:0"
                        self.device = torch.device(guard_dev)
                        self.logger.info(f"Using device: {self.device} (visible GPUs: {n_gpu})")

                        # Display GPU information
                        dev_idx = self.device.index or 0
                        gpu_name = torch.cuda.get_device_name(dev_idx)
                        gpu_memory = (
                            torch.cuda.get_device_properties(dev_idx).total_memory / 1e9
                        )
                        self.logger.info(
                            f"GPU: {gpu_name}, Memory: {gpu_memory:.2f} GB"
                        )

                    try:
                        # Load processor
                        self.logger.info("Loading processor...")
                        self.judge_processor = AutoProcessor.from_pretrained(
                            llama_guard_path
                        )

                        # Load model to specified device
                        self.logger.info("Loading model...")
                        self.judge_model = _AutoVisionModel.from_pretrained(
                            llama_guard_path,
                            torch_dtype=torch.bfloat16,
                            device_map=None,  # Don't use automatic device mapping to avoid competition
                        )

                        # Manually move model to device
                        self.judge_model = self.judge_model.to(self.device)
                        self.judge_model.eval()

                        # Log model information
                        model_params = sum(
                            p.numel() for p in self.judge_model.parameters()
                        )
                        self.logger.info(
                            f"Model loaded, parameters: {model_params / 1e9:.2f}B"
                        )
                        self.logger.info(
                            f"Model device: {next(self.judge_model.parameters()).device}"
                        )

                        # Mark model as loaded
                        self._model_loaded = True

                    except Exception as e:
                        self.logger.error(f"Model loading failed: {e}")
                        raise

    def apply_defense(self, test_case: TestCase, **kwargs) -> TestCase:
        """Apply defense method - thread-safe version"""

        idx = test_case.test_case_id
        attack_image_path = test_case.image_path
        attack_prompt = test_case.prompt

        # Step 1: Get the victim model's response.
        # 통합 평가 모드: 앞 단계(response_generation)에서 저장해 둔 응답이
        # metadata 로 넘어오면, 타깃 모델을 GPU 에 다시 올리지 않고 그대로 사용한다.
        # (output-side 가드를 타깃과 동시 적재하지 않게 해 OOM 을 피하고,
        #  모든 가드가 '동일한 응답'을 평가하도록 보장한다.)
        precomputed = (
            (test_case.metadata or {}).get("precomputed_response")
            or (test_case.metadata or {}).get("target_response")
            or kwargs.get("precomputed_response")
        )

        if precomputed:
            self.logger.debug(f"[{idx}] Step 1: Using precomputed victim response (no target load)")
            original_output = precomputed
        else:
            self.logger.debug(f"[{idx}] Step 1: No precomputed response -> using target model")
            # 타깃 모델을 케이스마다 새로 만들면 GPU 에 모델이 계속 쌓여 OOM 이 난다.
            # → 클래스 레벨에 한 번만 생성해 캐싱하고, 이후 케이스는 재사용한다.
            #   (가드 self.judge_model 이 한 번만 로딩되는 것과 동일한 패턴)
            target_model = type(self)._shared_target_model
            if target_model is None:
                target_model_name = self.config["target_model_name"]
                model_config = get_model_config(target_model_name)
                if not model_config:
                    model_params = self.config.get("model_params", {})
                    model_config = model_params.get(target_model_name, {})
                target_model = UNIFIED_REGISTRY.create_model(target_model_name, model_config)
                if target_model is None:
                    raise RuntimeError(
                        f"[{idx}] target model '{target_model_name}' could not be created "
                        f"and no precomputed_response was provided in metadata."
                    )
                type(self)._shared_target_model = target_model
                self.logger.info(f"[{idx}] Target model loaded once and cached for reuse.")
            original_output = generate_output(
                attack_image_path, attack_prompt, target_model
            )

        # Step 2: Build Llama-Guard-3 input
        self.logger.debug(f"[{idx}] Step 2: Build input")
        image = Image.open(attack_image_path).convert("RGB")

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": attack_prompt},
                    {"type": "image"},
                ],
            },
            {
                "role": "agent",
                "content": [{"type": "text", "text": original_output}],
            },
        ]

        # Step 3: Call Llama-Guard-3 for judgment (protected by lock)
        self.logger.debug(f"[{idx}] Step 3: Prepare model input")
        input_prompt = self.judge_processor.apply_chat_template(
            conversation, return_tensors="pt"
        )

        inputs = self.judge_processor(
            text=input_prompt, images=image, return_tensors="pt"
        ).to(self.device)

        prompt_len = len(inputs["input_ids"][0])

        # Key: Use lock to protect model inference
        self.logger.debug(
            f"[{idx}] Waiting for inference lock, thread: {threading.get_ident()}"
        )
        with self._inference_lock:
            self.logger.debug(
                f"[{idx}] Starting inference, thread: {threading.get_ident()}"
            )

            # Use no_grad to reduce memory usage
            with torch.no_grad():
                judge_outputs = self.judge_model.generate(
                    **inputs,
                    max_new_tokens=20,
                    pad_token_id=0,
                    do_sample=False,  # Greedy decoding, faster and more stable
                    num_beams=1,  # Don't use beam search
                    use_cache=True,  # Enable cache
                    output_scores=True,             # 확률 추출용 logits
                    return_dict_in_generate=True,   # scores 를 받으려면 필요
                )

            seq = judge_outputs.sequences
            generated_tokens = seq[:, prompt_len:]
            judge_response = self.judge_processor.decode(generated_tokens[0])

            # --- P(unsafe) 확률 추출 (pilot 과 동일 로직) ---
            # Llama-Guard 출력은 '\n\n' 다음에 safe/unsafe 가 온다.
            # safe/unsafe 토큰이 처음 등장한 step 의 logits 에서 확률을 읽는다.
            unsafe_prob = safe_prob = unsafe_prob_norm = None
            try:
                tok = self.judge_processor.tokenizer
                safe_id = tok.convert_tokens_to_ids("safe")
                unsafe_id = tok.convert_tokens_to_ids("unsafe")
                gen_ids = generated_tokens[0].tolist()
                decision_step = None
                for i, tid in enumerate(gen_ids):
                    if tid in (safe_id, unsafe_id):
                        decision_step = i
                        break
                if decision_step is not None and decision_step < len(judge_outputs.scores):
                    logits = judge_outputs.scores[decision_step][0].float()
                    probs = torch.softmax(logits, dim=-1)
                    unsafe_prob = float(probs[unsafe_id])
                    safe_prob = float(probs[safe_id])
                    denom = unsafe_prob + safe_prob
                    unsafe_prob_norm = (unsafe_prob / denom) if denom > 0 else None
            except Exception as _e:
                self.logger.debug(f"[{idx}] prob extraction skipped: {_e}")

            # GPU 메모리 누적 방지: 가드 추론 텐서를 해제한다.
            del judge_outputs, generated_tokens, seq
            for _v in list(inputs.values()):
                del _v
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Step 4: Make decision based on judgment result
        self.logger.debug(f"[{idx}] Step 4: Make decision based on judgment result")
        # 판정 결과에 확률 + 타깃 원래 응답을 함께 기록.
        #   [중요] 가드가 막으면 model_response 가 거부 메시지로 교체되어
        #   타깃의 진짜 응답이 사라진다. 그러면 judge 가 거부 메시지를 채점해
        #   무조건 무해로 보게 되어 ASR/recall 이 왜곡된다.
        #   → 타깃 원래 응답을 target_response 로 항상 보존한다.
        prob_meta = {
            "guard_unsafe_prob": unsafe_prob,
            "guard_safe_prob": safe_prob,
            "guard_unsafe_prob_norm": unsafe_prob_norm,
            "target_response": original_output,   # 타깃 진짜 응답 (가드 차단과 무관하게 보존)
        }
        if "unsafe" in judge_response.lower():
            result = self.block_input(test_case)
        else:
            result = self.reply_directly(test_case, response_text=original_output)
        # 확률 + 타깃 응답 정보를 metadata 에 병합
        try:
            if result.metadata is None:
                result.metadata = {}
            result.metadata.update(prob_meta)
        except Exception:
            pass
        return result