from abc import ABC, abstractmethod
from typing import List, Optional, Union, Dict, Any
import inspect
import backoff
from core.base_classes import BaseModel as CoreBaseModel
from core.data_formats import TestCase, ModelResponse


class BaseModel(CoreBaseModel):
    """Base class for all model implementations."""

    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_CONTENT_REJECTION_OUTPUT = (
        "[ERROR] Prompt detected as harmful content, refusing to answer"
    )
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 10
    API_TIMEOUT = 600

    # Content policy detection keywords (common across providers)
    CONTENT_POLICY_KEYWORDS = [
        "content policy",
        "safety",
        "harmful",
        "unsafe",
        "violation",
        "moderation",
        "data_inspection_failed",
        "inappropriate content",
    ]

    # Provider-specific additional keywords (subclasses can override)
    PROVIDER_SPECIFIC_KEYWORDS = []

    def __init__(self, model_name: str, api_key: str = None, base_url: str = None):
        # Call parent class __init__, pass empty configuration
        super().__init__(config={})
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.model_type = self._determine_model_type()

    @classmethod
    def from_config(cls, name: str, config: Dict[str, Any] | None = None):
        """
        Unified factory method: build model instance from a config dict.

        This is intentionally signature-aware: it only passes kwargs that the concrete
        model class' __init__ accepts, so providers with different constructor
        signatures won't break.
        """
        config = config or {}

        # model_name in config takes priority; fall back to the alias name
        model_name = config.get("model_name", name)
        api_key = config.get("api_key", "")
        base_url = config.get("base_url", None)

        init_kwargs = {
            "model_name": model_name,
            "api_key": api_key,
            "base_url": base_url,
        }

        # Filter kwargs by the concrete __init__ signature
        try:
            sig = inspect.signature(cls.__init__)
            accepted = {
                p.name
                for p in sig.parameters.values()
                if p.name != "self"
                and p.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            }
            filtered_kwargs = {k: v for k, v in init_kwargs.items() if k in accepted}
            return cls(**filtered_kwargs)
        except Exception:
            # Fallback: best-effort positional init
            return cls(model_name=model_name, api_key=api_key, base_url=base_url)

    def _determine_model_type(self):
        """Determine model type: api (API call) or local (local loading)"""
        # Default implementation: determine based on whether api_key exists
        # Subclasses can override this method
        if self.api_key or self.base_url:
            return "api"
        else:
            return "local"

    def _is_content_policy_rejection(self, error: Exception) -> bool:
        """Check if an exception represents a content policy rejection.

        Args:
            error: The exception to check

        Returns:
            True if the error indicates content policy rejection
        """
        error_str = str(error).lower()

        # Combine common and provider-specific keywords
        all_keywords = self.CONTENT_POLICY_KEYWORDS + self.PROVIDER_SPECIFIC_KEYWORDS

        return any(keyword in error_str for keyword in all_keywords)

    def _handle_content_rejection(self) -> str:
        """Return the standard content rejection output."""
        return self.API_CONTENT_REJECTION_OUTPUT

    def _handle_content_rejection_stream(self):
        """Return a generator that yields the content rejection placeholder."""

        def error_generator():
            yield self.API_CONTENT_REJECTION_OUTPUT

        return error_generator()

    def _handle_api_error(self) -> str:
        """Return the standard API error output."""
        return self.API_ERROR_OUTPUT

    def _handle_api_error_stream(self):
        """Return a generator that yields the API error placeholder."""

        def error_generator():
            yield self.API_ERROR_OUTPUT

        return error_generator()

    def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with retry logic and exponential backoff using backoff library."""

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=self.API_MAX_RETRY,
            max_time=self.API_TIMEOUT * 2,
            base=2,
            factor=30,
            on_backoff=lambda details: print(
                f"Attempt {details['tries']} failed: {details['exception'].__class__.__name__}: {details['exception']}"
            ),
            on_giveup=lambda details: print(
                f"Final attempt failed after {details['tries']} tries: {details['exception'].__class__.__name__}: {details['exception']}"
            ),
        )
        def _execute():
            return func(*args, **kwargs)

        try:
            return _execute()
        except Exception as e:
            raise

    @abstractmethod
    def _generate_single(
        self,
        messages: List[dict],
        **kwargs,
    ) -> str:
        """Generate response for a single prompt."""
        pass

    @abstractmethod
    def _generate_stream(
        self,
        messages: List[dict],
        **kwargs,
    ):
        """Generate streaming response for a single prompt."""
        pass

    def generate_response(self, test_case: TestCase, **kwargs) -> ModelResponse:
        """
        Generate model response

        Args:
            test_case: Test case
            **kwargs: Other parameters

        Returns:
            ModelResponse: Model response
        """
        # Convert TestCase to message list
        messages = self._test_case_to_messages(test_case)

        # Generate response
        response = self.generate(messages, **kwargs)
        response_text = self._extract_text(response)

        return ModelResponse(
            test_case_id=test_case.test_case_id,
            model_response=response_text,
            model_name=self.model_name,
            metadata=test_case.metadata,
        )

    def _extract_text(self, response_obj: Any) -> str:
        """
        Extract plain text from various provider SDK response objects.

        Why this exists:
        - Some providers return OpenAI-like objects with `.choices[0].message.content`
        - Some return `.text` (e.g., certain Gemini SDK objects)
        - Some return `content` blocks (e.g., Anthropic)
        - Our retry/error handling may return a string placeholder directly
        """
        if response_obj is None:
            return ""

        # Placeholder / already-a-text
        if isinstance(response_obj, str):
            return response_obj

        # Dict-like responses
        if isinstance(response_obj, dict):
            # OpenAI-like dict
            if "choices" in response_obj and response_obj["choices"]:
                try:
                    choice0 = response_obj["choices"][0]
                    # message.content
                    if isinstance(choice0, dict):
                        msg = choice0.get("message")
                        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                            return msg["content"]
                        delta = choice0.get("delta")
                        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                            return delta["content"]
                except Exception:
                    pass
            if isinstance(response_obj.get("text"), str):
                return response_obj["text"]
            content = response_obj.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, str):
                        texts.append(block)
                    elif isinstance(block, dict) and isinstance(block.get("text"), str):
                        texts.append(block["text"])
                if texts:
                    return "".join(texts)

            return str(response_obj)

        # OpenAI-like objects: response.choices[0].message.content
        try:
            choices = getattr(response_obj, "choices", None)
            if choices and len(choices) > 0:
                choice0 = choices[0]
                msg = getattr(choice0, "message", None)
                if msg is not None:
                    content = getattr(msg, "content", None)
                    if isinstance(content, str):
                        return content
                # Some SDKs expose delta for streaming chunks
                delta = getattr(choice0, "delta", None)
                if delta is not None:
                    delta_content = getattr(delta, "content", None)
                    if isinstance(delta_content, str):
                        return delta_content
        except Exception:
            pass

        # Gemini-like objects: response.text
        try:
            text = getattr(response_obj, "text", None)
            if isinstance(text, str):
                return text
        except Exception:
            pass

        # Anthropic-like objects: response.content is a list of blocks with .text
        try:
            content = getattr(response_obj, "content", None)
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = []
                for block in content:
                    if isinstance(block, str):
                        texts.append(block)
                    else:
                        block_text = getattr(block, "text", None)
                        if isinstance(block_text, str):
                            texts.append(block_text)
                        elif isinstance(block, dict) and isinstance(block.get("text"), str):
                            texts.append(block["text"])
                if texts:
                    return "".join(texts)
        except Exception:
            pass

        # Last resort: stringify
        return str(response_obj)

    def generate_responses_batch(
        self, test_cases: List[TestCase], **kwargs
    ) -> List[ModelResponse]:
        """
        Batch generate model responses (for locally loaded models)

        Args:
            test_cases: List of test cases
            **kwargs: Other parameters

        Returns:
            List[ModelResponse]: List of model responses
        """
        # Default implementation: loop calling single generation
        # Local models should override this method to implement true batch inference
        responses = []
        for test_case in test_cases:
            response = self.generate_response(test_case, **kwargs)
            responses.append(response)
        return responses

    def _test_case_to_messages(self, test_case: TestCase) -> List[Dict[str, Any]]:
        """Convert TestCase to message list"""
        messages = []

        # If there is an image, add image message
        if test_case.image_path:
            try:
                # Check if image file exists
                import os
                from pathlib import Path

                image_path = Path(test_case.image_path)

                # Load image and encode as base64
                from PIL import Image
                import base64
                from io import BytesIO

                # Open image
                image = Image.open(image_path)

                # Encode image as base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Create data URL
                data_url = f"data:image/png;base64,{img_str}"

                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": test_case.prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": data_url},
                            },
                        ],
                    }
                )

            except Exception as e:
                self.logger.error(f"Error processing image: {e}")
                # Fallback to plain text on error
                messages.append({"role": "user", "content": test_case.prompt})
        else:
            # Plain text message
            messages.append({"role": "user", "content": test_case.prompt})

        return messages

    def generate(
        self,
        messages: Union[List[dict], List[List[dict]]],
        use_tqdm: bool = False,
        stream: bool = False,
        **kwargs,
    ) -> Union[str, List[str]]:
        """Generate responses for multiple prompts.

        Args:
            messages: Single message list or list of message lists
            use_tqdm: Whether to show progress bar
            stream: Whether to use streaming output
            **kwargs: Additional model-specific parameters

        Returns:
            Single response string or list of generated responses
        """
        if isinstance(messages, list) and all(
            isinstance(msg, dict) for msg in messages
        ):
            if stream:
                return self._generate_stream(messages, **kwargs)
            else:
                return self._generate_single(messages, **kwargs)

        if use_tqdm:
            from tqdm import tqdm

            messages = tqdm(messages)

        if stream:
            # For multiple messages with streaming, return a list of generators
            return [self._generate_stream(msg_set, **kwargs) for msg_set in messages]
        else:
            return [self._generate_single(msg_set, **kwargs) for msg_set in messages]
