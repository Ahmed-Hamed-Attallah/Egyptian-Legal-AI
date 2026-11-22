"""
Local Hugging Face Transformers provider implementation.
"""
import logging
import threading
import time
from typing import Any, Dict, List

from transformers import AutoModelForCausalLM, AutoTokenizer

from src.llm.base_provider import BaseLLMProvider
from src.domain.responses import GenerationRequest, GenerationResponse
from src.domain.exceptions import ProviderError, ProviderUnavailableError
from src.config.settings import settings
from src.config.model_config import resolve_device

logger = logging.getLogger(__name__)

_model_cache: Dict[str, tuple] = {}
_cache_lock = threading.Lock()


class TransformersProvider(BaseLLMProvider):
    """Local Hugging Face Transformers provider with conversation memory support."""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = (
            model_name
            or getattr(settings, "LOCAL_MODEL_NAME", "")
            or "Qwen/Qwen2.5-1.5B-Instruct"
        )
        self.device = device or resolve_device()
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        global _model_cache
        with _cache_lock:
            if self.model_name in _model_cache:
                self.tokenizer, self.model = _model_cache[self.model_name]
                logger.info(f"Loaded local model {self.model_name} from cache.")
                return
            logger.info(f"Loading local model: {self.model_name} on {self.device}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype="auto", device_map=self.device
                )
                _model_cache[self.model_name] = (tokenizer, model)
                self.tokenizer, self.model = tokenizer, model
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                raise ProviderError(f"Model loading failed: {e}", provider="transformers")

    def _build_prompt_from_messages(self, messages: List[Dict[str, str]]) -> str:
        """Convert a messages list into a prompt string.
        
        Uses the tokenizer's chat template if available (preferred for
        instruct models), otherwise falls back to a simple concatenation.
        """
        # Try chat template first (works with Qwen, Llama, Mistral, etc.)
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(
            self.tokenizer, "chat_template", None
        ):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                logger.warning(f"Chat template failed, using fallback: {e}")

        # Fallback: simple concatenation for models without chat templates
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"[System]\n{content}")
            elif role == "user":
                parts.append(f"[User]\n{content}")
            elif role == "assistant":
                parts.append(f"[Assistant]\n{content}")
        parts.append("[Assistant]\n")  # Prompt the model to generate
        return "\n\n".join(parts)

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self.model or not self.tokenizer:
            raise ProviderUnavailableError("Model not loaded", provider="transformers")
        start_time = time.time()
        try:
            # Get the full conversation (with memory if provided)
            messages = request.get_messages()

            # Build prompt from messages
            prompt = self._build_prompt_from_messages(messages)

            inputs = self.tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=2048
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Resolve generation parameters: request overrides -> settings defaults
            do_sample = (
                request.do_sample
                if getattr(request, "do_sample", None) is not None
                else settings.LOCAL_MODEL_DO_SAMPLE
            )
            temperature = (
                request.temperature
                if getattr(request, "temperature", None) is not None
                else settings.LOCAL_MODEL_TEMPERATURE
            )
            max_new_tokens = request.max_tokens or settings.LOCAL_MODEL_MAX_NEW_TOKENS

            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "repetition_penalty": settings.LOCAL_MODEL_REPETITION_PENALTY,
                "no_repeat_ngram_size": settings.LOCAL_MODEL_NO_REPEAT_NGRAM_SIZE,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }

            if do_sample:
                generate_kwargs["temperature"] = max(temperature, 1e-5)
                generate_kwargs["top_p"] = settings.LOCAL_MODEL_TOP_P

            outputs = self.model.generate(**inputs, **generate_kwargs)

            # Decode only the generated portion
            generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
            content = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            latency_ms = (time.time() - start_time) * 1000
            tokens_used = len(generated_tokens)

            return GenerationResponse(
                content=content,
                provider="transformers",
                model=self.model_name,
                tokens_used=tokens_used,
                latency_ms=latency_ms,
                metadata={"message_count": len(messages)},
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"CUDA out of memory: {e}")
                raise ProviderError(f"GPU memory exhausted: {e}", provider="transformers")
            raise
        except Exception as e:
            logger.error(f"Transformers generation failed: {e}")
            raise ProviderError(f"Generation failed: {e}", provider="transformers")

    def get_provider_info(self) -> Dict[str, Any]:
        return {
            "provider": "transformers",
            "model": self.model_name,
            "device": self.device,
            "supports_streaming": False,
            "max_tokens": 2048,
        }

    def health_check(self) -> bool:
        try:
            return self.model is not None and self.tokenizer is not None
        except Exception:
            return False

    def get_name(self) -> str:
        return "transformers"