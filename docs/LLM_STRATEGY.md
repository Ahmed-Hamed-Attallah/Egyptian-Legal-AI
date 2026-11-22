# Egyptian Legal AI — LLM Strategy

## 1. Objective

Implement a provider-independent LLM layer supporting API-based generation as the primary option and local Transformers inference as a fallback.

No model fine-tuning is planned initially.

## 2. Provider Architecture

```text
             LLM REQUEST
                  |
                  v
           PROVIDER ROUTER
                  |
                  v
          PRIMARY API PROVIDER
                  |
           +------+------+
           |             |
         SUCCESS        FAILURE
           |             |
           v             v
         RETURN       RETRY POLICY
                         |
                    FALLBACK MANAGER
                         |
                         v
                LOCAL TRANSFORMERS
                         |
                    +----+----+
                    |         |
                  SUCCESS   FAILURE
                    |         |
                    v         v
                  RETURN   CONTROLLED
                           ERROR
```

## 3. Initial Providers

### Primary: Groq

Responsibilities:
- API-based text generation.
- Prompt submission.
- Response handling.
- Timeout handling.
- Usage tracking.

### Fallback: Hugging Face Transformers

Responsibilities:
- Local model loading.
- Local text generation.
- Device selection.
- CPU fallback.
- Memory-aware inference.

### Optional Future Providers

- OpenAI.
- Gemini.
- Additional compatible inference providers.

Do not implement all providers before validating the common interface.

## 4. Common Provider Interface

Every provider should implement consistent methods:

```python
class BaseLLMProvider:

    def generate(self, messages, **kwargs):
        raise NotImplementedError

    def generate_stream(self, messages, **kwargs):
        raise NotImplementedError

    def health_check(self):
        raise NotImplementedError

    def get_model_info(self):
        raise NotImplementedError
```

Use appropriate abstract base classes and typed request/response models in the final implementation.

## 5. Fallback Policy

| Failure | Action |
|---|---|
| Successful API request | Return API response |
| Temporary timeout | Retry, then fallback |
| HTTP 429 | Apply retry/backoff policy, then fallback |
| HTTP 5xx | Retry, then fallback |
| Invalid API key | Mark provider unavailable and fallback |
| Invalid user request | Return validation error |
| Local model unavailable | Return controlled service error |

Do not hide failures or silently return fabricated answers.

## 6. Local Model Constraints

Hardware:
- 32 GB RAM.
- 4 GB GPU VRAM.

Model selection must consider:
- Parameter count.
- Quantization.
- Context length.
- Inference speed.
- Arabic language performance.
- Available system memory.
- CPU inference capability.

A small instruct model may be suitable for initial experiments, but it must be evaluated on Arabic legal questions before adoption.

## 7. Legal Generation Rules

The LLM must:
- Use retrieved context as the factual basis.
- Cite supporting legal sources.
- Preserve article numbers and law references.
- Distinguish current and historical legal text.
- Avoid inventing legal provisions.
- State when the retrieved context is insufficient.
- Distinguish information from legal advice.

## 8. Monitoring

Track:
- Active provider.
- Active model.
- Request latency.
- Token usage when available.
- Error type.
- Retry count.
- Fallback activation.
- Fallback success rate.

## 9. Configuration

Provider settings must come from centralized application configuration.

Never hardcode API keys.

The system should expose which provider generated each response through internal metadata and appropriate operational logs.