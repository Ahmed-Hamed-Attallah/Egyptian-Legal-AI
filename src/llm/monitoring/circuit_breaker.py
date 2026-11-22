import time
import threading
from enum import Enum
from typing import Dict, Any, Optional
from src.config.settings import settings

class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, provider: str):
        self.provider = provider
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.cooldown_until = 0.0
        self.lock = threading.Lock()
        
        self.threshold = settings.LLM_CIRCUIT_BREAKER_THRESHOLD
        self.recovery_timeout = settings.LLM_CIRCUIT_BREAKER_RECOVERY_SECONDS

    def allow_request(self) -> bool:
        with self.lock:
            now = time.time()
            if now < self.cooldown_until:
                return False
                
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                if now - self.last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True
            return False

    def record_success(self):
        with self.lock:
            self.failure_count = 0
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
            self.cooldown_until = 0.0

    def record_failure(self, category: str, retry_after: Optional[int] = None):
        with self.lock:
            self.last_failure_time = time.time()
            
            cooldown = settings.LLM_PROVIDER_COOLDOWN_SECONDS
            if category == "RATE_LIMITED":
                cooldown = max(settings.LLM_RATE_LIMIT_COOLDOWN_SECONDS, retry_after or 0)
            elif category in ("QUOTA_EXHAUSTED", "BILLING_LIMIT_REACHED", "INVALID_API_KEY"):
                cooldown = settings.LLM_QUOTA_COOLDOWN_SECONDS
            
            self.cooldown_until = self.last_failure_time + cooldown
            
            # Immediate trip for fatal config/billing errors
            if category in ("QUOTA_EXHAUSTED", "BILLING_LIMIT_REACHED", "INVALID_API_KEY", "MISSING_API_KEY"):
                self.state = CircuitState.OPEN
                self.failure_count = self.threshold
                return

            self.failure_count += 1
            if self.failure_count >= self.threshold:
                self.state = CircuitState.OPEN

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "state": self.state.value,
                "failure_count": self.failure_count,
                "cooldown_until": self.cooldown_until,
                "allow_request": self.allow_request()
            }
            
    def reset(self):
        with self.lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.cooldown_until = 0.0