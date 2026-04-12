"""
llm/llm_router.py
Hybrid LLM Router: Groq (primary) → Ollama (fallback)

Logic:
1. Try Groq API first (fast, cloud-based)
2. On rate limit / connection error → switch to Ollama (local)
3. Caches responses to minimize API calls
"""

import time
from typing import Optional, List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from utils.logger import logger
from utils.cache import cache_manager
from config.settings import settings


class GroqClient:
    """Wrapper for Groq API."""

    def __init__(self):
        self.available = False
        self.client = None
        self._init()

    def _init(self):
        if not settings.is_groq_configured():
            logger.warning("Groq API key not configured.")
            return
        try:
            from groq import Groq
            self.client = Groq(api_key=settings.llm.groq_api_key)
            self.available = True
            logger.info(f"Groq initialized: {settings.llm.groq_model}")
        except ImportError:
            logger.warning("groq package not installed.")
        except Exception as e:
            logger.warning(f"Groq init failed: {e}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> Optional[str]:
        """Generate response from Groq."""
        if not self.available or not self.client:
            return None
        try:
            response = self.client.chat.completions.create(
                model=settings.llm.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e).lower()
            if "rate_limit" in error_str or "429" in error_str:
                logger.warning("Groq rate limit hit. Switching to Ollama.")
            elif "authentication" in error_str or "401" in error_str:
                logger.error("Groq authentication failed. Check API key.")
                self.available = False
            else:
                logger.error(f"Groq error: {e}")
            return None


class OllamaClient:
    """Wrapper for local Ollama."""

    def __init__(self):
        self.available = False
        self._init()

    def _init(self):
        try:
            import httpx
            # Quick health check
            response = httpx.get(
                f"{settings.llm.ollama_base_url}/api/tags",
                timeout=3.0
            )
            if response.status_code == 200:
                self.available = True
                logger.info(f"Ollama initialized: {settings.llm.ollama_model}")
            else:
                logger.warning("Ollama server not responding.")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        max_tokens: int = 1000,
        temperature: float = 0.1,
    ) -> Optional[str]:
        """Generate response from Ollama."""
        if not self.available:
            return None
        try:
            import httpx
            payload = {
                "model": settings.llm.ollama_model,
                "prompt": f"System: {system_prompt}\n\nUser: {prompt}",
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                },
            }
            response = httpx.post(
                f"{settings.llm.ollama_base_url}/api/generate",
                json=payload,
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return None


class LLMRouter:
    """
    Intelligently routes LLM calls between Groq and Ollama.
    
    Priority:
    1. Groq (fast, cloud)
    2. Ollama (local fallback)
    3. Error message (if both fail)
    """

    def __init__(self):
        self.groq = GroqClient()
        self.ollama = OllamaClient()
        self._consecutive_groq_failures = 0
        self._groq_cooldown_until = 0

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful AI assistant.",
        max_tokens: int = 1000,
        temperature: float = 0.1,
        use_cache: bool = True,
    ) -> str:
        """
        Route to best available LLM with caching.
        Returns empty string if all fail.
        """
        # Check cache
        if use_cache:
            cache_key_data = {
                "prompt": prompt[:500],
                "system": system_prompt[:200],
                "max_tokens": max_tokens,
            }
            cached = cache_manager.get(
                cache_manager._make_key("llm_response", cache_key_data)
            )
            if cached is not None:
                return cached

        response = None

        # Try Groq (unless in cooldown)
        if self.groq.available and time.time() > self._groq_cooldown_until:
            response = self.groq.generate(prompt, system_prompt, max_tokens, temperature)
            if response:
                self._consecutive_groq_failures = 0
                logger.debug("Response from: Groq")
            else:
                self._consecutive_groq_failures += 1
                if self._consecutive_groq_failures >= 3:
                    self._groq_cooldown_until = time.time() + 60  # 1 min cooldown
                    logger.warning("Groq in 60s cooldown. Using Ollama.")

        # Fallback to Ollama
        if not response and self.ollama.available:
            response = self.ollama.generate(prompt, system_prompt, max_tokens, temperature)
            if response:
                logger.debug("Response from: Ollama (fallback)")

        if not response:
            logger.error("Both LLMs failed to generate a response.")
            response = "I was unable to generate a response. Please check your LLM configuration."

        # Cache successful response
        if use_cache and response:
            cache_key_data = {
                "prompt": prompt[:500],
                "system": system_prompt[:200],
                "max_tokens": max_tokens,
            }
            cache_manager.set(
                cache_manager._make_key("llm_response", cache_key_data),
                response,
                ttl=300,  # 5 min for LLM responses
            )

        return response

    def get_active_provider(self) -> str:
        """Return which LLM is currently primary."""
        if self.groq.available and time.time() > self._groq_cooldown_until:
            return "Groq"
        elif self.ollama.available:
            return "Ollama"
        return "None"

    def health_check(self) -> Dict[str, bool]:
        """Check status of both providers."""
        return {
            "groq": self.groq.available,
            "ollama": self.ollama.available,
            "active": self.get_active_provider(),
        }


# Singleton
llm_router = LLMRouter()

__all__ = ["llm_router", "LLMRouter"]
