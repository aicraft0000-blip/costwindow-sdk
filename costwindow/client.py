import requests
from datetime import datetime, timezone
from typing import Optional
from costwindow.exceptions import CostWindowProcessDisabledError
from costwindow.logger import get_local_logger

# Must point at the live backend — failures in .log() are swallowed by design,
# so a wrong default means customers silently record nothing.
DEFAULT_BACKEND_URL = "https://costwindow-site-production.up.railway.app"


class CostWindow:
    def __init__(self, api_key: str, backend_url: str = DEFAULT_BACKEND_URL):
        self.api_key     = api_key
        self.backend_url = backend_url.rstrip("/")
        self._logger     = get_local_logger()

    def log(
        self,
        label: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        called_at: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Report a completed API call to CostWindow.

        Raises CostWindowProcessDisabledError if the process has been killed.
        All other errors (network, timeout, server errors) are logged locally
        and swallowed — CostWindow will never break your script.
        """
        payload = {
            "label":         label,
            "provider":      provider,
            "model":         model,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
            "cached_tokens": cached_tokens,
            "called_at":     (called_at or datetime.now(timezone.utc)).isoformat(),
            "metadata":      metadata,
        }
        self._send(label, payload)

    def _send(self, label: str, payload: dict) -> None:
        try:
            response = requests.post(
                f"{self.backend_url}/ingest",
                json=payload,
                headers={"X-Api-Key": self.api_key},
                timeout=5,
            )
            data = response.json()
            if data.get("status") == "blocked":
                raise CostWindowProcessDisabledError(label)
        except CostWindowProcessDisabledError:
            raise
        except Exception as exc:
            self._logger.warning(
                "CostWindow unreachable — data not recorded. "
                "label=%s error=%s payload=%s", label, exc, payload
            )

    def is_active(self, label: str) -> bool:
        """
        Returns True if the process is enabled in CostWindow, False if kill-switched.
        Fails open: returns True if CostWindow is unreachable, so cron jobs aren't
        accidentally stopped by a network hiccup.
        """
        try:
            resp = requests.get(
                f"{self.backend_url}/processes/status/{label}",
                headers={"X-Api-Key": self.api_key},
                timeout=3,
            )
            if resp.status_code == 200:
                return bool(resp.json().get("is_active", True))
            return True
        except Exception:
            return True

    def wrap_anthropic(self, client, label: Optional[str] = None):
        """Return a wrapped Anthropic client that auto-captures token usage."""
        from costwindow.wrappers.anthropic import TrackedAnthropicClient
        return TrackedAnthropicClient(client, self, default_label=label)

    def wrap_openai(self, client, label: Optional[str] = None):
        """Return a wrapped OpenAI client that auto-captures token usage."""
        from costwindow.wrappers.openai import TrackedOpenAIClient
        return TrackedOpenAIClient(client, self, default_label=label)

    def track(self, label: str, provider: str = "anthropic"):
        """Context manager: track all calls in a block under one label."""
        from costwindow.wrappers.context import TrackingContext
        return TrackingContext(self, label, provider)
