from typing import Optional


class _TrackedMessages:
    """Proxy for `client.messages` that intercepts `.create()` calls."""

    def __init__(self, real_messages, cw, default_label: Optional[str]):
        self._real    = real_messages
        self._cw      = cw
        self._default = default_label

    def create(self, *args, **kwargs):
        label = kwargs.pop("cw_label", self._default)

        # Make the real Anthropic API call — cw_label stripped
        response = self._real.create(*args, **kwargs)

        # Capture token usage from the response
        usage = getattr(response, "usage", None)
        if usage and label:
            input_tokens  = getattr(usage, "input_tokens",  0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            cached_tokens = getattr(usage, "cache_read_input_tokens", 0) or 0
            model = getattr(response, "model", kwargs.get("model", "unknown"))

            self._cw.log(
                label=label,
                provider="anthropic",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=cached_tokens,
            )

        return response

    def stream(self, *args, **kwargs):
        """Pass-through for streaming calls."""
        kwargs.pop("cw_label", None)
        return self._real.stream(*args, **kwargs)


class TrackedAnthropicClient:
    """
    A thin proxy around an `anthropic.Anthropic()` client.
    Use exactly like the original — add `cw_label="..."` to any `.messages.create()` call.
    """

    def __init__(self, real_client, cw, default_label: Optional[str] = None):
        self._client  = real_client
        self._cw      = cw
        self.messages = _TrackedMessages(real_client.messages, cw, default_label)

    def __getattr__(self, name: str):
        return getattr(self._client, name)
