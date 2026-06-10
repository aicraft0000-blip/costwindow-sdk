from typing import Optional


class _TrackedChatCompletions:
    def __init__(self, real_completions, cw, default_label: Optional[str]):
        self._real    = real_completions
        self._cw      = cw
        self._default = default_label

    def create(self, *args, **kwargs):
        label = kwargs.pop("cw_label", self._default)

        response = self._real.create(*args, **kwargs)

        usage = getattr(response, "usage", None)
        if usage and label:
            input_tokens  = getattr(usage, "prompt_tokens",     0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            model = getattr(response, "model", kwargs.get("model", "unknown"))

            self._cw.log(
                label=label,
                provider="openai",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_tokens=0,
            )

        return response


class _TrackedChat:
    def __init__(self, real_chat, cw, default_label: Optional[str]):
        self._real_chat  = real_chat
        self.completions = _TrackedChatCompletions(real_chat.completions, cw, default_label)

    def __getattr__(self, name: str):
        return getattr(self._real_chat, name)


class TrackedOpenAIClient:
    """
    A thin proxy around an `openai.OpenAI()` client.
    Use exactly like the original — add `cw_label="..."` to any `.chat.completions.create()` call.
    """

    def __init__(self, real_client, cw, default_label: Optional[str] = None):
        self._client = real_client
        self._cw     = cw
        self.chat    = _TrackedChat(real_client.chat, cw, default_label)

    def __getattr__(self, name: str):
        return getattr(self._client, name)
