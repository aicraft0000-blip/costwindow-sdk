class TrackingContext:
    """
    Context manager that sets a default label for all CostWindow.log() calls
    within the block. Useful when you can't modify individual API call sites.

    Usage:
        with cw.track("hermes-anthropic-cron", provider="anthropic"):
            response = tracked_client.messages.create(...)
    """

    def __init__(self, cw, label: str, provider: str = "anthropic"):
        self._cw       = cw
        self._label    = label
        self._provider = provider

    def __enter__(self):
        self._cw._active_label    = self._label
        self._cw._active_provider = self._provider
        return self

    def __exit__(self, *_):
        self._cw._active_label    = None
        self._cw._active_provider = None
