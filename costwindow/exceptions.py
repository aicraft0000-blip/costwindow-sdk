class CostWindowProcessDisabledError(Exception):
    """
    Raised when an ingest call returns status='blocked'.
    This means the process was intentionally disabled via the kill switch.
    Catch this to gracefully skip further API calls for this process.
    """
    def __init__(self, label: str):
        self.label = label
        super().__init__(
            f"Process '{label}' is disabled. Re-enable it in the CostWindow dashboard."
        )
