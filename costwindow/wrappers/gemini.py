# sdk/costwindow/wrappers/gemini.py


class TrackedGeminiClient:
    """
    Wraps a google.generativeai Model instance to track token usage.

    Usage:
        import google.generativeai as genai
        from costwindow import CostWindow
        from costwindow.wrappers.gemini import TrackedGeminiClient

        cw = CostWindow(api_key="cw_live_...")
        model = genai.GenerativeModel("gemini-1.5-pro")
        tracked = TrackedGeminiClient(model, cw=cw, label="my-process", model_name="gemini-1.5-pro")
        response = tracked.generate_content("Your prompt here")
    """

    def __init__(self, model, *, cw, label: str, model_name: str = "gemini-1.5-pro"):
        self._model = model
        self._cw = cw
        self._label = label
        self._model_name = model_name

    def generate_content(self, prompt, **kwargs):
        response = self._model.generate_content(prompt, **kwargs)
        try:
            usage = response.usage_metadata
            self._cw.log(
                label=self._label,
                provider="google",
                model=self._model_name,
                input_tokens=getattr(usage, "prompt_token_count", 0),
                output_tokens=getattr(usage, "candidates_token_count", 0),
            )
        except Exception:
            pass
        return response
