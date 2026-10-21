def test_gemini_wrapper_intercepts_call(requests_mock):
    requests_mock.post(
        "https://costwindow-site-production.up.railway.app/ingest",
        json={"status": "ok", "cost_usd": 0.01}
    )
    from costwindow import CostWindow
    from costwindow.wrappers.gemini import TrackedGeminiClient

    cw = CostWindow(api_key="cw_live_test")

    class FakeResponse:
        usage_metadata = type("u", (), {"prompt_token_count": 100, "candidates_token_count": 50})()

    class FakeGeminiModel:
        def generate_content(self, prompt, **kwargs):
            return FakeResponse()

    tracked = TrackedGeminiClient(FakeGeminiModel(), cw=cw, label="test-gemini", model_name="gemini-3.8-flash")
    resp = tracked.generate_content("Hello")
    assert resp.usage_metadata.prompt_token_count == 100
    assert requests_mock.call_count == 1


def test_gemini_wrapper_default_model_is_current(requests_mock):
    # gemini-1.5-* is retired; an untagged call should be priced as a live model.
    requests_mock.post(
        "https://costwindow-site-production.up.railway.app/ingest",
        json={"status": "ok", "cost_usd": 0.01}
    )
    from costwindow import CostWindow
    from costwindow.wrappers.gemini import TrackedGeminiClient

    class FakeResponse:
        usage_metadata = type("u", (), {"prompt_token_count": 10, "candidates_token_count": 5})()

    class FakeGeminiModel:
        def generate_content(self, prompt, **kwargs):
            return FakeResponse()

    TrackedGeminiClient(FakeGeminiModel(), cw=CostWindow(api_key="cw_live_test"), label="t").generate_content("hi")
    assert requests_mock.last_request.json()["model"] == "gemini-3.8-flash"
