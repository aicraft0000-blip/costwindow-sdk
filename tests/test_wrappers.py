import pytest
from unittest.mock import MagicMock, patch


def _make_mock_anthropic_response(input_tokens=450, output_tokens=180, cache_read=0):
    """Create a mock Anthropic messages.create() response with usage."""
    response = MagicMock()
    response.usage = MagicMock()
    response.usage.input_tokens            = input_tokens
    response.usage.output_tokens           = output_tokens
    response.usage.cache_read_input_tokens = cache_read
    response.model = "claude-sonnet-4-6"
    return response


def test_wrap_anthropic_returns_tracked_client():
    from costwindow import CostWindow
    from costwindow.wrappers.anthropic import TrackedAnthropicClient
    cw      = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")
    mock_c  = MagicMock()
    wrapped = cw.wrap_anthropic(mock_c)
    assert isinstance(wrapped, TrackedAnthropicClient)


def test_anthropic_wrap_calls_real_api():
    from costwindow import CostWindow
    cw      = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")
    mock_c  = MagicMock()
    wrapped = cw.wrap_anthropic(mock_c)

    mock_response = _make_mock_anthropic_response()
    mock_c.messages.create.return_value = mock_response

    with patch.object(cw, "log") as mock_log:
        result = wrapped.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello"}],
            cw_label="test-process",
        )

    assert result is mock_response
    mock_c.messages.create.assert_called_once()


def test_anthropic_wrap_strips_cw_label_from_api_call():
    from costwindow import CostWindow
    cw      = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")
    mock_c  = MagicMock()
    wrapped = cw.wrap_anthropic(mock_c)
    mock_c.messages.create.return_value = _make_mock_anthropic_response()

    with patch.object(cw, "log"):
        wrapped.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
            cw_label="my-label",
        )

    call_kwargs = mock_c.messages.create.call_args[1]
    assert "cw_label" not in call_kwargs


def test_anthropic_wrap_logs_usage_automatically():
    from costwindow import CostWindow
    cw      = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")
    mock_c  = MagicMock()
    wrapped = cw.wrap_anthropic(mock_c)
    mock_c.messages.create.return_value = _make_mock_anthropic_response(
        input_tokens=450, output_tokens=180
    )

    with patch.object(cw, "log") as mock_log:
        wrapped.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello"}],
            cw_label="quesi-horoscope",
        )

    mock_log.assert_called_once()
    kwargs = mock_log.call_args[1]
    assert kwargs["label"]         == "quesi-horoscope"
    assert kwargs["provider"]      == "anthropic"
    assert kwargs["input_tokens"]  == 450
    assert kwargs["output_tokens"] == 180


def test_anthropic_wrap_captures_cache_tokens():
    from costwindow import CostWindow
    cw      = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")
    mock_c  = MagicMock()
    wrapped = cw.wrap_anthropic(mock_c)
    mock_c.messages.create.return_value = _make_mock_anthropic_response(
        input_tokens=1000, output_tokens=200, cache_read=500
    )

    with patch.object(cw, "log") as mock_log:
        wrapped.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{"role": "user", "content": "Hello"}],
            cw_label="cached-process",
        )

    kwargs = mock_log.call_args[1]
    assert kwargs["cached_tokens"] == 500


def test_anthropic_wrap_uses_default_label():
    from costwindow import CostWindow
    cw      = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")
    mock_c  = MagicMock()
    wrapped = cw.wrap_anthropic(mock_c, label="default-label")
    mock_c.messages.create.return_value = _make_mock_anthropic_response()

    with patch.object(cw, "log") as mock_log:
        wrapped.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=100,
            messages=[{"role": "user", "content": "Hi"}],
            # No cw_label — falls back to default
        )

    kwargs = mock_log.call_args[1]
    assert kwargs["label"] == "default-label"


# ── OpenAI wrapper tests ──────────────────────────────────────────────────────

def _make_mock_openai_response(prompt_tokens=200, completion_tokens=100):
    response = MagicMock()
    response.usage = MagicMock()
    response.usage.prompt_tokens     = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.model = "gpt-4o"
    return response


def test_wrap_openai_returns_tracked_client():
    from costwindow import CostWindow
    from costwindow.wrappers.openai import TrackedOpenAIClient
    cw      = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")
    mock_c  = MagicMock()
    wrapped = cw.wrap_openai(mock_c)
    assert isinstance(wrapped, TrackedOpenAIClient)


def test_openai_wrap_strips_cw_label():
    from costwindow import CostWindow
    cw      = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")
    mock_c  = MagicMock()
    wrapped = cw.wrap_openai(mock_c)
    mock_c.chat.completions.create.return_value = _make_mock_openai_response()

    with patch.object(cw, "log"):
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            cw_label="test-openai",
        )

    call_kwargs = mock_c.chat.completions.create.call_args[1]
    assert "cw_label" not in call_kwargs


def test_openai_wrap_logs_usage():
    from costwindow import CostWindow
    cw      = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")
    mock_c  = MagicMock()
    wrapped = cw.wrap_openai(mock_c)
    mock_c.chat.completions.create.return_value = _make_mock_openai_response(200, 100)

    with patch.object(cw, "log") as mock_log:
        wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            cw_label="my-openai-process",
        )

    kwargs = mock_log.call_args[1]
    assert kwargs["label"]         == "my-openai-process"
    assert kwargs["provider"]      == "openai"
    assert kwargs["input_tokens"]  == 200
    assert kwargs["output_tokens"] == 100
