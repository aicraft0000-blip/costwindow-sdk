import pytest
from unittest.mock import patch, MagicMock


def test_costwindow_stores_api_key():
    from costwindow import CostWindow
    cw = CostWindow(api_key="cw_live_testkey123")
    assert cw.api_key == "cw_live_testkey123"


def test_costwindow_default_backend_url():
    from costwindow import CostWindow
    cw = CostWindow(api_key="cw_live_test")
    assert cw.backend_url == "https://costwindow-site-production.up.railway.app"


def test_log_posts_to_ingest_endpoint():
    from costwindow import CostWindow
    cw = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")

    with patch("costwindow.client.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"status": "ok", "cost_usd": 0.0001, "process_id": "abc"}

        cw.log(
            label="test-process",
            provider="anthropic",
            model="claude-sonnet-4-6",
            input_tokens=450,
            output_tokens=180,
        )

    assert mock_post.called
    url = mock_post.call_args[0][0]
    assert url.endswith("/ingest")
    payload = mock_post.call_args[1]["json"]
    assert payload["label"]         == "test-process"
    assert payload["input_tokens"]  == 450
    assert payload["output_tokens"] == 180


def test_log_silently_continues_on_network_error():
    from costwindow import CostWindow
    cw = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")

    with patch("costwindow.client.requests.post") as mock_post:
        mock_post.side_effect = Exception("Connection refused")
        # Must NOT raise
        cw.log(label="test", provider="anthropic", model="claude-sonnet-4-6",
                input_tokens=100, output_tokens=50)


def test_log_raises_process_disabled_error_on_blocked_response():
    from costwindow import CostWindow, CostWindowProcessDisabledError
    cw = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")

    with patch("costwindow.client.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {
            "status": "blocked", "reason": "process_disabled"
        }

        with pytest.raises(CostWindowProcessDisabledError) as exc_info:
            cw.log(label="killed-process", provider="anthropic",
                   model="claude-sonnet-4-6", input_tokens=100, output_tokens=50)

    assert "killed-process" in str(exc_info.value)


def test_log_passes_cached_tokens():
    from costwindow import CostWindow
    cw = CostWindow(api_key="cw_live_test", backend_url="http://localhost:8001")

    with patch("costwindow.client.requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"status": "ok"}

        cw.log(label="t", provider="anthropic", model="claude-sonnet-4-6",
                input_tokens=1000, output_tokens=200, cached_tokens=500)

    payload = mock_post.call_args[1]["json"]
    assert payload["cached_tokens"] == 500


def test_is_active_returns_true_on_200(requests_mock):
    from costwindow import CostWindow
    requests_mock.get(
        "https://costwindow-site-production.up.railway.app/processes/status/my-cron",
        json={"label": "my-cron", "is_active": True},
    )
    cw = CostWindow(api_key="cw_live_test")
    assert cw.is_active("my-cron") is True


def test_is_active_returns_false_when_disabled(requests_mock):
    from costwindow import CostWindow
    requests_mock.get(
        "https://costwindow-site-production.up.railway.app/processes/status/my-cron",
        json={"label": "my-cron", "is_active": False},
    )
    cw = CostWindow(api_key="cw_live_test")
    assert cw.is_active("my-cron") is False


def test_is_active_returns_true_on_network_error(requests_mock):
    """Fail open — if CostWindow unreachable, don't stop the cron."""
    import requests
    from costwindow import CostWindow
    requests_mock.get(
        "https://costwindow-site-production.up.railway.app/processes/status/my-cron",
        exc=requests.ConnectionError,
    )
    cw = CostWindow(api_key="cw_live_test")
    assert cw.is_active("my-cron") is True
