# CostWindow SDK

Track what your AI scripts actually cost. This is the Python SDK for [CostWindow](https://costwindow.com) — it reports token usage from your Anthropic/OpenAI/Gemini calls to your CostWindow dashboard.

## Install

```bash
pip install git+https://github.com/aicraft0000-blip/costwindow-sdk.git
```

## Use

```python
from costwindow import CostWindow
import anthropic

cw = CostWindow(api_key="cw_live_...")  # from your CostWindow Settings page
client = cw.wrap_anthropic(anthropic.Anthropic())

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello"}],
    cw_label="my-script",
)
```

Every call is now tracked under `my-script` in your dashboard, with costs calculated automatically. `cw.wrap_openai()` and `cw.wrap_gemini()` work the same way.

### Kill switch for cron jobs

```python
import sys
if not cw.is_active("my-script"):
    sys.exit(0)  # process was switched off in the dashboard
```

CostWindow never breaks your script: if the service is unreachable, errors are logged to `~/.costwindow/logs/` and swallowed. The only exception that propagates is `CostWindowProcessDisabledError` — raised when you've kill-switched the process in the dashboard.

Get a subscription at [costwindow.com](https://costwindow.com).
