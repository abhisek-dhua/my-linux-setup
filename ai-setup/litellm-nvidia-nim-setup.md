# LiteLLM + NVIDIA NIM Setup Guide

## What This Does

LiteLLM acts as a local proxy between your AI agent (opencode, Claude Code, etc.) and NVIDIA NIM. It handles 429 backoff, retries, and throttling so your agent never hits rate limits directly.

```
[Your Agent] → [LiteLLM :4000] → [NVIDIA NIM]
               (handles 429s)
```

---

## Prerequisites

- Python 3.9+
- An NVIDIA NIM API key (`https://build.nvidia.com` → Settings → API key)
- pip

---

## Step 1: Install LiteLLM

```bash
pip install 'litellm[proxy]'
```

> LiteLLM is free and open-source (Apache-2.0).

---

## Step 2: Create the config file

Create `litellm_config.yaml` anywhere you like:

```yaml
model_list:
  - model_name: "*"
    litellm_params:
      model: "openai/nvidia_nemotron-3-ultra-550b-a55b"
      api_key: os.environ/NVIDIA_API_KEY
      api_base: "https://integrate.api.nvidia.com/v1"

general_settings:
  max_concurrent_requests: 5    # throttle — lower = fewer 429s
  num_retries: 5                # retry up to 5 times
  retry_after: 2                # wait 2s minimum between retries

router_settings:
  enable_retries: true
  fallbacks: []
```

> Replace the model name with the NVIDIA NIM model you want (e.g., `meta/llama-3.1-nemotron-70b-instruct`, `google/gemma-2-27b-it`).

---

## Step 3: Export your API key

```bash
export NVIDIA_API_KEY="nvapi-your-key-here"
```

To make this permanent, add it to `~/.bashrc` or `~/.zshrc`.

---

## Step 4: Start LiteLLM

```bash
litellm --config litellm_config.yaml
```

It will start listening on **http://localhost:4000**.

---

## Step 5: Point your agent at LiteLLM

### opencode

Edit `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "provider": {
    "nvidia": {
      "options": {
        "baseURL": "http://localhost:4000/v1",
        "apiKey": "anything"
      }
    }
  }
}
```

### Claude Code (if using NVIDIA NIM backend)

```bash
export OPENAI_BASE_URL=http://localhost:4000/v1
export OPENAI_API_KEY=anything
claude --api-base-url http://localhost:4000/v1
```

---

## Step 6: Test it

```bash
curl http://localhost:4000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/nvidia_nemotron-3-ultra-550b-a55b",
    "messages": [{"role": "user", "content": "Hello, world!"}]
  }'
```

You should see a JSON response from the model. If you get a 429, LiteLLM will silently retry and back off — your agent never sees the error.

---

## Key Settings Explained

| Setting | Default | What it does |
|---|---|---|
| `max_concurrent_requests` | (unlimited) | Caps parallel requests — avoids burst-rate 429s |
| `num_retries` | 0 (opencode default) | Number of retries on 429/timeout |
| `retry_after` | 0 | Minimum seconds between retries |
| `timeout` | 600s | Request timeout in seconds |

---

## Useful NVIDIA NIM Models

Common ones to set in `model:` field:
- `openai/nvidia_nemotron-3-ultra-550b-a55b`
- `openai/meta/llama-3.1-nemotron-70b-instruct`
- `openai/google/gemma-2-27b-it`
- `openai/qwen/qwen2.5-72b-instruct`

Check yours at: `https://build.nvidia.com`

---

## Troubleshooting

**LiteLLM won't start?**
```bash
pip install --upgrade litellm
```

**Still getting 429s?** Lower `max_concurrent_requests` to `3` or `2`.

**Agent says "connection refused"?** Make sure LiteLLM is running on port 4000 and your agent's `baseURL` matches it exactly.

---

## One-liner to start everything

```bash
export NVIDIA_API_KEY="nvapi-..." && \
  litellm --config litellm_config.yaml
```
