#!/usr/bin/env python3
"""Automatically update opencode.jsonc with latest free models from OpenRouter and all NVIDIA models.

Keys are read from:
  1. Environment variables: OPENROUTER_API_KEY, NVIDIA_API_KEY
  2. Existing opencode.jsonc config
  3. Interactive prompt (hidden input)

Keys are NEVER logged or printed.
"""

import json
import re
import sys
import os
import getpass
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' module required. Install with: pip install requests")
    sys.exit(1)

STORE_KEYS = False  # When True, writes apiKey into opencode.jsonc

CONFIG_PATHS = [
    Path.home() / ".config" / "opencode" / "opencode.jsonc",
    Path.cwd() / "opencode.jsonc",
    Path.cwd() / ".opencode" / "opencode.jsonc",
]

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"


def strip_jsonc_comments(text):
    """Strip comments from JSONC content for parsing."""
    result = []
    i = 0
    in_string = False
    escape = False
    while i < len(text):
        c = text[i]
        if escape:
            result.append(c)
            escape = False
            i += 1
            continue
        if c == '\\' and in_string:
            result.append(c)
            escape = True
            i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
            i += 1
            continue
        if not in_string:
            if c == '/' and i + 1 < len(text) and text[i + 1] == '/':
                while i < len(text) and text[i] != '\n':
                    i += 1
                continue
            if c == '/' and i + 1 < len(text) and text[i + 1] == '*':
                i += 2
                while i + 1 < len(text) and not (text[i] == '*' and text[i + 1] == '/'):
                    i += 1
                i += 2
                continue
        result.append(c)
        i += 1
    cleaned = ''.join(result)
    cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
    return cleaned


def read_config(path):
    """Read and parse opencode.jsonc."""
    try:
        raw = path.read_text(encoding="utf-8")
    except PermissionError:
        print(f"Error: Permission denied reading {path}")
        sys.exit(1)
    except OSError as e:
        print(f"Error reading {path}: {e}")
        sys.exit(1)

    if not raw.strip():
        print(f"Warning: Config file is empty, using defaults")
        return {}, raw

    clean = strip_jsonc_comments(raw)
    try:
        return json.loads(clean), raw
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in config: {e}")
        print(f"  at line {e.lineno}, column {e.colno} (char {e.pos})")
        sys.exit(1)


def write_config(path, config):
    """Write config as JSON."""
    text = json.dumps(config, indent=2, ensure_ascii=False)
    text = re.sub(r',(\s*[}\]])', r'\1', text)
    try:
        path.write_text(text, encoding="utf-8")
    except PermissionError:
        print(f"Error: Permission denied writing {path}")
        sys.exit(1)
    except OSError as e:
        print(f"Error writing {path}: {e}")
        sys.exit(1)


def fetch_openrouter_free_models():
    """Fetch all :free models from OpenRouter."""
    print("Fetching OpenRouter models...")
    resp = requests.get(OPENROUTER_MODELS_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    free_models = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        if model_id.endswith(":free"):
            free_models.append(model_id)
    free_models.sort()
    print(f"  Found {len(free_models)} free models")
    return free_models


def fetch_nvidia_models(api_key):
    """Fetch all models from NVIDIA API."""
    print("Fetching NVIDIA models...")
    resp = requests.get(
        NVIDIA_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    models = []
    for m in data.get("data", []):
        model_id = m.get("id", "")
        if model_id:
            models.append(model_id)
    models.sort()
    print(f"  Found {len(models)} models")
    return models


def get_api_key(provider, env_var, existing_key=None):
    """Get API key from env var, existing config, or prompt. Never returns None if possible."""
    env_val = os.environ.get(env_var, "").strip()
    if env_val:
        return env_val
    if existing_key:
        return existing_key
    print(f"\n  API key not found for {provider}.")
    print(f"  Set env var {env_var} or enter below.")
    key = getpass.getpass(f"  Enter {provider} API key: ").strip()
    if not key:
        print(f"  No key provided. Skipping {provider} update.")
        return None
    return key


def update_openrouter(config, free_models):
    """Update OpenRouter whitelist and models with free models."""
    provider = config.setdefault("provider", {})
    or_config = provider.setdefault("openrouter", {})
    or_config.setdefault("api", "https://openrouter.ai/api/v1")
    options = or_config.setdefault("options", {})

    api_key = get_api_key("OpenRouter", "OPENROUTER_API_KEY", options.get("apiKey"))
    if not api_key:
        return False
    if STORE_KEYS:
        options["apiKey"] = api_key
    else:
        options.pop("apiKey", None)

    or_config["whitelist"] = free_models
    models = {}
    for mid in free_models:
        models[mid] = {"id": mid}
    or_config["models"] = models

    print(f"  Updated OpenRouter: {len(free_models)} free models")
    return True


def update_nvidia(config, nvidia_models, api_key=None):
    """Update NVIDIA whitelist and models."""
    provider = config.setdefault("provider", {})
    nv_config = provider.setdefault("nvidia", {})
    nv_config.setdefault("api", "https://integrate.api.nvidia.com/v1")
    options = nv_config.setdefault("options", {})

    if api_key is None:
        api_key = get_api_key("NVIDIA", "NVIDIA_API_KEY", options.get("apiKey"))
    if not api_key:
        return False
    if STORE_KEYS:
        options["apiKey"] = api_key
    else:
        options.pop("apiKey", None)

    nv_config["whitelist"] = nvidia_models
    models = {}
    for mid in nvidia_models:
        models[mid] = {"id": mid}
    nv_config["models"] = models

    print(f"  Updated NVIDIA: {len(nvidia_models)} models")
    return True


def find_config():
    """Find existing config file or create one."""
    for p in CONFIG_PATHS:
        if p.exists():
            return p
    target = CONFIG_PATHS[0]
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error: Permission denied creating config directory {target.parent}")
        sys.exit(1)
    except OSError as e:
        print(f"Error creating config directory: {e}")
        sys.exit(1)

    try:
        target.write_text('{\n  "$schema": "https://opencode.ai/config.json"\n}\n', encoding="utf-8")
    except PermissionError:
        print(f"Error: Permission denied writing {target}")
        sys.exit(1)
    except OSError as e:
        print(f"Error writing initial config: {e}")
        sys.exit(1)

    print(f"Created new config: {target}")
    return target


def main():
    print("=== OpenCODE Model Updater ===\n")

    config_path = find_config()
    print(f"Config: {config_path}\n")

    config, _ = read_config(config_path)
    config.setdefault("$schema", "https://opencode.ai/config.json")
    config.setdefault("provider", {})

    or_models = []
    nv_models = []
    nv_key = None

    try:
        or_models = fetch_openrouter_free_models()
    except Exception as e:
        print(f"  Error fetching OpenRouter models: {e}")

    try:
        existing_nv_key = config.get("provider", {}).get("nvidia", {}).get("options", {}).get("apiKey")
        nv_key = get_api_key("NVIDIA", "NVIDIA_API_KEY", existing_nv_key)
        if nv_key:
            nv_models = fetch_nvidia_models(nv_key)
    except Exception as e:
        print(f"  Error fetching NVIDIA models: {e}")

    print()
    or_ok = update_openrouter(config, or_models) if or_models else False
    nv_ok = update_nvidia(config, nv_models, nv_key) if nv_models else False

    if or_ok or nv_ok:
        write_config(config_path, config)
        print(f"\nConfig saved: {config_path}")
    else:
        print("\nNo updates made.")

    print("\nRestart opencode for changes to take effect.")


if __name__ == "__main__":
    main()
