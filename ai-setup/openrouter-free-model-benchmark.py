#!/usr/bin/env python3
"""
OpenRouter Free Model Benchmark Script
Tests all available free models on OpenRouter and generates a benchmark report
"""

import requests
import time
import json
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics
import sys

# OpenRouter API Configuration
BASE_URL = "https://openrouter.ai/api/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
MODELS_URL = f"{BASE_URL}/models"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY environment variable not set")
    exit(1)

# Test configuration
TEST_PROMPT = """Write a short Python function to calculate the factorial of a number.
Include error handling for negative numbers and non-integer inputs."""

MAX_TOKENS = 500
TIMEOUT = 90  # seconds
CONCURRENT_REQUESTS = 3  # Number of concurrent tests

# Headers for API requests
HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}


def get_available_models():
    """Fetch list of available free models from OpenRouter API"""
    try:
        response = requests.get(
            MODELS_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()
        # Filter only free models
        return [model["id"] for model in data.get("data", []) if ":free" in model["id"]]
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def filter_working_models(models: list, test_count: int = 5) -> list:
    """Quick test to filter out models that return errors"""
    print(f"\nPre-checking {min(test_count, len(models))} free models for availability...")
    working = []

    test_payload = {
        "model": "",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }

    for i, model in enumerate(models[:test_count]):
        test_payload["model"] = model
        try:
            response = requests.post(
                CHAT_URL,
                headers=HEADERS,
                json=test_payload,
                timeout=10
            )
            if response.status_code == 200:
                working.append(model)
                print(f"  ✓ {model} - Available")
            else:
                print(f"  ✗ {model} - Not available (HTTP {response.status_code})")
        except Exception as e:
            print(f"  ✗ {model} - Error: {str(e)[:50]}")

    print(f"\nFound {len(working)} working free models out of {test_count} tested")
    return working


def test_model(model_id: str) -> dict:
    """Test a single model and return benchmark results"""
    results = {
        "model": model_id,
        "success": False,
        "latency_ms": None,
        "tokens_per_second": None,
        "error": None,
        "timestamp": datetime.now().isoformat()
    }

    payload = {
        "model": model_id,
        "messages": [
            {"role": "user", "content": TEST_PROMPT}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "stream": False
    }

    try:
        start_time = time.time()
        response = requests.post(
            CHAT_URL,
            headers=HEADERS,
            json=payload,
            timeout=TIMEOUT
        )
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000
        results["latency_ms"] = round(latency_ms, 2)

        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                # Get response content (handle both content and reasoning fields)
                msg = data["choices"][0].get("message", {})
                content = (msg.get("content") or "").strip()
                if not content:
                    content = (msg.get("reasoning") or "").strip()
                content = content[:200] + "..." if len(content) > 200 else content

                token_count = data.get("usage", {}).get("completion_tokens", 0)

                if token_count > 0 and latency_ms > 0:
                    tokens_per_sec = (token_count / (latency_ms / 1000))
                    results["tokens_per_second"] = round(tokens_per_sec, 2)

                results["success"] = True
                results["response_preview"] = content or "(empty response)"
            else:
                results["error"] = "Empty response"
        else:
            results["error"] = f"HTTP {response.status_code}: {response.text[:200]}"

    except requests.exceptions.Timeout:
        results["error"] = f"Timeout after {TIMEOUT}s"
    except requests.exceptions.RequestException as e:
        results["error"] = f"Request error: {str(e)}"
    except Exception as e:
        results["error"] = f"Unexpected error: {str(e)}"

    return results


def run_benchmark(models: list, max_models: int = None) -> list:
    """Run benchmarks on multiple models concurrently"""
    if max_models:
        models = models[:max_models]

    results = []
    completed = 0
    total = len(models)

    print(f"\n{'='*80}")
    print(f"Testing {total} free models...")
    print(f"{'='*80}\n")
    sys.stdout.flush()

    with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        future_to_model = {executor.submit(test_model, model): model for model in models}

        for future in as_completed(future_to_model):
            model = future_to_model[future]
            completed += 1
            try:
                result = future.result()
                results.append(result)

                status = "✓" if result["success"] else "✗"
                print(f"[{completed}/{total}] {status} {model}")
                sys.stdout.flush()

                if result["success"]:
                    print(f"    Latency: {result['latency_ms']}ms | "
                          f"Tokens/s: {result['tokens_per_second']}")
                else:
                    print(f"    Error: {result['error'][:100]}")

            except Exception as e:
                print(f"[{completed}/{total}] ✗ {model} - Exception: {e}")
                results.append({
                    "model": model,
                    "success": False,
                    "error": str(e)
                })

    return results


def generate_report(results: list) -> str:
    """Generate a formatted benchmark report"""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    # Calculate statistics for successful models
    latencies = [r["latency_ms"] for r in successful if r["latency_ms"]]
    tps_values = [r["tokens_per_second"] for r in successful if r["tokens_per_second"]]

    report = []
    report.append("\n" + "="*80)
    report.append("OPENROUTER FREE MODEL BENCHMARK REPORT")
    report.append("="*80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Free Models Tested: {len(results)}")
    report.append(f"Successful: {len(successful)}")
    report.append(f"Failed: {len(failed)}")
    report.append(f"Success Rate: {(len(successful)/len(results)*100):.1f}%" if results else "Success Rate: 0%")
    report.append("")

    if successful:
        report.append("-"*80)
        report.append("PERFORMANCE STATISTICS (Successful Models Only)")
        report.append("-"*80)

        if latencies:
            report.append(f"Latency (ms):")
            report.append(f"  Mean:   {statistics.mean(latencies):.2f}")
            report.append(f"  Median: {statistics.median(latencies):.2f}")
            report.append(f"  Min:    {min(latencies):.2f}")
            report.append(f"  Max:    {max(latencies):.2f}")

        if tps_values:
            report.append(f"Tokens/Second:")
            report.append(f"  Mean:   {statistics.mean(tps_values):.2f}")
            report.append(f"  Median: {statistics.median(tps_values):.2f}")
            report.append(f"  Min:    {min(tps_values):.2f}")
            report.append(f"  Max:    {max(tps_values):.2f}")

        report.append("")
        report.append("-"*80)
        report.append("ALL WORKING FREE MODELS (by Latency)")
        report.append("-"*80)

        sorted_by_latency = sorted(successful, key=lambda x: x["latency_ms"])
        for i, result in enumerate(sorted_by_latency, 1):
            report.append(f"{i:2d}. {result['model']:<50} "
                         f"{result['latency_ms']:>8.2f}ms "
                         f"({result.get('tokens_per_second', 0):.2f} tok/s)")

        report.append("")
        report.append("-"*80)
        report.append("ALL WORKING FREE MODELS (by Tokens/Second)")
        report.append("-"*80)

        sorted_by_tps = sorted(successful,
                           key=lambda x: x.get("tokens_per_second", 0) or 0,
                           reverse=True)
        for i, result in enumerate(sorted_by_tps, 1):
            report.append(f"{i:2d}. {result['model']:<50} "
                         f"{result.get('tokens_per_second', 0):>8.2f} tok/s "
                         f"({result['latency_ms']:.2f}ms)")

    if failed:
        report.append("")
        report.append("-"*80)
        report.append(f"FAILED FREE MODELS ({len(failed)})")
        report.append("-"*80)
        for result in failed:
            report.append(f"  ✗ {result['model']}")
            report.append(f"    Error: {result['error'][:80]}")

    report.append("")
    report.append("="*80)
    report.append("END OF REPORT")
    report.append("="*80)

    return "\n".join(report)


def save_results(results: list, filename: str = None):
    """Save detailed results to JSON file (no-op for now)"""
    pass


def main():
    """Main entry point"""
    print("OpenRouter Free Model Benchmark Tool")
    print("="*80)

    # Get available free models
    print("\nFetching available free models from OpenRouter...")
    models = get_available_models()

    if not models:
        print("No free models found or error fetching models.")
        return

    print(f"Found {len(models)} free models")
    print(f"\nPre-checking all {len(models)} free models to find working ones...")
    working = filter_working_models(models, test_count=len(models))
    if working:
        print(f"\nProceeding with {len(working)} confirmed working free models")
        models = working
    else:
        print("\nWarning: No working free models found. Testing all anyway...")

    # Run benchmarks
    results = run_benchmark(models)

    # Generate and display report
    report = generate_report(results)
    print(report)

    # Save report as text only
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"openrouter_free_benchmark_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
