"""
CloneShield Render Latency Benchmark
=====================================
Tests latency for two deployed models on Render:
  1. Protection Model  → POST /protect
  2. Cloner (XTTS v2)  → POST /clone  (Render proxy → local ngrok → XTTS)

Categories per model:
  A) Cold Start  – first request after the Render service has been idle/spun down
  B) Warm Server – subsequent requests while the server is already running

Usage:
    python test_latency_render.py [--warm-runs 5] [--audio path/to/file.wav]

Requirements:
    pip install requests torchaudio pandas tabulate colorama
"""

import argparse
import io
import os
import sys
import time
import tempfile
import statistics
from pathlib import Path
from datetime import datetime

import requests
import torchaudio
import torch
import pandas as pd

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        GREEN = YELLOW = RED = CYAN = MAGENTA = WHITE = RESET = ""
    class Style:
        BRIGHT = RESET_ALL = ""

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION — edit these to match your deployment
# ─────────────────────────────────────────────────────────────────────────────
RENDER_BASE_URL   = "https://fakeless-api.onrender.com"   # ← your Render URL
PROTECT_ENDPOINT  = f"{RENDER_BASE_URL}/protect"
CLONE_ENDPOINT    = f"{RENDER_BASE_URL}/clone"

# Render free-tier cold-start can take 50–90 s; set a generous timeout
COLD_START_TIMEOUT = 300   # seconds
WARM_TIMEOUT       = 200    # seconds

# Default audio file – replaced by CLI arg or auto-generated sine wave
DEFAULT_AUDIO = "sound_samples/input/speaker.wav"

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def banner(text: str, color=None):
    line = "=" * 60
    c = color or (Fore.CYAN if HAS_COLOR else "")
    print(f"\n{c}{Style.BRIGHT}{line}")
    print(f"  {text}")
    print(f"{line}{Style.RESET_ALL}\n")


def info(msg):
    print(f"{Fore.WHITE}  [INFO]  {msg}{Style.RESET_ALL}")


def ok(msg):
    print(f"{Fore.GREEN}  [ OK ]  {msg}{Style.RESET_ALL}")


def warn(msg):
    print(f"{Fore.YELLOW}  [WARN]  {msg}{Style.RESET_ALL}")


def err(msg):
    print(f"{Fore.RED}  [ERR ]  {msg}{Style.RESET_ALL}")


def generate_sine_wav(path: str, duration_s: float = 3.0, sr: int = 16000):
    """Create a simple sine-wave WAV so tests run even without a real speaker file."""
    t = torch.linspace(0, duration_s, int(sr * duration_s))
    wave = (0.5 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0)
    torchaudio.save(path, wave, sr)
    info(f"Generated synthetic audio → {path}  ({os.path.getsize(path)/1024:.1f} KB)")


def load_audio_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()

# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE REQUEST HELPERS  (return dict with timing details)
# ─────────────────────────────────────────────────────────────────────────────

def call_protect(audio_bytes: bytes, timeout: int, filter_strength: float = 1.0) -> dict:
    """POST to /protect and return latency info."""
    result = {
        "endpoint": "/protect",
        "timeout_used": timeout,
        "status_code": None,
        "latency_ms": None,
        "ttfb_ms": None,      # time-to-first-byte (approx via response elapsed)
        "response_size_kb": None,
        "success": False,
        "error": None,
    }
    files = {"file": ("input.wav", io.BytesIO(audio_bytes), "audio/wav")}
    data  = {"filter_strength": str(filter_strength)}

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            PROTECT_ENDPOINT,
            files=files,
            data=data,
            timeout=timeout,
        )
        total_ms = (time.perf_counter() - t0) * 1000
        result["status_code"]      = resp.status_code
        result["latency_ms"]       = round(total_ms, 2)
        result["ttfb_ms"]          = round(resp.elapsed.total_seconds() * 1000, 2)
        result["response_size_kb"] = round(len(resp.content) / 1024, 1)
        result["success"]          = resp.status_code == 200
        if not result["success"]:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:120]}"
    except requests.exceptions.Timeout:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        result["latency_ms"] = f"≥{elapsed_ms}"
        result["error"] = f"Timed out after {timeout}s  (≥{elapsed_ms} ms elapsed)"
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        result["latency_ms"] = f"≥{elapsed_ms}"
        result["error"] = str(e)

    return result


def call_clone(audio_bytes: bytes, timeout: int, text: str = "Hello this is a latency test.") -> dict:
    """POST to /clone (Render proxy → local XTTS) and return latency info."""
    result = {
        "endpoint": "/clone",
        "timeout_used": timeout,
        "status_code": None,
        "latency_ms": None,
        "ttfb_ms": None,
        "response_size_kb": None,
        "success": False,
        "error": None,
    }
    files = {"speaker": ("speaker.wav", io.BytesIO(audio_bytes), "audio/wav")}
    data  = {"text": text}

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            CLONE_ENDPOINT,
            files=files,
            data=data,
            timeout=timeout,
        )
        total_ms = (time.perf_counter() - t0) * 1000
        result["status_code"]      = resp.status_code
        result["latency_ms"]       = round(total_ms, 2)
        result["ttfb_ms"]          = round(resp.elapsed.total_seconds() * 1000, 2)
        result["response_size_kb"] = round(len(resp.content) / 1024, 1)
        result["success"]          = resp.status_code == 200

        # /clone can return JSON {error:...} even on 200
        if result["success"]:
            try:
                body = resp.json()
                if "error" in body:
                    result["success"] = False
                    result["error"]   = body["error"]
            except Exception:
                pass  # Binary audio response → that is fine too

        if not result["success"] and result["error"] is None:
            result["error"] = f"HTTP {resp.status_code}: {resp.text[:120]}"
    except requests.exceptions.Timeout:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        result["latency_ms"] = f"≥{elapsed_ms}"
        result["error"] = f"Timed out after {timeout}s  (≥{elapsed_ms} ms elapsed)"
    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        result["latency_ms"] = f"≥{elapsed_ms}"
        result["error"] = str(e)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  COLD-START TEST  (single shot with long timeout)
# ─────────────────────────────────────────────────────────────────────────────

def run_cold_start_test(caller, audio_bytes: bytes, label: str) -> dict:
    banner(f"COLD START — {label}", Fore.MAGENTA)
    warn("Cold-start requests can take 30–90 s on Render Free Tier.")
    info(f"Sending first request to {label} endpoint now…")

    result = caller(audio_bytes, timeout=COLD_START_TIMEOUT)
    result["category"] = "Cold Start"
    result["model"]    = label

    if result["success"]:
        ok(f"Cold-start completed in {result['latency_ms']} ms  "
           f"(TTFB: {result['ttfb_ms']} ms | "
           f"Response: {result['response_size_kb']} KB)")
    else:
        err(f"Cold-start FAILED: {result['error']}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  WARM-SERVER TESTS  (N sequential requests after server is up)
# ─────────────────────────────────────────────────────────────────────────────

def run_warm_tests(caller, audio_bytes: bytes, label: str, n_runs: int = 5) -> list[dict]:
    banner(f"WARM SERVER — {label}  ({n_runs} runs)", Fore.CYAN)
    info("Server is already warm from the cold-start call above.")
    results = []

    for i in range(1, n_runs + 1):
        info(f"Warm run {i}/{n_runs}…")
        result = caller(audio_bytes, timeout=WARM_TIMEOUT)
        result["category"] = "Warm Server"
        result["model"]    = label
        result["run"]      = i

        if result["success"]:
            ok(f"  Run {i}: {result['latency_ms']} ms  "
               f"(TTFB: {result['ttfb_ms']} ms)")
        else:
            err(f"  Run {i} FAILED: {result['error']}")

        results.append(result)
        # Small pause to avoid hammering (free tier rate-limits)
        if i < n_runs:
            time.sleep(1.5)

    return results


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY & REPORT
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(cold_results: list[dict], warm_results_all: list[list[dict]]) -> tuple:
    """
    Returns two DataFrames:
      - runs_df   : one row per individual request (no aggregate row)
      - agg_df    : one row per model with proper numeric aggregate columns
    """
    run_rows = []
    agg_rows = []

    # ── Cold-start rows ──────────────────────────────────────────────────────
    for cr in cold_results:
        run_rows.append({
            "Model":        cr["model"],
            "Category":     "Cold Start",
            "Run":          1,
            "Latency (ms)": cr["latency_ms"] if cr["success"] else "FAILED",
            "TTFB (ms)":    cr["ttfb_ms"]    if cr["success"] else None,
            "Resp (KB)":    cr["response_size_kb"] if cr["success"] else None,
            "Status":       "OK" if cr["success"] else f"FAILED: {cr['error'][:60]}",
        })

    # ── Warm-server rows + per-model aggregates ───────────────────────────────
    for warm_runs in warm_results_all:
        successful = [r for r in warm_runs if r["success"]]
        model_name = warm_runs[0]["model"] if warm_runs else "?"

        for r in warm_runs:
            run_rows.append({
                "Model":        r["model"],
                "Category":     "Warm Server",
                "Run":          r.get("run", "?"),
                "Latency (ms)": r["latency_ms"] if r["success"] else "FAILED",
                "TTFB (ms)":    r["ttfb_ms"]    if r["success"] else None,
                "Resp (KB)":    r["response_size_kb"] if r["success"] else None,
                "Status":       "OK" if r["success"] else f"FAILED: {r['error'][:60]}",
            })

        # Aggregate row with proper numeric columns
        if successful:
            lats = [r["latency_ms"] for r in successful]
            ttfbs = [r["ttfb_ms"] for r in successful if r["ttfb_ms"] is not None]
            agg_rows.append({
                "Model":            model_name,
                "Category":         "Warm Server",
                "Runs OK":          f"{len(successful)}/{len(warm_runs)}",
                "Avg Latency (ms)": round(statistics.mean(lats), 2),
                "Min Latency (ms)": round(min(lats), 2),
                "Max Latency (ms)": round(max(lats), 2),
                "P50 Latency (ms)": round(statistics.median(lats), 2),
                "StdDev (ms)":      round(statistics.stdev(lats), 2) if len(lats) > 1 else 0.0,
                "Avg TTFB (ms)":    round(statistics.mean(ttfbs), 2) if ttfbs else None,
            })
        else:
            agg_rows.append({
                "Model":            model_name,
                "Category":         "Warm Server",
                "Runs OK":          f"0/{len(warm_runs)}",
                "Avg Latency (ms)": None,
                "Min Latency (ms)": None,
                "Max Latency (ms)": None,
                "P50 Latency (ms)": None,
                "StdDev (ms)":      None,
                "Avg TTFB (ms)":    None,
            })

    return pd.DataFrame(run_rows), pd.DataFrame(agg_rows)


def print_summary(runs_df: pd.DataFrame, agg_df: pd.DataFrame):
    banner("INDIVIDUAL RUNS", Fore.CYAN)
    if HAS_TABULATE:
        print(tabulate(runs_df, headers="keys", tablefmt="fancy_grid", showindex=False))
    else:
        print(runs_df.to_string(index=False))

    if not agg_df.empty:
        banner("WARM SERVER AGGREGATE STATS", Fore.GREEN)
        if HAS_TABULATE:
            print(tabulate(agg_df, headers="keys", tablefmt="fancy_grid", showindex=False))
        else:
            print(agg_df.to_string(index=False))


def save_csv(runs_df: pd.DataFrame, agg_df: pd.DataFrame, base_path: str):
    # Individual runs CSV
    runs_df.to_csv(base_path, index=False)
    ok(f"Individual runs saved  → {base_path}")

    # Aggregate stats CSV (separate file, clean numeric columns)
    agg_path = base_path.replace(".csv", "_aggregate.csv")
    agg_df.to_csv(agg_path, index=False)
    ok(f"Aggregate stats saved  → {agg_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CloneShield Render Latency Benchmark")
    p.add_argument("--audio",      default=DEFAULT_AUDIO,
                   help="Path to a .wav file to use as test audio")
    p.add_argument("--warm-runs",  type=int, default=5,
                   help="Number of warm-server runs per model (default: 5)")
    p.add_argument("--protect-only", action="store_true",
                   help="Only test the /protect endpoint")
    p.add_argument("--clone-only",   action="store_true",
                   help="Only test the /clone endpoint")
    p.add_argument("--output-csv", default="latency_results.csv",
                   help="CSV file to write results to (default: latency_results.csv)")
    p.add_argument("--render-url", default=RENDER_BASE_URL,
                   help="Override the Render base URL")
    return p.parse_args()


def main():
    args = parse_args()

    # Allow CLI override of base URL
    global RENDER_BASE_URL, PROTECT_ENDPOINT, CLONE_ENDPOINT
    RENDER_BASE_URL  = args.render_url.rstrip("/")
    PROTECT_ENDPOINT = f"{RENDER_BASE_URL}/protect"
    CLONE_ENDPOINT   = f"{RENDER_BASE_URL}/clone"

    banner(
        f"CloneShield Latency Benchmark  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        Fore.CYAN,
    )
    info(f"Render URL  : {RENDER_BASE_URL}")
    info(f"Audio file  : {args.audio}")
    info(f"Warm runs   : {args.warm_runs}")
    info(f"Cold timeout: {COLD_START_TIMEOUT}s  |  Warm timeout: {WARM_TIMEOUT}s")

    # ── Prepare audio ───────────────────────────────────────────────────────
    audio_path = args.audio
    if not os.path.exists(audio_path):
        warn(f"Audio file not found: {audio_path}")
        warn("Generating a 3-second synthetic sine-wave WAV as fallback…")
        audio_path = "test_latency_audio.wav"
        generate_sine_wav(audio_path)

    audio_bytes = load_audio_bytes(audio_path)
    info(f"Audio loaded: {len(audio_bytes)/1024:.1f} KB")

    run_protect = not args.clone_only
    run_clone   = not args.protect_only

    cold_results     = []
    warm_results_all = []

    # ── Protection Model ────────────────────────────────────────────────────
    if run_protect:
        # Cold start
        cr = run_cold_start_test(call_protect, audio_bytes, "Protection Model (/protect)")
        cold_results.append(cr)

        # Warm runs (only if cold start succeeded or partial)
        warm_results = run_warm_tests(
            call_protect, audio_bytes, "Protection Model (/protect)", n_runs=args.warm_runs
        )
        warm_results_all.append(warm_results)

    # ── Cloner Model ────────────────────────────────────────────────────────
    if run_clone:
        # Cold start
        cr = run_cold_start_test(call_clone, audio_bytes, "Cloner Model (/clone)")
        cold_results.append(cr)

        # Warm runs
        warm_results = run_warm_tests(
            call_clone, audio_bytes, "Cloner Model (/clone)", n_runs=args.warm_runs
        )
        warm_results_all.append(warm_results)

    # ── Report ───────────────────────────────────────────────────────────────
    runs_df, agg_df = build_summary(cold_results, warm_results_all)
    print_summary(runs_df, agg_df)
    save_csv(runs_df, agg_df, args.output_csv)


if __name__ == "__main__":
    main()
