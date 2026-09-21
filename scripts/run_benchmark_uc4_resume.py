"""
UC4 RESUME SCRIPT — runs only Llama-3.1-8B and Llama-3.3-70B (models 6 & 7)
then merges with existing partial results.
"""

import os, csv, json, time, re, httpx, glob
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

GOLD_SET_PATH  = "data/gold_sets/uc4_product_review_sentiment.csv"
RAW_OUTPUT_DIR = "data/raw_outputs"
RESULTS_DIR    = "data/results"
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR,    exist_ok=True)

TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
RAW_FILE     = os.path.join(RAW_OUTPUT_DIR, f"uc4_raw_{TIMESTAMP}.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR,    f"uc4_summary_{TIMESTAMP}.csv")

TEMPERATURE = 0.0
MAX_TOKENS  = 128
SEEDS       = [42, 43, 44]
RUNS        = len(SEEDS)

# ── Only the 2 remaining models ──────────────────────────────────────────────
MODELS = [
    { "name": "Llama-3.1-8B",  "model_id": "llama3.1:8b",             "provider": "ollama", "base_url": "http://localhost:11434/v1", "api_key": "ollama", "tier": "SLM", "params": "8B",  "delay": 1.0, "timeout": 300.0 },
    { "name": "Llama-3.3-70B", "model_id": "llama-3.3-70b-versatile", "provider": "groq",   "base_url": "https://api.groq.com/openai/v1", "api_key": os.getenv("GROQ_API_KEY"), "tier": "LLM", "params": "70B", "delay": 2.0, "timeout": 30.0 },
]

SYSTEM_PROMPT = (
    "You are a sentiment classification system. "
    "Classify the product review as POSITIVE, NEGATIVE, or NEUTRAL. "
    "POSITIVE = overall satisfied, recommends the product. "
    "NEGATIVE = overall dissatisfied, does not recommend. "
    "NEUTRAL = mixed feelings, neither clearly positive nor negative. "
    "Reply with exactly ONE word: POSITIVE, NEGATIVE, or NEUTRAL. "
    "No explanation. No punctuation. Just one word."
)

VALID_LABELS = {"POSITIVE", "NEGATIVE", "NEUTRAL"}

def find_partial_file():
    """Find the most recent partial UC4 file with 450 rows."""
    files = sorted(glob.glob(os.path.join(RAW_OUTPUT_DIR, "uc4_raw_*.csv")), reverse=True)
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
            if len(rows) == 450:
                print(f"  Found partial file: {f} ({len(rows)} rows)")
                return f, rows
    return None, []

def load_test_items():
    items = []
    with open(GOLD_SET_PATH, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                items.append(row)
    return items

def clean_response(raw):
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().upper()
    for label in VALID_LABELS:
        if label in cleaned:
            return label
    return "INVALID"

def run_inference(model_cfg, item, run_idx):
    try:
        client = OpenAI(
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            timeout=httpx.Timeout(model_cfg["timeout"], connect=10.0)
        )
        start    = time.time()
        response = client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": f"Classify this product review: {item['text']}"}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEEDS[run_idx] if model_cfg["provider"] == "groq" else None,
        )
        latency = round((time.time() - start) * 1000)
        raw     = response.choices[0].message.content
        result  = clean_response(raw)
        return result, latency, raw, None
    except Exception as e:
        return "ERROR", 0, "", str(e)[:120]

def run_model_benchmark(model_cfg, test_items):
    print(f"\n  -- {model_cfg['name']} ({model_cfg['tier']} - {model_cfg['params']}) --")
    print(f"     {len(test_items)} items x {RUNS} runs = {len(test_items)*RUNS} inferences")

    rows      = []
    latencies = []

    for item in test_items:
        for run_idx in range(RUNS):
            result, latency, raw, error = run_inference(model_cfg, item, run_idx)
            correct = (result == item["label"]) if result not in ["ERROR","INVALID"] else False
            rows.append({
                "timestamp":    datetime.now().isoformat(),
                "model_name":   model_cfg["name"],
                "model_tier":   model_cfg["tier"],
                "model_params": model_cfg["params"],
                "provider":     model_cfg["provider"],
                "item_id":      item["id"],
                "item_text":    item["text"][:80] + "..." if len(item["text"]) > 80 else item["text"],
                "gold_label":   item["label"],
                "category":     item["category"],
                "difficulty":   item["difficulty"],
                "run_number":   run_idx + 1,
                "prediction":   result,
                "correct":      correct,
                "latency_ms":   latency,
                "raw_output":   raw[:200] if raw else "",
                "error":        error or "",
            })
            latencies.append(latency)
            icon = "OK" if correct else "X" if result != "ERROR" else "E"
            print(f"     [{item['id']}] run{run_idx+1}: {result:<10} {icon}  ({latency}ms)")
        time.sleep(model_cfg["delay"])

    total_inf = len(test_items) * RUNS
    total_cor = sum(1 for r in rows if r["correct"])
    accuracy  = round(total_cor / total_inf * 100, 1)
    valid_lats = sorted([l for l in latencies if l > 0])
    p50 = valid_lats[len(valid_lats)//2]        if valid_lats else 0
    p95 = valid_lats[int(len(valid_lats)*0.95)] if valid_lats else 0
    print(f"\n     Done - Accuracy: {accuracy}%  |  P50: {p50}ms  |  P95: {p95}ms")
    return rows, accuracy, p50, p95

def save_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    print()
    print("=" * 65)
    print("  UC4 RESUME — Running models 6 & 7 only")
    print("  Llama-3.1-8B + Llama-3.3-70B (180 inferences)")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Load existing 450 rows
    partial_file, existing_rows = find_partial_file()
    if not existing_rows:
        print("\n  ERROR: Could not find partial file with 450 rows.")
        print("  Please run the full script: python3 scripts/run_benchmark_uc4.py")
        exit(1)

    test_items = load_test_items()
    print(f"  Loaded {len(test_items)} test items")
    print(f"  Existing rows: {len(existing_rows)}")

    new_rows     = []
    summary_rows = []
    llm_accuracy = None

    for model_cfg in MODELS:
        if model_cfg["provider"] == "ollama":
            print(f"\n  Warming up {model_cfg['name']}...")
            warmup_client = OpenAI(
                api_key=model_cfg["api_key"],
                base_url=model_cfg["base_url"],
                timeout=httpx.Timeout(model_cfg["timeout"], connect=10.0)
            )
            try:
                warmup_client.chat.completions.create(
                    model=model_cfg["model_id"],
                    messages=[{"role": "user", "content": "Reply with one word: POSITIVE"}],
                    max_tokens=16, temperature=0.0
                )
                print(f"  Warmup OK")
            except Exception as e:
                print(f"  Warmup warning: {e}")

        raw_rows, accuracy, p50, p95 = run_model_benchmark(model_cfg, test_items)
        new_rows.extend(raw_rows)

        summary_rows.append({
            "model_name":       model_cfg["name"],
            "tier":             model_cfg["tier"],
            "params":           model_cfg["params"],
            "provider":         model_cfg["provider"],
            "accuracy_pct":     accuracy,
            "latency_p50":      p50,
            "latency_p95":      p95,
            "total_inferences": len(test_items) * RUNS,
        })
        if model_cfg["tier"] == "LLM":
            llm_accuracy = accuracy

        # Save merged file after each model
        all_rows = existing_rows + new_rows
        save_csv(RAW_FILE, all_rows)
        print(f"\n  Saved merged file: {RAW_FILE} ({len(all_rows)} rows)")

    # Final merged file
    all_rows = existing_rows + new_rows
    save_csv(RAW_FILE, all_rows)
    save_csv(SUMMARY_FILE, summary_rows)

    print()
    print("=" * 65)
    print("  RESUME COMPLETE")
    print(f"  Total rows in merged file: {len(all_rows)} (should be 630)")
    print(f"  Merged file: {RAW_FILE}")
    print()
    if llm_accuracy:
        print(f"  LLM (Llama-3.3-70B) accuracy: {llm_accuracy}%")
    for r in summary_rows:
        print(f"  {r['model_name']}: {r['accuracy_pct']}%")
    print()
    print("  NEXT STEP:")
    print("  => python3 scripts/evaluate_uc4.py")
    print("=" * 65)
