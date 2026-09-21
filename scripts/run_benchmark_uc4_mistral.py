"""
UC4 MISTRAL-ONLY SCRIPT — runs only Mistral-7B then merges with existing 630-row file.
"""

import os, csv, time, re, httpx, glob
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

TEMPERATURE = 0.0
MAX_TOKENS  = 128
SEEDS       = [42, 43, 44]
RUNS        = len(SEEDS)

MISTRAL = {
    "name": "Mistral-7B",
    "model_id": "mistral:latest",
    "provider": "ollama",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "tier": "SLM",
    "params": "7B",
    "delay": 1.0,
    "timeout": 300.0
}

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

def find_complete_file():
    """Find the most recent UC4 file with 630 rows."""
    files = sorted(glob.glob(os.path.join(RAW_OUTPUT_DIR, "uc4_raw_*.csv")), reverse=True)
    for f in files:
        with open(f, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
            if len(rows) == 630:
                print(f"  Found complete file: {f} ({len(rows)} rows)")
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
        )
        latency = round((time.time() - start) * 1000)
        raw     = response.choices[0].message.content
        result  = clean_response(raw)
        return result, latency, raw, None
    except Exception as e:
        return "ERROR", 0, "", str(e)[:120]

def save_csv(path, rows):
    if not rows: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

if __name__ == "__main__":
    print()
    print("=" * 65)
    print("  UC4 MISTRAL-ONLY RUN")
    print("  Mistral-7B x 30 items x 3 seeds = 90 inferences")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # Load existing 630 rows
    complete_file, existing_rows = find_complete_file()
    if not existing_rows:
        print("\n  ERROR: Could not find complete file with 630 rows.")
        print("  Please check data/raw_outputs/ folder.")
        exit(1)

    test_items = load_test_items()
    print(f"  Loaded {len(test_items)} test items")
    print(f"  Existing rows: {len(existing_rows)}")

    # Warmup
    print(f"\n  Warming up Mistral-7B...")
    warmup_client = OpenAI(
        api_key=MISTRAL["api_key"],
        base_url=MISTRAL["base_url"],
        timeout=httpx.Timeout(MISTRAL["timeout"], connect=10.0)
    )
    try:
        warmup_client.chat.completions.create(
            model=MISTRAL["model_id"],
            messages=[{"role": "user", "content": "Reply with one word: POSITIVE"}],
            max_tokens=16, temperature=0.0
        )
        print("  Warmup OK")
    except Exception as e:
        print(f"  Warmup warning: {e}")

    # Run Mistral
    print(f"\n  -- Mistral-7B (SLM - 7B) --")
    print(f"     {len(test_items)} items x {RUNS} runs = {len(test_items)*RUNS} inferences")

    mistral_rows = []
    latencies    = []

    for item in test_items:
        for run_idx in range(RUNS):
            result, latency, raw, error = run_inference(MISTRAL, item, run_idx)
            correct = (result == item["label"]) if result not in ["ERROR", "INVALID"] else False
            mistral_rows.append({
                "timestamp":    datetime.now().isoformat(),
                "model_name":   MISTRAL["name"],
                "model_tier":   MISTRAL["tier"],
                "model_params": MISTRAL["params"],
                "provider":     MISTRAL["provider"],
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
        time.sleep(MISTRAL["delay"])

    # Results
    total_cor = sum(1 for r in mistral_rows if r["correct"])
    accuracy  = round(total_cor / (len(test_items) * RUNS) * 100, 1)
    valid_lats = sorted([l for l in latencies if l > 0])
    p50 = valid_lats[len(valid_lats)//2]        if valid_lats else 0
    p95 = valid_lats[int(len(valid_lats)*0.95)] if valid_lats else 0

    print(f"\n  Done - Mistral-7B Accuracy: {accuracy}%  |  P50: {p50}ms  |  P95: {p95}ms")

    # Merge and save
    all_rows = existing_rows + mistral_rows
    save_csv(RAW_FILE, all_rows)

    print()
    print("=" * 65)
    print(f"  Total rows in merged file: {len(all_rows)} (should be 720)")
    print(f"  Mistral-7B accuracy: {accuracy}%")
    print(f"  Merged file: {RAW_FILE}")
    print()
    print("  NEXT STEP:")
    print("  => python3 scripts/evaluate_uc4.py")
    print("=" * 65)
