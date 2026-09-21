"""
run_benchmark_new_ucs_llama8b.py

Runs ONLY Llama-3.1-8B (the slowest model from the original benchmark) plus
the Llama-3.3-70B baseline, across all 8 new use cases. Run this AFTER
run_benchmark_new_ucs_main.py completes, so the heaviest model runs alone.

Adapted from run_benchmark_uc4.py — reuses the exact same MODELS list,
run_inference, and clean_response logic, generalized to loop across all
8 new use cases (each with its own gold-set file and system prompt).

Usage:
    python run_benchmark_new_ucs.py --use_case uc1
    python run_benchmark_new_ucs.py --use_case uc2
    ... etc, through uc8
    python run_benchmark_new_ucs.py --use_case all    (runs all 8 in sequence)
"""

import os, csv, json, time, re, httpx, argparse
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime
from benchmark_utils import log_memory_state, check_memory, unload_ollama_model, warm_up_inference

load_dotenv()

RAW_OUTPUT_DIR = "data/raw_outputs"
RESULTS_DIR = "data/results"
os.makedirs(RAW_OUTPUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

TEMPERATURE = 0.0
MAX_TOKENS = 128
SEEDS = [42, 43, 44]
RUNS = len(SEEDS)

# ---- Same MODELS list as run_benchmark_uc4.py, unchanged ----
MODELS = [
    {"name": "Llama-3.1-8B", "model_id": "llama3.1:8b", "provider": "ollama",
     "base_url": "http://localhost:11434/v1", "api_key": "ollama", "tier": "SLM",
     "params": "8B", "delay": 1.0, "timeout": 300.0},
    {"name": "Llama-3.3-70B", "model_id": "llama-3.3-70b-versatile", "provider": "groq",
     "base_url": "https://api.groq.com/openai/v1", "api_key": os.getenv("GROQ_API_KEY"),
     "tier": "LLM", "params": "70B", "delay": 2.0, "timeout": 30.0},
]
# NOTE: Run this AFTER run_benchmark_new_ucs_main.py has completed. Llama-3.1-8B
# was the slowest model in the original benchmark; running it alone reduces
# memory pressure. The Llama-3.3-70B baseline is included here too so this
# script's raw output is self-contained and directly comparable.

# ---- Per-use-case configuration: gold set path, system prompt, valid labels ----
# NOTE: gold set CSVs must be reformatted to columns: id, text, label, category, difficulty, split
# (matching the uc4 format) -- already done for these 8 files.

USE_CASES = {
    "uc1": {
        "gold_path": "data/gold_sets/uc1_phishing_detection.csv",
        "system_prompt": (
            "You are a phishing detection system. "
            "Classify the email as PHISHING or LEGITIMATE. "
            "Reply with exactly ONE word: PHISHING or LEGITIMATE. "
            "No explanation. No punctuation. Just one word."
        ),
        "valid_labels": {"PHISHING", "LEGITIMATE"},
        "user_prefix": "Classify this email:",
    },
    "uc2": {
        "gold_path": "data/gold_sets/uc2_receipt_extraction.csv",
        "system_prompt": (
            "You are a receipt field extraction system. "
            "Given OCR-extracted receipt text, extract the company name, date, and total amount. "
            "Reply in exactly this format: company=<name>; date=<date>; total=<amount>. "
            "No explanation."
        ),
        "valid_labels": None,  # free-form extraction, not a fixed label set -- needs custom scoring
        "user_prefix": "Extract fields from this receipt text:",
    },
    "uc3": {
        "gold_path": "data/gold_sets/uc3_it_ticket_classification.csv",
        "system_prompt": (
            "You are an IT support ticket classification system. "
            "Classify the ticket into exactly one category: Fileservice, Support general, "
            "Software, O365, Active Directory, Computer-Services, or EOL. "
            "Reply with exactly the category name. No explanation."
        ),
        "valid_labels": {"Fileservice", "Support general", "Software", "O365",
                          "Active Directory", "Computer-Services", "EOL"},
        "user_prefix": "Classify this IT ticket:",
    },
    "uc4": {
        "gold_path": "data/gold_sets/uc4_product_review_sentiment.csv",
        "system_prompt": (
            "You are a sentiment classification system. "
            "Classify the product review as POSITIVE, NEGATIVE, or NEUTRAL. "
            "POSITIVE = overall satisfied, recommends the product. "
            "NEGATIVE = overall dissatisfied, does not recommend. "
            "NEUTRAL = mixed feelings, neither clearly positive nor negative. "
            "Reply with exactly ONE word: POSITIVE, NEGATIVE, or NEUTRAL. "
            "No explanation. No punctuation. Just one word."
        ),
        "valid_labels": {"POSITIVE", "NEGATIVE", "NEUTRAL"},
        "user_prefix": "Classify this product review:",
    },
    "uc5": {
        "gold_path": "data/gold_sets/uc5_cve_cwe_classification.csv",
        "system_prompt": (
            "You are a vulnerability classification system. "
            "Given a CVE description, identify the most likely CWE category "
            "(e.g., CWE-79, CWE-89, CWE-119, NVD-CWE-Other, etc.). "
            "Reply with exactly the CWE identifier. No explanation."
        ),
        "valid_labels": None,  # open-ended CWE ID space -- needs custom/fuzzy scoring
        "user_prefix": "Classify the CWE category for this vulnerability description:",
    },
    "uc6": {
        "gold_path": "data/gold_sets/uc6_symptom_disease_classification.csv",
        "system_prompt": (
            "You are a clinical symptom-to-diagnosis classification system. "
            "Given a list of patient symptoms, identify the single most likely diagnosis. "
            "Reply with exactly the diagnosis name. No explanation."
        ),
        "valid_labels": None,  # open-ended diagnosis space -- needs custom/fuzzy scoring
        "user_prefix": "Identify the most likely diagnosis given these symptoms:",
    },
    "uc7": {
        "gold_path": "data/gold_sets/uc7_contract_clause_classification.csv",
        "system_prompt": (
            "You are a legal contract clause classification system. "
            "Given a question about whether a specific clause type is present in a contract, "
            "answer PRESENT or NOT PRESENT. "
            "Reply with exactly ONE phrase: PRESENT or NOT PRESENT. No explanation."
        ),
        "valid_labels": {"PRESENT", "NOT PRESENT"},
        "user_prefix": "Answer this contract clause question:",
    },
    "uc8": {
        "gold_path": "data/gold_sets/uc8_financial_news_sentiment.csv",
        "system_prompt": (
            "You are a financial news sentiment classification system. "
            "Classify the sentiment of this financial news sentence as POSITIVE, NEGATIVE, or NEUTRAL. "
            "Reply with exactly ONE word: POSITIVE, NEGATIVE, or NEUTRAL. "
            "No explanation. No punctuation. Just one word."
        ),
        "valid_labels": {"POSITIVE", "NEGATIVE", "NEUTRAL"},
        "user_prefix": "Classify the sentiment of this financial news sentence:",
    },
}


def load_test_items(gold_path):
    items = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                items.append(row)
    return items


def clean_response(raw, valid_labels):
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().upper()
    if valid_labels:
        for label in valid_labels:
            if label.upper() in cleaned:
                return label
        return "INVALID"
    else:
        # Free-form task (extraction/open-label) -- return raw cleaned text,
        # scoring handled separately (see NOTE below)
        return cleaned[:200]


def run_inference(model_cfg, item, run_idx, system_prompt, user_prefix, valid_labels):
    try:
        client = OpenAI(
            api_key=model_cfg["api_key"],
            base_url=model_cfg["base_url"],
            timeout=httpx.Timeout(model_cfg["timeout"], connect=10.0)
        )
        start = time.time()
        response = client.chat.completions.create(
            model=model_cfg["model_id"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prefix}\n{item['text']}"}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            seed=SEEDS[run_idx] if model_cfg["provider"] == "groq" else None,
        )
        latency = round((time.time() - start) * 1000)
        raw = response.choices[0].message.content
        result = clean_response(raw, valid_labels)
        return result, latency, raw, None
    except Exception as e:
        return "ERROR", 0, "", str(e)[:120]


def run_model_benchmark(model_cfg, test_items, system_prompt, user_prefix, valid_labels):
    print(f"\n-- {model_cfg['name']} ({model_cfg['tier']} - {model_cfg['params']}) --")
    print(f"  {len(test_items)} items x {RUNS} runs = {len(test_items)*RUNS} inferences")
    rows = []
    latencies = []

    for item in test_items:
        for run_idx in range(RUNS):
            result, latency, raw, error = run_inference(
                model_cfg, item, run_idx, system_prompt, user_prefix, valid_labels
            )
            if valid_labels:
                correct = (result == item["label"]) if result not in ["ERROR", "INVALID"] else False
            else:
                # NOTE: for free-form tasks (uc2 receipt extraction, uc5 CWE,
                # uc6 diagnosis) exact-match on raw text is too strict.
                # Use evaluate_new_ucs.py (LLM-judge or fuzzy match) for real
                # scoring, same pattern as the original UC8 in the paper.
                correct = (result.strip().upper() == str(item["label"]).strip().upper())

            rows.append({
                "timestamp": datetime.now().isoformat(),
                "model_name": model_cfg["name"],
                "model_tier": model_cfg["tier"],
                "model_params": model_cfg["params"],
                "provider": model_cfg["provider"],
                "item_id": item["id"],
                "item_text": item["text"][:80] + "..." if len(item["text"]) > 80 else item["text"],
                "gold_label": item["label"],
                "category": item["category"],
                "difficulty": item["difficulty"],
                "run_number": run_idx + 1,
                "prediction": result,
                "correct": correct,
                "latency_ms": latency,
                "raw_output": raw[:200] if raw else "",
                "error": error or "",
            })
            latencies.append(latency)
            icon = "OK" if correct else "X" if result != "ERROR" else "E"
            print(f"  [{item['id']}] run{run_idx+1}: {str(result)[:20]:<20} {icon} ({latency}ms)", end="\r")
            time.sleep(model_cfg["delay"])

    total_inf = len(test_items) * RUNS
    total_cor = sum(1 for r in rows if r["correct"])
    accuracy = round(total_cor / total_inf * 100, 1) if total_inf else 0
    valid_lats = sorted([l for l in latencies if l > 0])
    p50 = valid_lats[len(valid_lats)//2] if valid_lats else 0
    p95 = valid_lats[int(len(valid_lats)*0.95)] if valid_lats else 0
    print(f"\n  Done - Accuracy: {accuracy}% | P50: {p50}ms | P95: {p95}ms")
    return rows, accuracy, p50, p95


def save_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def run_one_use_case(uc_key):
    cfg = USE_CASES[uc_key]
    test_items = load_test_items(cfg["gold_path"])
    print(f"\n{'='*60}\nRunning {uc_key} -- {len(test_items)} test items\n{'='*60}")

    all_rows = []
    summary_rows = []
    for model_cfg in MODELS:
        rows, accuracy, p50, p95 = run_model_benchmark(
            model_cfg, test_items, cfg["system_prompt"], cfg["user_prefix"], cfg["valid_labels"]
        )
        all_rows.extend(rows)
        summary_rows.append({
            "use_case": uc_key, "model_name": model_cfg["name"], "tier": model_cfg["tier"],
            "params": model_cfg["params"], "accuracy_pct": accuracy, "p50_ms": p50, "p95_ms": p95,
        })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(RAW_OUTPUT_DIR, f"{uc_key}_raw_{timestamp}.csv")
    summary_path = os.path.join(RESULTS_DIR, f"{uc_key}_summary_{timestamp}.csv")
    save_csv(raw_path, all_rows)
    save_csv(summary_path, summary_rows)
    print(f"\nSaved raw output to {raw_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_case", required=True, choices=list(USE_CASES.keys()) + ["all"])
    args = parser.parse_args()

    if args.use_case == "all":
        for uc_key in USE_CASES:
            run_one_use_case(uc_key)
    else:
        run_one_use_case(args.use_case)
