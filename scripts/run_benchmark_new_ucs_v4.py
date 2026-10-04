"""
run_benchmark_new_ucs_v2.py

Improved version: lets you pick WHICH models to run per command (so you can
do small batches, e.g. 3 at a time), AND saves results after EACH model
finishes (not just at the end of a whole use case) -- so if anything
interrupts the run, nothing already completed is lost.

Usage examples:
    # Run just 3 models on uc1:
    python run_benchmark_new_ucs_v2.py --use_case uc1 --models Llama-3.2-3B,Phi4-Mini,Gemma3-4B

    # Then run the remaining models on uc1 later:
    python run_benchmark_new_ucs_v2.py --use_case uc1 --models Qwen2.5-7B,Mistral-7B,Llama-3.3-70B

    # Run all 7 models (default) on uc2:
    python run_benchmark_new_ucs_v2.py --use_case uc2

    # List available model names:
    python run_benchmark_new_ucs_v2.py --list_models
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
MAX_TOKENS = 700
# NOTE: was 128 -- far too small for GPT-OSS-120B, a reasoning model that
# spends part of its token budget "thinking" before answering. At 128, ~60%
# of its responses came back completely empty (all budget consumed by
# reasoning, none left for the actual answer). 700 leaves room for both.
SEEDS = [42, 43, 44]
RUNS = len(SEEDS)

# ---- All 7 models available. Pick a subset per run using --models ----
ALL_MODELS = [
    {"name": "Llama-3.2-3B", "model_id": "llama3.2:3b", "provider": "ollama",
     "base_url": "http://localhost:11434/v1", "api_key": "ollama", "tier": "SLM",
     "params": "3B", "delay": 1.0, "timeout": 300.0},
    {"name": "Phi4-Mini", "model_id": "phi4-mini:latest", "provider": "ollama",
     "base_url": "http://localhost:11434/v1", "api_key": "ollama", "tier": "SLM",
     "params": "3.8B", "delay": 1.0, "timeout": 300.0},
    {"name": "Gemma3-4B", "model_id": "gemma3:4b", "provider": "ollama",
     "base_url": "http://localhost:11434/v1", "api_key": "ollama", "tier": "SLM",
     "params": "4B", "delay": 1.0, "timeout": 300.0},
    {"name": "Qwen2.5-7B", "model_id": "qwen2.5:7b", "provider": "ollama",
     "base_url": "http://localhost:11434/v1", "api_key": "ollama", "tier": "SLM",
     "params": "7B", "delay": 1.0, "timeout": 300.0},
    {"name": "Mistral-7B", "model_id": "mistral:latest", "provider": "ollama",
     "base_url": "http://localhost:11434/v1", "api_key": "ollama", "tier": "SLM",
     "params": "7B", "delay": 1.0, "timeout": 300.0},
    {"name": "Llama-3.1-8B", "model_id": "llama3.1:8b", "provider": "ollama",
     "base_url": "http://localhost:11434/v1", "api_key": "ollama", "tier": "SLM",
     "params": "8B", "delay": 1.0, "timeout": 300.0},
    {"name": "GPT-OSS-120B", "model_id": "openai/gpt-oss-120b", "provider": "groq",
     "base_url": "https://api.groq.com/openai/v1", "api_key": os.getenv("GROQ_API_KEY"),
     "tier": "LLM", "params": "70B", "delay": 5.0, "timeout": 60.0},
]
MODEL_LOOKUP = {m["name"]: m for m in ALL_MODELS}

# ---- Per-use-case configuration ----
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
        "valid_labels": None,
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
        "gold_path": "data/gold_sets/uc4_amazon_review_sentiment.csv",
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
        "valid_labels": None,
        "user_prefix": "Classify the CWE category for this vulnerability description:",
    },
    "uc6": {
        "gold_path": "data/gold_sets/uc6_symptom_disease_classification.csv",
        "system_prompt": (
            "You are a clinical symptom-to-diagnosis classification system. "
            "Given a list of patient symptoms, identify the single most likely diagnosis. "
            "Reply with exactly the diagnosis name. No explanation."
        ),
        "valid_labels": None,
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


def load_test_items(gold_path, item_start=None, item_end=None):
    items = []
    with open(gold_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] == "test":
                items.append(row)
    if item_start is not None or item_end is not None:
        start_idx = (item_start - 1) if item_start else 0
        end_idx = item_end if item_end else len(items)
        items = items[start_idx:end_idx]
    return items


def clean_response(raw, valid_labels):
    cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip().upper()
    if valid_labels:
        for label in valid_labels:
            if label.upper() in cleaned:
                return label
        return "INVALID"
    else:
        return cleaned[:200]


def run_inference(model_cfg, item, run_idx, system_prompt, user_prefix, valid_labels,
                   max_retries=6):
    client = OpenAI(
        api_key=model_cfg["api_key"],
        base_url=model_cfg["base_url"],
        timeout=httpx.Timeout(model_cfg["timeout"], connect=10.0)
    )
    for attempt in range(max_retries):
        try:
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
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                # Exponential backoff: 10s, 20s, 40s, 80s, 160s, then give up
                wait_time = 10 * (2 ** attempt)
                print(f"\n  [Rate limited -- waiting {wait_time}s before retry "
                      f"{attempt+2}/{max_retries}]", end="")
                time.sleep(wait_time)
                continue
            return "ERROR", 0, "", err_str[:120]
    return "ERROR", 0, "", "Max retries exceeded"


def append_csv(path, rows):
    """Append rows to a CSV, writing the header only if the file is new."""
    if not rows:
        return
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def run_model_benchmark(model_cfg, test_items, system_prompt, user_prefix, valid_labels,
                         uc_key, raw_path, summary_path):
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

    # ---- KEY CHANGE: save immediately after this one model finishes ----
    append_csv(raw_path, rows)
    append_csv(summary_path, [{
        "use_case": uc_key, "model_name": model_cfg["name"], "tier": model_cfg["tier"],
        "params": model_cfg["params"], "accuracy_pct": accuracy, "p50_ms": p50, "p95_ms": p95,
    }])
    print(f"  -> Saved {model_cfg['name']}'s results to {raw_path} (appended)")

    return accuracy, p50, p95


def run_one_use_case(uc_key, selected_models, item_start=None, item_end=None):
    cfg = USE_CASES[uc_key]
    test_items = load_test_items(cfg["gold_path"], item_start, item_end)
    range_note = f" (items {item_start or 1}-{item_end or len(test_items)})" if (item_start or item_end) else ""
    print(f"\n{'='*60}\nRunning {uc_key}{range_note} -- {len(test_items)} test items")
    print(f"Models this run: {', '.join(m['name'] for m in selected_models)}\n{'='*60}")

    # One fixed filename per use case (not per-run timestamp) so repeated
    # partial runs all append to the SAME file instead of creating a new
    # timestamped file each time.
    raw_path = os.path.join(RAW_OUTPUT_DIR, f"newuc{uc_key[2:]}_raw.csv")
    summary_path = os.path.join(RESULTS_DIR, f"newuc{uc_key[2:]}_summary.csv")

    for model_cfg in selected_models:
        run_model_benchmark(
            model_cfg, test_items, cfg["system_prompt"], cfg["user_prefix"],
            cfg["valid_labels"], uc_key, raw_path, summary_path
        )
        # Free memory before loading the next model -- critical given limited
        # available RAM (Ollama's own log reported only ~598 MB free at one
        # point). NOTE: benchmark_utils.py's check_memory()/log_memory_state()
        # only support Mac/Linux and always report 0MB on Windows, so they
        # are intentionally NOT used here -- they would show a false "low
        # memory" warning every time on this Windows machine.
        if model_cfg["provider"] == "ollama":
            print(f"  Unloading {model_cfg['name']} to free memory before next model...")
            try:
                unload_ollama_model(model_cfg["model_id"])
                time.sleep(3.0)  # give the OS a moment to actually reclaim memory
            except Exception as e:
                print(f"  Warning: could not unload {model_cfg['name']}: {e}")

    print(f"\nAll requested models finished for {uc_key}.")
    print(f"Cumulative raw output: {raw_path}")
    print(f"Cumulative summary: {summary_path}")


def check_already_done(uc_key, model_names):
    """Check the summary file to see which of the requested models already
    have results for this use case, so you don't accidentally re-run them."""
    summary_path = os.path.join(RESULTS_DIR, f"newuc{uc_key[2:]}_summary.csv")
    if not os.path.isfile(summary_path):
        return set()
    done = set()
    with open(summary_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(row["model_name"])
    return done & set(model_names)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_case", choices=list(USE_CASES.keys()) + ["all"])
    parser.add_argument("--models", default=None,
                         help="Comma-separated model names to run (default: all 7). "
                              "Use --list_models to see valid names.")
    parser.add_argument("--list_models", action="store_true")
    parser.add_argument("--force", action="store_true",
                         help="Re-run models even if already marked done for this use case.")
    parser.add_argument("--item_start", type=int, default=None,
                         help="Only run items starting from this number (1-indexed). "
                              "Use with --item_end to run a smaller batch, e.g. "
                              "--item_start 1 --item_end 10 runs just items 1-10.")
    parser.add_argument("--item_end", type=int, default=None,
                         help="Only run items up to and including this number.")
    args = parser.parse_args()

    if args.list_models:
        print("Available model names:")
        for m in ALL_MODELS:
            print(f"  {m['name']}")
        exit()

    if not args.use_case:
        parser.error("--use_case is required (or use --list_models)")

    if args.models:
        requested_names = [n.strip() for n in args.models.split(",")]
        unknown = [n for n in requested_names if n not in MODEL_LOOKUP]
        if unknown:
            print(f"Unknown model name(s): {unknown}")
            print("Run --list_models to see valid names.")
            exit(1)
        selected_models = [MODEL_LOOKUP[n] for n in requested_names]
    else:
        selected_models = ALL_MODELS

    use_case_list = list(USE_CASES.keys()) if args.use_case == "all" else [args.use_case]

    for uc_key in use_case_list:
        model_names = [m["name"] for m in selected_models]
        already_done = check_already_done(uc_key, model_names)
        if already_done and not args.force:
            print(f"\nSkipping already-completed models for {uc_key}: {already_done}")
            print("(use --force to re-run them anyway)")
        to_run = [m for m in selected_models if args.force or m["name"] not in already_done]
        if not to_run:
            print(f"\nAll requested models already done for {uc_key}. Nothing to run.")
            continue
        run_one_use_case(uc_key, to_run, args.item_start, args.item_end)
