"""
evaluate_open_ended_ucs.py

LLM-judge evaluation for the three open-ended new use cases where exact-match
scoring is too strict: UC2 (receipt field extraction), UC5 (CVE-to-CWE
classification), UC6 (symptom-to-diagnosis classification).

This is NEW code -- it follows the same general "LLM-judge + human audit"
pattern described in the paper for the original UC8, but is not a direct
port of an existing script (no existing evaluate_uc8.py was found to reuse).
Please review the judge prompt and spot-check its verdicts against a sample
of items before treating its output as final, the same way UC8's original
audit worked.

Usage:
    python evaluate_open_ended_ucs.py --raw_file data/raw_outputs/uc2_raw_TIMESTAMP.csv --use_case uc2
"""

import os, csv, json, time, argparse
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

JUDGE_MODEL = "openai/gpt-oss-120b"
JUDGE_PROVIDER_BASE_URL = "https://api.groq.com/openai/v1"
JUDGE_API_KEY = os.getenv("GROQ_API_KEY")

# Per-use-case judge instructions -- tell the judge what "close enough" means
# for this specific task type.
JUDGE_PROMPTS = {
    "uc2": (
        "You are auditing a receipt field extraction system. "
        "You will see the GOLD (correct) answer and the MODEL's answer, both "
        "in the format 'company=...; date=...; total=...'. "
        "Judge the model's answer CORRECT if the company name, date, and "
        "total amount all reasonably match the gold answer (minor formatting "
        "differences, e.g. '25/12/2018' vs 'Dec 25, 2018', or abbreviated "
        "company names, should still count as correct). "
        "Judge INCORRECT if any of the three fields is substantively wrong "
        "or missing. "
        "Reply with exactly one word: CORRECT or INCORRECT."
    ),
    "uc5": (
        "You are auditing a vulnerability classification system. "
        "You will see the GOLD (correct) CWE category and the MODEL's answer. "
        "Judge the model's answer CORRECT if it names the same CWE ID, or a "
        "CWE ID that is a reasonable synonym/parent-category match (e.g. "
        "'CWE-79' and 'Cross-site Scripting' referring to the same "
        "vulnerability type should count as correct). "
        "Judge INCORRECT if the model names a clearly different vulnerability "
        "category. "
        "Reply with exactly one word: CORRECT or INCORRECT."
    ),
    "uc6": (
        "You are auditing a clinical symptom-to-diagnosis classification "
        "system. You will see the GOLD (correct) diagnosis and the MODEL's "
        "answer. Judge the model's answer CORRECT if it names the same "
        "diagnosis, or a clearly equivalent/synonymous diagnosis (e.g. "
        "'heart attack' and 'myocardial infarction' should count as correct). "
        "Judge INCORRECT if the model names a substantively different "
        "diagnosis. "
        "Reply with exactly one word: CORRECT or INCORRECT."
    ),
}


def judge_one(use_case, gold_answer, model_answer):
    client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_PROVIDER_BASE_URL)
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_PROMPTS[use_case] +
                    " Think briefly if needed, but you MUST end your reply "
                    "with a final line containing only the single word "
                    "CORRECT or INCORRECT."},
                {"role": "user", "content": f"GOLD: {gold_answer}\nMODEL: {model_answer}"}
            ],
            temperature=0.0,
            max_tokens=500,  # GPT-OSS-120B is a reasoning model and needs
                              # room to think before its final verdict word;
                              # 10 tokens (the old value) cut it off before
                              # it ever reached CORRECT/INCORRECT.
        )
        full_text = response.choices[0].message.content.strip().upper()
        # Reasoning models put the final answer LAST, after any thinking --
        # so check the end of the response, not just anywhere in it.
        tail = full_text[-30:]
        if "INCORRECT" in tail:
            return "INCORRECT"
        elif "CORRECT" in tail:
            return "CORRECT"
        # Fallback: search the whole response if the verdict wasn't in the tail
        elif "INCORRECT" in full_text:
            return "INCORRECT"
        elif "CORRECT" in full_text:
            return "CORRECT"
        else:
            return f"JUDGE_UNCLEAR: {full_text[:80]}"
    except Exception as e:
        return f"JUDGE_ERROR: {str(e)[:80]}"


def evaluate_raw_file(raw_path, use_case, output_path=None):
    rows = []
    with open(raw_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gold = row["gold_label"]
            model_answer = row["prediction"]
            verdict = judge_one(use_case, gold, model_answer)
            row["judge_verdict"] = verdict
            row["judge_correct"] = (verdict == "CORRECT")
            rows.append(row)
            print(f"  item {row['item_id']} | model={model_answer[:40]:<40} | judge={verdict}")
            time.sleep(1.0)  # rate-limit courtesy delay, matches original script's pacing

    if output_path is None:
        output_path = raw_path.replace("_raw_", "_judged_")

    if rows:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    total = len(rows)
    correct = sum(1 for r in rows if r["judge_correct"])
    accuracy = round(correct / total * 100, 1) if total else 0
    print(f"\nJudged {total} items -- Judge-assessed accuracy: {accuracy}%")
    print(f"Saved to {output_path}")
    print(f"\nIMPORTANT: spot-check a random sample of the judge's verdicts "
          f"by hand before treating {accuracy}% as final, the same way the "
          f"original UC8 combined LLM-judge scoring with a manual audit.")
    return rows, accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_file", required=True, help="Path to the raw benchmark output CSV")
    parser.add_argument("--use_case", required=True, choices=["uc2", "uc5", "uc6"])
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    evaluate_raw_file(args.raw_file, args.use_case, args.output_file)
