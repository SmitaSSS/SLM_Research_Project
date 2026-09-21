import os, re, glob

reports = {
    'UC1': 'evaluation/uc1_report_20260310_221837.txt',
    'UC2': 'evaluation/uc2_report_20260413.txt',
    'UC3': 'evaluation/uc3_report_20260311_094029.txt',
    'UC4': 'evaluation/uc4_report_20260415_092524.txt',
    'UC5': 'evaluation/uc5_report_20260413_113200.txt',
    'UC6': 'evaluation/uc6_report_20260413.txt',
    'UC7': 'evaluation/uc7_report_20260413_121937.txt',
    'UC8': 'evaluation/uc8_report_20260311_152838.txt',
}

print()
print("=" * 80)
print("  S3 BENCHMARK — COMBINED RESULTS SUMMARY")
print("=" * 80)
print(f"  {'UC':<5} {'Best SLM Model':<20} {'SLM Acc':>8} {'LLM Acc':>8} {'Parity':>8}  Verdict")
print(f"  {'-'*5} {'-'*20} {'-'*8} {'-'*8} {'-'*8}  {'-'*15}")

for uc, path in reports.items():
    try:
        txt = open(path, encoding='utf-8').read()

        # Extract best SLM accuracy
        slm_match = re.search(r'Best SLM[:\s]+([A-Za-z0-9\-\.]+)[^\d]*(\d+\.\d+)%', txt)
        # Extract LLM accuracy
        llm_match = re.search(r'LLM baseline accuracy\s*[:\|]+\s*(\d+\.\d+)%', txt, re.IGNORECASE)
        if not llm_match:
            llm_match = re.search(r'Llama-3\.3-70B.*?(\d+\.\d+)%', txt)

        verdict = 'PURE SLM' if 'PURE SLM CONFIRMED' in txt else \
                  'HYBRID'   if 'HYBRID' in txt.upper() else \
                  'LLM ONLY' if 'LLM ONLY' in txt.upper() or 'LLM-ONLY' in txt.upper() else '?'

        slm_acc = float(slm_match.group(2)) if slm_match else None
        llm_acc = float(llm_match.group(1)) if llm_match else None
        model   = slm_match.group(1) if slm_match else '?'
        parity  = f"{slm_acc/llm_acc*100:.1f}%" if slm_acc and llm_acc else '?'
        slm_str = f"{slm_acc:.1f}%" if slm_acc else '?'
        llm_str = f"{llm_acc:.1f}%" if llm_acc else '?'

        print(f"  {uc:<5} {model:<20} {slm_str:>8} {llm_str:>8} {parity:>8}  {verdict}")

    except FileNotFoundError:
        print(f"  {uc:<5} {'FILE NOT FOUND':<20}")
    except Exception as e:
        print(f"  {uc:<5} ERROR: {e}")

print("=" * 80)
print()
