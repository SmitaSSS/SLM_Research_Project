"""
sensitivity_analysis.py
Multi-profile sensitivity analysis for S³ tier assignments.

Computes S³ scores for all 8 use cases under 5 different weight profiles
to test whether tier assignments are robust to reasonable weight variations.

Method: Triantaphyllou [35] sensitivity analysis — computes the minimum
weight change needed to flip each tier assignment.

Outputs:
  evaluation/sensitivity_matrix.csv     — S³ scores under all profiles
  evaluation/sensitivity_report.txt     — Full analysis report
"""

import os
import csv
from datetime import datetime

EVAL_DIR = "evaluation"
os.makedirs(EVAL_DIR, exist_ok=True)
TIMESTAMP    = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILE     = os.path.join(EVAL_DIR, f"sensitivity_matrix_{TIMESTAMP}.csv")
REPORT_FILE  = os.path.join(EVAL_DIR, f"sensitivity_report_{TIMESTAMP}.txt")

# ── Tier Boundaries ───────────────────────────────────────────
TAU_1, TAU_2 = 3.0, 3.7   # Pure SLM: S3 <= TAU_1; Hybrid: TAU_1 < S3 <= TAU_2; LLM Only: S3 > TAU_2

# OS and LT are reverse-scored (a HIGH raw score favours the small model), so the
# formula uses 6 - score for these two dimensions. Raw (1-5) scores are stored below.
REVERSE_SCORED = ("OS", "LT")

# ── S³ raw dimension scores for the 8 use cases (final values used in the paper; OS and LT are reverse-scored in the formula) ─────
USE_CASES = {
    "UC1_SMS_Threat":       {"TC": 2, "OS": 3, "SK": 4, "DS": 3, "LT": 4, "VL": 5},
    "UC2_Invoice_Extract":  {"TC": 2, "OS": 4, "SK": 2, "DS": 3, "LT": 2, "VL": 4},
    "UC3_Ticket_Routing":   {"TC": 1, "OS": 4, "SK": 3, "DS": 2, "LT": 3, "VL": 4},
    "UC4_Review_Sentiment": {"TC": 1, "OS": 3, "SK": 2, "DS": 1, "LT": 3, "VL": 3},
    "UC5_Code_Review":      {"TC": 4, "OS": 4, "SK": 4, "DS": 2, "LT": 2, "VL": 3},
    "UC6_Clinical_Triage":  {"TC": 4, "OS": 4, "SK": 5, "DS": 4, "LT": 5, "VL": 1},
    "UC7_Legal_Contract":   {"TC": 4, "OS": 2, "SK": 3, "DS": 4, "LT": 2, "VL": 2},
    "UC8_Financial_Report": {"TC": 5, "OS": 1, "SK": 4, "DS": 3, "LT": 2, "VL": 1},
}

# ── Weight Profiles ───────────────────────────────────────────
WEIGHT_PROFILES = {
    "Default":        {"TC": 3, "OS": 2, "SK": 4, "DS": 2, "LT": 3, "VL": 1},
    "Security-First": {"TC": 2, "OS": 1, "SK": 5, "DS": 4, "LT": 2, "VL": 1},
    "Latency-First":  {"TC": 2, "OS": 2, "SK": 3, "DS": 1, "LT": 5, "VL": 3},
    "Balanced":       {"TC": 3, "OS": 3, "SK": 3, "DS": 3, "LT": 3, "VL": 3},
    "Volume-Heavy":   {"TC": 2, "OS": 2, "SK": 3, "DS": 2, "LT": 2, "VL": 5},
}

DIMS = ["TC", "OS", "SK", "DS", "LT", "VL"]


def compute_s3(scores, weights):
    """Compute S³ score using dynamic-denominator WSM."""
    numerator = sum(((6 - scores[d]) if d in REVERSE_SCORED else scores[d]) * weights[d] for d in DIMS)
    denominator = sum(5 * weights[d] for d in DIMS)
    return round(numerator / denominator * 5, 4)


def assign_tier_formula(s3_score):
    """Assign tier by formula only (no gate rules)."""
    if s3_score <= TAU_1:
        return "Pure SLM"
    elif s3_score <= TAU_2:
        return "Hybrid"
    else:
        return "LLM Only"


def assign_tier_with_gates(s3_score, scores):
    """Assign tier with pre-screening gate rules."""
    if scores.get("SK", 0) == 5:
        return "LLM Only"
    if scores.get("TC", 0) == 5 and scores.get("SK", 0) >= 4:
        return "LLM Only"
    if scores.get("SK", 0) >= 4:
        return "Hybrid" if s3_score <= TAU_2 else "LLM Only"
    return assign_tier_formula(s3_score)


def is_hard_rule_locked(scores):
    """True when a hard rule (SK=5, or TC=5 with SK>=4) fixes the tier regardless of weights."""
    return scores.get("SK", 0) == 5 or (scores.get("TC", 0) == 5 and scores.get("SK", 0) >= 4)


def compute_triantaphyllou_margin(scores, weights, current_tier):
    """
    Compute the minimum percentage change to any single weight
    that would flip the tier assignment (Triantaphyllou method).

    Returns (min_change_pct, dimension, new_tier) or (inf, None, None)
    if no single-weight change can flip the tier.
    """
    s3_current = compute_s3(scores, weights)
    min_change = float("inf")
    flip_dim = None
    flip_tier = None

    for dim in DIMS:
        original_w = weights[dim]
        if original_w == 0:
            continue

        # Try increasing and decreasing the weight
        for delta_pct in range(1, 500):  # 1% to 500%
            for direction in [1, -1]:
                new_w = original_w * (1 + direction * delta_pct / 100)
                if new_w < 0:
                    continue

                test_weights = dict(weights)
                test_weights[dim] = new_w
                new_s3 = compute_s3(scores, test_weights)
                new_tier = assign_tier_formula(new_s3)

                # Check if gate rules apply (they depend on scores, not weights)
                if scores.get("SK", 0) == 5:
                    new_tier = "LLM Only"
                elif scores.get("TC", 0) == 5 and scores.get("SK", 0) >= 4:
                    new_tier = "LLM Only"
                elif scores.get("SK", 0) >= 4:
                    new_tier = "Hybrid" if new_s3 <= TAU_2 else "LLM Only"

                if new_tier != current_tier:
                    if delta_pct < min_change:
                        min_change = delta_pct
                        flip_dim = f"{dim} {'+'if direction>0 else '-'}{delta_pct}%"
                        flip_tier = new_tier
                    break  # Found flip point for this direction

    return min_change, flip_dim, flip_tier


def run_sensitivity():
    """Run full sensitivity analysis across all UCs and profiles."""
    # Step 1: Compute S³ for all UCs under all profiles
    matrix = {}
    for uc_name, scores in USE_CASES.items():
        matrix[uc_name] = {}
        for profile_name, weights in WEIGHT_PROFILES.items():
            s3 = compute_s3(scores, weights)
            tier = assign_tier_with_gates(s3, scores)
            matrix[uc_name][profile_name] = {"s3": s3, "tier": tier}

    # Step 2: Determine stability
    stability = {}
    for uc_name in USE_CASES:
        tiers = set(matrix[uc_name][p]["tier"] for p in WEIGHT_PROFILES)
        is_stable = len(tiers) == 1
        stability[uc_name] = {
            "stable": is_stable,
            "tiers": tiers,
            "unanimous_tier": tiers.pop() if is_stable else None,
        }

    # Step 3: Triantaphyllou margins (default profile)
    margins = {}
    default_weights = WEIGHT_PROFILES["Default"]
    for uc_name, scores in USE_CASES.items():
        s3 = compute_s3(scores, default_weights)
        tier = assign_tier_with_gates(s3, scores)
        min_change, flip_dim, flip_tier = compute_triantaphyllou_margin(
            scores, default_weights, tier
        )
        margins[uc_name] = {
            "s3": s3,
            "tier": tier,
            "min_change_pct": min_change,
            "flip_dim": flip_dim,
            "flip_tier": flip_tier,
        }

    return matrix, stability, margins


def save_csv(matrix):
    """Save sensitivity matrix to CSV."""
    rows = []
    for uc_name, profiles in matrix.items():
        row = {"use_case": uc_name}
        for profile_name, data in profiles.items():
            row[f"{profile_name}_s3"] = data["s3"]
            row[f"{profile_name}_tier"] = data["tier"]
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(matrix, stability, margins):
    """Build full sensitivity analysis report."""
    lines = []
    W = 76

    lines.append("=" * W)
    lines.append("  S3 MULTI-PROFILE SENSITIVITY ANALYSIS")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("  Method: Triantaphyllou [35] — single-weight perturbation")
    lines.append("=" * W)

    # ── Weight profiles ───────────────────────────────────────
    lines.append("")
    lines.append("  WEIGHT PROFILES")
    lines.append("  " + "-" * 70)
    lines.append(f"  {'Profile':<18} {'TC':>4} {'OS':>4} {'SK':>4} {'DS':>4} {'LT':>4} {'VL':>4}  {'Rationale'}")
    lines.append("  " + "-" * 70)
    rationales = {
        "Default":        "Paper baseline (default profile)",
        "Security-First": "Regulated industries prioritize SK, DS",
        "Latency-First":  "Real-time apps prioritize LT, VL",
        "Balanced":       "Equal weights — no dimension priority",
        "Volume-Heavy":   "High-throughput batch processing",
    }
    for name, w in WEIGHT_PROFILES.items():
        lines.append(
            f"  {name:<18} {w['TC']:>4} {w['OS']:>4} {w['SK']:>4} "
            f"{w['DS']:>4} {w['LT']:>4} {w['VL']:>4}  {rationales[name]}"
        )

    # ── Sensitivity Matrix ────────────────────────────────────
    lines.append("")
    lines.append("  TABLE 1 — S3 Scores Under Each Weight Profile")
    lines.append("  " + "-" * 72)
    header = f"  {'Use Case':<22}"
    for p in WEIGHT_PROFILES:
        header += f" {p[:8]:>10}"
    header += "  Stable?"
    lines.append(header)
    lines.append("  " + "-" * 72)

    stable_count = 0
    for uc_name in USE_CASES:
        row_str = f"  {uc_name:<22}"
        for p in WEIGHT_PROFILES:
            s3 = matrix[uc_name][p]["s3"]
            row_str += f" {s3:>10.2f}"
        is_stable = stability[uc_name]["stable"]
        flag = "YES" if is_stable else "NO"
        if is_stable:
            stable_count += 1
        row_str += f"  {flag}"
        lines.append(row_str)
    lines.append("  " + "-" * 72)
    lines.append(f"  Stable assignments: {stable_count}/{len(USE_CASES)}")

    # ── Tier Assignments ──────────────────────────────────────
    lines.append("")
    lines.append("  TABLE 2 — Tier Assignments Under Each Weight Profile")
    lines.append("  " + "-" * 72)
    header2 = f"  {'Use Case':<22}"
    for p in WEIGHT_PROFILES:
        header2 += f" {p[:8]:>10}"
    lines.append(header2)
    lines.append("  " + "-" * 72)
    for uc_name in USE_CASES:
        row_str = f"  {uc_name:<22}"
        for p in WEIGHT_PROFILES:
            tier = matrix[uc_name][p]["tier"]
            short = {"Pure SLM": "Pure", "Hybrid": "Hybrid", "LLM Only": "LLM"}
            row_str += f" {short.get(tier, tier):>10}"
        lines.append(row_str)

    # ── Triantaphyllou Margins ────────────────────────────────
    lines.append("")
    lines.append("  TABLE 3 — Triantaphyllou Stability Margins (Default Profile)")
    lines.append("  Minimum single-weight change to flip tier assignment")
    lines.append("  " + "-" * 72)
    lines.append(f"  {'Use Case':<22} {'S3':>5} {'Tier':<10} {'Min Change':>11} {'Via':>12} {'Flips To':<10}")
    lines.append("  " + "-" * 72)

    for uc_name in USE_CASES:
        m = margins[uc_name]
        if m["min_change_pct"] == float("inf"):
            if is_hard_rule_locked(USE_CASES[uc_name]):
                change_str, via_str = "LOCKED", "hard rule"
            else:
                change_str, via_str = ">500%", "none found"
            flip_str = "N/A"
        else:
            change_str = f"{m['min_change_pct']}%"
            via_str = m["flip_dim"] or "N/A"
            flip_str = m["flip_tier"] or "N/A"
        lines.append(
            f"  {uc_name:<22} {m['s3']:>4.2f} {m['tier']:<10} "
            f"{change_str:>11} {via_str:>12} {flip_str:<10}"
        )
    lines.append("  " + "-" * 72)

    # Interpret margins
    lines.append("")
    lines.append("  MARGIN INTERPRETATION:")
    for uc_name in USE_CASES:
        m = margins[uc_name]
        if m["min_change_pct"] == float("inf"):
            if is_hard_rule_locked(USE_CASES[uc_name]):
                lines.append(f"    {uc_name}: LOCKED by hard rule — weight changes cannot flip tier")
            else:
                lines.append(f"    {uc_name}: ROBUST — no single-weight change up to 500% flips the tier")
        elif m["min_change_pct"] >= 100:
            lines.append(f"    {uc_name}: ROBUST — requires >{m['min_change_pct']}% weight change to flip")
        elif m["min_change_pct"] >= 30:
            lines.append(f"    {uc_name}: STABLE — requires {m['min_change_pct']}% weight change to flip")
        else:
            lines.append(f"    {uc_name}: SENSITIVE — only {m['min_change_pct']}% weight change flips tier")

    # ── Boundary Cases ────────────────────────────────────────
    lines.append("")
    lines.append("  BOUNDARY CASE ANALYSIS")
    lines.append("  " + "-" * 60)
    lines.append("  Use cases near tier boundaries (within 0.3 of tau_1=3.0 or tau_2=3.7):")
    for uc_name in USE_CASES:
        m = margins[uc_name]
        s3 = m["s3"]
        dist_tau1 = abs(s3 - TAU_1)
        dist_tau2 = abs(s3 - TAU_2)
        min_dist = min(dist_tau1, dist_tau2)
        if min_dist < 0.3:
            nearest = "tau_1 (3.0)" if dist_tau1 < dist_tau2 else "tau_2 (3.7)"
            lines.append(f"    {uc_name}: S3={s3:.2f}, distance to {nearest} = {min_dist:.2f}")

    # ── Summary ───────────────────────────────────────────────
    lines.append("")
    lines.append("  SUMMARY")
    lines.append("  " + "-" * 60)
    locked = sum(1 for uc, m in margins.items() if m["min_change_pct"] == float("inf") and is_hard_rule_locked(USE_CASES[uc]))
    robust = sum(1 for uc, m in margins.items() if m["min_change_pct"] >= 100 and not (m["min_change_pct"] == float("inf") and is_hard_rule_locked(USE_CASES[uc])))
    stable = sum(1 for m in margins.values() if 30 <= m["min_change_pct"] < 100)
    sensitive = sum(1 for m in margins.values() if m["min_change_pct"] < 30)

    lines.append(f"  Hard-rule locked (cannot flip):  {locked}/8")
    lines.append(f"  Robust (>100% change needed):    {robust}/8")
    lines.append(f"  Stable (30-100% change needed):  {stable}/8")
    lines.append(f"  Sensitive (<30% change needed):   {sensitive}/8")
    lines.append(f"  Profile-stable (same tier all 5): {stable_count}/8")
    lines.append("")

    if sensitive == 0:
        lines.append("  CONCLUSION: All tier assignments are robust or locked.")
        lines.append("  The S3 scoring framework produces stable deployment recommendations")
        lines.append("  even under substantial weight variations. This supports the argument")
        lines.append("  that the geometric thresholds (3.0, 3.7) are adequate for N<50.")
    else:
        sensitive_ucs = [uc for uc, m in margins.items() if m["min_change_pct"] < 30]
        lines.append(f"  CAUTION: {sensitive} UC(s) are sensitive to weight changes:")
        for uc in sensitive_ucs:
            lines.append(f"    - {uc}: requires only {margins[uc]['min_change_pct']}% change")
        lines.append("  These cases would benefit from UTADIS calibration at N>=50.")

    lines.append("")
    lines.append("=" * W)
    lines.append("  END OF SENSITIVITY ANALYSIS")
    lines.append("=" * W)

    return "\n".join(lines)


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  S3 Multi-Profile Sensitivity Analysis")
    print("  5 weight profiles × 8 use cases = 40 S3 computations")
    print("=" * 60)

    matrix, stability, margins = run_sensitivity()

    save_csv(matrix)
    print(f"\n  Saved: {CSV_FILE}")

    report = build_report(matrix, stability, margins)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved: {REPORT_FILE}")

    print()
    print(report)

    print()
    print("  FILES SAVED:")
    print(f"    {CSV_FILE}")
    print(f"    {REPORT_FILE}")
    print("=" * 60)
    print()
