# S³ Scoring Worksheet — All 8 Use Cases (final scores used in the paper)

Raw 1-5 dimension scores for each use case, the polarity adjustment, and the resulting S³ score and tier. These reproduce Table 6 of the paper.

**Formula**: S³ = Σ(Adjusted scoreᵢ × wᵢ) / Σ(5 × wᵢ) × 5
**Polarity**: Output Structure (OS) and Latency Tolerance (LT) are reverse-scored: adjusted score = 6 − raw score. The other four dimensions use the raw score. The tables below show both.
**Default weights**: TC=3, OS=2, SK=4, DS=2, LT=3, VL=1, so Σ(5×wᵢ) = 75
**Tier boundaries**: τ₁=3.0, τ₂=3.7. Pure SLM: S³ ≤ 3.0; Hybrid: 3.0 < S³ ≤ 3.7; LLM Only: S³ > 3.7
**Gate rules** (evaluated first): SK=5 → LLM Only (Hard Rule 1); TC=5 and SK≥4 → LLM Only (Hard Rule 2); SK≥4 → minimum tier Hybrid (Flag Rule)

## Scoring Scale Reference (paper Table 1)

| Dim | 1 | 2 | 3 | 4 | 5 |
|-----|---|---|---|---|---|
| **TC** | Binary classification | Few-class classification | Multi-class with edge cases | Multi-step reasoning | Expert judgment; compositional reasoning |
| **OS** | Free-form paragraph | Soft-structured paragraph | Named fields / semi-schema | Defined labels; closed vocabulary | Single word/value; strict schema |
| **SK** | No consequence; informational | Minor; easily corrected | Business impact; rework required | Significant harm; affects persons | Severe, irreversible, legally consequential |
| **DS** | Fully public data | Semi-public / internal | Personal data; not regulated | Regulated PII/PHI | Classified, ITAR-controlled |
| **LT** | Batch; hours acceptable | Background; minutes acceptable | Interactive; P95 < 2s | Near real-time; P95 < 500ms | Real-time; P95 < 100ms |
| **VL** | Dozens per day | Hundreds per day | Tens of thousands/day | Hundreds of thousands/day | Millions per day |

---

## UC1 — SMS Threat Detection
**Task**: Binary classification of SMS as THREAT or BENIGN
**Benchmark**: Mistral-7B 90.0%, LLM 90.0%, parity 100.0%

| Dim | Raw | Adj. | Weight | Weighted |
|-----|-----|------|--------|----------|
| TC | 2 | 2 | 3 | 6 |
| OS | 3 | 3 | 2 | 6 |
| SK | 4 | 4 | 4 | 16 |
| DS | 3 | 3 | 2 | 6 |
| LT | 4 | 2 | 3 | 6 |
| VL | 5 | 5 | 1 | 5 |

**Weighted sum**: 2×3 + 3×2 + 4×4 + 3×2 + 2×3 + 5×1 = **45**
**S³**: 45/75 × 5 = **3.00**
**Gate**: Flag Rule (SK>=4); formula alone gives Pure SLM
**Tier**: **Hybrid**

---

## UC2 — Invoice Field Extraction
**Task**: Extract structured JSON fields from invoice text
**Benchmark**: Phi-4-Mini 92.2%, LLM 91.1%, parity 101.2%

| Dim | Raw | Adj. | Weight | Weighted |
|-----|-----|------|--------|----------|
| TC | 2 | 2 | 3 | 6 |
| OS | 4 | 2 | 2 | 4 |
| SK | 2 | 2 | 4 | 8 |
| DS | 3 | 3 | 2 | 6 |
| LT | 2 | 4 | 3 | 12 |
| VL | 4 | 4 | 1 | 4 |

**Weighted sum**: 2×3 + 2×2 + 2×4 + 3×2 + 4×3 + 4×1 = **40**
**S³**: 40/75 × 5 = **2.67**
**Gate**: No gate; formula only
**Tier**: **Pure SLM**

---

## UC3 — Support Ticket Routing
**Task**: 6-class classification of support tickets (BILLING, TECHNICAL, ACCOUNT, SHIPPING, RETURNS, GENERAL)
**Benchmark**: 4 SLMs tied at 86.7%, LLM 83.3%, parity 104.1%

| Dim | Raw | Adj. | Weight | Weighted |
|-----|-----|------|--------|----------|
| TC | 1 | 1 | 3 | 3 |
| OS | 4 | 2 | 2 | 4 |
| SK | 3 | 3 | 4 | 12 |
| DS | 2 | 2 | 2 | 4 |
| LT | 3 | 3 | 3 | 9 |
| VL | 4 | 4 | 1 | 4 |

**Weighted sum**: 1×3 + 2×2 + 3×4 + 2×2 + 3×3 + 4×1 = **36**
**S³**: 36/75 × 5 = **2.40**
**Gate**: No gate; formula only
**Tier**: **Pure SLM**

---

## UC4 — Product Review Sentiment
**Task**: 3-class sentiment classification (POSITIVE, NEGATIVE, NEUTRAL)
**Benchmark**: Mistral-7B 94.4%, LLM 96.7%, parity 97.7%

| Dim | Raw | Adj. | Weight | Weighted |
|-----|-----|------|--------|----------|
| TC | 1 | 1 | 3 | 3 |
| OS | 3 | 3 | 2 | 6 |
| SK | 2 | 2 | 4 | 8 |
| DS | 1 | 1 | 2 | 2 |
| LT | 3 | 3 | 3 | 9 |
| VL | 3 | 3 | 1 | 3 |

**Weighted sum**: 1×3 + 3×2 + 2×4 + 1×2 + 3×3 + 3×1 = **31**
**S³**: 31/75 × 5 = **2.07**
**Gate**: No gate; formula only
**Tier**: **Pure SLM**

---

## UC5 — Automated Code Review
**Task**: 5-class code quality classification (SECURITY, LOGIC_ERROR, PERFORMANCE, BEST_PRACTICE, CLEAN)
**Benchmark**: Llama-3.1-8B 46.7%, LLM 61.1%, parity 76.4%

| Dim | Raw | Adj. | Weight | Weighted |
|-----|-----|------|--------|----------|
| TC | 4 | 4 | 3 | 12 |
| OS | 4 | 2 | 2 | 4 |
| SK | 4 | 4 | 4 | 16 |
| DS | 2 | 2 | 2 | 4 |
| LT | 2 | 4 | 3 | 12 |
| VL | 3 | 3 | 1 | 3 |

**Weighted sum**: 4×3 + 2×2 + 4×4 + 2×2 + 4×3 + 3×1 = **51**
**S³**: 51/75 × 5 = **3.40**
**Gate**: Flag Rule (SK>=4) not binding; formula already gives Hybrid
**Tier**: **Hybrid**

---

## UC6 — Healthcare Clinical Triage
**Task**: 4-class triage priority (CRITICAL, URGENT, SEMIURGENT, NONURGENT)
**Benchmark**: Llama-3.1-8B 73.3%, LLM 68.9%, parity 106.4% (aggregate masks URGENT recall: SLM 0-64% vs LLM 82%)

| Dim | Raw | Adj. | Weight | Weighted |
|-----|-----|------|--------|----------|
| TC | 4 | 4 | 3 | 12 |
| OS | 4 | 2 | 2 | 4 |
| SK | 5 | 5 | 4 | 20 |
| DS | 4 | 4 | 2 | 8 |
| LT | 5 | 1 | 3 | 3 |
| VL | 1 | 1 | 1 | 1 |

**Weighted sum**: 4×3 + 2×2 + 5×4 + 4×2 + 1×3 + 1×1 = **48**
**S³**: 48/75 × 5 = **3.20**
**Gate**: Hard Rule 1 (SK=5)
**Tier**: **LLM Only**

---

## UC7 — Legal Contract Risk Analysis
**Task**: 4-class contract clause risk (HIGH_RISK, MEDIUM_RISK, LOW_RISK, STANDARD)
**Benchmark**: Qwen2.5-7B 53.3%, LLM 57.8%, parity 92.3%

| Dim | Raw | Adj. | Weight | Weighted |
|-----|-----|------|--------|----------|
| TC | 4 | 4 | 3 | 12 |
| OS | 2 | 4 | 2 | 8 |
| SK | 3 | 3 | 4 | 12 |
| DS | 4 | 4 | 2 | 8 |
| LT | 2 | 4 | 3 | 12 |
| VL | 2 | 2 | 1 | 2 |

**Weighted sum**: 4×3 + 4×2 + 3×4 + 4×2 + 4×3 + 2×1 = **54**
**S³**: 54/75 × 5 = **3.60**
**Gate**: No gate; formula only
**Tier**: **Hybrid**

---

## UC8 — Financial Report Drafting
**Task**: Generate a 2-3 paragraph quarterly earnings summary from financial data
**Benchmark**: Llama-3.2-3B 72.3%, LLM 80.6%, parity 89.7%

| Dim | Raw | Adj. | Weight | Weighted |
|-----|-----|------|--------|----------|
| TC | 5 | 5 | 3 | 15 |
| OS | 1 | 5 | 2 | 10 |
| SK | 4 | 4 | 4 | 16 |
| DS | 3 | 3 | 2 | 6 |
| LT | 2 | 4 | 3 | 12 |
| VL | 1 | 1 | 1 | 1 |

**Weighted sum**: 5×3 + 5×2 + 4×4 + 3×2 + 4×3 + 1×1 = **60**
**S³**: 60/75 × 5 = **4.00**
**Gate**: Hard Rule 2 (TC=5, SK>=4)
**Tier**: **LLM Only**

---
## Summary Table

| UC | Task | TC | OS | SK | DS | LT | VL | Weighted sum | S³ | Gate / basis | Tier |
|----|------|----|----|----|----|----|----|--------------|----|--------------|------|
| 1 | SMS Threat Detection | 2 | 3 | 4 | 3 | 4 | 5 | 45 | 3.00 | Flag Rule (SK>=4); formula alone gives Pure SLM | **Hybrid** |
| 2 | Invoice Field Extraction | 2 | 4 | 2 | 3 | 2 | 4 | 40 | 2.67 | No gate; formula only | **Pure SLM** |
| 3 | Support Ticket Routing | 1 | 4 | 3 | 2 | 3 | 4 | 36 | 2.40 | No gate; formula only | **Pure SLM** |
| 4 | Product Review Sentiment | 1 | 3 | 2 | 1 | 3 | 3 | 31 | 2.07 | No gate; formula only | **Pure SLM** |
| 5 | Automated Code Review | 4 | 4 | 4 | 2 | 2 | 3 | 51 | 3.40 | Flag Rule (SK>=4) not binding; formula already gives Hybrid | **Hybrid** |
| 6 | Healthcare Clinical Triage | 4 | 4 | 5 | 4 | 5 | 1 | 48 | 3.20 | Hard Rule 1 (SK=5) | **LLM Only** |
| 7 | Legal Contract Risk Analysis | 4 | 2 | 3 | 4 | 2 | 2 | 54 | 3.60 | No gate; formula only | **Hybrid** |
| 8 | Financial Report Drafting | 5 | 1 | 4 | 3 | 2 | 1 | 60 | 4.00 | Hard Rule 2 (TC=5, SK>=4) | **LLM Only** |

Raw scores are shown (before the OS/LT inversion).

### Benchmark validation
All eight predicted tiers match the benchmark outcomes (8/8). See the README benchmark table and paper Table 9.
