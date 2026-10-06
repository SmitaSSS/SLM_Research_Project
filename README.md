# Suitability Framework: Decision Framework for Small Language Model Deployment

**Suitability Score for SLM Selection (S3) and SDDF Runtime Routing**

A complete two-stage deployment decision architecture for enterprise Small Language Model (SLM) systems. This project benchmarks SLMs (3B-8B parameters) against LLMs (70B) across eight enterprise use cases to validate whether task type — not model scale — governs SLM deployment suitability.

> **Paper**: *Decision Framework and Industrial Deployment Protocol for Small Language Models in Agentic AI Systems*
> **Authors**: Smitha Rajappa, Riddhima Ramasahayam Reddy, Rohit Savant, Yashraj Saxena, Smita Sengupta 
> **Professor**: Dr. Ashim Bose
> **Institution**: University of Texas at Dallas

---

## Table of Contents

- [Overview](#overview)
- [The S3 Formula](#the-s3-formula)
  - [Dynamic Denominator WSM](#dynamic-denominator-wsm)
  - [Scoring Dimensions](#scoring-dimensions)
  - [Weight Assignment](#weight-assignment)
  - [Tier Boundaries](#tier-boundaries)
  - [Pre-Screening Gate](#pre-screening-gate)
  - [Worked Example](#worked-example)
- [Use Cases](#use-cases)
- [Project Structure](#project-structure)
- [Hardware Requirements](#hardware-requirements)
- [Installation Guide](#installation-guide)
  - [1. System Prerequisites](#1-system-prerequisites)
  - [2. Install Ollama](#2-install-ollama)
  - [3. Download Models](#3-download-models)
  - [4. Python Environment](#4-python-environment)
  - [5. API Keys](#5-api-keys)
- [Running Benchmarks](#running-benchmarks)
  - [Step 1: Verify Infrastructure](#step-1-verify-infrastructure)
  - [Step 2: Build Gold Sets](#step-2-build-gold-sets)
  - [Step 3: Run Benchmarks](#step-3-run-benchmarks)
  - [Step 4: Evaluate Results](#step-4-evaluate-results)
- [Model Registry](#model-registry)
- [Inference Configuration](#inference-configuration)
- [Key Results](#key-results)
- [Known Limitations](#known-limitations)
- [Citation](#citation)
- [License](#license)

---

## Overview

Enterprise adoption of LLMs faces a fundamental scaling constraint: API cost differentials across model tiers reach two orders of magnitude. This project asks: **for which tasks can a 3-8B parameter SLM replace a 70B LLM without meaningful quality loss?**

The S3 framework provides a **pre-deployment scoring instrument** that predicts the answer before running any inference. The SDDF framework provides **runtime query-level routing** that operationalises the prediction in production.

**Core finding**: Task type accounts for the majority of SLM performance variance. The same 3B model moves 36.6 percentage points between two use cases; the 70B LLM moves only 6.7pp across the same two tasks. S3 predicts this variance before any model is run.

---

## The S3 Formula

### Dynamic Denominator WSM

S3 uses a Weighted Sum Model (WSM) with a dynamic denominator:

```
S3 = [ Sum(Score_i * w_i) ] / [ Sum(5 * w_i) ] * 5
```

Where:
- `Score_i` is the polarity-adjusted score (1-5) for dimension `i`: the raw score for TC, SK, DS and VL, and `6 - raw` for OS and LT (see Scoring Dimensions)
- `w_i` is the organisational weight (integer 1-5) for dimension `i`
- `Sum(5 * w_i)` is the dynamic denominator (the exact maximum possible weighted sum)

**Key property**: The dynamic denominator guarantees `S3 in [1, 5]` under **any** valid weight configuration:
- When all scores = 1: S3 = 1 exactly
- When all scores = 5: S3 = 5 exactly
- The output range is identical regardless of the specific weight values chosen

This enables **threshold portability** across organisations with different risk profiles.

### Scoring Dimensions

Each task is scored on six dimensions. Score 1 = SLM optimal; Score 5 = LLM likely required, with one exception: **Output Structure (OS) and Latency Tolerance (LT) are reverse-scored.** Their anchors are written so that a *high* raw score (rigid output format, strict response-time requirement) is more easily satisfied by a small model, so the formula uses `6 - score` for these two dimensions. Raw 1-5 scores are what you record; the inversion is applied inside the scoring code.

| Dimension | Score 1 | Score 2 | Score 3 | Score 4 | Score 5 | What It Measures |
|-----------|---------|---------|---------|---------|---------|------------------|
| **Task Complexity (TC)** | Binary classification | Multi-class classification | Multi-step extraction | Reasoning with context | Expert-level judgment under ambiguity | Cognitive demand on the model at inference time |
| **Output Structure (OS)** | Free-form text | Loose paragraph | One-of-N labels | Strict JSON schema | Exact single value/code token | Degree of output format constraint |
| **Stakes (SK)** | No consequence | Minor inconvenience | Business impact, rework required | Significant harm, affects persons | Severe, irreversible, legally consequential | Maximum consequence of incorrect output |
| **Data Sensitivity (DS)** | Fully public | Semi-public/internal | Personal data, not regulated | Regulated PII/PHI | Classified/ITAR-controlled | Governs deployment location |
| **Latency Requirement (LT)** | Batch (hours) | Background (minutes) | Interactive (P95 < 2s) | Near real-time (P95 < 500ms) | Real-time (P95 < 100ms) | P95 SLA compliance |
| **Volume (VL)** | Dozens/day | Hundreds/day | Thousands/day | Tens of thousands/day | Millions/day | Throughput scale |

**Scoring rule**: Score dimensions for the specific inference the model performs in production. Score the hardest representative item type appearing in more than 5% of production volume.

### Weight Assignment

Weights are integers 1-5, developed through structured expert elicitation informed by Analytic Hierarchy Process principles (Saaty, 1980), without a formal pairwise comparison matrix or consistency ratio. Domain experts compared the relative importance of each dimension pair and the judgments were translated into the 1-5 integer scale. Edwards and Barron (1994) is cited in the paper for the robustness of simple rating approaches to weight imprecision, not as the elicitation method. Each weight reflects how much that dimension influences deployment tier decisions in your organisation.

| Weight | Label | Meaning |
|--------|-------|---------|
| 1 | Negligible | Minimal influence on tier decision |
| 2 | Low | Considered but rarely drives escalation |
| 3 | Moderate | Important, regularly influences the score |
| 4 | High | Frequently determines the tier outcome |
| 5 | Dominant | Almost always determines the deployment tier |

**Hard constraint**: `w_SK >= w_TC` must hold for any valid weight profile. The swing from SK=1 (no consequence) to SK=5 (irreversible harm) is at least as consequential as TC=1 to TC=5 for any rational enterprise decision-maker.

**Default weight profile used in this study**:

| Dimension | TC | OS | SK | DS | LT | VL |
|-----------|----|----|----|----|----|----|
| Weight | 3 | 2 | 4 | 2 | 3 | 1 |

Dynamic denominator = 5 x (3 + 2 + 4 + 2 + 3 + 1) = **75**

### Tier Boundaries

S3 maps to three deployment tiers:

```
S3 <= 3.0          -->  Pure SLM     (SLM handles task autonomously)
3.0 < S3 <= 3.7    -->  Hybrid       (SLM primary + LLM fallback)
S3 > 3.7           -->  LLM Only     (LLM required for quality/safety)
```

The thresholds (tau_1 = 3.0, tau_2 = 3.7) were calibrated on an independent calibration set of eight use cases (`data/gold_sets/`, new use cases; Table 12 of the paper) and then applied to the eight original use cases as held-out validation. They are **provisional** — full empirical calibration via UTADIS linear programming is planned at N >= 50 confirmed deployment outcomes.

**Decision sequence.** Gate rules are evaluated first. Hard Rule 1 and Hard Rule 2 assign LLM Only directly. The Flag Rule (SK >= 4) sets a minimum tier of Hybrid but does not lower a higher tier. If no hard rule applies, the S3 score is mapped to a tier with the boundaries above (a score exactly equal to a threshold falls in the lower tier).

### Pre-Screening Gate

Before computing S3, a non-compensatory gate is applied. This prevents the weighted average from masking disqualifying conditions:

| Stakes (SK) | Task Complexity (TC) | Gate Decision |
|-------------|---------------------|---------------|
| 5 | Any | **DISQUALIFY** — Hard Rule 1. No autonomous deployment. |
| >= 4 | 5 | **DISQUALIFY** — Hard Rule 2. Expert reasoning under severe consequences. |
| >= 4 | <= 4 | **PASS WITH FLAG** — Minimum tier = Hybrid regardless of S3 score. |
| <= 3 | Any | **PASS** — Proceed to S3 weighted scoring. |

**Why the gate exists**: WSM is a fully compensatory model — a high latency score could mathematically offset a catastrophic stakes score. The gate makes S3 non-compensatory where it matters most (safety) and compensatory everywhere else.

### Worked Example

**UC4 — Product Review Sentiment** vs **UC1 — SMS Threat Detection**

Both use the same default weight profile. Scores below are polarity-adjusted (OS and LT already converted with `6 - raw`; UC4 raw OS=3, LT=3; UC1 raw OS=3, LT=4):

| Dimension | UC4 Score | UC1 Score | Weight | UC4 Score x Weight | UC1 Score x Weight |
|-----------|-----------|-----------|--------|--------------------|--------------------|
| TC | 1 | 2 | 3 | 3 | 6 |
| OS (adj.) | 3 | 3 | 2 | 6 | 6 |
| SK | 2 | 4 | 4 | 8 | 16 |
| DS | 1 | 3 | 2 | 2 | 6 |
| LT (adj.) | 3 | 2 | 3 | 9 | 6 |
| VL | 3 | 5 | 1 | 3 | 5 |
| **Sum** | | | | **31** | **45** |

```
UC4: S3 = 31/75 x 5 = 2.07  -->  Pure SLM (benchmark: best SLM 97.7% of LLM accuracy)
UC1: S3 = 45/75 x 5 = 3.00  -->  Formula alone sits exactly at tau_1 (Pure SLM),
                                   but SK=4 triggers the Flag Rule --> Hybrid
                                   (benchmark: best SLM matches the LLM at 90.0%)
```

The 14-point numerator difference comes mostly from Stakes (+8). UC1's tier is set by the Flag Rule, which is why Stakes carries the highest weight and why the Flag Rule exists.

---

## Use Cases

Eight enterprise use cases spanning all three deployment tiers (S3 uses the default weight profile and polarity-adjusted scores; see Scoring Dimensions):

| UC | Domain | Task Type | S³ Score | Predicted Tier | Gate Rule | Status |
|----|--------|-----------|----------|----------------|-----------|--------|
| UC1 | SMS Threat Detection | Binary classification | 3.00 | Hybrid | Flag Rule (SK=4): formula alone gives Pure SLM | Confirmed |
| UC2 | Invoice Field Extraction | Structured JSON extraction | 2.67 | Pure SLM | None | Confirmed |
| UC3 | Support Ticket Routing | 6-way classification | 2.40 | Pure SLM | None | Confirmed |
| UC4 | Product Review Sentiment | 3-way classification | 2.07 | Pure SLM | None | Confirmed |
| UC5 | Automated Code Review | 5-way classification | 3.40 | Hybrid | None binding (SK=4 Flag floor already satisfied) | Confirmed |
| UC6 | Healthcare Clinical Triage | 4-way classification | 3.20 | LLM Only | SK=5 Hard Rule 1 | Confirmed |
| UC7 | Legal Contract Analysis | 4-way risk classification | 3.60 | Hybrid | None (SK=3) | Confirmed |
| UC8 | Financial Report Drafting | Free-form generation | 4.00 | LLM Only | TC=5, SK=4 Hard Rule 2 (formula also gives LLM Only) | Confirmed |

**Note on historical scores.** The `build_gold_set_uc[N].py` scripts, their metadata files and some evaluation reports record the S3 scores and thresholds as they stood when each gold set was built (March 2026, earlier weighting and scoring scheme). They are kept unchanged as a historical record. The scores in the table above are the final scores used in the paper, and `scripts/sensitivity_analysis.py` reproduces them.

**Datasets.** See [`DATASETS.md`](DATASETS.md) for the source and location of every dataset used in the S³ paper (the calibration-set sources and the team-authored validation sets). Benchmark outputs for the calibration set are in `data/raw_outputs/newuc*` and `data/results/newuc*`.

**Note on `data/results/newuc5_summary.csv`.** The file contains the LLM rows from earlier attempts as well as the final run. The baseline reported in the paper for UC5 is the final run (46.7%).

---

## Project Structure

```
SLM_Research_Project/
|
|-- configs/
|   |-- models.json               # Model registry (SLM + LLM tiers)
|   |-- inference_config.json      # Locked inference parameters
|
|-- scripts/
|   |-- verify_apis.py             # Health check for all models
|   |-- build_gold_set_uc[N].py    # Generate gold sets (100 items each)
|   |-- run_benchmark_uc[N].py     # Run inference benchmarks
|   |-- evaluate_uc[N].py          # Compute evaluation metrics
|   |-- benchmark_utils.py         # Shared utilities (memory monitoring, warm-up)
|   |-- s3_sddf_bridge.py          # S3-SDDF cross-framework convergence analysis
|   |-- sensitivity_analysis.py    # Multi-profile sensitivity analysis
|   |-- capture_hardware.py        # Hardware spec capture
|
|-- data/
|   |-- gold_sets/                 # Gold test sets (CSV + metadata JSON)
|   |   |-- uc[N]_*.csv            # 100-item gold sets
|   |   |-- uc[N]_metadata.json    # S3 scores, hypotheses, dimensions
|   |-- raw_outputs/               # Raw inference results per benchmark run
|   |-- results/                   # Aggregated per-model summary metrics
|   |-- hardware_spec.json         # Captured hardware configuration
|
|-- evaluation/                    # Evaluation reports (TXT + CSV)
|   |-- s3_sddf_bridge_*.csv/txt   # Bridge analysis results
|   |-- sensitivity_matrix_*.csv   # Sensitivity analysis results
|
|-- docs/
|   |-- s3_scoring_worksheet.md    # Canonical dimension scores with rationale
|
|-- README.md                      # This file
```

**Workflow for each use case**:
```
build_gold_set_ucN.py  -->  run_benchmark_ucN.py  -->  evaluate_ucN.py
   (creates gold set)      (runs 630 inferences)     (computes metrics + report)
```

---

## Hardware Requirements

### Minimum Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **RAM** | 16 GB | 32 GB | 16 GB causes memory pressure with 7-8B models |
| **CPU** | 8 cores | 10+ cores | Apple Silicon or modern x86_64 |
| **Disk** | 30 GB free | 50 GB free | For Ollama model files |
| **OS** | macOS 13+ / Ubuntu 22.04+ | Latest | Ollama requires modern OS |
| **GPU** | Not required | NVIDIA 8GB+ VRAM | Dramatically reduces latency |

### Hardware Used in This Study

```
Machine:    MacBook Pro (Model 14,9)
Chip:       Apple M2 Pro
Cores:      10 (6 Performance + 4 Efficiency)
RAM:        16 GB unified memory
GPU:        Integrated (Apple Neural Engine) — no discrete GPU
OS:         macOS (arm64)
Inference:  CPU-only via Ollama (no GPU acceleration)
```

### Memory Budget per Model

Ollama uses quantised models (typically Q4_K_M). Approximate memory requirements:

| Model | Parameters | Quantised Size | RAM Needed (weights + KV cache) |
|-------|-----------|---------------|--------------------------------|
| Llama-3.2-3B | 3B | ~2.0 GB | ~3-4 GB |
| Phi4-Mini | 3.8B | ~2.5 GB | ~4-5 GB |
| Gemma3-4B | 4B | ~2.8 GB | ~4-5 GB |
| Qwen2.5-7B | 7B | ~4.5 GB | ~6-8 GB |
| Mistral-7B | 7B | ~4.5 GB | ~6-8 GB |
| Llama-3.1-8B | 8B | ~5.0 GB | ~7-9 GB |

**Important**: On a 16 GB machine, 7-8B models leave minimal headroom for the OS and KV cache, causing memory pressure and swap. This affects **latency** (not accuracy). See [Known Limitations](#known-limitations).

### Latency Expectations

All latency numbers below are from CPU-only inference on M2 Pro 16GB. GPU deployments are expected to be 10-50x faster.

| Model | UC1 (128 tok) P50/P95 | UC8 (1024 tok) P50/P95 |
|-------|----------------------|----------------------|
| Llama-3.2-3B | 210 / 978ms | 9,055 / 11,802ms |
| Mistral-7B | 155 / 418ms | 17,045 / 20,654ms |
| Llama-3.1-8B | 331 / 2,413ms | 19,793 / 25,680ms |
| Llama-3.3-70B (Groq) | 319 / 2,428ms | 1,332 / 1,540ms |

---

## Installation Guide

### 1. System Prerequisites

```bash
# Verify system
uname -m          # Should show arm64 (Apple Silicon) or x86_64
python3 --version # Requires Python 3.8+
```

### 2. Install Ollama

Ollama runs LLMs locally. It manages model downloads, quantisation, and serving.

**macOS**:
```bash
# Download and install from https://ollama.com/download
# Or via Homebrew:
brew install ollama
```

**Linux**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Verify installation**:
```bash
ollama --version
# Expected: ollama version 0.6.x or later
```

**Start the Ollama server**:
```bash
# Start in background (required before running benchmarks)
ollama serve

# In a separate terminal, verify it's running:
curl http://localhost:11434
# Expected: "Ollama is running"
```

### 3. Download Models

Download all six SLM models and verify each one responds correctly:

```bash
# Small models (3-4B) — fast download, fits easily in 16GB RAM
ollama pull llama3.2:3b
ollama pull phi4-mini:latest
ollama pull gemma3:4b

# Medium models (7-8B) — larger download, needs 8GB+ free RAM
ollama pull qwen2.5:7b
ollama pull mistral:latest
ollama pull llama3.1:8b
```

**Verify each model works**:
```bash
# Quick test — should return a classification
ollama run llama3.2:3b "Classify as THREAT or BENIGN: You won a free iPhone! Click here."

# Check model sizes
ollama list
```

**Expected output from `ollama list`**:
```
NAME                 SIZE
llama3.2:3b          2.0 GB
phi4-mini:latest     2.5 GB
gemma3:4b            3.3 GB
qwen2.5:7b           4.7 GB
mistral:latest       4.1 GB
llama3.1:8b          4.9 GB
```

**Total disk space**: ~21.5 GB for all six models.

### 4. Python Environment

```bash
# Clone the repository
git clone https://github.com/<your-org>/SLM_Research_Project.git
cd SLM_Research_Project

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install openai httpx python-dotenv psutil
```

**Required packages**:

| Package | Version | Purpose |
|---------|---------|---------|
| `openai` | >= 1.0 | OpenAI-compatible API client (works with Ollama and Groq) |
| `httpx` | >= 0.24 | HTTP client with timeout support |
| `python-dotenv` | >= 1.0 | Load API keys from .env file |
| `psutil` | >= 5.9 | Memory monitoring (optional but recommended) |

### 5. API Keys

The LLM baseline (Llama-3.3-70B) runs on Groq's cloud API. You need a free API key:

1. Go to [console.groq.com](https://console.groq.com)
2. Create a free account
3. Generate an API key
4. Create a `.env` file in the project root:

```bash
echo "GROQ_API_KEY=gsk_your_key_here" > .env
```

**Free tier limits**: 14,400 requests/day — sufficient for all benchmarks.

**Running without Groq**: If you only want to benchmark local SLMs (without the LLM baseline), the scripts will skip the Groq model and log errors for it. All local SLM results remain valid.

---

## Running Benchmarks

### Step 1: Verify Infrastructure

```bash
# Ensure Ollama is running
ollama serve &

# Run the verification script
python3 scripts/verify_apis.py
```

**Expected output**:
```
=== LOCAL MODELS (Ollama) ===
  Llama-3.2-3B    ✓  (latency: 210ms)
  Phi4-Mini       ✓  (latency: 222ms)
  Gemma3-4B       ✓  (latency: 311ms)
  Qwen2.5-7B      ✓  (latency: 221ms)
  Mistral-7B      ✓  (latency: 155ms)
  Llama-3.1-8B    ✓  (latency: 331ms)

=== CLOUD MODELS (Groq) ===
  Llama-3.3-70B   ✓  (latency: 319ms)
```

**Troubleshooting**:
- `Connection refused`: Ollama server not running. Start with `ollama serve`.
- `Model not found`: Model not downloaded. Run `ollama pull <model_name>`.
- `Groq 401`: Invalid API key. Check `.env` file.
- `Groq 429`: Rate limit hit. Wait 60 seconds and retry.

### Step 2: Build Gold Sets

Gold sets are fixed test datasets. They only need to be built once:

```bash
# Build all gold sets (creates CSV + metadata JSON files)
python3 scripts/build_gold_set_uc1.py
python3 scripts/build_gold_set_uc2.py
python3 scripts/build_gold_set_uc3.py
python3 scripts/build_gold_set_uc4.py
python3 scripts/build_gold_set_uc5.py
python3 scripts/build_gold_set_uc6.py
python3 scripts/build_gold_set_uc7.py
python3 scripts/build_gold_set_uc8.py
```

Each script creates:
- `data/gold_sets/uc[N]_*.csv` — 100-item gold set
- `data/gold_sets/uc[N]_metadata.json` — S3 scores, hypotheses, dimensions

### Step 3: Run Benchmarks

Each benchmark runs **7 models x 30 test items x 3 seeds = 630 inferences**:

```bash
# Classification tasks (Category 1, max_tokens=128) — ~15-30 min each
python3 scripts/run_benchmark_uc1.py    # SMS Threat Detection
python3 scripts/run_benchmark_uc3.py    # Support Ticket Routing
python3 scripts/run_benchmark_uc4.py    # Product Review Sentiment
python3 scripts/run_benchmark_uc5.py    # Code Review
python3 scripts/run_benchmark_uc6.py    # Healthcare Clinical Triage
python3 scripts/run_benchmark_uc7.py    # Legal Contract Analysis

# Structured extraction (Category 2, max_tokens=512) — ~30-60 min
python3 scripts/run_benchmark_uc2.py    # Invoice Field Extraction

# Long-form generation (Category 3, max_tokens=1024) — ~60-120 min
python3 scripts/run_benchmark_uc8.py    # Financial Report Drafting
```

**Important notes**:
- Run benchmarks **one at a time** to avoid memory contention between models.
- Close other memory-intensive applications before running.
- On 16GB RAM, 7-8B models will cause memory pressure. This affects latency, not accuracy.
- Results are saved incrementally after each model completes (crash-safe).
- The `--resume` flag (UC6, UC8) allows continuing interrupted runs.

**Output files**:
- `data/raw_outputs/uc[N]_raw_[TIMESTAMP].csv` — every inference result
- `data/results/uc[N]_summary_[TIMESTAMP].csv` — per-model aggregated metrics

### Step 4: Evaluate Results

```bash
# Generate evaluation reports
python3 scripts/evaluate_uc1.py
python3 scripts/evaluate_uc2.py
python3 scripts/evaluate_uc3.py
python3 scripts/evaluate_uc4.py
python3 scripts/evaluate_uc5.py
python3 scripts/evaluate_uc6.py
python3 scripts/evaluate_uc7.py
python3 scripts/evaluate_uc8.py
```

Each evaluation script:
1. Loads the most recent raw results file
2. Computes per-model metrics (accuracy, F1, precision, recall, latency percentiles)
3. Computes per-difficulty and per-category breakdowns
4. Checks the hypotheses recorded in each use case's metadata
5. Generates evaluation report (TXT) and metrics (CSV)

**Output files**:
- `evaluation/uc[N]_report_[TIMESTAMP].txt` — human-readable report
- `evaluation/uc[N]_evaluation_[TIMESTAMP].csv` — machine-readable metrics

---

## Model Registry

### Local SLMs (Ollama)

| Model | Parameters | Architecture | Quantisation | Strengths |
|-------|-----------|--------------|-------------|-----------|
| Llama-3.2-3B | 3B | Decoder-only | Q4_K_M | Lowest parameter boundary; demonstrates task-type dominance over scale |
| Phi4-Mini | 3.8B | Decoder-only | Q4_K_M | Microsoft instruction-tuned; strong structured output |
| Gemma3-4B | 4B | Decoder-only | Q4_K_M | Google efficient architecture; consistent mid-tier |
| Qwen2.5-7B | 7B | Decoder-only | Q4_K_M | Alibaba multilingual; strong structured extraction |
| Mistral-7B | 7B | Decoder-only | Q4_K_M | Best SLM across multiple UCs; UC3 parity 104% |
| Llama-3.1-8B | 8B | Decoder-only | Q4_K_M | Largest local SLM; UC4 exact parity with LLM |

### Cloud LLM (Groq)

| Model | Parameters | Backend | Role |
|-------|-----------|---------|------|
| Llama-3.3-70B | 70B | Groq LPU Cloud | Performance ceiling for parity computation |

---

## Inference Configuration

All parameters are **locked** as of 2 March 2026:

```json
{
  "temperature": 0.0,
  "top_p": 1.0,
  "seeds": [42, 43, 44],
  "runs_per_item": 3,
  "max_tokens_by_category": {
    "C1": 128,
    "C2": 512,
    "C3": 1024
  }
}
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.0 | Deterministic output for reproducibility |
| Seeds | [42, 43, 44] | Three runs per item for variance estimation |
| Runs per item | 3 | Statistical reliability without excessive cost |
| Test items | 30 per UC | From 100-item gold set (70 train / 30 test) |
| Total inferences | 630 per UC | 7 models x 30 items x 3 runs |

**Acceptance criteria** (defined in the use-case metadata):
- Accuracy >= 95% of LLM baseline on valid outputs
- P95 latency <= task-specific SLA
- Valid output rate >= 95%

---

## Key Results

### S³ Tier Prediction: 100% Accuracy on All 8 Confirmed Cases

| UC | S³ Score | Predicted | Gate Rule | Best SLM | LLM Baseline | Parity | Confirmed? |
|----|----------|-----------|-----------|----------|-------------|--------|-----------|
| UC4 | 2.07 | Pure SLM | None | Mistral-7B: 94.4% | 96.7% | 97.7% | Yes |
| UC3 | 2.40 | Pure SLM | None | Multiple: 86.7% | 83.3% | 104.1% | Yes |
| UC2 | 2.67 | Pure SLM | None | Phi4-Mini: 92.2% | 91.1% | 101.2% | Yes |
| UC1 | 3.00 | Hybrid | Flag (SK=4) | Mistral-7B: 90.0% | 90.0% | 100% | Yes |
| UC6 | 3.20 | LLM Only | Hard Rule 1 | Llama-3.1-8B: 73.3% | 68.9% | 106.4%* | Yes |
| UC5 | 3.40 | Hybrid | Flag floor not binding | Llama-3.1-8B: 46.7% | 61.1% | 76.4% | Yes |
| UC7 | 3.60 | Hybrid | None | Qwen2.5-7B: 53.3% | 57.8% | 92.3% | Yes |
| UC8 | 4.00 | LLM Only | Hard Rule 2 | Llama-3.2-3B: 72.3% | 80.6% | 89.7% | Yes |

*UC6 parity of 106.4% masks URGENT class recall failure (SLM 0-64% vs LLM 82%)

### Gate Rules Are Load-Bearing

Three use cases demonstrate why gate rules are essential beyond the formula:

- **UC6** (S³=3.20): The formula alone gives Hybrid. Hard Rule 1 (SK=5) escalates to LLM Only. Overall accuracy masks dangerous URGENT under-triage by SLMs.
- **UC1** (S³=3.00): The formula alone sits exactly at tau_1 (Pure SLM). The Flag Rule (SK=4) sets the minimum tier at Hybrid. Benchmark: best SLM matches the LLM (100% parity).
- **UC8** (S³=4.00): Hard Rule 2 (TC=5, SK=4) and the formula both give LLM Only, so the gate reinforces an assignment the formula also reaches. Benchmark: 89.7% parity.

### S³-SDDF Bridge (exploratory)

`scripts/s3_sddf_bridge.py` maps S³ dimension scores onto SDDF task families. It is an exploratory script and is not part of the results reported in the S³ paper.

### Sensitivity Analysis

Five weight profiles (Default, Security-First, Latency-First, Balanced, Volume-Heavy) tested across all 8 UCs:

- **7/8 profile-stable** — same tier under all 5 profiles
- **2/8 hard-rule locked** (UC6, UC8) — cannot flip regardless of weight changes
- **UC2 is the only profile-sensitive case** — it moves from Pure SLM to Hybrid under the Latency-First profile (a conservative shift)
- Profiles are those in Table 2 of the paper; run `python scripts/sensitivity_analysis.py` to reproduce

### Cross-Task Finding

**Task type dominates model scale**: Llama-3.2-3B achieves 93.3% on UC4 (S³=2.07) and 56.7% on UC1 (S³=3.00) — a 36.6pp swing from the same model, same hardware, same prompt structure. The 70B LLM moves only 6.7pp across the same two tasks. The S³ scores are assigned from task properties, not from model outputs.

---

## Known Limitations

### Memory Pressure on 16GB Systems

With 16GB RAM and CPU-only inference, 7-8B models cause memory pressure (swap). Evidence from benchmark data:

- **Periodic latency spikes**: Every 3rd inference shows 3-5x latency increase due to memory decompression after inter-item sleep
- **Catastrophic swap events**: Occasional 10-50x latency spikes (e.g., 13,591ms vs typical 331ms for Llama-3.1-8B)
- **UC8 latency explosion**: Local models show 40-110x slowdown on 1024-token generation vs 128-token classification, far exceeding the expected 8x from token count alone

**Impact**: Latency metrics (P50, P95) are contaminated by swap. Accuracy metrics are unaffected.

**Mitigation**: For reliable latency measurement, use 32GB+ RAM or GPU-accelerated inference. All SDDF latency metrics require GPU recalibration before use in production SLA decisions.

### Groq API Reliability

The Groq cloud API (Llama-3.3-70B baseline) experienced cascading failures during UC8 benchmarking:
- Rate limit errors (HTTP 429) after sustained high-volume requests
- DNS resolution failures following rate limit exhaustion
- ~18 out of 630 UC8 inferences failed (97.1% success rate)

UC1-UC6 benchmarks completed without Groq failures.

### Threshold Calibration

Tier boundaries (3.0 and 3.7) were calibrated on an independent eight-use-case calibration set and validated on the eight original use cases. Sensitivity analysis across 5 weight profiles shows 7/8 UCs are profile-stable. Full empirical calibration via UTADIS linear programming requires N >= 50 confirmed deployment outcomes. See paper Section 9 for details.

### Dimensional Independence

The WSM additive formula assumes preferential independence between dimensions. Correlations likely exist in practice (e.g., high-Stakes tasks tend to involve regulated data). This is documented as a known limitation with conservative impact (double-counting risk factors errs toward escalation, which is the safe direction).

---

## Citation

```bibtex
@article{rajappa2026s3framework,
  title     = {Decision Framework and Industrial Deployment Protocol
               for Small Language Models in Agentic AI Systems},
  author    = {Rajappa, Smitha and Ramasahayam Reddy, Riddhima and
               Savant, Rohit and Saxena, Yashraj and Sengupta, Smita},
  year      = {2026},
  institution = {University of Texas at Dallas},
  keywords  = {small language models, enterprise AI deployment,
               model selection framework, dynamic denominator WSM,
               hybrid routing, SDDF, threshold calibration, agentic AI}
}
```

---

## License

This project is part of academic research at the University of Texas at Dallas under the guidance of Professor Ashim Bose. Please cite the paper if you use this framework or data in your work.
