# An Agentic AI Framework for Autonomous Auditing and Error Detection in Phylogenetic Inference Models

An autonomous **neuro-symbolic Agentic AI auditing framework** designed to inspect, verify, and diagnose topological errors and evolutionary failure modes in deep-learning phylogenetic inference models (**Fusang** and **Phyloformer**) alongside classical maximum-likelihood baselines (**IQ-TREE 2**, **RAxML-NG**, and **FastTree 2**).

---

## 1. Why This Framework Exists

Modern deep-learning phylogenetic models predict evolutionary trees from Multiple Sequence Alignments (MSAs) orders of magnitude faster than classical algorithms:
* **Fusang** ($\le 40$ taxa) predicts discrete tree topologies directly from gapped alignments, but outputs **no biological branch lengths** and is vulnerable to indel-boundary artifacts, Long-Branch Attraction (LBA), and attention instability near $N = 40$ taxa.
* **Phyloformer** ($> 40$ taxa) predicts an $N \times N$ pairwise evolutionary distance matrix $\mathbf{D}$ from ungapped sites (converted into a tree via **BIONJ**), but ignores shared derived insertions/deletions (indel synapomorphies) and can fail under rapid-radiation short internal branches ($\ell_{\text{int}} < 0.005$), extreme rate heterogeneity ($\alpha = 0.2$), and localized recombination.

However, asking a standard Large Language Model (LLM) to explain *why* a deep-learning tree split is right or wrong causes severe **coordinate hallucinations** (inventing non-existent alignment columns or wrong ancestral mutations).

### How Neuro-Symbolic Agent Solves This
Instead of letting an LLM guess from raw FASTA text, this framework separates **symbolic mathematical computation** from **neural reasoning** via a **4-State LangGraph State Machine**:
1. **Deterministic Python Tools** compute the exact Robinson-Foulds distance, ancestral character states (`IQ-TREE 2 --ancestral`), Fitch binary gap parsimony states, Consistency/Retention Indices ($CI_i, RI_i$), LBA branch ratios ($\rho_{\text{LBA}}$), and local gap densities.
2. **The Local LLM Orchestrator** (e.g., `qwen/qwen3-14b` via LM Studio) synthesizes the structured tool evidence into a biological causal diagnosis.
3. **A Hard Symbolic Verification Gate (State 4)** cross-checks every cited column coordinate and mutation (`ancestral_state -> derived_state`) against the ground-truth alignment and ASR table *before* any report is emitted. If a hallucination is detected, the gate blocks the output and forces the agent to self-correct until **Claimed-Coordinate Verification Rate ($\text{CCVR}$) = $1.00$**.

---

## 2. Architecture & 4-Step Pipeline

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Ground-Truth & Evolutionary Feature Extraction                        │
│  • Tier 1 & 2 (Synthetic): IQ-TREE 2 (--alisim) + Zipfian Indels + Recomb     │
│  • Tier 3 (Empirical FASTA): 3-Engine Consensus (IQ-TREE 2 + RAxML + FastTree)│
│  • Outputs: T_true_rooted.nwk, M_true.tsv (ASR + Fitch Indel MP), A_true.json │
└──────────────────────────────────────┬────────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Target Model Inference, B=100 Bootstraps & Rules 1–3 Standardization │
│  • Generates B = 100 non-parametric bootstrap replicate MSAs                 │
│  • Runs Fusang (N ≤ 40) or Phyloformer (NxN matrix -> scikit-bio + BIONJ)    │
│  • Rule 1 (Outgroup Rooting), Rule 2 (Post-Order N1..Nk), Rule 3 (Support %) │
│  • Outputs: <model>_normalized.nwk, <model>_metadata.json                    │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: 4-State Neuro-Symbolic LangGraph Diagnostic Agent                    │
│  • State 1 (Topological Triage)     -> Tool 1: audit_tree_topology           │
│  • State 2 (Ancestral Attribution)  -> Tool 2: extract_branch_synapomorphies │
│  • State 3 (Conditional Stressor)   -> Tool 3: diagnose_evolutionary_stressor│
│                                     -> Tool 4: inspect_alignment_window      │
│  • State 4 (Verification Gate)      -> Tool 5: verify_claim_coordinates      │
└──────────────────────────────────────┬───────────────────────────────────────┘
                                       ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Multi-Dimensional Evaluation & Statistical Validation                │
│  • Node Metrics: CCVR, BSF, Feature F1, Bayesian IG, Faithfulness FGS, G_node│
│  • Tree Metrics: Logical Consistency Rate (LCR), Reconciled Goodness (G_tree)│
│  • Statistical Tests: Paired Wilcoxon Signed-Rank, McNemar's, BH-FDR (q<0.05)│
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```text
SP_Env/
├── README.md                              # Project documentation & usage guide
├── pytest.ini                             # Pytest configuration
├── envs/
│   └── env_bio_core.yml                   # Conda environment specification
├── src/
│   ├── preprocessing/
│   │   ├── ground_truth_gen.py            # Step 1: ASR, CI/RI, Fitch gap MP, Shannon MI
│   │   ├── normalizer_wrapper.py          # Step 2: B=100 bootstrap, BIONJ, Rules 1–3 standardization
│   │   ├── synthetic_generator.py         # Step A: Tier 1 & Tier 2 AliSim + Indel/LBA/Recomb generator
│   │   ├── empirical_consensus.py         # Step B: Tier 3 3-engine consensus Silver Standard builder
│   │   └── target_model_runner.py         # Step B: Fusang, Phyloformer & classical model runner
│   ├── tools/
│   │   └── bio_tools.py                   # Step 3: All 5 deterministic Python tools
│   ├── agent/
│   │   ├── diagnostic_agent.py            # Step 3: 4-State LangGraph state machine orchestrator
│   │   └── ai_baselines.py                # Step C: AI Baseline 1 (Zero-Shot) & Baseline 2 (Generic Agent)
│   ├── evaluation/
│   │   └── metrics.py                     # Step 4: CCVR, BSF, F1, Bayesian IG, FGS, G_node, LCR, G_tree, EDR
│   └── schemas/
│       └── audit_schema.py                # Strict Pydantic v2 schema for JSON audit reports
├── scripts/
│   ├── generate_synthetic_benchmarks.py   # CLI: Generate Tier 1 & Tier 2 synthetic stressor datasets
│   ├── sanitize_tier3_fasta.py            # CLI: [Tier 3 Compatibility Only] Clean raw NCBI/TreeBASE/OrthoMaM FASTAs
│   ├── prepare_empirical_tier3.py         # CLI: Build Tier 3 3-engine consensus from real FASTAs
│   ├── run_target_models.py               # CLI: Run Fusang, Phyloformer, IQ-TREE2, RAxML-NG, FastTree2
│   ├── run_live_agent_audit.py            # CLI: Run single live agent audit via LM Studio
│   └── run_ai_baselines.py                # CLI: Run 3-way comparison (Proposed vs. Baseline 1 vs. Baseline 2)
├── data/
│   ├── 00_golden_toy/                     # 15-taxon development & smoke-test dataset
│   ├── 01_raw_alignments/                 # Raw FASTA alignments (tier1/, tier2/, tier3/)
│   ├── 02_ground_truths/                  # Ground truths (T_true_rooted.nwk, M_true.tsv, A_true.json)
│   ├── 03_standardized_trees/             # Standardized model predictions (<model>_normalized.nwk & .json)
│   └── 04_audit_reports/                  # Output JSON audit reports & statistical tables
│       ├── proposed_agent/                # Proposed 4-State Agent reports
│       ├── baseline1_zeroshot_llm/        # AI Baseline 1 (Zero-Shot LLM) reports
│       ├── baseline2_generic_agent/       # AI Baseline 2 (Generic Agent) reports
│       └── statistical_comparisons/       # Paired Wilcoxon, McNemar & BH-FDR summaries + master TSV
└── tests/                                 # Automated unit & integration test suite (30 tests)
    ├── test_bionj.py                      # Validates custom BIONJ against FastME across 17 cases
    ├── test_tools.py                      # Validates Step 1 ground truth & all 5 deterministic tools
    ├── test_agent.py                      # Validates 4-State LangGraph agent, hallucination gate & metrics
    ├── test_synthetic_generator.py        # Validates Tier 1 & Tier 2 synthetic stressor generator
    ├── test_step_b.py                     # Validates Tier 3 3-engine consensus & target model runner
    └── test_ai_baselines.py               # Validates 3-way baseline comparison & separate JSON storage
```

---

## 4. Environment & LM Studio Setup

### 4.1 Conda Environment (`env_bio_core`)
All bioinformatics binaries (`iqtree2`, `raxml-ng`, `fasttree`, `fastme`) and Python dependencies (`ete3`, `dendropy`, `biopython`, `scikit-bio`, `langgraph`, `langchain-openai`, `scipy`, `statsmodels`, `pydantic`, `pytest`) are installed in `/home/neal/miniconda3/envs/env_bio_core`.

To activate the environment in your terminal:
```bash
conda activate env_bio_core
```
*(Or invoke Python/pytest directly via `/home/neal/miniconda3/envs/env_bio_core/bin/python`).*

### 4.2 Connecting a Local LLM via LM Studio
1. Open **LM Studio** on your host machine.
2. Download and load an instruction-tuned LLM (recommended: **`qwen/qwen3-14b` (`Q4_K_M`)** for 16 GB VRAM GPUs, or `qwen2.5-14b-instruct`).
3. Go to the **Developer / Local Server** tab in LM Studio and click **Start Server** on port `1234` (`http://localhost:1234/v1`).
4. *(Optional)* Enable **Just-in-Time (JIT) Model Loading** in LM Studio so the scripts can automatically switch between different models if you test multiple LLMs for the Zero-Shot baseline.

---

## 5. Quick Start: Running Tests & Smoke Audit

### 5.1 Run the Full Automated Test Suite (30 Tests)
Verify that all 4 pipeline steps, BIONJ, synthetic generators, empirical consensus, and evaluation metrics pass:
```bash
/home/<USER>/miniconda3/envs/env_bio_core/bin/pytest -v
```

### 5.2 Run a Live Single-Tree Audit Connected to LM Studio
With LM Studio running on `http://localhost:1234/v1`, run the 4-State Agent on the 15-taxon Golden Toy dataset:
```bash
/home/<USER>/miniconda3/envs/env_bio_core/bin/python scripts/run_live_agent_audit.py
```
This audits all 13 internal nodes (`N1..N13`) and writes the validated report to `data/04_audit_reports/live_lmstudio_toy_audit_report.json`.

---

## 6. Complete Workflows for Data Gathering

### Workflow A: Controlled Synthetic Benchmarks (Tier 1 & Tier 2)

#### 1. Generate Synthetic Alignments & Ground Truths
* **Quick Smoke Test (3 datasets):**
  ```bash
  /home/neal/miniconda3/envs/env_bio_core/bin/python scripts/generate_synthetic_benchmarks.py \
    --quick-smoke \
    --compute-ground-truths
  ```
* **Full Production Benchmark Grid (Tier 1 & Tier 2):**
  ```bash
  /home/neal/miniconda3/envs/env_bio_core/bin/python scripts/generate_synthetic_benchmarks.py \
    --tier both \
    --replicates 1 \
    --seq-type DNA \
    --compute-ground-truths
  ```
  * **Tier 1** simulates $N \in \{15, 30, 40, 50\}$, $L = 1000$, Zipfian indels ($L_{\text{gap}} \in [1, 30]$) at gap proportions $G \in \{5\%, 15\%, 30\%\}$, and LBA terminal branch multipliers $k \in \{2, 10, 40\}$.
  * **Tier 2** simulates $N \in \{50, 100, 200\}$, $L \in \{1000, 2000\}$, short internal radiation branches ($\ell_{\text{int}} = 0.002 < 0.005$), extreme gamma rate heterogeneity ($\alpha = 0.2$), and a localized $15\%$ recombinant block swap.
  * Outputs raw FASTAs to `data/01_raw_alignments/` and ground truths (`T_true_rooted.nwk`, `M_true.tsv`, `A_true.json`) to `data/02_ground_truths/`.

#### 2. Run Target Models (`Fusang`, `Phyloformer`) & Classical Baselines
```bash
/home/neal/miniconda3/envs/env_bio_core/bin/python scripts/run_target_models.py \
  --models Fusang Phyloformer IQ-TREE2 RAxML-NG FastTree2 \
  --bootstraps 100
```
* Automatically skips `Fusang` when $N > 40$.
* Converts `Phyloformer` $N \times N$ distance matrices into trees via validated **BIONJ** (`distance_matrix_to_bionj_newick`).
* Applies **Rule 1** (Outgroup rooting), **Rule 2** (Post-order `N1..Nk` indexing), and **Rule 3** ($[0, 100\%]$ support mapping & unit branch lengths `1.0` for `Fusang`) into `data/03_standardized_trees/`.

---

### Workflow B: Real Empirical Biological FASTAs (Tier 3)

For real-world biological alignments (**TreeBASE** fungal/plant loci, **OrthoMaM v12** vertebrate exons, or **SARS-CoV-2 / HIV-1** viral recombinants):

#### 1. Sanitize Your Downloaded FASTA for Tier 3 Compatibility (`sanitize_tier3_fasta.py`)
> **Note (Tier 3 Compatibility Only):** This sanitizer is **only** for external real-world FASTA files downloaded from public databases (which often contain spaces, colons, parentheses, `?`/`.` gap symbols, or 100%-gap columns that crash Newick/FASTA parsers). It is **not** needed for Tier 1 or Tier 2 synthetic benchmarks.

```bash
/home/neal/miniconda3/envs/env_bio_core/bin/python scripts/sanitize_tier3_fasta.py \
  --input-fasta /path/to/downloaded_raw_database_file.fasta \
  --output-fasta data/01_raw_alignments/tier3/My_Empirical_Dataset/alignment.fa \
  --outgroup Designated_Outgroup_Taxon
```
This outputs a clean `data/01_raw_alignments/tier3/My_Empirical_Dataset/alignment.fa` plus `alignment_tier3_sanitized_report.json` mapping original database headers to clean taxon IDs.

#### 2. Build the 3-Engine Consensus Silver Standard (`IQ-TREE 2` + `RAxML-NG` + `FastTree 2`)
```bash
/home/neal/miniconda3/envs/env_bio_core/bin/python scripts/prepare_empirical_tier3.py \
  --fasta data/01_raw_alignments/tier3/My_Empirical_Dataset/alignment.fa \
  --dataset-id My_Empirical_Dataset \
  --outgroup Designated_Outgroup_Taxon \
  --ufboot 1000 \
  --raxml-bs 100
```
*(If auditing a viral alignment with known recombinant breakpoints, append `--recomb-breakpoints "[[start_col, end_col]]"`).*
* Saves the 3-engine concordance summary (`tier3_consensus_summary.json`) and full ground-truth suite (`T_true_rooted.nwk`, `M_true.tsv`, `A_true.json`) to `data/02_ground_truths/tier3/My_Empirical_Dataset/`.

#### 3. Run `Fusang` and `Phyloformer` on the Empirical FASTA
```bash
/home/neal/miniconda3/envs/env_bio_core/bin/python scripts/run_target_models.py \
  --fasta data/01_raw_alignments/tier3/My_Empirical_Dataset/alignment.fa \
  --dataset-id My_Empirical_Dataset \
  --outgroup Designated_Outgroup_Taxon \
  --models Fusang Phyloformer \
  --bootstraps 100
```

---

### Workflow C: Running the 3-Way Auditor Comparison & Statistical Tests

To evaluate the **Proposed 4-State Diagnostic Agent** against **AI Baseline 1 (Zero-Shot LLM)** and **AI Baseline 2 (Generic Unconstrained Agent)** on any dataset:

```bash
/home/neal/miniconda3/envs/env_bio_core/bin/python scripts/run_ai_baselines.py \
  --dataset-id TOY_DEV_15TAXA \
  --target-model Fusang \
  --fasta data/00_golden_toy/toy_alignment.fa \
  --pred-newick data/03_normalized_preds/toy_fusang_normalized.nwk \
  --pred-meta data/03_normalized_preds/toy_fusang_metadata.json \
  --true-newick data/02_ground_truths/toy/T_true_rooted.nwk \
  --asr-state data/02_ground_truths/toy/M_true.tsv \
  --stressor-gt data/02_ground_truths/toy/A_true.json \
  --agent-model "qwen/qwen3-14b" \
  --zeroshot-model "qwen/qwen3-14b" \
  --generic-agent-model "qwen/qwen3-14b" \
  --require-live-llm
```

* **Using a Different LLM for Zero-Shot:** Pass any model identifier available in your LM Studio to `--zeroshot-model` (for example, `--zeroshot-model "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"`).
* **Offline Deterministic Mode (No LM Studio required):** Pass `--base-url none` (and omit `--require-live-llm`).

#### Where Output Reports Are Saved
Each auditor method writes its own isolated JSON report so results never overwrite one another:
* **Proposed 4-State Agent:** `data/04_audit_reports/proposed_agent/<dataset>_<model>_proposed_<llm>_report.json`
* **AI Baseline 1 (Zero-Shot LLM):** `data/04_audit_reports/baseline1_zeroshot_llm/<dataset>_<model>_zeroshot_<llm>_report.json`
* **AI Baseline 2 (Generic Agent):** `data/04_audit_reports/baseline2_generic_agent/<dataset>_<model>_generic_agent_<llm>_report.json`
* **Paired Statistical Tests & Master Table:**
  * `data/04_audit_reports/statistical_comparisons/<dataset>_<model>_comparison_summary.json`
  * `data/04_audit_reports/statistical_comparisons/master_benchmark_metrics_table.tsv`

---

## 7. Summary of Evaluation Metrics in Every Audit Report

| Metric | Symbol | Range | What It Measures |
| :--- | :--- | :--- | :--- |
| **Normalized Robinson-Foulds** | $\text{RF}_{\text{norm}}$ | $[0, 1]$ | Global topological error between $T_{\text{pred}}$ and $T_{\text{true}}$ ($0.0 =$ identical topologies). |
| **False-Confidence Index** | $\text{FCI}$ | $[0, 1]$ | Fraction of high-support splits ($\ge 85\%$) that are topologically wrong (hallucinated clades). |
| **Error Detection Rate** | $\text{EDR}$ | $[0, 1]$ | Recall of the auditor in flagging true topological errors and their underlying stressor. |
| **Claimed-Coordinate Verification Rate** | $\text{CCVR}$ | $[0, 1]$ | Fraction of cited `(site, ancestral_state, derived_state)` claims verified in `M_true.tsv` and `alignment.fa`. |
| **Bootstrap Support Fidelity** | $\text{BSF}$ | $\{0, 1\}$ | Binary gate checking whether the text's confidence phrasing matches $\phi(\text{Support}(b_k))$ (`Low`, `Moderate`, `Definitive/High`). |
| **Site Identification F1** | $\text{F1}(b_k)$ | $[0, 1]$ | Harmonic mean of Precision and Recall of cited synapomorphic/homoplastic columns vs. `M_true.tsv`. |
| **Bayesian Transition Info Gain** | $\text{IG}(b_k)$ | $[0, 1]$ | Proportion of branch site-log-likelihood support ($\Delta\text{SLS}_i$) captured by the agent's cited coordinates, paired with CTMC transition log-likelihood $\log P(\ell_k)$ and Shannon entropy reduction (bits). |
| **Faithfulness / Grounding Score** | $\text{FGS}(b_k)$ | $[0, 1]$ | Fraction of atomic clauses in the explanation entailed by the deterministic tool outputs $\mathcal{O}_{\text{tools}}(b_k)$. |
| **Per-Node Explanation Goodness** | $G_{\text{node}}(b_k)$ | $[0, 1]$ | $\text{CCVR}(b_k) \cdot \text{BSF}(b_k) \cdot \frac{1}{3}(\text{F1}(b_k) + \text{IG}(b_k) + \text{FGS}(b_k))$. Collapses to $0$ if any coordinate is hallucinated. |
| **Logical Consistency Rate** | $\text{LCR}$ | $[0, 1]$ | Global check verifying zero cross-node hierarchical or support-threshold contradictions across the whole tree. |
| **Overall Tree Goodness Score** | $G_{\text{tree}}$ | $[0, 1]$ | $\text{LCR}(T_{\text{pred}}) \cdot \sum_{k=1}^{K} w_k G_{\text{node}}(b_k)$, weighted by branch phylogenetic information $I_{\text{branch}}(b_k)$. |
