# An Agentic AI Framework for Autonomous Auditing and Error Detection in Phylogenetic Inference Models

An autonomous **neuro-symbolic Agentic AI auditing framework** designed to inspect, verify, and diagnose topological errors and evolutionary failure modes in deep-learning phylogenetic inference models (**Fusang** and **Phyloformer**) alongside classical maximum-likelihood baselines (**IQ-TREE 2**, **RAxML-NG**, and **FastTree 2**).

---

## 1. Overview & Neuro-Symbolic Architecture

Deep-learning phylogenetic models reconstruct trees orders of magnitude faster than classical likelihood searches, yet suffer from unquantified failure boundaries and opaque support values:
* **Fusang** ($\le 40$ taxa) predicts discrete tree topologies without evolutionary branch lengths and is sensitive to indel boundaries, Long-Branch Attraction (LBA), and taxon scaling near $N = 40$.
* **Phyloformer** ($> 40$ taxa) predicts an $N \times N$ pairwise evolutionary distance matrix $\hat{\mathbf{D}}$ (converted into a tree via **BIONJ**), compromising topological accuracy on short radiation branches ($\ell_{\text{int}} < 0.005$), extreme rate heterogeneity ($\alpha = 0.2$), and localized recombination.

Rather than letting an LLM read raw FASTA strings and hallucinate sequence coordinates, this framework couples a **4-State LangGraph State Machine** with **5 Deterministic Python Bioinformatics Tools**:
1. **State 1 — Topological & Confidence Triage (`audit_tree_topology`):** Computes $\text{RF}_{\text{norm}}$, $\text{FCI}_{\text{global}}$, and $\text{FCI}_{\text{cond}}$ (respecting 3-engine consensus masks on empirical alignments).
2. **State 2 — Clade-Isolated Mutation & Indel Attribution (`extract_branch_synapomorphies`):** Queries exact Felsenstein site log-likelihood transitions ($\Delta\text{SLS}_i(b) > 0$, `IQ-TREE 2 --ancestral -wsl`) and Fitch gap synapomorphies on concordant splits, or extracts clade-partitioning convergent/indel columns on discordant splits without cross-clade node-ID contamination.
3. **State 3 — Clade-Local Stressor Interrogation (`diagnose_evolutionary_stressor` + `inspect_alignment_window`):** Isolates LBA, indel/gap-masking, recombination breakpoint overlap, and short-branch homoplasy using informative-site $CI_i$ and clade-local metrics.
4. **State 4 — Strict Branch & Prose Verification Gate (`verify_claim_coordinates`):** Verifies every cited `(site, ancestral_state, derived_state)` tuple in both JSON `draft_claims` and `explanation` prose before emitting a Pydantic-validated JSON Audit Report.

---

## 2. Repository Structure

```text
SP_Env/
├── README.md                              # Project documentation & usage guide
├── pytest.ini                             # Pytest configuration
├── envs/
│   ├── env_bio_core.yml                   # Orchestrator, bioinformatics CLIs & LangGraph (Python 3.11)
│   ├── env_fusang.yml                     # Isolated PyTorch/CUDA Conda env for Fusang
│   └── env_phyloformer.yml                # Isolated PyTorch/CUDA Conda env for Phyloformer
├── src/
│   ├── utils/
│   │   └── io_cache.py                    # mtime-cached I/O, Conda binary & WSL2 LM Studio discovery
│   ├── preprocessing/
│   │   ├── ground_truth_gen.py            # Step 1: ASR, exact .sitelh delta_SLS, bit-parallel Fitch MP
│   │   ├── normalizer_wrapper.py          # Step 2: B=100 bootstrap, vectorized BIONJ, Rules 1–3
│   │   ├── synthetic_generator.py         # Step A: AliSim + native CTMC Zipfian indels & recombination
│   │   ├── empirical_consensus.py         # Step B: Tier 3 ModelFinder (-m MFP) 3-engine consensus
│   │   └── target_model_runner.py         # Step B: Batch GPU adapter hooks & parallel classical runner
│   ├── tools/
│   │   └── bio_tools.py                   # Step 3: All 5 deterministic Python bioinformatics tools
│   ├── agent/
│   │   ├── diagnostic_agent.py            # Step 3: 4-State LangGraph state machine orchestrator
│   │   └── ai_baselines.py                # Step C: Unassisted Zero-Shot (B1) & Generic Agent (B2)
│   ├── evaluation/
│   │   └── metrics.py                     # Step 4: Unbiased shared evaluator & replicate statistical suite
│   └── schemas/
│       └── audit_schema.py                # Strict Pydantic v2 schema for JSON audit reports
├── scripts/
│   ├── adapters/
│   │   ├── run_fusang_cli.py              # Single-load batch adapter for Jerry-0591/Fusang
│   │   └── run_phyloformer_cli.py         # Single-load batch GPU adapter for lucanest/Phyloformer
│   ├── run_full_pipeline.py               # CLI: Master end-to-end 4-step runner
│   ├── generate_synthetic_benchmarks.py   # CLI: Generate Tier 1 & Tier 2 DNA/AA stressor datasets
│   ├── sanitize_tier3_fasta.py            # CLI: Clean external empirical DNA/AA FASTA files
│   ├── prepare_empirical_tier3.py         # CLI: Build Tier 3 3-engine consensus Silver Standard
│   ├── run_target_models.py               # CLI: Run Fusang, Phyloformer & classical ML baselines
│   ├── run_live_agent_audit.py            # CLI: Run single live agent audit via LM Studio
│   └── run_ai_baselines.py                # CLI: Run 3-way auditor comparison & statistical validation
└── tests/                                 # Isolated tmp_path unit, integration & anti-loophole tests
```

---

## 3. Quick Start

### 3.1 Run the Full Automated Test Suite
```bash
/home/neal/miniconda3/envs/env_bio_core/bin/pytest -v --disable-warnings
```

### 3.2 Run End-to-End Pipeline (`run_full_pipeline.py`)
* **With Live Local LLM in LM Studio (`http://localhost:1234/v1`) and Real Model Adapter:**
  ```bash
  conda activate env_bio_core
  python scripts/run_full_pipeline.py \
    --dataset-id TOY_DEV_15TAXA \
    --target-model Fusang \
    --fusang-batch-cmd "conda run -n env_fusang python scripts/adapters/run_fusang_cli.py --fusang-repo /path/to/Fusang --msa-dir {msa_dir} --out-dir {out_dir}" \
    --require-live-llm
  ```
* **Quick Offline Smoke Test (Surrogate + Deterministic Mock Guardrails Explicitly Enabled):**
  ```bash
  python scripts/run_full_pipeline.py \
    --dataset-id TOY_DEV_15TAXA \
    --target-model Fusang \
    --quick \
    --use-surrogate \
    --offline-mock
  ```

### 3.3 Generate Tier 1 & Tier 2 Synthetic Benchmarks (DNA & Protein)
```bash
python scripts/generate_synthetic_benchmarks.py \
  --tier both \
  --seq-type both \
  --replicates 1 \
  --compute-ground-truths
```
