"""
Live CLI Runner for Pipeline Step 3 (LangGraph + Local Qwen 2.5 via LM Studio).
Usage:
    python scripts/run_live_agent_audit.py --base-url http://localhost:1234/v1 --model qwen2.5-14b-instruct
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.ground_truth_gen import run_ground_truth_pipeline
from src.preprocessing.normalizer_wrapper import standardize_and_root_tree
from src.agent.diagnostic_agent import run_diagnostic_audit


def _detect_wsl_host_ip() -> str | None:
    """Detects the Windows host IP from inside WSL2 (/etc/resolv.conf or default route)."""
    try:
        resolv = Path("/etc/resolv.conf").read_text()
        for line in resolv.splitlines():
            if line.strip().startswith("nameserver"):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1].strip()
    except Exception:
        pass
    return None


def check_lm_studio_server(base_url: str) -> tuple[str, list[str]]:
    """
    Queries GET /v1/models on LM Studio to verify the local server is running and list loaded models.
    Automatically tries both localhost and the WSL2 Windows host IP if running inside WSL2.
    """
    candidates = [base_url.rstrip("/")]
    wsl_ip = _detect_wsl_host_ip()
    if wsl_ip and "localhost" in base_url:
        candidates.append(base_url.replace("localhost", wsl_ip).rstrip("/"))
    if "localhost" in base_url:
        candidates.append(base_url.replace("localhost", "127.0.0.1").rstrip("/"))

    last_err = None
    for cand_url in candidates:
        models_url = f"{cand_url}/models"
        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("id", "") for m in data.get("data", [])]
                return cand_url, models
        except Exception as exc:
            last_err = exc
    raise RuntimeError(f"Could not connect to {candidates}: {last_err}")


def main():
    parser = argparse.ArgumentParser(description="Run Live LangGraph + LM Studio Phylogenetic Diagnostic Audit")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="LM Studio OpenAI-compatible server URL")
    parser.add_argument("--model", default=None, help="Model identifier in LM Studio (auto-detected if omitted)")
    parser.add_argument(
        "--output",
        default="data/04_audit_reports/live_lmstudio_toy_audit_report.json",
        help="Path to write validated JSON Audit Report",
    )
    args = parser.parse_args()

    print(f"[1/4] Checking LM Studio server at {args.base_url} ...")
    try:
        resolved_base_url, loaded_models = check_lm_studio_server(args.base_url)
    except Exception as exc:
        print(
            f"❌ Could not reach LM Studio server ({exc}).\n"
            "   Please open LM Studio on Windows -> Developer / Local Server tab ->\n"
            "   Enable 'Serve on Local Network' (for WSL2 access) -> Load Qwen 2.5 -> Click 'Start Server' (port 1234)."
        )
        sys.exit(1)

    if not loaded_models:
        print("❌ LM Studio server is running, but no model is currently loaded in memory.")
        sys.exit(1)

    model_id = args.model or loaded_models[0]
    print(f"✅ Connected to LM Studio at {resolved_base_url}! Active model: {model_id}")

    print("[2/4] Ensuring Golden Toy Ground Truths (Step 1) and Normalized Tree (Step 2) exist...")
    if not Path("data/02_ground_truths/toy/M_true.tsv").exists():
        run_ground_truth_pipeline(
            "data/00_golden_toy/toy_alignment.fa",
            "data/00_golden_toy/T_true.nwk",
            "data/02_ground_truths/toy",
        )
    if not Path("data/03_normalized_preds/toy_fusang_normalized.nwk").exists():
        with open("data/02_ground_truths/toy/T_true_rooted.nwk") as f:
            nwk_str = f.read().strip()
        standardize_and_root_tree(
            nwk_str,
            [nwk_str] * 10,
            "Taxon_01",
            "Fusang",
            "data/03_normalized_preds/toy_fusang",
        )

    print(f"[3/4] Running LangGraph 4-State Diagnostic Agent with live LLM ({model_id})...")
    report = run_diagnostic_audit(
        dataset_id="toy_00_live_lmstudio",
        target_model="Fusang",
        fasta_path="data/00_golden_toy/toy_alignment.fa",
        pred_newick_path="data/03_normalized_preds/toy_fusang_normalized.nwk",
        pred_meta_path="data/03_normalized_preds/toy_fusang_metadata.json",
        true_newick_path="data/02_ground_truths/toy/T_true_rooted.nwk",
        asr_state_path="data/02_ground_truths/toy/M_true.tsv",
        stressor_gt_path="data/02_ground_truths/toy/A_true.json",
        output_json_path=args.output,
        orchestrator_llm=model_id,
        lm_studio_base_url=resolved_base_url,
        require_live_llm=True,
    )

    gm = report["global_metrics"]
    print("\n[4/4] ✅ Live Audit Complete! Validated JSON Report Summary:")
    print(f"  • Output File:          {args.output}")
    print(f"  • Nodes Audited:        {gm['total_internal_nodes_audited']}")
    print(f"  • Normalized RF (Eq.1): {gm['normalized_rf_distance']}")
    print(f"  • Global FCI (Eq.2):    {gm['false_confidence_index_fci']}")
    print(f"  • Global CCVR (Eq.8):   {gm['claim_to_coordinate_verification_rate']}")
    print(f"  • Global LCR (Eq.10):   {gm['logical_consistency_rate_lcr']}")
    print(f"  • Tree Goodness G_tree: {gm['overall_tree_goodness_score']}")
    print("\nSample Node N1 Causal Explanation from Live LLM:")
    print(f"  \"{report['node_audits'][0]['diagnosed_failure_mode']['agent_causal_explanation']}\"")


if __name__ == "__main__":
    main()
