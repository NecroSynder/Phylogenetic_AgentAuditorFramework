from typing import List, Literal, Optional
from pydantic import BaseModel, Field

TargetModelName = Literal["Fusang", "Phyloformer", "IQ-TREE2", "RAxML-NG", "FastTree2"]


class AuditMetadata(BaseModel):
    dataset_id: str
    target_model: TargetModelName
    taxon_count: int = Field(..., gt=3)
    sequence_length: int = Field(..., gt=0)
    orchestrator_llm: str


class GlobalMetrics(BaseModel):
    normalized_rf_distance: float = Field(..., ge=0.0, le=1.0)
    false_confidence_index_fci: float = Field(..., ge=0.0, le=1.0)
    conditional_fci: float = Field(default=0.0, ge=0.0, le=1.0)
    error_detection_rate_edr: float = Field(default=1.0, ge=0.0, le=1.0)
    total_internal_nodes_audited: int = Field(..., ge=0)
    false_confidence_nodes_count: int = Field(..., ge=0)
    total_hallucination_retries_triggered: int = Field(default=0, ge=0)
    gate_rescued_nodes_count: int = Field(default=0, ge=0)
    pre_gate_ccvr: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    claim_to_coordinate_verification_rate: float = Field(..., ge=0.0, le=1.0)
    logical_consistency_rate_lcr: float = Field(..., ge=0.0, le=1.0)
    overall_tree_goodness_score: float = Field(..., ge=0.0, le=1.0)


class BayesianTransitionMetrics(BaseModel):
    ctmc_branch_transition_loglik: float
    clade_shannon_entropy_reduction_bits: float
    transition_information_gain_ig: float = Field(..., ge=0.0, le=1.0)


class ExplanationQualityVector(BaseModel):
    ccvr_local: float = Field(..., ge=0.0, le=1.0)
    pre_gate_ccvr_local: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    bsf_local: int = Field(..., ge=0, le=1)
    feature_f1: float = Field(..., ge=0.0, le=1.0)
    information_gain_ig: float = Field(..., ge=0.0, le=1.0)
    faithfulness_fgs: float = Field(..., ge=0.0, le=1.0)
    verification_retries: int = Field(default=0, ge=0)


class CausalSequenceEvidence(BaseModel):
    site_coordinate: int = Field(..., ge=1)
    ancestral_state: str
    derived_state: str
    posterior_probability: float = Field(..., ge=0.0, le=1.0)
    consistency_index_ci: float = Field(..., ge=0.0, le=1.0)
    mutation_class: Literal["SYNAPOMORPHY", "INDEL", "HOMOPLASY", "HOMOPLASTIC_CONVERGENCE"]


class DiagnosedFailureMode(BaseModel):
    primary_stressor: Literal[
        "NONE",
        "LBA",
        "INDEL_GAP",
        "RECOMBINATION",
        "SHORT_BRANCH_HOMOPLASY",
        "SHORT_BRANCH_HOMOPLASY_AND_LBA",
    ]
    lba_branch_ratio: float = Field(..., ge=0.0)
    local_gap_density_pct: float = Field(..., ge=0.0, le=100.0)
    agent_causal_explanation: str


class NodeAuditRecord(BaseModel):
    node_id: str
    predicted_clade_taxa: List[str]
    model_support_pct: float = Field(..., ge=0.0, le=100.0)
    topological_status: Literal["CONCORDANT", "DISCORDANT_LOW_SUPPORT", "FALSE_CONFIDENCE_ERROR"]
    bayesian_transition_metrics: BayesianTransitionMetrics
    explanation_quality_vector: ExplanationQualityVector
    node_goodness_score: float = Field(..., ge=0.0, le=1.0)
    causal_sequence_evidence: List[CausalSequenceEvidence]
    diagnosed_failure_mode: DiagnosedFailureMode


class AuditReport(BaseModel):
    audit_metadata: AuditMetadata
    global_metrics: GlobalMetrics
    node_audits: List[NodeAuditRecord]
