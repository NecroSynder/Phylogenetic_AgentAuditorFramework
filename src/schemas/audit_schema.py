from typing import List, Literal
from pydantic import BaseModel, Field

class AuditMetadata(BaseModel):
    dataset_id: str
    target_model: Literal["Fusang", "Phyloformer"]
    taxon_count: int = Field(..., gt=3)
    sequence_length: int = Field(..., gt=0)
    orchestrator_llm: str

class GlobalMetrics(BaseModel):
    normalized_rf_distance: float = Field(..., ge=0.0, le=1.0)
    false_confidence_index_fci: float = Field(..., ge=0.0, le=1.0)
    total_internal_nodes_audited: int
    false_confidence_nodes_count: int
    claim_to_coordinate_verification_rate: float = Field(..., ge=0.0, le=1.0)
    logical_consistency_rate_lcr: float = Field(..., ge=0.0, le=1.0)
    overall_tree_goodness_score: float = Field(..., ge=0.0, le=1.0)

class BayesianTransitionMetrics(BaseModel):
    ctmc_branch_transition_loglik: float
    clade_shannon_entropy_reduction_bits: float
    transition_information_gain_ig: float = Field(..., ge=0.0, le=1.0)

class ExplanationQualityVector(BaseModel):
    ccvr_local: float = Field(..., ge=0.0, le=1.0)
    bsf_local: int = Field(..., ge=0, le=1)
    feature_f1: float = Field(..., ge=0.0, le=1.0)
    information_gain_ig: float = Field(..., ge=0.0, le=1.0)
    faithfulness_fgs: float = Field(..., ge=0.0, le=1.0)

class CausalSequenceEvidence(BaseModel):
    site_coordinate: int = Field(..., ge=1)
    ancestral_state: str
    derived_state: str
    posterior_probability: float = Field(..., ge=0.0, le=1.0)
    consistency_index_ci: float = Field(..., gt=0.0, le=1.0)
    mutation_class: Literal["SYNAPOMORPHY", "INDEL", "HOMOPLASY", "HOMOPLASTIC_CONVERGENCE"]

class DiagnosedFailureMode(BaseModel):
    primary_stressor: Literal[
        "NONE", "LBA", "INDEL_GAP", "RECOMBINATION", 
        "SHORT_BRANCH_HOMOPLASY", "SHORT_BRANCH_HOMOPLASY_AND_LBA"
    ]
    lba_branch_ratio: float
    local_gap_density_pct: float
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