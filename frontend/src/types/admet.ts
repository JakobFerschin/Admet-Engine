/** Per-endpoint ML prediction with calibrated probability & epistemic uncertainty. */
export interface MLEndpointDetail {
  probability: number;
  epistemic_std: number;
  ci_95: [number, number];
  risk_tier: "HIGH" | "MODERATE" | "LOW";
  member_probabilities: Record<string, number>;
}

/** Nearest ChEMBL v37 training compound from Applicability Domain lookup. */
export interface NearestNeighborInfo {
  smiles: string;
  tanimoto: number;
  herg_active: boolean | null;
  cyp3a4_active: boolean | null;
}

/** Vectorized ECFP4 Tanimoto Applicability Domain assessment. */
export interface ApplicabilityDomain {
  status: "In-Domain" | "Borderline" | "Out-of-Domain";
  max_tanimoto: number;
  mean_top5_tanimoto: number;
  nearest_neighbor: NearestNeighborInfo | null;
}

/** Deterministic RDKit medicinal chemistry profile. */
export interface CheminformaticsProfile {
  identifiers: {
    canonical_smiles: string;
    molecular_formula: string;
    inchi_key: string;
    murcko_scaffold: string;
  };
  physchem: {
    mw: number;
    clogp: number;
    tpsa: number;
    hbd: number;
    hba: number;
    rotatable_bonds: number;
    aromatic_rings: number;
    heavy_atoms: number;
    formal_charge: number;
    molar_refractivity: number;
    fsp3: number;
    qed: number;
    bertz_ct: number;
  };
  rules: {
    lipinski_pass: boolean;
    lipinski_violations: number;
    lipinski_details: string[];
    veber_pass: boolean;
    pfizer_3_75_warning: boolean;
    gsk_4_400_pass: boolean;
  };
  alerts: {
    pains_count: number;
    pains_alerts: string[];
    brenk_count: number;
    brenk_alerts: string[];
    total_alerts: number;
  };
  boiled_egg: {
    tpsa: number;
    wlogp: number;
    hia_absorbed: boolean;
    bbb_permeant: boolean;
    region: "yolk" | "white" | "outside";
    classification: string;
  };
  svg_2d: string | null;
}

/** Single molecule result from POST /predict/admet. */
export interface MoleculePrediction {
  smiles: string;
  status: "success" | "error";
  predictions: Record<string, number> | null;
  ml_endpoints?: Record<string, MLEndpointDetail> | null;
  applicability_domain?: ApplicabilityDomain | null;
  cheminformatics?: CheminformaticsProfile | null;
  error: string | null;
}

/** Batch response from POST /predict/admet. */
export interface AdmetResponse {
  results: MoleculePrediction[];
  model_version: string;
  prediction_time_ms: number;
}

/** Binary classification metrics from held-out Bemis-Murcko scaffold evaluation. */
export interface BinaryMetrics {
  n_samples: number;
  n_positive: number;
  prevalence: number;
  roc_auc: number;
  pr_auc: number;
  brier_score: number;
  ece: number;
  f1: number;
  balanced_accuracy: number;
  sensitivity: number;
  specificity: number;
}

export interface ClinicalBenchmarkRow {
  name: string;
  clinical_class: string;
  expected_risk: string;
  smiles: string;
  hERG_prob: number;
  hERG_std: number;
  CYP3A4_prob: number;
  CYP3A4_std: number;
  max_tanimoto: number;
  domain_status: string;
}

/** Validation & provenance metadata returned by GET /model/metadata. */
export interface ModelMetadata {
  model_version: string;
  trained_at_utc?: string;
  dataset?: string;
  split_method?: string;
  architecture?: {
    members: string[];
    calibration: string;
    uncertainty: string;
    applicability_domain: string;
  };
  scaffold_stats?: {
    total_compounds: number;
    total_scaffolds: number;
    train_compounds: number;
    train_scaffolds: number;
    val_compounds: number;
    val_scaffolds: number;
    test_compounds: number;
    test_scaffolds: number;
  };
  endpoints?: Record<
    string,
    {
      short_name: string;
      target_id: string;
      weights: Record<string, number>;
      temperatures: Record<string, number>;
      test_metrics: BinaryMetrics;
    }
  >;
  ablation_table?: Record<string, Record<string, BinaryMetrics>>;
  clinical_benchmark?: ClinicalBenchmarkRow[];
  train_duration_sec?: number;
}

export const ML_ENDPOINT_META: Record<
  string,
  { title: string; target: string; assay: string; description: string }
> = {
  hERG_inhibition: {
    title: "hERG Channel Blockade",
    target: "KCNH2 · CHEMBL240",
    assay: "Patch-clamp / binding (pChEMBL ≥ 5.0)",
    description: "Cardiac Kv11.1 potassium channel inhibition risk linked to QT prolongation & Torsades de Pointes.",
  },
  CYP3A4_inhibition: {
    title: "CYP3A4 Metabolic Inhibition",
    target: "CYP3A4 · CHEMBL340",
    assay: "Microsomal / recombinant IC50/Ki (pChEMBL ≥ 5.0)",
    description: "Primary hepatic cytochrome P450 isozyme responsible for ~50% of clinical drug clearance & DDIs.",
  },
};

export const MODEL_MEMBER_LABELS: Record<string, string> = {
  dmpnn: "Chemprop One-Hot DMPNN (Graph)",
  ecfp4_mlp: "Residual ECFP4 + PhysChem MLP",
  extra_trees: "ExtraTrees Non-Linear QSAR",
  linear_qsar: "L2-Regularized Logistic QSAR",
  consensus_ensemble: "Calibrated Consensus Ensemble",
};
