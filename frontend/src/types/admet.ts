/** Single molecule prediction from the ADMET-Engine API. */
export interface MoleculePrediction {
  smiles: string;
  status: "success" | "error";
  predictions: Record<string, number> | null;
  error: string | null;
}

/** Batch response from POST /predict/admet */
export interface AdmetResponse {
  results: MoleculePrediction[];
  model_version: string;
  prediction_time_ms: number;
}

/** All 12 ADMET endpoint labels */
export const ENDPOINT_LABELS = [
  "hepatotoxicity",
  "hERG_inhibition",
  "AMES_mutagenicity",
  "CYP3A4_inhibition",
  "nephrotoxicity",
  "CYP2D6_inhibition",
  "CYP2C9_inhibition",
  "blood_brain_barrier",
  "plasma_protein_binding",
  "caco2_permeability",
  "logD_logP",
  "P_gp_substrate_inhibitor",
] as const;

export type EndpointLabel = (typeof ENDPOINT_LABELS)[number];

/** Subset used in the radar chart (6 key toxicology endpoints) */
export const RADAR_ENDPOINTS: EndpointLabel[] = [
  "hepatotoxicity",
  "hERG_inhibition",
  "AMES_mutagenicity",
  "CYP3A4_inhibition",
  "nephrotoxicity",
  "blood_brain_barrier",
];

/** Pretty display names for endpoints */
export const ENDPOINT_DISPLAY: Record<EndpointLabel, string> = {
  hepatotoxicity: "Hepatotox",
  hERG_inhibition: "hERG",
  AMES_mutagenicity: "AMES",
  CYP3A4_inhibition: "CYP3A4",
  nephrotoxicity: "Nephrotox",
  CYP2D6_inhibition: "CYP2D6",
  CYP2C9_inhibition: "CYP2C9",
  blood_brain_barrier: "BBB",
  plasma_protein_binding: "PPB",
  caco2_permeability: "Caco-2",
  logD_logP: "logD/logP",
  P_gp_substrate_inhibitor: "P-gp",
};

/** Triage colour thresholds */
export const TRIAGE_THRESHOLD = {
  HIGH: 0.8,    // 🔴 ≥ 0.8
  MEDIUM: 0.5,  // 🟡 ≥ 0.5
  // < 0.5       // 🟢
} as const;
