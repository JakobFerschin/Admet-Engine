import { useState } from "react";
import { ChevronDown, ChevronUp, Compass, Info } from "lucide-react";
import type {
  ApplicabilityDomain,
  MLEndpointDetail,
  ModelMetadata,
} from "../types/admet";
import { ML_ENDPOINT_META, MODEL_MEMBER_LABELS } from "../types/admet";

interface EndpointPredictionPanelProps {
  mlEndpoints: Record<string, MLEndpointDetail>;
  applicabilityDomain?: ApplicabilityDomain | null;
  metadata?: ModelMetadata | null;
}

function tierStyles(tier: "HIGH" | "MODERATE" | "LOW") {
  if (tier === "HIGH") {
    return {
      badge: "bg-crimson-soft text-crimson border-crimson-border",
      bar: "bg-crimson",
      text: "text-crimson",
    };
  }
  if (tier === "MODERATE") {
    return {
      badge: "bg-ochre-soft text-ochre border-ochre-border",
      bar: "bg-ochre",
      text: "text-ochre",
    };
  }
  return {
    badge: "bg-teal-soft text-teal-academic border-teal-border",
    bar: "bg-teal-academic",
    text: "text-teal-academic",
  };
}

function domainStyles(status: ApplicabilityDomain["status"]) {
  if (status === "In-Domain") {
    return "bg-teal-soft text-teal-academic border-teal-border";
  }
  if (status === "Borderline") {
    return "bg-ochre-soft text-ochre border-ochre-border";
  }
  return "bg-crimson-soft text-crimson border-crimson-border";
}

export default function EndpointPredictionPanel({
  mlEndpoints,
  applicabilityDomain,
  metadata,
}: EndpointPredictionPanelProps) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});

  const endpointKeys = Object.keys(mlEndpoints);

  return (
    <div className="card space-y-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="section-kicker">ChEMBL v37 Multi-Modal ML Ensemble</p>
          <h3 className="font-serif text-lg font-semibold text-ink mt-0.5">
            Calibrated Toxicity &amp; Metabolism Predictions
          </h3>
        </div>
        <span className="data-mono text-[10px] text-ink-muted bg-subtle px-2 py-1 rounded border border-hairline">
          4-Model Consensus · ±95% CI
        </span>
      </div>

      {/* ML Endpoint Cards */}
      <div className="space-y-4">
        {endpointKeys.map((key) => {
          const ep = mlEndpoints[key];
          const meta = ML_ENDPOINT_META[key] ?? {
            title: key,
            target: key,
            assay: "ChEMBL v37 Binary Classification",
            description: "",
          };
          const valAuc = metadata?.endpoints?.[key]?.test_metrics?.roc_auc;
          const style = tierStyles(ep.risk_tier);
          const pct = Math.round(ep.probability * 1000) / 10;
          const ciLo = Math.round(ep.ci_95[0] * 100);
          const ciHi = Math.round(ep.ci_95[1] * 100);
          const isOpen = !!expanded[key];

          return (
            <div
              key={key}
              className="rounded-md border border-hairline bg-canvas/60 p-4 space-y-3"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2 flex-wrap">
                    <h4 className="font-sans text-sm font-semibold text-ink">
                      {meta.title}
                    </h4>
                    <span className="data-mono text-[10px] text-ink-muted">
                      {meta.target}
                    </span>
                    {valAuc != null && (
                      <span
                        className="data-mono text-[10px] text-navy bg-navy-soft border border-navy-border px-1.5 py-0.5 rounded"
                        title="Held-out Bemis-Murcko scaffold test set ROC-AUC"
                      >
                        Scaffold AUC {valAuc.toFixed(3)}
                      </span>
                    )}
                  </div>
                  <p className="text-[11px] text-ink-muted mt-0.5">{meta.description}</p>
                </div>

                <div className="text-right shrink-0">
                  <div className="flex items-baseline justify-end gap-1.5">
                    <span className={`data-mono text-xl font-semibold ${style.text}`}>
                      {pct.toFixed(1)}%
                    </span>
                    <span className="data-mono text-[11px] text-ink-muted">
                      ± {(ep.epistemic_std * 100).toFixed(1)}%
                    </span>
                  </div>
                  <span
                    className={`inline-block mt-1 data-mono text-[10px] font-semibold px-2 py-0.5 rounded border ${style.badge}`}
                  >
                    {ep.risk_tier} RISK
                  </span>
                </div>
              </div>

              {/* Calibrated Probability Bar + 95% Epistemic CI Whisker */}
              <div className="space-y-1">
                <div className="relative h-2.5 w-full rounded-full bg-slate-200/80 overflow-visible">
                  {/* Main probability fill */}
                  <div
                    className={`h-full rounded-full transition-all duration-300 ${style.bar}`}
                    style={{ width: `${Math.max(2, Math.min(100, ep.probability * 100))}%` }}
                  />
                  {/* Threshold markers at 0.45 and 0.70 */}
                  <div
                    className="absolute top-0 bottom-0 w-px bg-slate-400/70"
                    style={{ left: "45%" }}
                    title="Moderate threshold (0.45)"
                  />
                  <div
                    className="absolute top-0 bottom-0 w-px bg-slate-500/70"
                    style={{ left: "70%" }}
                    title="High risk threshold (0.70)"
                  />
                  {/* 95% Epistemic CI whisker */}
                  <div
                    className="absolute -top-1 h-4.5 border-l-2 border-r-2 border-ink/70 flex items-center pointer-events-none"
                    style={{
                      left: `${ciLo}%`,
                      width: `${Math.max(1, ciHi - ciLo)}%`,
                    }}
                    title={`95% epistemic interval: [${ep.ci_95[0].toFixed(3)}, ${ep.ci_95[1].toFixed(3)}]`}
                  >
                    <div className="w-full h-[1.5px] bg-ink/60" />
                  </div>
                </div>
                <div className="flex justify-between data-mono text-[10px] text-ink-muted">
                  <span>0.00 (Inactive)</span>
                  <span>
                    95% CI [{ep.ci_95[0].toFixed(2)}, {ep.ci_95[1].toFixed(2)}]
                  </span>
                  <span>1.00 (Active)</span>
                </div>
              </div>

              {/* Per-Model Consensus Breakdown Toggle */}
              <div className="pt-1 border-t border-hairline/80">
                <button
                  type="button"
                  onClick={() => setExpanded((p) => ({ ...p, [key]: !p[key] }))}
                  className="w-full flex items-center justify-between text-[11px] text-ink-secondary hover:text-ink"
                >
                  <span className="inline-flex items-center gap-1.5 font-medium">
                    <Info className="w-3.5 h-3.5 text-ink-muted" />
                    Inspect 4-Model Consensus Breakdown
                  </span>
                  {isOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                </button>

                {isOpen && (
                  <div className="mt-2.5 grid grid-cols-2 gap-2 animate-fade-in">
                    {Object.entries(ep.member_probabilities).map(([mKey, mProb]) => (
                      <div
                        key={mKey}
                        className="bg-paper border border-hairline rounded px-2.5 py-1.5 flex items-center justify-between gap-2"
                      >
                        <span className="text-[11px] text-ink-secondary truncate">
                          {MODEL_MEMBER_LABELS[mKey] ?? mKey}
                        </span>
                        <span className="data-mono text-xs font-semibold text-ink">
                          {(mProb * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Applicability Domain Section */}
      {applicabilityDomain && (
        <div className="rounded-md border border-hairline bg-paper p-3.5 space-y-2.5">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <div className="flex items-center gap-2">
              <Compass className="w-4 h-4 text-navy" />
              <span className="text-xs font-semibold text-ink">
                Applicability Domain (ECFP4 Tanimoto)
              </span>
            </div>
            <span
              className={`data-mono text-[10px] font-semibold px-2 py-0.5 rounded border ${domainStyles(
                applicabilityDomain.status,
              )}`}
            >
              {applicabilityDomain.status.toUpperCase()}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div className="bg-canvas rounded px-3 py-2 border border-hairline">
              <p className="text-[10px] font-mono uppercase text-ink-muted">
                Max Similarity (T_max)
              </p>
              <p className="data-mono text-sm font-semibold text-ink mt-0.5">
                {applicabilityDomain.max_tanimoto.toFixed(3)}
              </p>
            </div>
            <div className="bg-canvas rounded px-3 py-2 border border-hairline">
              <p className="text-[10px] font-mono uppercase text-ink-muted">
                Top-5 Mean Similarity
              </p>
              <p className="data-mono text-sm font-semibold text-ink mt-0.5">
                {applicabilityDomain.mean_top5_tanimoto.toFixed(3)}
              </p>
            </div>
          </div>

          {applicabilityDomain.nearest_neighbor && (
            <div className="text-[11px] pt-1.5 border-t border-hairline space-y-1">
              <div className="flex items-center justify-between gap-2">
                <span className="text-ink-muted font-medium">
                  Nearest ChEMBL v37 Training Analog:
                </span>
                <span className="data-mono text-[10px] text-ink-secondary">
                  {applicabilityDomain.nearest_neighbor.herg_active !== null &&
                    `hERG: ${applicabilityDomain.nearest_neighbor.herg_active ? "Active" : "Inactive"}`}
                  {applicabilityDomain.nearest_neighbor.herg_active !== null &&
                    applicabilityDomain.nearest_neighbor.cyp3a4_active !== null &&
                    " · "}
                  {applicabilityDomain.nearest_neighbor.cyp3a4_active !== null &&
                    `CYP3A4: ${applicabilityDomain.nearest_neighbor.cyp3a4_active ? "Active" : "Inactive"}`}
                </span>
              </div>
              <code
                className="block data-mono text-[10px] text-ink-secondary bg-canvas px-2 py-1 rounded border border-hairline truncate"
                title={applicabilityDomain.nearest_neighbor.smiles}
              >
                {applicabilityDomain.nearest_neighbor.smiles}
              </code>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
