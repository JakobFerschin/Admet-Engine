import { FlaskConical, ShieldCheck, Layers } from "lucide-react";
import type { ModelMetadata } from "../types/admet";
import { ML_ENDPOINT_META, MODEL_MEMBER_LABELS } from "../types/admet";

interface ModelValidationViewProps {
  metadata: ModelMetadata | null;
  onAnalyzeDrug: (smiles: string, name: string) => void;
}

export default function ModelValidationView({
  metadata,
  onAnalyzeDrug,
}: ModelValidationViewProps) {
  if (!metadata || !metadata.ablation_table) {
    return (
      <div className="card py-12 text-center space-y-2">
        <p className="font-serif text-lg text-ink">
          Model Validation Metadata Unavailable
        </p>
        <p className="text-xs text-ink-muted">
          Start the FastAPI backend (<code className="data-mono">uvicorn src.api.main:app --port 8000</code>) to inspect live ChEMBL v37 scaffold-split benchmarks.
        </p>
      </div>
    );
  }

  const sc = metadata.scaffold_stats;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Methodology Overview */}
      <section className="card space-y-4">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="section-kicker">Model Provenance &amp; Rigor</p>
            <h2 className="font-serif text-2xl font-semibold text-ink mt-0.5">
              Bemis-Murcko Scaffold Validation &amp; Ablation Study
            </h2>
            <p className="text-xs text-ink-secondary mt-1.5 max-w-3xl leading-relaxed">
              Unlike random train/test splits that leak congeneric chemical series and inflate reported AUC,
              all models in <span className="font-medium text-ink">ADMET-Engine v2.0</span> are partitioned strictly
              by canonical Bemis-Murcko core scaffolds (zero core overlap between train, validation, and test).
              Temperature scaling and ensemble weights are fitted exclusively on the validation split; all tables
              below report unbiased performance on completely unseen scaffolds.
            </p>
          </div>
          <div className="bg-canvas border border-hairline rounded px-3.5 py-2.5 text-right shrink-0">
            <p className="font-mono text-[10px] uppercase text-ink-muted">Active Artifact</p>
            <p className="data-mono text-xs font-semibold text-ink mt-0.5">{metadata.model_version}</p>
            {metadata.trained_at_utc && (
              <p className="data-mono text-[10px] text-ink-muted mt-0.5">{metadata.trained_at_utc}</p>
            )}
          </div>
        </div>

        {/* Scaffold Split Stat Cards */}
        {sc && (
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2">
            <div className="bg-canvas border border-hairline rounded p-3">
              <p className="section-kicker">Curated Corpus</p>
              <p className="data-mono text-lg font-semibold text-ink mt-1">
                {sc.total_compounds.toLocaleString()}
              </p>
              <p className="text-[11px] text-ink-muted">
                {sc.total_scaffolds.toLocaleString()} unique scaffolds
              </p>
            </div>
            <div className="bg-canvas border border-hairline rounded p-3">
              <p className="section-kicker">Train Partition (80%)</p>
              <p className="data-mono text-lg font-semibold text-ink mt-1">
                {sc.train_compounds.toLocaleString()}
              </p>
              <p className="text-[11px] text-ink-muted">
                {sc.train_scaffolds.toLocaleString()} core scaffolds
              </p>
            </div>
            <div className="bg-canvas border border-hairline rounded p-3">
              <p className="section-kicker">Calibration Val (10%)</p>
              <p className="data-mono text-lg font-semibold text-ink mt-1">
                {sc.val_compounds.toLocaleString()}
              </p>
              <p className="text-[11px] text-ink-muted">
                {sc.val_scaffolds.toLocaleString()} unseen scaffolds
              </p>
            </div>
            <div className="bg-teal-soft border border-teal-border rounded p-3">
              <p className="section-kicker text-teal-academic">Held-Out Test (10%)</p>
              <p className="data-mono text-lg font-semibold text-teal-academic mt-1">
                {sc.test_compounds.toLocaleString()}
              </p>
              <p className="text-[11px] text-teal-academic/80">
                {sc.test_scaffolds.toLocaleString()} unseen scaffolds · 0 overlap
              </p>
            </div>
          </div>
        )}
      </section>

      {/* Table 1: Model Ablation & Calibration Benchmark */}
      <section className="card space-y-5">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-crimson" />
          <div>
            <p className="section-kicker">Table 1 · Held-Out Bemis-Murcko Scaffold Test Set</p>
            <h3 className="font-serif text-lg font-semibold text-ink">
              Multi-Modal Architecture Ablation &amp; Calibration Metrics
            </h3>
          </div>
        </div>

        <div className="space-y-6">
          {Object.entries(metadata.ablation_table).map(([endpointKey, models]) => {
            const epMeta = ML_ENDPOINT_META[endpointKey];
            const weights = metadata.endpoints?.[endpointKey]?.weights;
            return (
              <div key={endpointKey} className="space-y-2">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h4 className="font-sans text-sm font-semibold text-ink">
                    {epMeta?.title ?? endpointKey}{" "}
                    <span className="data-mono text-xs font-normal text-ink-muted">
                      ({epMeta?.target ?? endpointKey})
                    </span>
                  </h4>
                  {weights && (
                    <span className="data-mono text-[10px] text-ink-muted">
                      Val-optimized weights:{" "}
                      {Object.entries(weights)
                        .map(([k, v]) => `${k}=${v.toFixed(2)}`)
                        .join(" · ")}
                    </span>
                  )}
                </div>

                <div className="overflow-x-auto">
                  <table className="academic-table">
                    <thead>
                      <tr>
                        <th>Model Architecture</th>
                        <th className="text-right">ROC-AUC</th>
                        <th className="text-right">PR-AUC</th>
                        <th className="text-right">Brier ↓</th>
                        <th className="text-right">ECE ↓</th>
                        <th className="text-right">F1</th>
                        <th className="text-right">Bal. Acc</th>
                        <th className="text-right">n (Test)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(models).map(([mKey, m]) => {
                        const isEnsemble = mKey === "consensus_ensemble";
                        return (
                          <tr
                            key={mKey}
                            className={
                              isEnsemble
                                ? "bg-navy-soft/70 font-semibold"
                                : "hover:bg-canvas/70"
                            }
                          >
                            <td className="text-ink">
                              {MODEL_MEMBER_LABELS[mKey] ?? mKey}
                            </td>
                            <td className="text-right data-mono text-ink">
                              {m.roc_auc.toFixed(4)}
                            </td>
                            <td className="text-right data-mono text-ink-secondary">
                              {m.pr_auc.toFixed(4)}
                            </td>
                            <td className="text-right data-mono text-ink-secondary">
                              {m.brier_score.toFixed(4)}
                            </td>
                            <td className="text-right data-mono text-ink-secondary">
                              {m.ece.toFixed(4)}
                            </td>
                            <td className="text-right data-mono text-ink-secondary">
                              {m.f1.toFixed(4)}
                            </td>
                            <td className="text-right data-mono text-ink-secondary">
                              {m.balanced_accuracy.toFixed(4)}
                            </td>
                            <td className="text-right data-mono text-ink-muted">
                              {m.n_samples}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Table 2: Clinical Reference Drug Benchmark */}
      {metadata.clinical_benchmark && metadata.clinical_benchmark.length > 0 && (
        <section className="card space-y-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-teal-academic" />
            <div>
              <p className="section-kicker">Table 2 · Clinical Pharmacology Verification</p>
              <h3 className="font-serif text-lg font-semibold text-ink">
                Withdrawn Drugs &amp; Safe Clinical Reference Panel
              </h3>
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="academic-table">
              <thead>
                <tr>
                  <th>Compound</th>
                  <th>Clinical Profile</th>
                  <th className="text-right">hERG P(active)</th>
                  <th className="text-right">CYP3A4 P(active)</th>
                  <th className="text-right">AD (T_max)</th>
                  <th className="text-right">Action</th>
                </tr>
              </thead>
              <tbody>
                {metadata.clinical_benchmark.map((row) => (
                  <tr key={row.name} className="hover:bg-canvas/70">
                    <td className="font-medium text-ink">{row.name}</td>
                    <td className="text-ink-secondary text-[11px]">{row.clinical_class}</td>
                    <td
                      className={`text-right data-mono font-semibold ${
                        row.hERG_prob >= 0.7
                          ? "text-crimson"
                          : row.hERG_prob >= 0.45
                            ? "text-ochre"
                            : "text-teal-academic"
                      }`}
                    >
                      {(row.hERG_prob * 100).toFixed(1)}%{" "}
                      <span className="text-ink-muted font-normal">
                        ±{(row.hERG_std * 100).toFixed(1)}
                      </span>
                    </td>
                    <td
                      className={`text-right data-mono font-semibold ${
                        row.CYP3A4_prob >= 0.7
                          ? "text-crimson"
                          : row.CYP3A4_prob >= 0.45
                            ? "text-ochre"
                            : "text-teal-academic"
                      }`}
                    >
                      {(row.CYP3A4_prob * 100).toFixed(1)}%{" "}
                      <span className="text-ink-muted font-normal">
                        ±{(row.CYP3A4_std * 100).toFixed(1)}
                      </span>
                    </td>
                    <td className="text-right data-mono text-[11px] text-ink-secondary">
                      {row.max_tanimoto.toFixed(2)} ({row.domain_status})
                    </td>
                    <td className="text-right">
                      <button
                        type="button"
                        onClick={() => onAnalyzeDrug(row.smiles, row.name)}
                        className="inline-flex items-center gap-1 text-[11px] font-medium text-navy hover:text-crimson transition-colors"
                      >
                        <FlaskConical className="w-3 h-3" />
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {/* Peer-Reviewed Citations */}
      <section className="card space-y-2.5">
        <p className="section-kicker">Selected Literature Basis</p>
        <ol className="list-decimal list-inside space-y-1.5 text-xs text-ink-secondary leading-relaxed">
          <li>
            Yang, K. et al. <em>Analyzing Learned Molecular Representations for Property Prediction</em> (Chemprop / DMPNN). <strong>J. Chem. Inf. Model.</strong> 2019, 59, 3370–3388.
          </li>
          <li>
            Daina, A.; Zoete, V. <em>A BOILED-Egg to Predict Gastrointestinal Absorption and Brain Penetration of Small Molecules</em>. <strong>ChemMedChem</strong> 2016, 11, 1117–1121.
          </li>
          <li>
            Bickerton, G. R. et al. <em>Quantifying the Chemical Beauty of Drugs</em> (QED). <strong>Nature Chemistry</strong> 2012, 4, 90–98.
          </li>
          <li>
            Baell, J. B.; Holloway, G. A. <em>New Substructure Filters for Removal of Pan Assay Interference Compounds (PAINS)</em>. <strong>J. Med. Chem.</strong> 2010, 53, 2719–2740.
          </li>
          <li>
            Hughes, J. D. et al. <em>Physiochemical Drug Properties Associated with In Vivo Toxicological Outcomes</em> (Pfizer 3/75 Rule). <strong>Bioorg. Med. Chem. Lett.</strong> 2008, 18, 4872–4875.
          </li>
          <li>
            Lovering, F.; Bikker, J.; Humblet, C. <em>Escape from Flatland: Increasing Saturation as an Approach to Improving Clinical Success</em> (Fsp³). <strong>J. Med. Chem.</strong> 2009, 52, 6752–6756.
          </li>
        </ol>
      </section>
    </div>
  );
}
