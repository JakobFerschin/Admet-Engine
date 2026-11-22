import { useState, useCallback } from "react";
import { GitCompare, X, AlertCircle } from "lucide-react";
import { useAdmetPrediction } from "./hooks/useAdmetPrediction";
import { useAnalysisHistory } from "./hooks/useAnalysisHistory";
import type { HistoryEntry } from "./hooks/useAnalysisHistory";
import DashboardLayout, { type WorkbenchTab } from "./components/DashboardLayout";
import SmilesInput from "./components/SmilesInput";
import MoleculeStructureCard from "./components/MoleculeStructureCard";
import EndpointPredictionPanel from "./components/EndpointPredictionPanel";
import BoiledEggChart from "./components/BoiledEggChart";
import PhyschemMpoPanel from "./components/PhyschemMpoPanel";
import ModelValidationView from "./components/ModelValidationView";
import PdfExportButton from "./components/PdfExportButton";
import HistoryPanel from "./components/HistoryPanel";

export default function App() {
  const { predict, loading, error, data, restoreEntry, health, metadata } =
    useAdmetPrediction();
  const { entries, addEntry, clearHistory } = useAnalysisHistory();

  const [activeTab, setActiveTab] = useState<WorkbenchTab>("workbench");
  const [inputSmiles, setInputSmiles] = useState<string[]>([]);
  const [compoundNames, setCompoundNames] = useState<string[]>([]);
  const [selectedMolecule, setSelectedMolecule] = useState(0);
  const [compareIndex, setCompareIndex] = useState<number | null>(null);
  const [historyOpen, setHistoryOpen] = useState(false);

  const handleSubmit = async (smilesList: string[], names?: string[]) => {
    setActiveTab("workbench");
    setInputSmiles(smilesList);
    setCompoundNames(names ?? smilesList.map((_, i) => `Compound ${i + 1}`));
    setSelectedMolecule(0);
    setCompareIndex(null);
    try {
      const result = await predict(smilesList);
      addEntry(smilesList, result);
    } catch {
      // Captured in hook error state
    }
  };

  const handleHistorySelect = useCallback(
    (entry: HistoryEntry) => {
      setActiveTab("workbench");
      setInputSmiles(entry.smiles);
      setCompoundNames(entry.smiles.map((_, i) => `Compound ${i + 1}`));
      setSelectedMolecule(0);
      setCompareIndex(null);
      restoreEntry(entry.response);
    },
    [restoreEntry],
  );

  const handleInspectFromValidation = (smiles: string, name: string) => {
    handleSubmit([smiles], [name]);
  };

  const indexedResults =
    data?.results.map((r, idx) => ({
      result: r,
      label: compoundNames[idx] || `Compound ${idx + 1}`,
    })) ?? [];

  const successItems = indexedResults.filter((x) => x.result.status === "success");
  const errorItems = indexedResults.filter((x) => x.result.status === "error");

  const safeIdx = Math.min(selectedMolecule, Math.max(0, successItems.length - 1));
  const currentItem = successItems[safeIdx];
  const overlayItem =
    compareIndex != null && compareIndex !== safeIdx ? successItems[compareIndex] : null;

  return (
    <DashboardLayout
      activeTab={activeTab}
      onTabChange={setActiveTab}
      online={Boolean(health?.model_loaded)}
      modelVersion={health?.model_version ?? data?.model_version}
      historyCount={entries.length}
      onToggleHistory={() => setHistoryOpen((p) => !p)}
    >
      {activeTab === "validation" ? (
        <ModelValidationView
          metadata={metadata}
          onAnalyzeDrug={handleInspectFromValidation}
        />
      ) : (
        <div className="space-y-6 animate-fade-in">
          {/* Input Panel */}
          <SmilesInput onSubmit={handleSubmit} loading={loading} />

          {/* Error Banner */}
          {error && (
            <div className="card border-crimson-border bg-crimson-soft">
              <div className="flex items-start gap-3">
                <AlertCircle className="w-4 h-4 text-crimson mt-0.5 shrink-0" />
                <div className="text-xs space-y-1">
                  <p className="font-semibold text-crimson">{error.message}</p>
                  {error.status === 0 && (
                    <p className="text-ink-secondary">
                      Launch the FastAPI server from the repository root:{" "}
                      <code className="data-mono bg-white px-1.5 py-0.5 rounded border border-crimson-border">
                        uvicorn src.api.main:app --port 8000
                      </code>
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Loading State */}
          {loading && (
            <div className="card py-12 text-center space-y-3">
              <div className="inline-block w-6 h-6 border-2 border-hairline border-t-crimson rounded-full animate-spin" />
              <p className="font-serif text-base text-ink">
                Running multi-modal inference &amp; cheminformatics profiling…
              </p>
              <p className="data-mono text-[11px] text-ink-muted">
                {inputSmiles.length} compound{inputSmiles.length !== 1 ? "s" : ""} · DMPNN + ECFP4/PhysChem Ensemble + RDKit
              </p>
            </div>
          )}

          {/* Initial Academic Overview when no batch has been submitted yet */}
          {!data && !loading && (
            <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="card space-y-1.5">
                <p className="section-kicker">01 · Graph &amp; QSAR Consensus</p>
                <h3 className="font-serif text-base font-semibold text-ink">
                  Scaffold-Validated ChEMBL v37 Ensemble
                </h3>
                <p className="text-xs text-ink-secondary leading-relaxed">
                  Combines a 45-dim one-hot Chemprop DMPNN, Residual ECFP4+PhysChem MLP,
                  ExtraTrees, and L2 Logistic QSAR — temperature-calibrated on unseen
                  Bemis-Murcko scaffolds with 95% epistemic uncertainty intervals.
                </p>
              </div>
              <div className="card space-y-1.5">
                <p className="section-kicker">02 · Applicability Domain</p>
                <h3 className="font-serif text-base font-semibold text-ink">
                  Tanimoto Analog Verification
                </h3>
                <p className="text-xs text-ink-secondary leading-relaxed">
                  Every prediction reports exact maximum ECFP4 Tanimoto similarity against the
                  ChEMBL v37 training corpus and surfaces the closest experimental training
                  analog so out-of-domain extrapolations are immediately flagged.
                </p>
              </div>
              <div className="card space-y-1.5">
                <p className="section-kicker">03 · Medicinal Chemistry Suite</p>
                <h3 className="font-serif text-base font-semibold text-ink">
                  BOILED-Egg, QED &amp; Toxicophore Alerts
                </h3>
                <p className="text-xs text-ink-secondary leading-relaxed">
                  Deterministic RDKit profiling computes 2D structures with PAINS/Brenk atom
                  highlighting, exact Daina &amp; Zoete BOILED-Egg permeation ellipses, Lipinski
                  Ro5, Veber, Pfizer 3/75, GSK 4/400, QED, and Fsp³.
                </p>
              </div>
            </section>
          )}

          {/* Results Workspace */}
          {data && !loading && successItems.length > 0 && currentItem && (
            <div className="space-y-5">
              {/* Batch Summary Bar + Compound Selector */}
              <div className="card py-3.5 px-4 flex flex-wrap items-center justify-between gap-3">
                <div className="flex items-center gap-2 flex-wrap">
                  {successItems.map((item, idx) => {
                    const isSelected = idx === safeIdx;
                    const isCompared = idx === compareIndex;
                    return (
                      <button
                        key={idx}
                        type="button"
                        onClick={() => setSelectedMolecule(idx)}
                        onContextMenu={(e) => {
                          e.preventDefault();
                          setCompareIndex((prev) => (prev === idx ? null : idx));
                        }}
                        title={`${item.result.smiles}\nRight-click to overlay on MPO radar`}
                        className={`px-3 py-1.5 rounded text-xs font-medium transition-colors border ${
                          isSelected
                            ? "bg-ink text-white border-ink"
                            : isCompared
                              ? "bg-navy-soft text-navy border-navy-border"
                              : "bg-canvas text-ink-secondary border-hairline hover:bg-subtle"
                        }`}
                      >
                        {item.label}
                        {isCompared && (
                          <span className="ml-1.5 data-mono text-[9px] uppercase opacity-80">
                            [Compare]
                          </span>
                        )}
                      </button>
                    );
                  })}

                  {overlayItem && (
                    <button
                      type="button"
                      onClick={() => setCompareIndex(null)}
                      className="inline-flex items-center gap-1 text-[11px] text-navy bg-navy-soft px-2 py-1 rounded border border-navy-border"
                    >
                      <GitCompare className="w-3 h-3" />
                      Overlay: {overlayItem.label}
                      <X className="w-3 h-3" />
                    </button>
                  )}
                </div>

                <div className="flex items-center gap-3">
                  <span className="data-mono text-[11px] text-ink-muted">
                    {data.model_version} · {data.prediction_time_ms.toFixed(1)} ms ·{" "}
                    {successItems.length}/{data.results.length} valid
                  </span>
                  <PdfExportButton response={data} compoundNames={compoundNames} />
                </div>
              </div>

              {/* Row 1: 2D Structure & Alerts + Calibrated ChEMBL ML Predictions & AD */}
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
                <div className="lg:col-span-5">
                  {currentItem.result.cheminformatics && (
                    <MoleculeStructureCard
                      compoundName={currentItem.label}
                      inputSmiles={currentItem.result.smiles}
                      chem={currentItem.result.cheminformatics}
                    />
                  )}
                </div>
                <div className="lg:col-span-7">
                  {currentItem.result.ml_endpoints ? (
                    <EndpointPredictionPanel
                      mlEndpoints={currentItem.result.ml_endpoints}
                      applicabilityDomain={currentItem.result.applicability_domain}
                      metadata={metadata}
                    />
                  ) : (
                    <div className="card">
                      <p className="text-xs text-ink-muted">
                        Legacy model active — restart backend with ChEMBL v37 ensemble bundle.
                      </p>
                    </div>
                  )}
                </div>
              </div>

              {/* Row 2: Analytical BOILED-Egg Permeation Map */}
              <BoiledEggChart
                molecules={successItems.map((x) => x.result)}
                labels={successItems.map((x) => x.label)}
                selectedIndex={safeIdx}
                onSelect={setSelectedMolecule}
              />

              {/* Row 3: Physicochemical Readout + Rule Compliance + MPO Comparison Radar */}
              {currentItem.result.cheminformatics && (
                <PhyschemMpoPanel
                  chem={currentItem.result.cheminformatics}
                  compoundName={currentItem.label}
                  overlayChem={overlayItem?.result.cheminformatics}
                  overlayName={overlayItem?.label}
                />
              )}

              {/* Partial Errors */}
              {errorItems.length > 0 && (
                <details className="card border-ochre-border bg-ochre-soft">
                  <summary className="text-xs font-semibold text-ochre cursor-pointer">
                    {errorItems.length} invalid SMILES entry excluded from batch — click to inspect
                  </summary>
                  <div className="mt-2.5 space-y-1">
                    {errorItems.map((item, i) => (
                      <p key={i} className="data-mono text-[11px] text-ink-secondary">
                        <span className="font-semibold">{item.result.smiles}</span> —{" "}
                        <span className="text-crimson">{item.result.error}</span>
                      </p>
                    ))}
                  </div>
                </details>
              )}
            </div>
          )}

          {/* All Failed State */}
          {data && !loading && successItems.length === 0 && (
            <div className="card border-crimson-border bg-crimson-soft space-y-1.5">
              <p className="text-sm font-semibold text-crimson">
                All {data.results.length} submitted SMILES failed RDKit standardization.
              </p>
              {errorItems.map((item, i) => (
                <p key={i} className="data-mono text-xs text-ink-secondary">
                  {item.result.smiles} → {item.result.error}
                </p>
              ))}
            </div>
          )}
        </div>
      )}

      <HistoryPanel
        entries={entries}
        onSelect={handleHistorySelect}
        onClear={clearHistory}
        isOpen={historyOpen}
        onToggle={() => setHistoryOpen((p) => !p)}
      />
    </DashboardLayout>
  );
}
