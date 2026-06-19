import { useState, useCallback } from "react";
import { GitCompare, X } from "lucide-react";
import { useAdmetPrediction } from "./hooks/useAdmetPrediction";
import { useAnalysisHistory } from "./hooks/useAnalysisHistory";
import type { HistoryEntry } from "./hooks/useAnalysisHistory";
import DashboardLayout from "./components/DashboardLayout";
import SmilesInput from "./components/SmilesInput";
import RadarChart from "./components/RadarChart";
import TriageBoard from "./components/TriageBoard";
import PdfExportButton from "./components/PdfExportButton";
import HistoryPanel from "./components/HistoryPanel";
import type { AdmetResponse } from "./types/admet";

const CHART_COLOURS = ["#22d3ee", "#a78bfa", "#f59e0b", "#f43f5e", "#10b981"];

export default function App() {
  const { predict, loading, error, data, restoreEntry } = useAdmetPrediction();
  const { entries, addEntry, clearHistory, loadEntry } = useAnalysisHistory();

  const [inputSmiles, setInputSmiles] = useState<string[]>([]);
  const [selectedMolecule, setSelectedMolecule] = useState(0);
  // Compare mode: up to 2 indices
  const [compareSlots, setCompareSlots] = useState<number[]>([]);
  // History panel toggle
  const [historyOpen, setHistoryOpen] = useState(false);

  const handleSubmit = async (smilesList: string[]) => {
    setInputSmiles(smilesList);
    setSelectedMolecule(0);
    setCompareSlots([]);
    try {
      const result = await predict(smilesList);
      addEntry(smilesList, result);
    } catch {
      // error state captured by hook
    }
  };

  const handleHistorySelect = useCallback(
    (entry: HistoryEntry) => {
      setInputSmiles(entry.smiles);
      setSelectedMolecule(0);
      setCompareSlots([]);
      restoreEntry(entry.response);
    },
    [restoreEntry],
  );

  const toggleCompare = (molIndex: number) => {
    setCompareSlots((prev) => {
      if (prev.includes(molIndex)) {
        return prev.filter((i) => i !== molIndex);
      }
      if (prev.length >= 2) {
        // Replace the oldest slot
        return [prev[1], molIndex];
      }
      return [...prev, molIndex];
    });
  };

  const successResults = data?.results.filter((r) => r.status === "success") ?? [];
  const errorResults = data?.results.filter((r) => r.status === "error") ?? [];
  const currentMolecule = successResults[Math.min(selectedMolecule, successResults.length - 1)];

  // Build compare data for RadarChart overlay
  const compareMoleculeA = compareSlots.length >= 1 ? successResults[compareSlots[0]] : null;
  const compareMoleculeB = compareSlots.length >= 2 ? successResults[compareSlots[1]] : null;

  return (
    <DashboardLayout>
      <div className="space-y-6 animate-fade-in">
        {/* ── Input Section ───────────────────────────────── */}
        <SmilesInput onSubmit={handleSubmit} loading={loading} />

        {/* ── Error Banner ─────────────────────────────────── */}
        {error && (
          <div className="card border-accent-rose/30 bg-accent-rose/5">
            <div className="flex items-start gap-3">
              <span className="font-mono text-accent-rose text-sm mt-0.5">⚠</span>
              <div>
                <p className="font-mono text-sm text-accent-rose">{error.message}</p>
                {error.status === 0 && (
                  <p className="text-xs text-gray-500 mt-1 font-mono">
                    Ensure the FastAPI backend is running:{" "}
                    <code className="text-gray-400">uvicorn src.api.main:app --port 8000</code>
                  </p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ── Loading State ────────────────────────────────── */}
        {loading && (
          <div className="card border-dashed border-surface-border/50 py-12 text-center">
            <div className="inline-block w-6 h-6 border-2 border-accent-cyan/30 border-t-accent-cyan rounded-full animate-spin mb-3" />
            <p className="font-mono text-sm text-gray-400">
              Running inference on {inputSmiles.length} molecule{inputSmiles.length !== 1 ? "s" : ""}…
            </p>
          </div>
        )}

        {/* ── Results ──────────────────────────────────────── */}
        {data && !loading && successResults.length > 0 && (
          <div className="space-y-4">
            {/* ── Meta bar ──────────────────────────────────── */}
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="text-[10px] font-mono text-gray-600">
                <span>
                  Model: <span className="text-gray-400">{data.model_version}</span>
                </span>
                <span className="ml-4">
                  {data.prediction_time_ms.toFixed(1)} ms · {successResults.length}/{data.results.length} succeeded
                </span>
              </div>
              <PdfExportButton response={data} />
            </div>

            {/* ── Molecule tabs + Compare ──────────────────── */}
            <div className="flex items-center gap-1.5 flex-wrap">
              {successResults.map((r, i) => {
                const isSelected = i === selectedMolecule;
                const isCompared = compareSlots.includes(i);
                const slotLabel = compareSlots[0] === i ? "A" : compareSlots[1] === i ? "B" : null;
                return (
                  <button
                    key={i}
                    onClick={() => setSelectedMolecule(i)}
                    onContextMenu={(e) => {
                      e.preventDefault();
                      toggleCompare(i);
                    }}
                    className={`relative px-3 py-1.5 rounded-md font-mono text-[11px] transition-all duration-150
                      ${isSelected
                        ? "bg-accent-cyan/15 border border-accent-cyan/40 text-accent-cyan"
                        : isCompared
                          ? "bg-accent-violet/10 border border-accent-violet/30 text-accent-violet"
                          : "bg-surface-overlay border border-surface-border text-gray-500 hover:border-gray-600 hover:text-gray-300"
                      }`}
                    title={`${r.smiles}\nRight-click to toggle compare (${isCompared ? "remove" : "add"})`}
                  >
                    {slotLabel && (
                      <span className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full 
                                       bg-accent-violet text-[9px] font-bold text-white flex items-center justify-center">
                        {slotLabel}
                      </span>
                    )}
                    Mol {i + 1}
                  </button>
                );
              })}

              {/* Compare mode controls */}
              {compareSlots.length > 0 && (
                <div className="flex items-center gap-1.5 ml-2">
                  <span className="text-[10px] font-mono text-accent-violet flex items-center gap-1">
                    <GitCompare className="w-3 h-3" />
                    Compare ({compareSlots.length}/2)
                  </span>
                  <button
                    onClick={() => setCompareSlots([])}
                    className="text-gray-600 hover:text-gray-400 transition-colors"
                  >
                    <X className="w-3 h-3" />
                  </button>
                </div>
              )}
            </div>

            {/* ── SMILES display ────────────────────────────── */}
            <div className="bg-surface-overlay border border-surface-border rounded-lg px-4 py-2.5 font-mono text-xs text-gray-400 break-all">
              {currentMolecule?.smiles}
            </div>

            {/* ── Compare mode hint ─────────────────────────── */}
            {compareSlots.length >= 1 && (
              <div className="text-[10px] font-mono text-gray-600 flex items-center gap-2">
                <GitCompare className="w-3 h-3 text-accent-violet" />
                <span>
                  Compare active —{" "}
                  <span className="text-accent-cyan">Cyan</span> = Mol {compareSlots[0] + 1}
                  {compareSlots[1] != null && (
                    <>
                      {" "}| <span className="text-fuchsia-400">Magenta</span> = Mol {compareSlots[1] + 1}
                    </>
                  )}
                </span>
                <span className="text-gray-700 ml-2">(Right-click tabs to toggle)</span>
              </div>
            )}

            {/* ── Charts + Triage grid ──────────────────────── */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {/* Compare mode: overlay two polygons */}
              {compareSlots.length >= 2 && compareMoleculeA && compareMoleculeB ? (
                <RadarChart
                  predictions={compareMoleculeA.predictions ?? {}}
                  moleculeLabel={`Mol ${compareSlots[0] + 1}`}
                  colour="#22d3ee"
                  overlayPredictions={compareMoleculeB.predictions ?? {}}
                  overlayLabel={`Mol ${compareSlots[1] + 1}`}
                  overlayColour="#d946ef"
                />
              ) : compareSlots.length === 1 && compareMoleculeA ? (
                <RadarChart
                  predictions={compareMoleculeA.predictions ?? {}}
                  moleculeLabel={`Mol ${compareSlots[0] + 1} (A)`}
                  colour="#22d3ee"
                />
              ) : (
                <RadarChart
                  predictions={currentMolecule?.predictions ?? {}}
                  moleculeLabel={`Mol ${selectedMolecule + 1}`}
                  colour={CHART_COLOURS[selectedMolecule % CHART_COLOURS.length]}
                />
              )}
              <TriageBoard
                predictions={currentMolecule?.predictions ?? {}}
                moleculeLabel={`Mol ${selectedMolecule + 1}`}
              />
            </div>

            {/* ── Error molecules summary ───────────────────── */}
            {errorResults.length > 0 && (
              <details className="card border-accent-amber/20 bg-accent-amber/5">
                <summary className="font-mono text-xs text-accent-amber cursor-pointer">
                  {errorResults.length} molecule{errorResults.length !== 1 ? "s" : ""} failed — click for details
                </summary>
                <div className="mt-2 space-y-1">
                  {errorResults.map((r, i) => (
                    <p key={i} className="text-[11px] font-mono text-gray-500">
                      <span className="text-gray-400">{r.smiles.slice(0, 50)}…</span>
                      <span className="text-accent-rose ml-2">{r.error}</span>
                    </p>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}

        {/* ── All-failed state ──────────────────────────────── */}
        {data && !loading && successResults.length === 0 && (
          <div className="card border-accent-rose/30 bg-accent-rose/5">
            <p className="font-mono text-sm text-accent-rose">
              All {data.results.length} molecule{data.results.length !== 1 ? "s" : ""} failed processing.
            </p>
            {errorResults.map((r, i) => (
              <p key={i} className="text-xs font-mono text-gray-500 mt-1">
                {r.smiles.slice(0, 40)}… → {r.error}
              </p>
            ))}
          </div>
        )}
      </div>

      {/* ── History Panel ───────────────────────────────────── */}
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
