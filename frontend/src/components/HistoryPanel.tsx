import { Clock, Trash2, ChevronRight } from "lucide-react";
import type { HistoryEntry } from "../hooks/useAnalysisHistory";

interface HistoryPanelProps {
  entries: HistoryEntry[];
  onSelect: (entry: HistoryEntry) => void;
  onClear: () => void;
  isOpen: boolean;
  onToggle: () => void;
}

export default function HistoryPanel({
  entries,
  onSelect,
  onClear,
  isOpen,
  onToggle,
}: HistoryPanelProps) {
  return (
    <>
      <aside
        className={`fixed right-0 top-0 h-full w-80 bg-paper border-l border-hairline
                    z-50 transform transition-transform duration-250 ease-out shadow-elevated
                    ${isOpen ? "translate-x-0" : "translate-x-full"}`}
      >
        <div className="flex flex-col h-full">
          <div className="flex items-center justify-between px-5 py-4 border-b border-hairline">
            <div>
              <p className="section-kicker">Session Log</p>
              <h2 className="font-serif text-base font-semibold text-ink">
                Analysis History
              </h2>
            </div>
            <div className="flex items-center gap-2">
              {entries.length > 0 && (
                <button
                  type="button"
                  onClick={onClear}
                  className="text-ink-muted hover:text-crimson transition-colors p-1"
                  title="Clear history"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              )}
              <button
                type="button"
                onClick={onToggle}
                className="text-ink-muted hover:text-ink transition-colors p-1"
                title="Close panel"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto">
            {entries.length === 0 ? (
              <div className="px-5 py-10 text-center">
                <Clock className="w-7 h-7 text-slate-300 mx-auto mb-2.5" />
                <p className="text-xs font-medium text-ink-secondary">No analyses recorded</p>
                <p className="text-[11px] text-ink-muted mt-1">
                  Completed batches are cached locally for quick recall.
                </p>
              </div>
            ) : (
              <div className="divide-y divide-hairline">
                {entries.map((entry) => (
                  <button
                    key={entry.id}
                    type="button"
                    onClick={() => {
                      onSelect(entry);
                      onToggle();
                    }}
                    className="w-full text-left px-5 py-3.5 hover:bg-canvas transition-colors"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="data-mono text-[10px] text-ink-muted">
                        {new Date(entry.timestamp).toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                      <span className="data-mono text-[10px] text-teal-academic font-medium">
                        {entry.response.results.filter((r) => r.status === "success").length}/
                        {entry.response.results.length} valid
                      </span>
                    </div>
                    <p className="data-mono text-[11px] text-ink truncate">
                      {entry.smiles.slice(0, 2).join(", ")}
                      {entry.smiles.length > 2 ? ` +${entry.smiles.length - 2} more` : ""}
                    </p>
                    <p className="data-mono text-[10px] text-ink-muted mt-0.5">
                      {entry.response.model_version} · {entry.response.prediction_time_ms.toFixed(0)} ms
                    </p>
                  </button>
                ))}
              </div>
            )}
          </div>

          <div className="px-5 py-3 border-t border-hairline bg-canvas">
            <p className="data-mono text-[10px] text-ink-muted">
              Browser localStorage · Retains last 10 runs
            </p>
          </div>
        </div>
      </aside>

      {isOpen && (
        <div
          className="fixed inset-0 bg-ink/20 backdrop-blur-[1px] z-40"
          onClick={onToggle}
        />
      )}
    </>
  );
}
