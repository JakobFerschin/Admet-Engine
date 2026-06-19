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
      {/* Toggle button */}
      <button
        onClick={onToggle}
        className="fixed right-0 top-20 z-40 bg-surface-raised border border-surface-border 
                   border-r-0 rounded-l-lg px-2 py-3 text-gray-500 hover:text-accent-cyan
                   transition-all duration-200"
        title="Analysis History"
      >
        <Clock className="w-4 h-4" />
      </button>

      {/* Slide-out panel */}
      <aside
        className={`fixed right-0 top-0 h-full w-80 bg-surface border-l border-surface-border 
                     z-50 transform transition-transform duration-300 ease-in-out shadow-2xl
                     ${isOpen ? "translate-x-0" : "translate-x-full"}`}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between px-5 py-4 border-b border-surface-border">
            <h2 className="font-mono text-xs text-gray-300 tracking-widest uppercase">
              Analysis History
            </h2>
            <div className="flex items-center gap-2">
              {entries.length > 0 && (
                <button
                  onClick={onClear}
                  className="text-gray-600 hover:text-accent-rose transition-colors"
                  title="Clear history"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
              <button
                onClick={onToggle}
                className="text-gray-500 hover:text-gray-300 transition-colors"
              >
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Entries */}
          <div className="flex-1 overflow-y-auto">
            {entries.length === 0 ? (
              <div className="px-5 py-8 text-center">
                <Clock className="w-8 h-8 text-gray-700 mx-auto mb-3" />
                <p className="text-xs font-mono text-gray-600">No analyses yet</p>
                <p className="text-[10px] text-gray-700 mt-1">
                  Run a prediction to see it here
                </p>
              </div>
            ) : (
              <div className="divide-y divide-surface-border">
                {entries.map((entry) => (
                  <button
                    key={entry.id}
                    onClick={() => onSelect(entry)}
                    className="w-full text-left px-5 py-3 hover:bg-surface-overlay transition-colors group"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-[10px] text-gray-500">
                        {new Date(entry.timestamp).toLocaleTimeString("de-AT", {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                      <span className="font-mono text-[10px] text-gray-600">
                        {entry.response.results.filter((r) => r.status === "success").length}
                        /{entry.response.results.length}
                      </span>
                    </div>
                    <p className="font-mono text-[11px] text-gray-400 truncate group-hover:text-gray-200 transition-colors">
                      {entry.smiles.slice(0, 3).join(", ")}
                      {entry.smiles.length > 3 ? ` +${entry.smiles.length - 3} more` : ""}
                    </p>
                    <p className="font-mono text-[10px] text-gray-600 mt-0.5">
                      {entry.response.model_version} · {entry.response.prediction_time_ms.toFixed(0)}ms
                    </p>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-5 py-3 border-t border-surface-border">
            <p className="text-[10px] font-mono text-gray-600">
              Stored in localStorage · Max {10} entries
            </p>
          </div>
        </div>
      </aside>

      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-45"
          onClick={onToggle}
        />
      )}
    </>
  );
}
