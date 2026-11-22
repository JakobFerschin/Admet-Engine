import { type ReactNode } from "react";
import { FlaskConical, BarChart3, Clock } from "lucide-react";

export type WorkbenchTab = "workbench" | "validation";

interface DashboardLayoutProps {
  children: ReactNode;
  activeTab: WorkbenchTab;
  onTabChange: (tab: WorkbenchTab) => void;
  online: boolean;
  modelVersion?: string;
  historyCount: number;
  onToggleHistory: () => void;
}

export default function DashboardLayout({
  children,
  activeTab,
  onTabChange,
  online,
  modelVersion,
  historyCount,
  onToggleHistory,
}: DashboardLayoutProps) {
  return (
    <div className="min-h-screen bg-canvas flex flex-col">
      {/* Subtle Harvard Crimson institutional top rule */}
      <div className="h-[3px] w-full bg-crimson" />

      {/* Academic Masthead */}
      <header className="bg-paper border-b border-hairline sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="flex items-baseline gap-2.5">
              <h1 className="font-serif text-xl font-semibold tracking-tight text-ink">
                ADMET-Engine
              </h1>
              <span className="hidden sm:inline font-mono text-[10px] uppercase tracking-[0.14em] text-ink-muted">
                Computational Medicinal Chemistry
              </span>
            </div>

            <nav className="flex items-center gap-1 ml-2 pl-4 border-l border-hairline">
              <button
                type="button"
                onClick={() => onTabChange("workbench")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  activeTab === "workbench"
                    ? "bg-ink text-white"
                    : "text-ink-secondary hover:bg-subtle hover:text-ink"
                }`}
              >
                <FlaskConical className="w-3.5 h-3.5" />
                Workbench
              </button>
              <button
                type="button"
                onClick={() => onTabChange("validation")}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium transition-colors ${
                  activeTab === "validation"
                    ? "bg-ink text-white"
                    : "text-ink-secondary hover:bg-subtle hover:text-ink"
                }`}
              >
                <BarChart3 className="w-3.5 h-3.5" />
                Validation &amp; Methods
              </button>
            </nav>
          </div>

          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={onToggleHistory}
              className="btn-secondary flex items-center gap-1.5 py-1.5"
              title="Analysis History"
            >
              <Clock className="w-3.5 h-3.5 text-ink-muted" />
              <span className="hidden sm:inline">History</span>
              {historyCount > 0 && (
                <span className="data-mono text-[10px] bg-subtle text-ink-secondary px-1.5 py-0.5 rounded">
                  {historyCount}
                </span>
              )}
            </button>

            <div className="hidden md:flex items-center gap-2 pl-3 border-l border-hairline">
              <span
                className={`w-2 h-2 rounded-full ${
                  online ? "bg-teal-academic" : "bg-crimson"
                }`}
              />
              <div className="leading-tight">
                <p className="data-mono text-[10px] font-medium text-ink">
                  {online ? "API READY" : "OFFLINE"}
                </p>
                {modelVersion && (
                  <p className="data-mono text-[9px] text-ink-muted">
                    {modelVersion}
                  </p>
                )}
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-7">
        {children}
      </main>

      {/* Academic Footer */}
      <footer className="border-t border-hairline bg-paper px-6 py-3.5 mt-8">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2 text-[11px] text-ink-muted">
          <span className="font-serif italic text-ink-secondary">
            ADMET-Engine v2.0 — Preclinical Safety &amp; Pharmacokinetics Platform
          </span>
          <span className="data-mono text-[10px]">
            ChEMBL v37 · Bemis-Murcko Scaffold Split · DMPNN + ECFP4 Ensemble · RDKit
          </span>
        </div>
      </footer>
    </div>
  );
}
