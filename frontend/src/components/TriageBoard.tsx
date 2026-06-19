import { AlertTriangle, CheckCircle2, ShieldAlert } from "lucide-react";
import type { EndpointLabel } from "../types/admet";
import { ENDPOINT_LABELS, ENDPOINT_DISPLAY, TRIAGE_THRESHOLD } from "../types/admet";

interface TriageBoardProps {
  predictions: Record<string, number>;
  moleculeLabel: string;
}

function getTriage(value: number) {
  if (value >= TRIAGE_THRESHOLD.HIGH)
    return { level: "high" as const, icon: ShieldAlert, colour: "text-accent-rose", bg: "bg-accent-rose/10", border: "border-accent-rose/30" };
  if (value >= TRIAGE_THRESHOLD.MEDIUM)
    return { level: "medium" as const, icon: AlertTriangle, colour: "text-accent-amber", bg: "bg-accent-amber/10", border: "border-accent-amber/30" };
  return { level: "low" as const, icon: CheckCircle2, colour: "text-accent-emerald", bg: "bg-accent-emerald/10", border: "border-accent-emerald/30" };
}

export default function TriageBoard({ predictions, moleculeLabel }: TriageBoardProps) {
  // Count per triage level
  const counts = { high: 0, medium: 0, low: 0 };
  ENDPOINT_LABELS.forEach((key) => {
    const v = predictions[key];
    if (v == null) return;
    if (v >= TRIAGE_THRESHOLD.HIGH) counts.high++;
    else if (v >= TRIAGE_THRESHOLD.MEDIUM) counts.medium++;
    else counts.low++;
  });

  return (
    <div className="card">
      {/* Header with counts */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-mono text-xs text-gray-400 tracking-widest uppercase">
          Triage — {moleculeLabel}
        </h3>
        <div className="flex gap-3 text-[10px] font-mono">
          <span className="text-accent-rose flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-accent-rose inline-block" />
            {counts.high}
          </span>
          <span className="text-accent-amber flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-accent-amber inline-block" />
            {counts.medium}
          </span>
          <span className="text-accent-emerald flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-accent-emerald inline-block" />
            {counts.low}
          </span>
        </div>
      </div>

      {/* Endpoint rows */}
      <div className="grid grid-cols-2 gap-1.5">
        {ENDPOINT_LABELS.map((key: EndpointLabel) => {
          const value = predictions[key];
          const displayName = ENDPOINT_DISPLAY[key];
          const triage = value != null ? getTriage(value) : null;

          return (
            <div
              key={key}
              className={`flex items-center justify-between px-3 py-2 rounded-md border
                ${triage ? `${triage.bg} ${triage.border}` : "bg-surface-overlay border-surface-border opacity-50"}
                transition-colors duration-200`}
            >
              <span className="text-[11px] font-mono text-gray-300 truncate mr-2">
                {displayName}
              </span>
              <span className={`text-xs font-mono font-bold tabular-nums ${triage?.colour ?? "text-gray-600"}`}>
                {value != null ? value.toFixed(4) : "—"}
              </span>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex gap-4 mt-4 pt-3 border-t border-surface-border">
        <span className="text-[10px] font-mono text-accent-rose flex items-center gap-1">
          <ShieldAlert className="w-3 h-3" /> ≥ 0.8
        </span>
        <span className="text-[10px] font-mono text-accent-amber flex items-center gap-1">
          <AlertTriangle className="w-3 h-3" /> ≥ 0.5
        </span>
        <span className="text-[10px] font-mono text-accent-emerald flex items-center gap-1">
          <CheckCircle2 className="w-3 h-3" /> &lt; 0.5
        </span>
      </div>
    </div>
  );
}
