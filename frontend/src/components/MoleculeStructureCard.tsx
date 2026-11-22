import { useState } from "react";
import { Copy, Check, ShieldAlert, CheckCircle2 } from "lucide-react";
import type { CheminformaticsProfile } from "../types/admet";

interface MoleculeStructureCardProps {
  compoundName: string;
  inputSmiles: string;
  chem: CheminformaticsProfile;
}

/** Format molecular formula with HTML subscripts (e.g. C32H41NO2 -> C₃₂H₄₁NO₂). */
function Formula({ formula }: { formula: string }) {
  const parts = formula.split(/(\d+)/g);
  return (
    <span className="data-mono font-semibold text-ink">
      {parts.map((part, idx) =>
        /^\d+$/.test(part) ? (
          <sub key={idx} className="text-[0.72em]">
            {part}
          </sub>
        ) : (
          <span key={idx}>{part}</span>
        ),
      )}
    </span>
  );
}

export default function MoleculeStructureCard({
  compoundName,
  inputSmiles,
  chem,
}: MoleculeStructureCardProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard?.writeText(chem.identifiers.canonical_smiles);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const hasAlerts = chem.alerts.total_alerts > 0;

  return (
    <div className="card flex flex-col justify-between gap-4">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="section-kicker">2D Standardized Structure</p>
          <h3 className="font-serif text-lg font-semibold text-ink mt-0.5">
            {compoundName}
          </h3>
        </div>
        <div className="text-right">
          <div className="text-sm">
            <Formula formula={chem.identifiers.molecular_formula} />
          </div>
          <p className="data-mono text-[11px] text-ink-muted">
            MW {chem.physchem.mw.toFixed(2)} Da
          </p>
        </div>
      </div>

      {/* Crisp white laboratory structure plate */}
      <div className="border border-hairline rounded bg-white p-3 min-h-[210px] flex items-center justify-center relative">
        {chem.svg_2d ? (
          <div
            className="svg-molecule w-full"
            dangerouslySetInnerHTML={{ __html: chem.svg_2d }}
          />
        ) : (
          <p className="data-mono text-xs text-ink-muted">2D structure unavailable</p>
        )}
        {hasAlerts && (
          <span className="absolute bottom-2 right-2 text-[10px] font-mono text-crimson bg-crimson-soft border border-crimson-border px-2 py-0.5 rounded">
            Highlighted: structural alert atoms
          </span>
        )}
      </div>

      {/* Structural Alerts Summary (PAINS & Brenk) */}
      <div
        className={`rounded border px-3 py-2.5 text-xs ${
          hasAlerts
            ? "bg-crimson-soft border-crimson-border"
            : "bg-teal-soft border-teal-border"
        }`}
      >
        <div className="flex items-center justify-between gap-2">
          <span
            className={`font-semibold inline-flex items-center gap-1.5 ${
              hasAlerts ? "text-crimson" : "text-teal-academic"
            }`}
          >
            {hasAlerts ? (
              <ShieldAlert className="w-3.5 h-3.5" />
            ) : (
              <CheckCircle2 className="w-3.5 h-3.5" />
            )}
            {hasAlerts
              ? `${chem.alerts.total_alerts} Substructure Alert${chem.alerts.total_alerts > 1 ? "s" : ""} Detected`
              : "Clean — 0 PAINS / 0 Brenk Alerts"}
          </span>
          <span className="data-mono text-[10px] text-ink-muted">
            PAINS: {chem.alerts.pains_count} · Brenk: {chem.alerts.brenk_count}
          </span>
        </div>

        {hasAlerts && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {chem.alerts.pains_alerts.map((a) => (
              <span
                key={`p-${a}`}
                className="data-mono text-[10px] px-2 py-0.5 rounded bg-white border border-crimson-border text-crimson"
              >
                PAINS: {a}
              </span>
            ))}
            {chem.alerts.brenk_alerts.map((a) => (
              <span
                key={`b-${a}`}
                className="data-mono text-[10px] px-2 py-0.5 rounded bg-white border border-ochre-border text-ochre"
              >
                Brenk: {a}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Identifiers metadata */}
      <dl className="space-y-2 pt-1 border-t border-hairline text-[11px]">
        <div className="flex items-start justify-between gap-2">
          <dt className="font-mono text-[10px] uppercase tracking-wider text-ink-muted shrink-0 pt-0.5">
            Canonical SMILES
          </dt>
          <dd className="flex items-center gap-1.5 min-w-0">
            <code className="data-mono text-ink-secondary truncate max-w-[260px]" title={chem.identifiers.canonical_smiles}>
              {chem.identifiers.canonical_smiles}
            </code>
            <button
              type="button"
              onClick={handleCopy}
              className="text-ink-muted hover:text-ink shrink-0"
              title="Copy canonical SMILES"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-teal-academic" /> : <Copy className="w-3.5 h-3.5" />}
            </button>
          </dd>
        </div>

        <div className="flex items-center justify-between gap-2">
          <dt className="font-mono text-[10px] uppercase tracking-wider text-ink-muted shrink-0">
            InChIKey
          </dt>
          <dd className="data-mono text-ink-secondary select-all">
            {chem.identifiers.inchi_key}
          </dd>
        </div>

        <div className="flex items-start justify-between gap-2">
          <dt className="font-mono text-[10px] uppercase tracking-wider text-ink-muted shrink-0">
            Bemis-Murcko Core
          </dt>
          <dd
            className="data-mono text-ink-secondary truncate max-w-[280px]"
            title={chem.identifiers.murcko_scaffold}
          >
            {chem.identifiers.murcko_scaffold}
          </dd>
        </div>

        {inputSmiles !== chem.identifiers.canonical_smiles && (
          <p className="text-[10px] text-ink-muted italic">
            Standardized from input SMILES (salt/tautomer canonicalized).
          </p>
        )}
      </dl>
    </div>
  );
}
