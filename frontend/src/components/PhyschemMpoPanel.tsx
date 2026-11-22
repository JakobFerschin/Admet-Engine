import {
  Radar,
  RadarChart as RechartsRadar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { CheckCircle2, AlertTriangle, XCircle } from "lucide-react";
import type { CheminformaticsProfile } from "../types/admet";

interface PhyschemMpoPanelProps {
  chem: CheminformaticsProfile;
  compoundName: string;
  overlayChem?: CheminformaticsProfile | null;
  overlayName?: string;
}

/** Normalize key physicochemical properties onto [0, 1] oral drug-like MPO scale (1 = ideal). */
function computeMpoScores(p: CheminformaticsProfile["physchem"]): Record<string, number> {
  const clamp = (v: number) => Math.max(0, Math.min(1, v));
  return {
    "MW ≤500": clamp(1 - Math.max(0, p.mw - 300) / 350),
    "cLogP ≤3": clamp(1 - Math.max(0, Math.abs(p.clogp - 2.0) - 1.0) / 4.0),
    "TPSA": clamp(p.tpsa >= 20 && p.tpsa <= 120 ? 1 : p.tpsa < 20 ? p.tpsa / 20 : 1 - (p.tpsa - 120) / 80),
    "QED": clamp(p.qed),
    "Fsp³": clamp(p.fsp3 / 0.5),
    "RotB ≤10": clamp(1 - Math.max(0, p.rotatable_bonds - 4) / 10),
  };
}

export default function PhyschemMpoPanel({
  chem,
  compoundName,
  overlayChem,
  overlayName,
}: PhyschemMpoPanelProps) {
  const { physchem: p, rules } = chem;
  const primaryMpo = computeMpoScores(p);
  const overlayMpo = overlayChem ? computeMpoScores(overlayChem.physchem) : null;
  const isCompare = Boolean(overlayMpo && overlayName);

  const radarData = Object.keys(primaryMpo).map((axis) => ({
    axis,
    [compoundName]: Number(primaryMpo[axis].toFixed(3)),
    ...(isCompare && overlayMpo && overlayName
      ? { [overlayName]: Number(overlayMpo[axis].toFixed(3)) }
      : {}),
  }));

  const ruleCards = [
    {
      name: "Lipinski Rule of 5",
      citation: "Adv. Drug Deliv. Rev. 1997",
      pass: rules.lipinski_pass,
      warn: false,
      detail:
        rules.lipinski_violations === 0
          ? "0 violations"
          : `${rules.lipinski_violations} violation${rules.lipinski_violations > 1 ? "s" : ""} (${rules.lipinski_details.join(", ")})`,
    },
    {
      name: "Veber Bioavailability",
      citation: "J. Med. Chem. 2002",
      pass: rules.veber_pass,
      warn: false,
      detail: `TPSA ${p.tpsa.toFixed(1)} ≤ 140 Å² · RotB ${p.rotatable_bonds} ≤ 10`,
    },
    {
      name: "Pfizer 3/75 Safety Rule",
      citation: "Hughes et al., BMCL 2008",
      pass: !rules.pfizer_3_75_warning,
      warn: rules.pfizer_3_75_warning,
      detail: rules.pfizer_3_75_warning
        ? `Flagged: cLogP ${p.clogp.toFixed(2)} > 3 & TPSA ${p.tpsa.toFixed(1)} < 75 Å²`
        : "Outside high-promiscuity lipophilic space",
    },
    {
      name: "GSK 4/400 ADMET Space",
      citation: "Gleeson, J. Med. Chem. 2008",
      pass: rules.gsk_4_400_pass,
      warn: false,
      detail: `MW ${p.mw.toFixed(0)} ≤ 400 Da · cLogP ${p.clogp.toFixed(2)} ≤ 4.0`,
    },
  ];

  const descriptorRows = [
    { label: "Molecular Weight", value: `${p.mw.toFixed(2)} Da`, target: "≤ 500" },
    { label: "Wildman-Crippen cLogP", value: p.clogp.toFixed(3), target: "0 – 3.5" },
    { label: "Ertl TPSA", value: `${p.tpsa.toFixed(2)} Å²`, target: "20 – 140" },
    { label: "H-Bond Donors / Acceptors", value: `${p.hbd} / ${p.hba}`, target: "≤ 5 / ≤ 10" },
    { label: "Rotatable Bonds", value: String(p.rotatable_bonds), target: "≤ 10" },
    { label: "QED Drug-Likeness", value: p.qed.toFixed(3), target: "≥ 0.50" },
    { label: "Fsp³ Saturation (Lovering)", value: p.fsp3.toFixed(3), target: "≥ 0.25" },
    { label: "Aromatic Rings / Heavy Atoms", value: `${p.aromatic_rings} / ${p.heavy_atoms}`, target: "≤ 3 rings" },
    { label: "Molar Refractivity", value: `${p.molar_refractivity.toFixed(2)}`, target: "40 – 130" },
    { label: "Bertz Complexity (CT)", value: p.bertz_ct.toFixed(1), target: "Topological" },
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      {/* Left: Physicochemical Readout + Medicinal Chemistry Rules */}
      <div className="card space-y-5">
        <div>
          <p className="section-kicker">Deterministic RDKit Cheminformatics</p>
          <h3 className="font-serif text-lg font-semibold text-ink mt-0.5">
            Physicochemical &amp; Rule Compliance Profile
          </h3>
        </div>

        {/* Rule Compliance 2x2 Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
          {ruleCards.map((rc) => (
            <div
              key={rc.name}
              className={`rounded border px-3 py-2.5 ${
                rc.pass
                  ? "bg-teal-soft/60 border-teal-border"
                  : rc.warn
                    ? "bg-ochre-soft border-ochre-border"
                    : "bg-crimson-soft/70 border-crimson-border"
              }`}
            >
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-ink">{rc.name}</span>
                {rc.pass ? (
                  <CheckCircle2 className="w-3.5 h-3.5 text-teal-academic shrink-0" />
                ) : rc.warn ? (
                  <AlertTriangle className="w-3.5 h-3.5 text-ochre shrink-0" />
                ) : (
                  <XCircle className="w-3.5 h-3.5 text-crimson shrink-0" />
                )}
              </div>
              <p className="text-[10px] text-ink-muted mt-0.5">{rc.citation}</p>
              <p className="data-mono text-[10px] text-ink-secondary mt-1.5 leading-snug">
                {rc.detail}
              </p>
            </div>
          ))}
        </div>

        {/* Quantitative Descriptor Table */}
        <div className="overflow-x-auto">
          <table className="academic-table">
            <thead>
              <tr>
                <th>Descriptor</th>
                <th className="text-right">Value</th>
                <th className="text-right">MedChem Guide</th>
              </tr>
            </thead>
            <tbody>
              {descriptorRows.map((row) => (
                <tr key={row.label} className="hover:bg-canvas/70">
                  <td className="text-ink-secondary font-medium">{row.label}</td>
                  <td className="text-right data-mono font-semibold text-ink">{row.value}</td>
                  <td className="text-right data-mono text-[11px] text-ink-muted">{row.target}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Right: Multiparameter Optimization (MPO) Radar */}
      <div className="card flex flex-col justify-between gap-4" id="mpo-radar-card">
        <div>
          <div className="flex items-start justify-between gap-2">
            <div>
              <p className="section-kicker">Multiparameter Optimization</p>
              <h3 className="font-serif text-lg font-semibold text-ink mt-0.5">
                {isCompare ? `MPO Comparison — ${compoundName} vs. ${overlayName}` : `Oral Drug-Likeness MPO — ${compoundName}`}
              </h3>
            </div>
            <span className="data-mono text-[10px] text-ink-muted bg-subtle px-2 py-0.5 rounded border border-hairline">
              QED {p.qed.toFixed(2)}
            </span>
          </div>
          <p className="text-[11px] text-ink-muted mt-1">
            Normalized desirability across six medicinal chemistry axes (1.00 = optimal oral lead-like space).
            Right-click compound tabs above to overlay a second molecule.
          </p>
        </div>

        <div className="h-[310px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <RechartsRadar data={radarData} cx="50%" cy="50%" outerRadius="74%">
              <PolarGrid stroke="#CBD5E1" strokeDasharray="3 3" />
              <PolarAngleAxis
                dataKey="axis"
                tick={{
                  fill: "#334155",
                  fontSize: 11,
                  fontFamily: '"JetBrains Mono", monospace',
                }}
              />
              <PolarRadiusAxis
                angle={30}
                domain={[0, 1]}
                tick={{
                  fill: "#64748B",
                  fontSize: 9,
                  fontFamily: '"JetBrains Mono", monospace',
                }}
                tickCount={5}
                stroke="#E2E4E9"
              />
              <Radar
                name={compoundName}
                dataKey={compoundName}
                stroke="#A51C30"
                strokeWidth={2}
                fill="#A51C30"
                fillOpacity={0.14}
                dot={{ r: 3, fill: "#A51C30", stroke: "#FFFFFF", strokeWidth: 1 }}
              />
              {isCompare && overlayName && (
                <Radar
                  name={overlayName}
                  dataKey={overlayName}
                  stroke="#1E3A5F"
                  strokeWidth={2}
                  strokeDasharray="5 3"
                  fill="#1E3A5F"
                  fillOpacity={0.1}
                  dot={{ r: 3, fill: "#1E3A5F", stroke: "#FFFFFF", strokeWidth: 1 }}
                />
              )}
              <Tooltip
                contentStyle={{
                  backgroundColor: "#FFFFFF",
                  border: "1px solid #E2E4E9",
                  borderRadius: "6px",
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: "11px",
                  color: "#0F172A",
                  boxShadow: "0 4px 12px rgba(15,23,42,0.08)",
                }}
              />
            </RechartsRadar>
          </ResponsiveContainer>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-hairline text-[10px] font-mono text-ink-muted">
          <div className="flex items-center gap-4">
            <span className="inline-flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-crimson inline-block" />
              {compoundName}
            </span>
            {isCompare && overlayName && (
              <span className="inline-flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-navy inline-block" />
                {overlayName}
              </span>
            )}
          </div>
          <span>QED: Bickerton et al., Nat. Chem. 2012</span>
        </div>
      </div>
    </div>
  );
}
