import { useMemo } from "react";
import type { MoleculePrediction } from "../types/admet";

interface BoiledEggChartProps {
  molecules: MoleculePrediction[];
  labels: string[];
  selectedIndex: number;
  onSelect: (index: number) => void;
}

/**
 * Exact BOILED-Egg analytical ellipses (Daina & Zoete, ChemMedChem 2016, 11, 1117-1121):
 *   White (HIA):  center (71.051, 2.292), semi-axes (71.0405, 4.370), angle -1.031325 deg
 *   Yolk  (BBB):  center (38.117, 3.177), semi-axes (41.0305, 2.7785), angle +0.171887 deg
 */
function sampleEllipsePath(
  cx: number,
  cy: number,
  a: number,
  b: number,
  angleDeg: number,
  mapX: (tpsa: number) => number,
  mapY: (wlogp: number) => number,
  steps = 96,
): string {
  const rad = (angleDeg * Math.PI) / 180;
  const cos = Math.cos(rad);
  const sin = Math.sin(rad);
  const pts: string[] = [];
  for (let i = 0; i <= steps; i++) {
    const t = (2 * Math.PI * i) / steps;
    const lx = a * Math.cos(t);
    const ly = b * Math.sin(t);
    const x = cx + lx * cos - ly * sin;
    const y = cy + lx * sin + ly * cos;
    pts.push(`${i === 0 ? "M" : "L"}${mapX(x).toFixed(2)},${mapY(y).toFixed(2)}`);
  }
  return pts.join(" ") + " Z";
}

export default function BoiledEggChart({
  molecules,
  labels,
  selectedIndex,
  onSelect,
}: BoiledEggChartProps) {
  const W = 480;
  const H = 310;
  const pad = { top: 20, right: 20, bottom: 44, left: 52 };
  const plotW = W - pad.left - pad.right;
  const plotH = H - pad.top - pad.bottom;

  const tpsaMin = -10;
  const tpsaMax = 200;
  const logpMin = -3.5;
  const logpMax = 8.0;

  const mapX = (tpsa: number) =>
    pad.left + ((Math.max(tpsaMin, Math.min(tpsaMax, tpsa)) - tpsaMin) / (tpsaMax - tpsaMin)) * plotW;
  const mapY = (wlogp: number) =>
    pad.top + ((logpMax - Math.max(logpMin, Math.min(logpMax, wlogp))) / (logpMax - logpMin)) * plotH;

  const { hiaPath, bbbPath } = useMemo(() => {
    return {
      hiaPath: sampleEllipsePath(71.051, 2.292, 71.0405, 4.37, -1.031325, mapX, mapY),
      bbbPath: sampleEllipsePath(38.117, 3.177, 41.0305, 2.7785, 0.171887, mapX, mapY),
    };
  }, []);

  const currentEgg = molecules[selectedIndex]?.cheminformatics?.boiled_egg;
  const xTicks = [0, 40, 80, 120, 160, 200];
  const yTicks = [-2, 0, 2, 4, 6, 8];

  return (
    <div className="card space-y-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="section-kicker">Analytical Pharmacokinetics</p>
          <h3 className="font-serif text-lg font-semibold text-ink mt-0.5">
            BOILED-Egg Permeation Map
          </h3>
        </div>
        {currentEgg && (
          <span
            className={`data-mono text-[10px] font-semibold px-2 py-0.5 rounded border ${
              currentEgg.region === "yolk"
                ? "bg-ochre-soft text-ochre border-ochre-border"
                : currentEgg.region === "white"
                  ? "bg-teal-soft text-teal-academic border-teal-border"
                  : "bg-subtle text-ink-secondary border-hairline"
            }`}
          >
            {currentEgg.classification}
          </span>
        )}
      </div>

      <p className="text-[11px] text-ink-muted">
        Daina &amp; Zoete (<span className="italic">ChemMedChem</span> 2016) WLOGP vs. Ertl TPSA
        ellipses predicting passive gastrointestinal absorption (White) and blood-brain barrier
        permeation (Yolk).
      </p>

      <div className="bg-white border border-hairline rounded p-2">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
          {/* Plot background (outside absorption zone) */}
          <rect
            x={pad.left}
            y={pad.top}
            width={plotW}
            height={plotH}
            fill="#F1F5F9"
            stroke="#CBD5E1"
            strokeWidth="1"
          />

          {/* Gridlines */}
          {xTicks.map((t) => (
            <g key={`x-${t}`}>
              <line
                x1={mapX(t)}
                y1={pad.top}
                x2={mapX(t)}
                y2={pad.top + plotH}
                stroke="#E2E8F0"
                strokeDasharray="2 3"
              />
              <text
                x={mapX(t)}
                y={pad.top + plotH + 15}
                textAnchor="middle"
                className="fill-slate-500 text-[10px] font-mono"
              >
                {t}
              </text>
            </g>
          ))}
          {yTicks.map((v) => (
            <g key={`y-${v}`}>
              <line
                x1={pad.left}
                y1={mapY(v)}
                x2={pad.left + plotW}
                y2={mapY(v)}
                stroke="#E2E8F0"
                strokeDasharray="2 3"
              />
              <text
                x={pad.left - 8}
                y={mapY(v) + 3.5}
                textAnchor="end"
                className="fill-slate-500 text-[10px] font-mono"
              >
                {v}
              </text>
            </g>
          ))}

          {/* Clip ellipses to plot area */}
          <defs>
            <clipPath id="boiled-egg-clip">
              <rect x={pad.left} y={pad.top} width={plotW} height={plotH} />
            </clipPath>
          </defs>

          <g clipPath="url(#boiled-egg-clip)">
            {/* White Ellipse — HIA */}
            <path
              d={hiaPath}
              fill="#FFFFFF"
              stroke="#64748B"
              strokeWidth="1.25"
            />
            {/* Yolk Ellipse — BBB */}
            <path
              d={bbbPath}
              fill="#FEF3C7"
              stroke="#D97706"
              strokeWidth="1.4"
            />

            {/* Region labels */}
            <text
              x={mapX(38)}
              y={mapY(3.2)}
              textAnchor="middle"
              className="fill-amber-800/70 text-[10px] font-mono font-semibold pointer-events-none"
            >
              BBB (Yolk)
            </text>
            <text
              x={mapX(108)}
              y={mapY(0.2)}
              textAnchor="middle"
              className="fill-slate-500 text-[10px] font-mono font-medium pointer-events-none"
            >
              HIA (White)
            </text>

            {/* Plotted molecules */}
            {molecules.map((mol, idx) => {
              const egg = mol.cheminformatics?.boiled_egg;
              if (!egg) return null;
              const cx = mapX(egg.tpsa);
              const cy = mapY(egg.wlogp);
              const isSelected = idx === selectedIndex;
              const label = labels[idx] || `Mol ${idx + 1}`;
              return (
                <g
                  key={idx}
                  className="cursor-pointer"
                  onClick={() => onSelect(idx)}
                >
                  {isSelected && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={10}
                      fill="#A51C30"
                      fillOpacity={0.14}
                    />
                  )}
                  <circle
                    cx={cx}
                    cy={cy}
                    r={isSelected ? 5.5 : 4}
                    fill={isSelected ? "#A51C30" : "#1E3A5F"}
                    stroke="#FFFFFF"
                    strokeWidth="1.5"
                  >
                    <title>
                      {`${label}\nTPSA: ${egg.tpsa} Å² | WLOGP: ${egg.wlogp}\n${egg.classification}`}
                    </title>
                  </circle>
                  <text
                    x={cx + 8}
                    y={cy + 3}
                    className={`text-[9.5px] font-mono pointer-events-none ${
                      isSelected ? "fill-crimson font-semibold" : "fill-slate-600"
                    }`}
                  >
                    {label.slice(0, 14)}
                  </text>
                </g>
              );
            })}
          </g>

          {/* Axis labels */}
          <text
            x={pad.left + plotW / 2}
            y={H - 8}
            textAnchor="middle"
            className="fill-slate-700 text-[11px] font-sans font-medium"
          >
            Topological Polar Surface Area — TPSA (Å²)
          </text>
          <text
            x={14}
            y={pad.top + plotH / 2}
            textAnchor="middle"
            transform={`rotate(-90, 14, ${pad.top + plotH / 2})`}
            className="fill-slate-700 text-[11px] font-sans font-medium"
          >
            Wildman-Crippen WLOGP
          </text>
        </svg>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] font-mono text-ink-muted">
        <div className="flex items-center gap-3">
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-amber-100 border border-amber-600 inline-block" />
            BBB Permeant
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-white border border-slate-500 inline-block" />
            High HIA
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-crimson inline-block" />
            Selected
          </span>
        </div>
        {currentEgg && (
          <span>
            TPSA {currentEgg.tpsa.toFixed(1)} Å² · WLOGP {currentEgg.wlogp.toFixed(2)}
          </span>
        )}
      </div>
    </div>
  );
}
