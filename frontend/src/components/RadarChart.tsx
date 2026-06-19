import {
  Radar,
  RadarChart as RechartsRadar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Tooltip,
  Legend,
} from "recharts";
import { RADAR_ENDPOINTS, ENDPOINT_DISPLAY } from "../types/admet";

interface RadarChartProps {
  /** Map of endpoint → predicted value (0–1) for primary molecule */
  predictions: Record<string, number>;
  /** Label for primary molecule */
  moleculeLabel: string;
  /** Colour for primary radar polygon */
  colour?: string;
  /** Optional overlay — second molecule for comparison */
  overlayPredictions?: Record<string, number> | null;
  overlayLabel?: string;
  overlayColour?: string;
}

export default function RadarChart({
  predictions,
  moleculeLabel,
  colour = "#22d3ee",
  overlayPredictions,
  overlayLabel,
  overlayColour = "#d946ef",
}: RadarChartProps) {
  const isCompare = overlayPredictions != null && overlayLabel != null;

  const data = RADAR_ENDPOINTS.map((key) => ({
    endpoint: ENDPOINT_DISPLAY[key],
    ...(isCompare
      ? { [moleculeLabel]: predictions[key] ?? 0, [overlayLabel!]: overlayPredictions![key] ?? 0 }
      : { value: predictions[key] ?? 0 }),
  }));

  return (
    <div className="card" id="radar-chart-card">
      <h3 className="font-mono text-xs text-gray-400 tracking-widest uppercase mb-2">
        Radar Profile — {isCompare ? `${moleculeLabel} vs ${overlayLabel}` : moleculeLabel}
      </h3>

      {isCompare && (
        <div className="flex items-center gap-4 mb-3 text-[10px] font-mono">
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 rounded-full inline-block" style={{ backgroundColor: colour }} />
            <span className="text-gray-300">{moleculeLabel}</span>
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-3 h-0.5 rounded-full inline-block" style={{ backgroundColor: overlayColour }} />
            <span className="text-gray-300">{overlayLabel}</span>
          </span>
        </div>
      )}

      <ResponsiveContainer width="100%" height={320}>
        <RechartsRadar data={data} cx="50%" cy="50%" outerRadius="75%">
          <PolarGrid stroke="#2a2a3a" strokeDasharray="3 3" />
          <PolarAngleAxis
            dataKey="endpoint"
            tick={{
              fill: "#9ca3af",
              fontSize: 11,
              fontFamily: '"JetBrains Mono", monospace',
            }}
          />
          <PolarRadiusAxis
            angle={30}
            domain={[0, 1]}
            tick={{ fill: "#4b5563", fontSize: 10, fontFamily: '"JetBrains Mono", monospace' }}
            tickCount={5}
            stroke="#1a1a24"
          />

          {isCompare ? (
            <>
              <Radar
                name={moleculeLabel}
                dataKey={moleculeLabel}
                stroke={colour}
                strokeWidth={2}
                fill={colour}
                fillOpacity={0.12}
                dot={{ r: 3, fill: colour, stroke: "#0a0a0f", strokeWidth: 1 }}
                activeDot={{ r: 5, fill: colour, stroke: "#0a0a0f", strokeWidth: 2 }}
              />
              <Radar
                name={overlayLabel!}
                dataKey={overlayLabel!}
                stroke={overlayColour}
                strokeWidth={2}
                strokeDasharray="5 3"
                fill={overlayColour}
                fillOpacity={0.10}
                dot={{ r: 3, fill: overlayColour, stroke: "#0a0a0f", strokeWidth: 1 }}
                activeDot={{ r: 5, fill: overlayColour, stroke: "#0a0a0f", strokeWidth: 2 }}
              />
              <Legend
                wrapperStyle={{
                  fontFamily: '"JetBrains Mono", monospace',
                  fontSize: "11px",
                  color: "#9ca3af",
                }}
              />
            </>
          ) : (
            <Radar
              name={moleculeLabel}
              dataKey="value"
              stroke={colour}
              strokeWidth={2}
              fill={colour}
              fillOpacity={0.15}
              dot={{ r: 3, fill: colour, stroke: "#0a0a0f", strokeWidth: 1 }}
              activeDot={{ r: 5, fill: colour, stroke: "#0a0a0f", strokeWidth: 2 }}
            />
          )}

          <Tooltip
            contentStyle={{
              backgroundColor: "#111118",
              border: "1px solid #2a2a3a",
              borderRadius: "8px",
              fontFamily: '"JetBrains Mono", monospace',
              fontSize: "12px",
              color: "#e5e7eb",
            }}
            formatter={(value) => [
              typeof value === "number" ? value.toFixed(4) : "—",
              "Score",
            ]}
          />
        </RechartsRadar>
      </ResponsiveContainer>
    </div>
  );
}
