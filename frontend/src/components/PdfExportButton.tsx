import { useCallback, useState } from "react";
import { FileDown, Loader2 } from "lucide-react";
import { jsPDF } from "jspdf";
import type { AdmetResponse } from "../types/admet";
import { ML_ENDPOINT_META } from "../types/admet";

interface PdfExportButtonProps {
  response: AdmetResponse;
  compoundNames: string[];
}

export default function PdfExportButton({
  response,
  compoundNames,
}: PdfExportButtonProps) {
  const [generating, setGenerating] = useState(false);

  const generatePdf = useCallback(async () => {
    setGenerating(true);
    try {
      const pdf = new jsPDF("p", "mm", "a4");
      const pageW = 210;
      const margin = 16;
      const now = new Date().toISOString().replace("T", " ").slice(0, 19);
      const reportId = crypto.randomUUID?.()?.slice(0, 8).toUpperCase() ?? "ADMET";
      const molecules = response.results
        .map((r, idx) => ({ r, name: compoundNames[idx] || `Compound ${idx + 1}` }))
        .filter((item) => item.r.status === "success");

      // ── Cover Header (Academic Dossier Style) ─────────────────
      pdf.setFillColor(165, 28, 48); // Harvard Crimson top rule
      pdf.rect(0, 0, pageW, 3.5, "F");

      pdf.setTextColor(15, 23, 42);
      pdf.setFont("times", "bold");
      pdf.setFontSize(22);
      pdf.text("ADMET-Engine", margin, 26);

      pdf.setFont("times", "italic");
      pdf.setFontSize(12);
      pdf.setTextColor(71, 85, 105);
      pdf.text("Computational Medicinal Chemistry & Preclinical Safety Dossier", margin, 33);

      pdf.setDrawColor(226, 228, 233);
      pdf.line(margin, 38, pageW - margin, 38);

      pdf.setFont("courier", "normal");
      pdf.setFontSize(8.5);
      pdf.setTextColor(51, 65, 85);
      const meta = [
        `Report ID:   ${reportId}`,
        `Timestamp:   ${now} UTC`,
        `Model:       ${response.model_version}`,
        `Compounds:   ${molecules.length} / ${response.results.length} valid`,
        `Runtime:     ${response.prediction_time_ms.toFixed(1)} ms`,
      ];
      let y = 46;
      for (const line of meta) {
        pdf.text(line, margin, y);
        y += 5.5;
      }

      y += 4;
      pdf.setDrawColor(15, 23, 42);
      pdf.line(margin, y, pageW - margin, y);
      y += 8;

      // ── Compound Summaries ────────────────────────────────────
      for (let i = 0; i < molecules.length; i++) {
        const { r: mol, name } = molecules[i];
        const chem = mol.cheminformatics;
        const ml = mol.ml_endpoints;
        const ad = mol.applicability_domain;

        if (y > 225) {
          pdf.addPage();
          pdf.setFillColor(165, 28, 48);
          pdf.rect(0, 0, pageW, 2.5, "F");
          y = 18;
        }

        pdf.setFont("times", "bold");
        pdf.setFontSize(13);
        pdf.setTextColor(15, 23, 42);
        pdf.text(`${i + 1}. ${name}`, margin, y);

        if (chem) {
          pdf.setFont("courier", "normal");
          pdf.setFontSize(8);
          pdf.setTextColor(71, 85, 105);
          pdf.text(
            `${chem.identifiers.molecular_formula}  |  MW ${chem.physchem.mw.toFixed(1)} Da  |  InChIKey ${chem.identifiers.inchi_key}`,
            margin,
            y + 5,
          );
          const smiLines = pdf.splitTextToSize(
            `SMILES: ${chem.identifiers.canonical_smiles}`,
            pageW - margin * 2,
          );
          pdf.text(smiLines, margin, y + 10);
          y += 10 + smiLines.length * 4;
        } else {
          y += 7;
        }

        // ML Endpoints
        if (ml) {
          pdf.setFont("helvetica", "bold");
          pdf.setFontSize(8.5);
          pdf.setTextColor(15, 23, 42);
          pdf.text("ChEMBL v37 Calibrated ML Predictions:", margin, y + 2);
          y += 6.5;

          pdf.setFont("courier", "normal");
          pdf.setFontSize(8);
          for (const [k, ep] of Object.entries(ml)) {
            const label = ML_ENDPOINT_META[k]?.title ?? k;
            const color: [number, number, number] =
              ep.risk_tier === "HIGH"
                ? [165, 28, 48]
                : ep.risk_tier === "MODERATE"
                  ? [180, 83, 9]
                  : [15, 118, 110];
            pdf.setTextColor(51, 65, 85);
            pdf.text(label, margin + 2, y);
            pdf.setTextColor(...color);
            pdf.text(
              `P = ${(ep.probability * 100).toFixed(1)}% (+/- ${(ep.epistemic_std * 100).toFixed(1)}%)  [${ep.risk_tier}]  95% CI [${ep.ci_95[0].toFixed(2)}, ${ep.ci_95[1].toFixed(2)}]`,
              margin + 58,
              y,
            );
            y += 5;
          }
        }

        if (ad) {
          pdf.setTextColor(71, 85, 105);
          pdf.setFont("courier", "normal");
          pdf.setFontSize(8);
          pdf.text(
            `Applicability Domain: ${ad.status} (Max Tanimoto = ${ad.max_tanimoto.toFixed(3)}, Top-5 Mean = ${ad.mean_top5_tanimoto.toFixed(3)})`,
            margin + 2,
            y + 1,
          );
          y += 6;
        }

        if (chem) {
          const p = chem.physchem;
          pdf.setTextColor(51, 65, 85);
          pdf.setFont("courier", "normal");
          pdf.setFontSize(7.8);
          pdf.text(
            `PhysChem: cLogP=${p.clogp.toFixed(2)} | TPSA=${p.tpsa.toFixed(1)} A^2 | HBD=${p.hbd} | HBA=${p.hba} | RotB=${p.rotatable_bonds} | QED=${p.qed.toFixed(3)} | Fsp3=${p.fsp3.toFixed(2)}`,
            margin + 2,
            y + 1,
          );
          y += 5;
          pdf.text(
            `Rules: Lipinski=${chem.rules.lipinski_pass ? "PASS" : "FAIL"} (${chem.rules.lipinski_violations} viol.) | Veber=${chem.rules.veber_pass ? "PASS" : "FAIL"} | Pfizer 3/75=${chem.rules.pfizer_3_75_warning ? "WARN" : "OK"} | BOILED-Egg=${chem.boiled_egg.classification}`,
            margin + 2,
            y + 1,
          );
          y += 5;
          pdf.text(
            `Alerts: PAINS=${chem.alerts.pains_count} | Brenk=${chem.alerts.brenk_count}${chem.alerts.total_alerts > 0 ? ` (${[...chem.alerts.pains_alerts, ...chem.alerts.brenk_alerts].slice(0, 3).join(", ")})` : " (Clean)"}`,
            margin + 2,
            y + 1,
          );
          y += 7;
        }

        pdf.setDrawColor(226, 228, 233);
        pdf.line(margin, y, pageW - margin, y);
        y += 6;
      }

      // Footer on last page
      pdf.setFont("times", "italic");
      pdf.setFontSize(8);
      pdf.setTextColor(100, 116, 139);
      pdf.text(
        "ADMET-Engine v2.0 — For Academic & Preclinical Research Use Only.",
        margin,
        288,
      );

      pdf.save(`ADMET_Dossier_${now.slice(0, 10)}_${reportId}.pdf`);
    } catch (err) {
      console.error("PDF export failed:", err);
    } finally {
      setGenerating(false);
    }
  }, [response, compoundNames]);

  const successCount = response.results.filter((r) => r.status === "success").length;

  return (
    <button
      type="button"
      onClick={generatePdf}
      disabled={generating || successCount === 0}
      className="btn-secondary flex items-center gap-1.5"
      title="Export publication-ready PDF compound dossier"
    >
      {generating ? (
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
      ) : (
        <FileDown className="w-3.5 h-3.5 text-crimson" />
      )}
      {generating ? "Generating…" : `Export Dossier PDF (${successCount})`}
    </button>
  );
}
