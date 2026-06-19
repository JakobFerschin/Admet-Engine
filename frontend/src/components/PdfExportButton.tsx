import { useCallback, useState } from "react";
import { FileText, Loader2 } from "lucide-react";
import html2canvas from "html2canvas";
import { jsPDF } from "jspdf";
import type { AdmetResponse } from "../types/admet";
import { ENDPOINT_LABELS, ENDPOINT_DISPLAY } from "../types/admet";

interface PdfExportButtonProps {
  response: AdmetResponse;
}

export default function PdfExportButton({ response }: PdfExportButtonProps) {
  const [generating, setGenerating] = useState(false);

  const generatePdf = useCallback(async () => {
    setGenerating(true);
    try {
      const pdf = new jsPDF("p", "mm", "a4");
      const pageW = 210;
      const margin = 14;
      const now = new Date().toISOString().replace("T", " ").slice(0, 19);
      const reportId = crypto.randomUUID?.()?.slice(0, 8) ?? "N/A";
      const molecules = response.results.filter((r) => r.status === "success");
      const errored = response.results.filter((r) => r.status === "error");

      // ══════════════════════════════════════════════════════════
      // PAGE 1 — COVER / DECKBLATT
      // ══════════════════════════════════════════════════════════
      pdf.setFillColor(15, 15, 25);
      pdf.rect(0, 0, pageW, 297, "F");

      // Accent line
      pdf.setFillColor(0, 180, 210);
      pdf.rect(0, 80, pageW, 2, "F");

      pdf.setTextColor(0, 180, 210);
      pdf.setFont("helvetica", "bold");
      pdf.setFontSize(22);
      pdf.text("ADMET-Engine", margin, 100);

      pdf.setTextColor(255, 255, 255);
      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(11);
      pdf.text("// Batch Analysis Report", margin, 112);

      pdf.setDrawColor(60, 60, 80);
      pdf.line(margin, 122, pageW - margin, 122);

      // Metadata block
      const metaLines = [
        `Generated:  ${now} UTC`,
        `Report ID:  ${reportId}`,
        `Model:      ${response.model_version}`,
        `Molecules:  ${response.results.length} total  (${molecules.length} succeeded, ${errored.length} failed)`,
        `Inference:  ${response.prediction_time_ms.toFixed(1)} ms`,
      ];

      pdf.setFontSize(9);
      pdf.setFont("courier", "normal");
      let metaY = 134;
      for (const line of metaLines) {
        pdf.setTextColor(180, 180, 200);
        pdf.text(line, margin, metaY);
        metaY += 7;
      }

      // Bottom line
      pdf.setDrawColor(60, 60, 80);
      pdf.line(margin, metaY + 6, pageW - margin, metaY + 6);

      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(7);
      pdf.setTextColor(100, 100, 120);
      pdf.text("ADMET-Engine v1.0  |  DMPNN + Set2Set  |  PyTorch Geometric", margin, metaY + 12);
      pdf.text("For Research Use Only — Not a Regulatory Submission", margin, metaY + 18);

      // ══════════════════════════════════════════════════════════
      // PER-MOLECULE PAGES — strict top-down layout
      // ══════════════════════════════════════════════════════════

      for (let i = 0; i < molecules.length; i++) {
        const mol = molecules[i];
        const preds = mol.predictions ?? {};

        pdf.addPage();

        // --- Page header bar ---
        pdf.setFillColor(15, 15, 25);
        pdf.rect(0, 0, pageW, 22, "F");
        pdf.setTextColor(0, 180, 210);
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(9);
        pdf.text("ADMET-Engine  //  Batch Report", margin, 10);
        pdf.setTextColor(200, 200, 210);
        pdf.setFont("courier", "normal");
        pdf.setFontSize(7);
        pdf.text(`Page ${pdf.getNumberOfPages() - 1}  |  Molecule ${i + 1}/${molecules.length}`, margin, 16);

        let y = 28;
        const pageBottom = 284; // leave room for footer

        // ── BLOCK 1: SMILES ──────────────────────────────────
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(9);
        pdf.setTextColor(20, 20, 30);
        pdf.text(`Molecule ${i + 1}`, margin, y);
        y += 6;

        pdf.setFont("courier", "normal");
        pdf.setFontSize(7);
        pdf.setTextColor(50, 50, 60);
        const smilesLines = pdf.splitTextToSize(mol.smiles, pageW - margin * 2);
        const smilesH = Math.min(smilesLines.length * 4, 18);
        pdf.text(smilesLines, margin, y);
        y += smilesH + 5;

        // ── BLOCK 2: TRIAGE TABLE ────────────────────────────
        // Page break if not enough room for header + rows + summary (~85mm)
        if (y + 85 > pageBottom) {
          pdf.addPage();
          y = 28;
        }

        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(9);
        pdf.setTextColor(20, 20, 30);
        pdf.text("ADMET Risk Assessment", margin, y);
        y += 7;

        const colX = [margin, margin + 52, margin + 104, margin + 148];
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(7);
        pdf.setTextColor(80, 80, 90);
        pdf.text("Endpoint", colX[0], y);
        pdf.text("Prediction", colX[1], y);
        pdf.text("Risk", colX[2], y);
        pdf.text("Status", colX[3], y);
        y += 3;
        pdf.setDrawColor(180, 180, 195);
        pdf.line(margin, y, pageW - margin, y);
        y += 4;

        let highCount = 0;
        let mediumCount = 0;
        let lowCount = 0;
        pdf.setFont("courier", "normal");
        pdf.setFontSize(7);

        for (const key of ENDPOINT_LABELS) {
          const value = preds[key];
          if (value == null) continue;

          let riskLevel: string;
          let riskColour: [number, number, number];
          let status: string;
          if (value >= 0.8) {
            riskLevel = "HIGH"; riskColour = [200, 30, 60]; status = "FAIL"; highCount++;
          } else if (value >= 0.5) {
            riskLevel = "MEDIUM"; riskColour = [200, 130, 0]; status = "WARN"; mediumCount++;
          } else {
            riskLevel = "LOW"; riskColour = [10, 150, 100]; status = "PASS"; lowCount++;
          }

          // Row-level page break
          if (y > pageBottom) {
            pdf.addPage();
            y = 28;
          }

          pdf.setTextColor(30, 30, 40);
          pdf.text(ENDPOINT_DISPLAY[key], colX[0], y);
          pdf.text(value.toFixed(4), colX[1], y);
          pdf.setTextColor(...riskColour);
          pdf.text(riskLevel, colX[2], y);
          pdf.text(status, colX[3], y);
          y += 5.5;
        }

        // Summary line
        y += 2;
        pdf.setDrawColor(180, 180, 195);
        pdf.line(margin, y, pageW - margin, y);
        y += 5;
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(7);
        pdf.setTextColor(200, 30, 60);
        pdf.text(`${highCount} HIGH`, margin, y);
        pdf.setTextColor(200, 130, 0);
        pdf.text(`${mediumCount} MEDIUM`, margin + 30, y);
        pdf.setTextColor(10, 150, 100);
        pdf.text(`${lowCount} LOW`, margin + 55, y);
        y += 8;

        // ── BLOCK 3: RADAR CHART (below table) ──────────────
        const radarEl = document.getElementById("radar-chart-card");
        if (radarEl) {
          // Page break if less than ~80mm remaining
          if (y + 80 > pageBottom) {
            pdf.addPage();
            y = 28;
          }

          pdf.setFont("helvetica", "bold");
          pdf.setFontSize(9);
          pdf.setTextColor(20, 20, 30);
          pdf.text("Radar Toxicity Profile", margin, y);
          y += 5;

          const canvas = await html2canvas(radarEl, {
            backgroundColor: "#0a0a0f",
            scale: 1.5,
          });
          const imgW = pageW - margin * 2;
          const imgH = (canvas.height / canvas.width) * imgW;
          // Safety: cap image height
          const maxImgH = pageBottom - y;
          const finalImgH = Math.min(imgH, maxImgH);

          pdf.addImage(
            canvas.toDataURL("image/png"),
            "PNG",
            margin, y,
            imgW, finalImgH,
          );
          y += finalImgH + 6;
        }

        // Footer
        pdf.setFont("helvetica", "normal");
        pdf.setFontSize(6);
        pdf.setTextColor(140, 140, 150);
        pdf.text(`${reportId}  |  Molecule ${i + 1}/${molecules.length}  |  ${now}`, margin, 290);
      }

      // ══════════════════════════════════════════════════════════
      // ERROR SUMMARY (if any)
      // ══════════════════════════════════════════════════════════
      if (errored.length > 0) {
        pdf.addPage();
        pdf.setFillColor(15, 15, 25);
        pdf.rect(0, 0, pageW, 22, "F");
        pdf.setTextColor(200, 60, 60);
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(9);
        pdf.text("ADMET-Engine  //  Processing Errors", margin, 10);

        let ey = 32;
        pdf.setFont("courier", "normal");
        pdf.setFontSize(8);
        for (const e of errored) {
          pdf.setTextColor(40, 40, 50);
          pdf.text(e.smiles.slice(0, 55), margin, ey);
          pdf.setTextColor(200, 60, 60);
          pdf.text(e.error ?? "Unknown error", margin + 75, ey);
          ey += 6;
        }
      }

      pdf.save(`ADMET_Batch_Report_${now.slice(0, 10)}.pdf`);
    } catch (err) {
      console.error("PDF generation failed:", err);
    } finally {
      setGenerating(false);
    }
  }, [response]);

  const successCount = response.results.filter((r) => r.status === "success").length;

  return (
    <button
      onClick={generatePdf}
      disabled={generating || successCount === 0}
      className="flex items-center gap-2 px-4 py-2 rounded-lg font-mono text-xs
                 border border-accent-cyan/30 text-accent-cyan bg-accent-cyan/5
                 hover:border-accent-cyan/50 hover:bg-accent-cyan/10
                 transition-all duration-200
                 disabled:opacity-30 disabled:cursor-not-allowed disabled:hover:border-accent-cyan/30 disabled:hover:bg-accent-cyan/5"
      title="Export complete batch as regulatory PDF report"
    >
      {generating ? (
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
      ) : (
        <FileText className="w-3.5 h-3.5" />
      )}
      {generating ? "Generating…" : `Batch-Bericht (${successCount})`}
    </button>
  );
}
