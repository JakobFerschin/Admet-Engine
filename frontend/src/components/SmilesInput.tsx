import { useState } from "react";
import { Play, Trash2, BookOpen } from "lucide-react";

interface SmilesInputProps {
  onSubmit: (smilesList: string[], labels?: string[]) => void;
  loading: boolean;
}

interface PresetPanel {
  id: string;
  label: string;
  subtitle: string;
  compounds: Array<{ name: string; smiles: string }>;
}

const PRESET_PANELS: PresetPanel[] = [
  {
    id: "cardiotox",
    label: "hERG Cardiotoxicity Panel",
    subtitle: "Withdrawn QT drugs vs. safe control",
    compounds: [
      {
        name: "Terfenadine",
        smiles: "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1",
      },
      {
        name: "Astemizole",
        smiles: "COc1ccc(CCN2CCC(Nc3nc4ccccc4n3Cc3ccc(F)cc3)CC2)cc1",
      },
      {
        name: "Cisapride",
        smiles: "COc1cc(N)c(Cl)cc1C(=O)NC1CCN(CCCOc2ccc(F)cc2)CC1OC",
      },
      {
        name: "Aspirin",
        smiles: "CC(=O)Oc1ccccc1C(=O)O",
      },
    ],
  },
  {
    id: "cyp3a4",
    label: "CYP3A4 DDI Reference Set",
    subtitle: "FDA index inhibitors vs. non-inhibitors",
    compounds: [
      {
        name: "Ketoconazole",
        smiles: "CC(=O)N1CCN(c2ccc(OC[C@H]3CO[C@](Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1",
      },
      {
        name: "Mibefradil",
        smiles: "COCC(=O)O[C@]1(CCN(C)CCc2nc3ccccc3[nH]2)CCc2cc(F)ccc2[C@@H]1C(C)C",
      },
      {
        name: "Caffeine",
        smiles: "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
      },
      {
        name: "Metformin",
        smiles: "CN(C)C(=N)NC(=N)N",
      },
    ],
  },
  {
    id: "alerts",
    label: "CNS, Drug-Likeness & PAINS",
    subtitle: "BOILED-Egg & toxicophore controls",
    compounds: [
      {
        name: "Diazepam",
        smiles: "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21",
      },
      {
        name: "Haloperidol",
        smiles: "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
      },
      {
        name: "Curcumin (PAINS)",
        smiles: "COc1cc(/C=C/C(=O)CC(=O)/C=C/c2ccc(O)c(OC)c2)ccc1O",
      },
      {
        name: "Acetaminophen",
        smiles: "CC(=O)Nc1ccc(O)cc1",
      },
    ],
  },
];

export default function SmilesInput({ onSubmit, loading }: SmilesInputProps) {
  const [raw, setRaw] = useState("");
  const [error, setError] = useState<string | null>(null);

  const parseEntries = (text: string): { smiles: string[]; names: string[] } => {
    const lines = text
      .split(/[\n;]+/)
      .map((l) => l.trim())
      .filter((l) => l.length > 0);

    const smiles: string[] = [];
    const names: string[] = [];

    for (const line of lines) {
      // Support optional "SMILES Name" or "SMILES\tName" notation common in cheminformatics
      const parts = line.split(/\s+/);
      smiles.push(parts[0]);
      names.push(parts.slice(1).join(" ") || "");
    }
    return { smiles, names };
  };

  const handleSubmit = () => {
    const { smiles, names } = parseEntries(raw);
    if (smiles.length === 0) {
      setError("Enter at least one SMILES string or select a clinical reference panel.");
      return;
    }
    if (smiles.length > 1000) {
      setError("Maximum batch size is 1,000 SMILES per request.");
      return;
    }
    setError(null);
    onSubmit(smiles, names);
  };

  const handlePreset = (preset: PresetPanel) => {
    const formatted = preset.compounds.map((c) => `${c.smiles} ${c.name}`).join("\n");
    setRaw(formatted);
    setError(null);
    onSubmit(
      preset.compounds.map((c) => c.smiles),
      preset.compounds.map((c) => c.name),
    );
  };

  const { smiles } = parseEntries(raw);

  return (
    <section className="card space-y-4">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <div>
          <p className="section-kicker">Compound Batch Input</p>
          <h2 className="font-serif text-lg font-medium text-ink mt-0.5">
            Molecular Structure Screening
          </h2>
        </div>
        <span className="data-mono text-[11px] text-ink-muted">
          {smiles.length > 0
            ? `${smiles.length} compound${smiles.length !== 1 ? "s" : ""} queued`
            : "Accepts SMILES (optional space-separated compound name)"}
        </span>
      </div>

      {/* Curated Academic Reference Panels */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex items-center gap-1 text-[11px] text-ink-muted mr-1">
          <BookOpen className="w-3.5 h-3.5" />
          Reference sets:
        </span>
        {PRESET_PANELS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            disabled={loading}
            onClick={() => handlePreset(preset)}
            className="text-left px-3 py-1.5 rounded border border-hairline bg-canvas
                       hover:bg-subtle hover:border-slate-300 transition-colors text-xs
                       disabled:opacity-40"
            title={preset.subtitle}
          >
            <span className="font-medium text-ink">{preset.label}</span>
          </button>
        ))}
      </div>

      <textarea
        id="smiles-batch-input"
        value={raw}
        onChange={(e) => {
          setRaw(e.target.value);
          if (error) setError(null);
        }}
        placeholder={
          "CC(C)(C)c1ccc(C(O)CCCN2CCC(C(O)(c3ccccc3)c3ccccc3)CC2)cc1 Terfenadine\nCC(=O)Oc1ccccc1C(=O)O Aspirin"
        }
        rows={4}
        className="input-field resize-y min-h-[104px]"
        disabled={loading}
        spellCheck={false}
      />

      {error && (
        <p className="text-xs font-medium text-crimson animate-fade-in">{error}</p>
      )}

      <div className="flex items-center justify-between gap-3 pt-1">
        <div className="flex items-center gap-2.5">
          <button
            id="run-analysis-btn"
            type="button"
            onClick={handleSubmit}
            disabled={loading || raw.trim().length === 0}
            className="btn-primary flex items-center gap-2"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            {loading ? "Evaluating…" : "Run ADMET Analysis"}
          </button>
          <button
            type="button"
            onClick={() => {
              setRaw("");
              setError(null);
            }}
            disabled={loading || raw.trim().length === 0}
            className="btn-secondary flex items-center gap-1.5"
            title="Clear input"
          >
            <Trash2 className="w-3.5 h-3.5" />
            Clear
          </button>
        </div>

        <p className="hidden sm:block text-[11px] text-ink-muted">
          RDKit MolStandardize · Counterion strip · Tautomer canonicalization
        </p>
      </div>
    </section>
  );
}
