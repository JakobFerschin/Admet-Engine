import { useState } from "react";
import { Send, Trash2 } from "lucide-react";

interface SmilesInputProps {
  onSubmit: (smilesList: string[]) => void;
  loading: boolean;
}

export default function SmilesInput({ onSubmit, loading }: SmilesInputProps) {
  const [raw, setRaw] = useState("");
  const [error, setError] = useState<string | null>(null);

  const parseSmiles = (text: string): string[] =>
    text
      .split(/[\n,;]+/)
      .map((s) => s.trim())
      .filter((s) => s.length > 0);

  const handleSubmit = () => {
    const smiles = parseSmiles(raw);
    if (smiles.length === 0) {
      setError("Enter at least one SMILES string");
      return;
    }
    if (smiles.length > 1000) {
      setError("Maximum 1000 SMILES per batch");
      return;
    }
    setError(null);
    onSubmit(smiles);
  };

  const handleClear = () => {
    setRaw("");
    setError(null);
  };

  const count = parseSmiles(raw).length;

  return (
    <div className="card space-y-3">
      <div className="flex items-center justify-between">
        <label className="font-mono text-xs text-gray-400 tracking-widest uppercase">
          SMILES Input
        </label>
        <span className="text-[10px] font-mono text-gray-600">
          {count > 0 ? `${count} molecule${count !== 1 ? "s" : ""}` : "Empty"}
        </span>
      </div>

      <textarea
        value={raw}
        onChange={(e) => {
          setRaw(e.target.value);
          if (error) setError(null);
        }}
        placeholder={`O=C(O)c1ccccc1O\nCN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21\nCC(=O)Oc1ccccc1C(=O)O`}
        rows={6}
        className="input-field resize-y min-h-[120px]"
        disabled={loading}
        spellCheck={false}
      />

      {error && (
        <p className="text-xs font-mono text-accent-rose animate-fade-in">{error}</p>
      )}

      <div className="flex gap-3">
        <button
          onClick={handleSubmit}
          disabled={loading || raw.trim().length === 0}
          className="btn-primary flex items-center gap-2"
        >
          <Send className="w-3.5 h-3.5" />
          Analysieren
        </button>
        <button
          onClick={handleClear}
          disabled={loading || raw.trim().length === 0}
          className="px-4 py-2 rounded-lg font-mono text-xs text-gray-500
                     border border-surface-border hover:border-gray-600 hover:text-gray-300
                     transition-all duration-200 disabled:opacity-30 disabled:cursor-not-allowed"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
}
