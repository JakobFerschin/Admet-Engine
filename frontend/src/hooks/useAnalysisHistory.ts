import { useState, useEffect, useCallback } from "react";
import type { AdmetResponse } from "../types/admet";

export interface HistoryEntry {
  id: string;
  timestamp: number;
  smiles: string[];
  response: AdmetResponse;
}

const STORAGE_KEY = "admet_analysis_history";
const MAX_ENTRIES = 10;

function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

function saveHistory(entries: HistoryEntry[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)));
  } catch {
    // localStorage full or unavailable — silently degrade
  }
}

interface UseAnalysisHistoryReturn {
  entries: HistoryEntry[];
  addEntry: (smiles: string[], response: AdmetResponse) => void;
  clearHistory: () => void;
  loadEntry: (id: string) => HistoryEntry | undefined;
}

export function useAnalysisHistory(): UseAnalysisHistoryReturn {
  const [entries, setEntries] = useState<HistoryEntry[]>(() => loadHistory());

  const addEntry = useCallback((smiles: string[], response: AdmetResponse) => {
    const entry: HistoryEntry = {
      id: crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`,
      timestamp: Date.now(),
      smiles,
      response,
    };
    setEntries((prev) => {
      const next = [entry, ...prev].slice(0, MAX_ENTRIES);
      saveHistory(next);
      return next;
    });
  }, []);

  const clearHistory = useCallback(() => {
    setEntries([]);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  const loadEntry = useCallback(
    (id: string) => entries.find((e) => e.id === id),
    [entries],
  );

  // Sync across tabs
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) {
        setEntries(loadHistory());
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  return { entries, addEntry, clearHistory, loadEntry };
}
