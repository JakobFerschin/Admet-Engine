import { useState, useCallback, useEffect } from "react";
import axios, { AxiosError } from "axios";
import type { AdmetResponse, ModelMetadata } from "../types/admet";

const API_BASE = (import.meta.env.VITE_API_URL as string | undefined) ?? "http://localhost:8000";

interface HealthStatus {
  status: string;
  model_loaded: boolean;
  model_version: string;
  mode: string;
  device: string;
}

interface UseAdmetPrediction {
  predict: (smilesList: string[]) => Promise<AdmetResponse>;
  restoreEntry: (response: AdmetResponse) => void;
  data: AdmetResponse | null;
  loading: boolean;
  error: { status: number; message: string } | null;
  health: HealthStatus | null;
  metadata: ModelMetadata | null;
}

export function useAdmetPrediction(): UseAdmetPrediction {
  const [data, setData] = useState<AdmetResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ status: number; message: string } | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function fetchServerInfo() {
      try {
        const [hRes, mRes] = await Promise.all([
          axios.get<HealthStatus>(`${API_BASE}/health`, { timeout: 4000 }),
          axios.get<ModelMetadata>(`${API_BASE}/model/metadata`, { timeout: 4000 }),
        ]);
        if (!cancelled) {
          setHealth(hRes.data);
          setMetadata(mRes.data);
        }
      } catch {
        if (!cancelled) {
          setHealth(null);
        }
      }
    }
    fetchServerInfo();
    return () => {
      cancelled = true;
    };
  }, []);

  const restoreEntry = useCallback((response: AdmetResponse) => {
    setError(null);
    setLoading(false);
    setData(response);
  }, []);

  const predict = useCallback(async (smilesList: string[]): Promise<AdmetResponse> => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post<AdmetResponse>(
        `${API_BASE}/predict/admet`,
        { smiles: smilesList, include_svg: true },
        { headers: { "Content-Type": "application/json" } },
      );
      setData(response.data);
      return response.data;
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: unknown }>;
      let status = 0;
      let message = "Unable to connect to inference backend.";

      if (axiosErr.response) {
        status = axiosErr.response.status;
        const detail = axiosErr.response.data?.detail;
        if (status === 422) {
          if (typeof detail === "string") {
            message = `Validation error (422): ${detail}`;
          } else if (Array.isArray(detail)) {
            const msgs = detail
              .map((d: { msg?: string; loc?: string[] }) =>
                d.msg ? `${d.msg} (${(d.loc ?? []).join(" → ")})` : "",
              )
              .filter(Boolean)
              .join("; ");
            message = `Validation error (422): ${msgs || "Invalid SMILES request"}`;
          } else {
            message = "Validation error (422): Check SMILES input format.";
          }
        } else if (status >= 500) {
          message = `Server error (${status}): Check FastAPI backend logs.`;
        } else {
          message = `Request failed (${status}): ${axiosErr.response.statusText}`;
        }
      } else if (axiosErr.code === "ECONNREFUSED" || axiosErr.code === "ERR_NETWORK") {
        message = `Backend unreachable at ${API_BASE} — start FastAPI via: uvicorn src.api.main:app --port 8000`;
      }

      setError({ status, message });
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { predict, restoreEntry, data, loading, error, health, metadata };
}
