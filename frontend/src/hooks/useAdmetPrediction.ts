import { useState, useCallback } from "react";
import axios, { AxiosError } from "axios";
import type { AdmetResponse } from "../types/admet";

const API_URL = "http://localhost:8000/predict/admet";

interface UseAdmetPrediction {
  /** Trigger a batch prediction. Returns the response or throws on network/API error. */
  predict: (smilesList: string[]) => Promise<AdmetResponse>;
  /** Inject a cached response (for history restore). */
  restoreEntry: (response: AdmetResponse) => void;
  /** The latest successful response, or null. */
  data: AdmetResponse | null;
  /** True while a request is in flight. */
  loading: boolean;
  /** Structured error { status, message } or null. */
  error: { status: number; message: string } | null;
}

export function useAdmetPrediction(): UseAdmetPrediction {
  const [data, setData] = useState<AdmetResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<{ status: number; message: string } | null>(null);

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
        API_URL,
        { smiles: smilesList },
        { headers: { "Content-Type": "application/json" } },
      );
      setData(response.data);
      return response.data;
    } catch (err) {
      const axiosErr = err as AxiosError<{ detail?: unknown }>;
      let status = 0;
      let message = "Unknown network error";

      if (axiosErr.response) {
        status = axiosErr.response.status;
        const detail = axiosErr.response.data?.detail;

        if (status === 422) {
          // FastAPI validation error — extract meaningful info
          if (typeof detail === "string") {
            message = `Error ${status}: ${detail}`;
          } else if (Array.isArray(detail)) {
            const msgs = detail
              .map((d: { msg?: string; loc?: string[] }) =>
                d.msg ? `${d.msg} (${(d.loc ?? []).join(" → ")})` : "",
              )
              .filter(Boolean)
              .join("; ");
            message = `Error ${status}: ${msgs || "Validation failed"}`;
          } else {
            message = `Error ${status}: Invalid request — check SMILES format`;
          }
        } else if (status === 500) {
          message = `Error 500: Internal server error — check backend logs`;
        } else if (status >= 400) {
          message = `Error ${status}: ${axiosErr.response.statusText || "Request failed"}`;
        }
      } else if (axiosErr.code === "ECONNREFUSED" || axiosErr.code === "ERR_NETWORK") {
        message = "Error: Cannot reach backend — is localhost:8000 running?";
      }

      setError({ status, message });
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { predict, restoreEntry, data, loading, error };
}
