import { type ReactNode } from 'react';
import { FlaskConical } from 'lucide-react';

interface DashboardLayoutProps {
  children: ReactNode;
}

export default function DashboardLayout({ children }: DashboardLayoutProps) {
  return (
    <div className="min-h-screen bg-surface flex flex-col">
      {/* ── Header ──────────────────────────────────────────── */}
      <header className="border-b border-surface-border bg-surface-raised/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <FlaskConical className="w-5 h-5 text-accent-cyan" />
            <span className="font-mono text-sm tracking-[0.2em] text-gray-300">
              ADMET-ENGINE
            </span>
            <span className="text-[10px] font-mono text-gray-600 tracking-widest">
              // V1.0
            </span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-accent-emerald animate-pulse" />
            <span className="text-[11px] font-mono text-gray-500 tracking-wide">ONLINE</span>
          </div>
        </div>
      </header>

      {/* ── Main Content ────────────────────────────────────── */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-6 py-8">
        {children}
      </main>

      {/* ── Footer ──────────────────────────────────────────── */}
      <footer className="border-t border-surface-border px-6 py-3">
        <div className="max-w-7xl mx-auto flex items-center justify-between text-[10px] font-mono text-gray-600">
          <span>ADMET-Engine © 2026</span>
          <span>DMPNN · PyTorch Geometric · FastAPI</span>
        </div>
      </footer>
    </div>
  );
}
