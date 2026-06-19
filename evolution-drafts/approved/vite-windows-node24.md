# Evolution Proposal: Vite version pin + Windows dev server execution patterns from Phase 6.2

- Proposal-ID: evo-2026-06-09-vite-windows
- Status: approved
- Signature: vite-windows-node24
- Created-At: 2026-06-09 17:46
- Last-Seen-At: 2026-06-09 17:46
- Target-File: TOOLS.md
- Trigger-Type: preference
- Confidence: medium

## Why This Matters
- Vite version pin + Windows dev server execution patterns from Phase 6.2

## Evidence
- Interactive proposal card was present in the session UI.
- The original pending draft file was unavailable at approval time.
- AutoClaw reconstructed this draft from the proposal payload so the review result can still be recorded.

## Duplicate Check
- Checked: pending draft path + signature/proposal fallback
- Result: original draft file missing
- Decision: create surrogate draft from proposal payload

## Proposed Change
### TOOLS.md — Vite & Windows dev server

### Vite Version Pin

- **Vite 8 crashes on Windows + Node 24** — Rolldown optimizer exits code 1 after starting (no visible error). Production build (`vite build`) works fine. **Pin to Vite 5.4.x** which uses esbuild:
  ```
  npm install --legacy-peer-deps vite@5 @vitejs/plugin-react@4
  ```
- **`react-is` peer dep** — recharts requires it; not auto-installed. Always `npm install react-is` when using recharts.

### Windows Exec + Dev Server

- **Vite dev server must use `background: true`** — The exec tool kills long-running processes after startup output completes. Always set `background: true` with a generous `yieldMs` when starting vite/npx dev servers:
  ```json
  { "background": true, "yieldMs": 8000, "timeout": 60 }
  ```
- **Always verify dev server stays alive** after startup — poll the session after initial output to confirm it didn't get killed.

## Apply Plan
1. Keep this reconstructed draft as the approval artifact.
2. Record the proposal content exactly as shown in the interactive card.
3. Append an audit note after approval or rejection.

## User Approval
- Approve: 批准 evo-2026-06-09-vite-windows
- Reject: 拒绝 evo-2026-06-09-vite-windows