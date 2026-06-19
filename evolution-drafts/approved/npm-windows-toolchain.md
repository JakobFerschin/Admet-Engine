# Evolution Proposal: Windows npm/PowerShell toolchain patterns learned from Phase 6.1 scaffolding

- Proposal-ID: evo-2026-06-09-npm-windows-toolchain
- Status: approved
- Signature: npm-windows-toolchain
- Created-At: 2026-06-09 16:37
- Last-Seen-At: 2026-06-09 16:37
- Target-File: TOOLS.md
- Trigger-Type: preference
- Confidence: medium

## Why This Matters
- Windows npm/PowerShell toolchain patterns learned from Phase 6.1 scaffolding

## Evidence
- Interactive proposal card was present in the session UI.
- The original pending draft file was unavailable at approval time.
- AutoClaw reconstructed this draft from the proposal payload so the review result can still be recorded.

## Duplicate Check
- Checked: pending draft path + signature/proposal fallback
- Result: original draft file missing
- Decision: create surrogate draft from proposal payload

## Proposed Change
### TOOLS.md — npm / Vite / Windows Patterns

### npm on Windows (PowerShell)

- **npm install fails silently** in PowerShell — exit code 1 with no stdout/stderr. Use `cmd /c` wrapper:
  ```powershell
  cd frontend; cmd /c "npm install --legacy-peer-deps 2>&1"
  ```
- **npm 11.x requires `--legacy-peer-deps`** on Windows for most scaffolded projects — peer dependency resolution breaks otherwise.
- **Always verify node_modules after npm install** — React packages can be missing from `package.json` deps despite being listed (especially with `--legacy-peer-deps`). Run `npm install react react-dom` if imports fail.

### Vite 8 Scaffolding (react-ts template)

- The latest `create-vite` templates may produce **vanilla TS projects** even with `--template react-ts`. Always verify:
  - `tsconfig.json` has `"jsx": "react-jsx"`
  - `main.tsx` uses `createRoot` from `react-dom/client`
  - `index.html` has `<div id="root">` and references `main.tsx`
  - Remove leftover vanilla template files (`main.ts`, `counter.ts`, `style.css`)
- **TypeScript strict checks to disable for React dev:** `verbatimModuleSyntax`, `erasableSyntaxOnly`, `noUnusedLocals`, `noUnusedParameters` can cause false positives during rapid component iteration.

## Apply Plan
1. Keep this reconstructed draft as the approval artifact.
2. Record the proposal content exactly as shown in the interactive card.
3. Append an audit note after approval or rejection.

## User Approval
- Approve: 批准 evo-2026-06-09-npm-windows-toolchain
- Reject: 拒绝 evo-2026-06-09-npm-windows-toolchain