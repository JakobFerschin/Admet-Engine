# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that is unique to your setup.

## What Goes Here

Things like:

- Camera names and locations
- SSH hosts and aliases
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras

- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH

- home-server → 192.168.1.100, user: admin

### TTS

- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

## npm on Windows (PowerShell)

- **npm install fails silently** in PowerShell — exit code 1 with no stdout/stderr. Use `cmd /c` wrapper:
  ```powershell
  cd frontend; cmd /c "npm install --legacy-peer-deps 2>&1"
  ```
- **npm 11.x requires `--legacy-peer-deps`** on Windows for most scaffolded projects — peer dependency resolution breaks otherwise.
- **Always verify node_modules after npm install** — React packages can be missing from `package.json` deps despite being listed (especially with `--legacy-peer-deps`). Run `npm install react react-dom` separately if imports fail.

## Vite 8 Scaffolding (react-ts template)

- The latest `create-vite` templates may produce **vanilla TS projects** even with `--template react-ts`. Always verify:
  - `tsconfig.json` has `"jsx": "react-jsx"`
  - `main.tsx` uses `createRoot` from `react-dom/client`
  - `index.html` has `<div id="root">` and references `main.tsx`
  - Remove leftover vanilla template files (`main.ts`, `counter.ts`, `style.css`)
- **TypeScript strict checks to disable for React dev:** `verbatimModuleSyntax`, `erasableSyntaxOnly`, `noUnusedLocals`, `noUnusedParameters` can cause false positives during rapid component iteration.

## Vite Version Pin

- **Vite 8 crashes on Windows + Node 24** — Rolldown optimizer exits code 1 after starting (no visible error). Production build (`vite build`) works fine. **Pin to Vite 5.4.x** which uses esbuild:
  ```
  npm install --legacy-peer-deps vite@5 @vitejs/plugin-react@4
  ```
- **`react-is` peer dep** — recharts requires it; not auto-installed. Always `npm install react-is` when using recharts.

## Windows Exec + Dev Server

- **Vite dev server must use `background: true`** — The exec tool kills long-running processes after startup output completes. Always set `background: true` with a generous `yieldMs` when starting vite/npx dev servers:
  ```json
  { "background": true, "yieldMs": 8000, "timeout": 60 }
  ```
- **Always verify dev server stays alive** after startup — poll the session after initial output to confirm it didn't get killed.

## ChEMBL v37 Schema & Large SQLite ETL

- **`confidence_score` is in `assays`, NOT `activities`** — the real ChEMBL v37 stores it in `assays`. Always `PRAGMA table_info()` before writing SQL against a new table.
- **Key activity columns**: `standard_flag` (1=standardised), `potential_duplicate` (0=clean), `pchembl_value` (NUMERIC 4,2), `standard_relation`, `standard_type`, `standard_value`, `standard_units`.
- **30 GB+ SQLite pattern**: filter at SQL level (`WHERE tid IN (...)`) — reduced 2B+ rows to 19K. Use `conn.row_factory = sqlite3.Row` + cursor iteration for zero-RAM streaming.
- **SQLite pragmas for large read-only DBs**: `PRAGMA query_only = ON`, `PRAGMA mmap_size = 268435456` (256 MB).
- **Path resolution in subdirectories**: when script is in `src/data/`, use `Path(__file__).resolve().parent.parent.parent` for project root (not `.parent.parent`).

---

Add whatever helps you do your job. This is your cheat sheet.
Do not store passwords, API keys, tokens, or secrets here in plain text.
For sensitive items, use aliases and safe context only.
