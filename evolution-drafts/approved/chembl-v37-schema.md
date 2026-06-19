# Evolution Proposal: ChEMBL v37 schema facts and large SQLite ETL patterns from Phase 7.1

- Proposal-ID: evo-2026-06-10-chembl-schema
- Status: approved
- Signature: chembl-v37-schema
- Created-At: 2026-06-10 11:08
- Last-Seen-At: 2026-06-10 11:08
- Target-File: TOOLS.md
- Trigger-Type: preference
- Confidence: medium

## Why This Matters
- ChEMBL v37 schema facts and large SQLite ETL patterns from Phase 7.1

## Evidence
- Interactive proposal card was present in the session UI.
- The original pending draft file was unavailable at approval time.
- AutoClaw reconstructed this draft from the proposal payload so the review result can still be recorded.

## Duplicate Check
- Checked: pending draft path + signature/proposal fallback
- Result: original draft file missing
- Decision: create surrogate draft from proposal payload

## Proposed Change
### TOOLS.md — ChEMBL v37 & large SQLite

### ChEMBL v37 Schema Notes

- **`confidence_score` is in `assays` table, NOT `activities`** — the mock DB placed it in activities but the real ChEMBL v37 stores it in `assays`. Always check `PRAGMA table_info()` before writing SQL.
- **Key activity columns**: `standard_flag` (1 = standardised), `potential_duplicate` (0 = clean), `pchembl_value` (NUMERIC 4,2), `standard_relation` ('=', '>', '<', etc.), `standard_type`, `standard_value`, `standard_units`.
- **No `confidence_score` in activities** — use `s.confidence_score >= 5` (on assays JOIN).

### Large SQLite ETL Pattern (30 GB+)

- **Filter at SQL level, not Python** — `WHERE tid IN (...)` on 30 GB DB reduced 2B+ rows to 19K. Never `SELECT *` without filters.
- **sqlite3 cursor iteration** — use `conn.row_factory = sqlite3.Row` + `for row in cursor:` for zero-RAM streaming.
- **`PRAGMA query_only = ON`** — prevents accidental writes on production DB.
- **`PRAGMA mmap_size = 268435456`** (256 MB) — speeds up large read-only queries.
- **Path resolution**: when script is in a subdirectory like `src/data/`, use `Path(__file__).resolve().parent.parent.parent` for project root (not `.parent.parent`).

## Apply Plan
1. Keep this reconstructed draft as the approval artifact.
2. Record the proposal content exactly as shown in the interactive card.
3. Append an audit note after approval or rejection.

## User Approval
- Approve: 批准 evo-2026-06-10-chembl-schema
- Reject: 拒绝 evo-2026-06-10-chembl-schema