# Evolution Proposal: Windows Python execution patterns — pre-written .py files, ASCII-only output, mandatary stdout flush to prevent repeated crashes

- Proposal-ID: evo-2026-06-10-windows-python
- Status: pending
- Signature: windows-python-exec-patterns
- Created-At: 2026-06-10 16:35
- Last-Seen-At: 2026-06-10 16:35
- Target-File: TOOLS.md
- Trigger-Type: preference
- Confidence: medium

## Why This Matters
- Windows Python execution patterns — pre-written .py files, ASCII-only output, mandatary stdout flush to prevent repeated crashes

## Evidence
- Interactive proposal card was present in the session UI.
- The original pending draft file was unavailable at approval time.
- AutoClaw reconstructed this draft from the proposal payload so the review result can still be recorded.

## Duplicate Check
- Checked: pending draft path + signature/proposal fallback
- Result: original draft file missing
- Decision: create surrogate draft from proposal payload

## Proposed Change
### TOOLS.md — Windows Python Rules

## Windows Python Execution Rules

**These patterns prevent 3 known crash types: Unicode encode errors, stdout buffering hangs, and inline quoting SyntaxErrors.**

### Rule 1: Write Scripts to .py Files
- PowerShell inline Python with escaped quotes reliably breaks (SyntaxError on `\"`).
- Always `write` a .py file first, then `exec` it.
- Exception: single-line imports or trivial expressions without quotes.

### Rule 2: ASCII-Only print() Output
- Windows cp1252 encoding cannot handle Unicode (★, Δ, →, ✅).
- `print("★")` → `UnicodeEncodeError` — use `print("**")` instead.
- Check with: `Select-String -Path script.py -Pattern "[^\x00-\x7F]"`
- Safe alternatives: `->` instead of `→`, `** BEST **` instead of `★ BEST`, `delta` instead of `Δ`.

### Rule 3: Mandatory stdout Flush
- Python print() can be buffered in subprocess, causing output to never appear.
- Always add `flush=True` to all print() calls.
- Also call `sys.stdout.flush()` after every print block.
- Logging modules (logging.info) are NOT affected by this — only print().

### Rule 4: Python Modules via `python -c`
- Single-line module versions OK: `python -c "import torch; print(torch.__version__)"`
- Multi-line: ALWAYS use a .py file.
- Avoid `exec(open('script.py').read())` — loses `__file__` context.

## Apply Plan
1. Keep this reconstructed draft as the approval artifact.
2. Record the proposal content exactly as shown in the interactive card.
3. Append an audit note after approval or rejection.

## User Approval
- Approve: 批准 evo-2026-06-10-windows-python
- Reject: 拒绝 evo-2026-06-10-windows-python