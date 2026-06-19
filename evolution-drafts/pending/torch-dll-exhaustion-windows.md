# Evolution Proposal: Windows DLL slot exhaustion from long-running Torch processes — requires reboot, prevent with process limits

- Proposal-ID: evo-2026-06-11-torch-dll-exhaustion
- Status: pending
- Signature: torch-dll-exhaustion-windows
- Created-At: 2026-06-11 13:35
- Last-Seen-At: 2026-06-11 13:35
- Target-File: TOOLS.md
- Trigger-Type: preference
- Confidence: medium

## Why This Matters
- Windows DLL slot exhaustion from long-running Torch processes — requires reboot, prevent with process limits

## Evidence
- Interactive proposal card was present in the session UI.
- The original pending draft file was unavailable at approval time.
- AutoClaw reconstructed this draft from the proposal payload so the review result can still be recorded.

## Duplicate Check
- Checked: pending draft path + signature/proposal fallback
- Result: original draft file missing
- Decision: create surrogate draft from proposal payload

## Proposed Change
### TOOLS.md — Torch DLL Exhaustion

## Torch on Windows — DLL Slot Exhaustion

**WinError 1114: DLL-Initialisierungsroutine fehlgeschlagen (c10.dll)**

### Cause
- Long-running Torch background processes (>1 hour) exhaust Windows DLL slots.
- Each Python process loads ~50-100 Torch DLLs. Multiple background processes or long runs accumulate.
- DLL slots are NOT freed immediately on process exit — requires reboot or extended idle.

### Prevention
- **Never run more than 2 Torch background processes simultaneously.** Kill old ones before starting new.
- **Add per-trial timeout:** After each trial in Optuna, explicitly `del model` + `gc.collect()` to release DLL handles.
- **Check before training:** `python -c "import torch; print('ok')"` — if it fails, reboot is required.
- **Prefer single long-running process over multiple restarts.** Each restart loads fresh DLLs.

### Recovery
- **Only fix:** System reboot. No `taskkill`, `scf`, or `DISM` workaround known.
- Scheduled long training runs should account for this: plan 1 dedicated overnight run, not multiple attempts.

## Apply Plan
1. Keep this reconstructed draft as the approval artifact.
2. Record the proposal content exactly as shown in the interactive card.
3. Append an audit note after approval or rejection.

## User Approval
- Approve: 批准 evo-2026-06-11-torch-dll-exhaustion
- Reject: 拒绝 evo-2026-06-11-torch-dll-exhaustion