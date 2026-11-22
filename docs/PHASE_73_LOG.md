# PHASE_73_LOG.md — Hyperparameter Optimization

## [2026-06-10 16:22] STEP 1: Environment Check

**Aktion:** Verified packages, data, models, directories.
**Ergebnis:**
- torch=2.10.0+cpu, pyg=2.8.0, optuna=4.9.0, matplotlib=3.10.8, rdkit=OK
- data/processed/chembl_admet_training.csv (16,321 rows)
- models/dmpnn_chembl_v1.pt (baseline, 1.63M params)
- models/dmpnn_chembl_v1_ext.pt (extended, +1 epoch)
- src/training/ + results/ created
**Problem / Fix:** optuna not installed — `pip install optuna` resolved.
**Naechster Schritt:** Write src/training/hparam_search.py
**Status:** DONE

## [2026-06-10 16:29] STEP 2: Smoke Test

**Aktion:** Single-trial test — model build, train 3 batches, validate.
**Ergebnis:** FlexiblePredictor(300, T=3, set2set) = 1,630,502 params. Train loss=1.31. Val AUC: hERG=0.488, CYP=0.464 (untrained, expected). No crashes.
**Problem / Fix:** None.
**Naechster Schritt:** Launch full 50-trial study.
**Status:** DONE

## [2026-06-10 16:30] STEP 3: Full 50-Trial Optuna Study

**Aktion:** Launched `src/training/hparam_search.py` as background process (pid 12048, session vivid-pine).
**Ergebnis:** Featurization started at 16:30. Study will run 50 trials x max 15 epochs each, MedianPruner active, TPESampler.
**Estimated duration:** 3-8 hours (CPU-only, ~85s/epoch, most trials pruned <5 epochs).
**Problem / Fix:** None yet.
**Naechster Schritt:** Monitor study, wait for completion.
**Status:** IN_PROGRESS

## [2026-06-10 16:38] STEP 3a: Crash-Proofing (A/B/C)

**Aktion A:** Checked hparam_search.py storage mode.
**Ergebnis:** storage=None (in-memory!). Crash would lose all trials.
**Fix:** Changed to `storage="sqlite:///models/optuna_study.db"` with `load_if_exists=True`, `study_name="admet_phase73_hparam"`.

**Aktion B:** Fixed double batch_size sampling — removed `batch_size` from `run_trial()` (already sampled in `objective()`).

**Aktion C:** Process check. vivid-pine (in-memory) killed, sharp-shell (SQLite) started at 16:38.
**Status:** DONE

**Naechster Schritt:** Write train_best.py + evaluate.py while study runs.

## [2026-06-10 16:40] STEP 4+5 (PRE-WRITTEN): train_best.py + evaluate.py

**Aktion:** Wrote both scripts while Optuna study runs (parallelization).
**Ergebnis:**
- `src/training/train_best.py`: loads best_hparams.json, trains 50 epochs patience=10, saves dmpnn_chembl_v2.pt + phase73_results.json. Uses FlexiblePredictor, masked loss, pos_weight support.
- `src/training/evaluate.py`: loads any checkpoint, computes ROC-AUC/PR-AUC/F1/Accuracy per task. Comparison table v1 vs v1_ext vs v2. ROC plot via matplotlib (PNG).
**Problem / Fix:** None.
**Naechster Schritt:** Wait for Optuna study to complete, then execute train_best.py.
**Status:** DONE (scripts ready, execution pending)

## [2026-06-10 18:32] STEP 3c: SQLite Bug Fix + Restart

**Aktion:** Optuna 4.9.0 SQLite storage threw `TrialModel.find_max_value_trial_id` error on Windows (WAL mode issue). Switched to in-memory storage + manual `phase73_trials.json` save after every completed trial via callback. Best config snapshot saved to `best_hparams.json` on each completion too.
**Ergebnis:** Study restarted (vivid-gulf, pid 20020). Crash-resilient: partial progress always saved to JSON.
**Problem / Fix:** Optuna v4.9.0 SQLite + Windows = bug. In-memory + callback saves = simpler and more robust.
**Naechster Schritt:** Monitor for first completed trials.
