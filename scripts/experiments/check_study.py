import optuna
study = optuna.load_study(study_name="admet_phase73_hparam", storage="sqlite:///models/optuna_study.db")
trials = study.trials
comp = [t for t in trials if t.state.name == "COMPLETE"]
prun = [t for t in trials if t.state.name == "PRUNED"]
runn = [t for t in trials if t.state.name == "RUNNING"]
print(f"Trials: {len(trials)} | Complete: {len(comp)} | Pruned: {len(prun)} | Running: {len(runn)}")
if comp:
    best = sorted(comp, key=lambda t: t.value or -1, reverse=True)[0]
    print(f"Best: score={best.value:.4f}  hERG={best.user_attrs.get('hERG_auc','?')}  CYP={best.user_attrs.get('CYP3A4_auc','?')}")
    print(f"Params: {best.params}")
for i, t in enumerate(comp[-3:]):
    print(f"  Trial {t.number}: score={t.value:.4f}  hERG={t.user_attrs.get('hERG_auc','?'):.4f}  CYP={t.user_attrs.get('CYP3A4_auc','?'):.4f}  params={t.params}")
