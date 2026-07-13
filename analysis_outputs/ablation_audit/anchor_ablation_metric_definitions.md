# Anchor Ablation Metric Definitions

- `num_total`: all 4,000 interactions, including unmapped GT and invalid model outputs.
- `num_evaluable`: interactions with at least one GT referent mapped into the scene anchor inventory.
- `Hit-All`: interactions with at least one mapped predicted/GT overlap divided by all interactions.
- `Hit-Mapped`: the same hit indicator divided by evaluable interactions only.
- `Exact`: exact predicted-set/GT-set equality over evaluable interactions.
- `Set-F1 macro`: arithmetic mean of per-interaction set F1 over evaluable interactions. No-overlap and empty predictions contribute 0. This differs from the historical evaluator `macro_f1`, which omitted blank/zero-F1 rows.
- `micro precision/recall/F1`: aggregate set TP/FP/FN across all interactions. Unmapped-GT interactions can contribute FP if a prediction is made.
- `valid_output`: `response_status == "ok"`. Invalid rows remain in all relevant denominators.
- All exported values are decimal fractions, not percentages.
