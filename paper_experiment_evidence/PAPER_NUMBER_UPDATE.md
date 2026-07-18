# Paper Number Synchronization

The attachment-specified source `vr_triref_aaai2027_full_draft(2).tex` is not present in the remote repository or under `/workspace/usr3`. The only matching VR-TriRef manuscript found was `/workspace/usr3/AAAI27_spj_layout_fixed/vr_triref_aaai2027.tex`.

That available manuscript was audited and updated in place with the minimum experiment-number changes:

- anchor inventory: `198` to `199` parsed canonical anchors;
- Exp.1 mapped coverage: `3,972/4,000` to `4,000/4,000`;
- Qwen3-VL-30B mention-first Exp.1: overall/mapped `76.60/77.14%` to `77.98/77.98%`;
- Exp.2 references: `8,330` to `8,519`, temporal F1 `73.33%` to `73.48%`, Point@100 `24.70%` to `24.97%`, Point@200 `43.64%` to `43.99%`, and Joint@200 `36.36%` to `36.74%`;
- scene tables and mapping-audit denominator text were updated to the rebuilt per-scene values.

No references, template parameters, figure placement, or unrelated prose were changed. The new authoritative tables are in `model_results.csv`, `model_results_by_scene.csv`, and `model_results_by_target_count.csv`.

## Compilation Status

The available environment has no `latexmk`, `pdflatex`, `xelatex`, `lualatex`, or `tectonic` executable. Therefore the requested two complete LaTeX compilations could not be executed here. No new LaTeX error, undefined citation/reference, or overfull-box report can be claimed. The pre-existing PDF was not presented as a rebuild.
