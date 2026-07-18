# Result Validation

Nine sample-level model files contain 4,000 unique `scene::row_index` keys each. Within every experiment, all three sample-set hashes and GT hashes match. Exp.1 has 4,000 mapped GT and zero unmapped rows; Exp.2 has zero missing prediction records; Exp.3 has 4,000 manifest rows and zero excluded IDs.

Qwen3-VL-30B Exp.2 has 3998 valid outputs and 2 invalid records. The two invalid records were produced on all three same-configuration retries and remain empty predictions in the 4,000-row corpus metrics. No output was manually repaired. Qwen3-VL-8B and InternVL3-38B have 4,000 valid Exp.2 outputs. All three Exp.3 runs have 4,000 valid outputs.

Machine-readable checks are in `validation/validation_results.csv` and `.json`; explicit invalid IDs are in `denominator_audit/invalid_output_ids.csv`. The 28 repaired mappings, `scene2::569`, anchor loader checks, aliases, and the sole new canonical anchor `drawer2 = (0.638, -1.227, 5.241)` are audited under `gt_completion/` and `location_region_audit/`.
