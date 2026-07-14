# Result Validation

The exporter read the final evaluator details and wrote nine files with unique `scene::row_index` keys. All Exp.1/Exp.2 files contain 4,000 rows; all Exp.3 files contain 3,971 rows. Validation results are in `validation/validation_results.csv` and `.json`.

Exp.1 has 3,972 mapped and 28 unmapped GT rows. Exp.2 keeps 4,000 denominator rows; Qwen3-VL-30B has 3,971 prediction records, 29 missing records, 7 parse failures, and 3,964 valid parses. Qwen3-VL-8B and InternVL3-38B each have 29 missing and no parse failures. Exp.3 excludes 29 rows from the 4,000-row interaction universe: 28 are unmapped in the Exp.3 anchor tables and one (`scene2::569`) is the mapped `drawer1` interaction absent from the Exp.3 scene2 anchor table (`missing_gt_anchor`). The explicit IDs are in `denominator_audit/exp3_excluded_ids.csv`.

The Exp.1 micro metrics in this bundle intentionally follow the audit definition and aggregate all rows, including false positives on unmapped GT rows.
