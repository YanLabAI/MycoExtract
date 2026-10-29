# MycoExtract data and analysis materials

Processed materials supporting the manuscript and its 29 October revision.

| File | Contents |
|---|---|
| 01_Database.xlsx | 247 records with values, units, assay context and source/evidence fields; 63 source identifiers. |
| 02_Extraction_validation.xlsx | Manual annotations and references, record/document ledgers, six-field counts, 158 numerical-source decisions and the seven-publication 43-record complex-table comparison. |
| 03_Kinetic_modelling.xlsx | 72 measurements, grouping/fold assignments and grouped/nested/joint held-out predictions. |
| 04_Analysis_scripts.zip | Scoring, unit conversion and modelling scripts, the frozen training feature matrix and input-preparation helper. |
| 05_Additional_predictions_and_figure_data.zip | All 50 additional retrospective predictions, candidate features and source annotations; Figures S5–S7 with source CSVs and portable verification/plotting scripts. |
| SHA256SUMS.txt | Checksums of the five data and analysis files. |

The first four files retain their earlier numerical contents. The new fifth
archive completes the additional-prediction and figure-source release.

## Evaluation denominators

Record results use the author-checked record ledger. Field results use the
original manual field annotations. The numerical-source audit contains 158
instances and 152 evaluable units; these denominators remain distinct.

## Current SI and fold source data

The fold distributions are presented in Figures S6 and S7. The two duplicated
fold-by-fold SI tables were removed from the 29 October revision; their complete
contents, including pooled rows and selected models, are retained in the fifth
archive as Fixed_Ridge_fold_and_pooled_metrics.csv and Nested_outer_fold_details.csv.
The separate Figure_S6_fold_metrics.csv and Figure_S7_outer_fold_metrics.csv contain
the 15 plotted points per figure. No fold value or figure input was changed.

Current Tables S6 and S7 retain pooled and summary model metrics. Current Tables
S8, S9 and S10 report the numerical audit, field evaluation and joint holdout.
Historical scripts and workbook sheet names may retain earlier identifiers.

## Reproduction

For the main summaries, extract 04_Analysis_scripts.zip into an analysis folder
beside the XLSX files, install its requirements, and run prepare_inputs.py followed
by reproduce_summary.py. The individual modelling scripts reproduce the grouped,
nested and joint analyses.

Extract 05_Additional_predictions_and_figure_data.zip into additional_predictions
beside these files. Its README describes how to verify all 50 additional predictions
using the frozen training and candidate features, and how to reproduce Figures
S5–S7. No LLM endpoint is required for this frozen-feature verification. The archive
includes verification_report.json with the observed numerical tolerance and runtime.

The frozen extraction configuration is ../config/extraction_config_v8.yaml;
the text/table prompts are in ../prompts/; biological curation and unit conversion
are in ../src/. Source article PDFs are not redistributed. PDF files in the fifth
archive are generated result plots.
