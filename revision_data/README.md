# MycoExtract revision data

Minimal processed materials for Reviewer 1 comment 9.

| File | Contents |
|---|---|
| 01_Database.xlsx | 247 records, values, units, assay context and source/evidence fields; 63 source identifiers. |
| 02_Extraction_validation.xlsx | Original manual annotations and reference records, scoring and document ledgers, six-field counts, 158 numerical-source decisions, and the seven-publication 43-record complex-table comparison. |
| 03_Kinetic_modelling.xlsx | 72 measurements, grouping and fold assignments, and grouped/nested/joint held-out predictions. |
| 04_Analysis_scripts.zip | Scoring, unit conversion and modelling scripts, with the frozen feature matrix and input preparation helper. |

Record results use the author-checked record ledger. Field results use the original manual field annotations; the numerical-source audit has 158 instances and 152 evaluable units. These denominators are retained separately. No annotation or score was changed when consolidating the files.

The frozen manuscript extraction configuration is ../config/extraction_config_v8.yaml; final text/table prompts are in ../prompts/; biological curation and unit-normalization code is in ../src/. These existing repository files provide the extraction protocol without duplicating the source tree in this data folder.

To recompute the supplied summaries, extract the scripts ZIP into an analysis folder beside the XLSX files, install its requirements, and run prepare_inputs.py followed by reproduce_summary.py. The individual modelling scripts rerun the grouped, nested and joint analyses. Source article PDFs are subject to publisher copyright and are not included.
