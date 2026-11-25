# MycoExtract

MycoExtract extracts and curates quantitative evidence on enzymatic mycotoxin detoxification from scientific literature. This repository contains the extraction workflow, its published configuration, and the data and scripts used to evaluate extraction and model kinetics.

## Repository layout

| Path | Contents |
| --- | --- |
| [src/](src/) | Extraction, aggregation, validation rules, and unit normalization |
| [config/](config/) and [prompts/](prompts/) | The v8 configuration and its text, table-image, review, and sequence-verification prompts |
| [scripts/](scripts/) | Extraction entry points and validation utilities |
| [scripts/analysis/](scripts/analysis/) | Scripts for preparing inputs, reproducing results, and drawing figures |
| [data/database/](data/database/) | Curated records and source-publication index in [mycoextract.xlsx](data/database/mycoextract.xlsx) |
| [data/validation/](data/validation/) | [Extraction validation workbook](data/validation/extraction_validation.xlsx) with manual annotations, evaluation ledgers, numerical source checks, and complex-table comparisons |
| [data/modeling/](data/modeling/) | [Kinetics workbook](data/modeling/kinetic_modeling.xlsx), original training CSV, frozen feature matrix, fold metrics, and retrospective prediction data |
| [figures/](figures/) | Released modeling figures and figures from the additional retrospective analysis |
| [tests/](tests/) | Extraction and curation regression tests |

## Extraction

Install the dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
```

The [v8 configuration](config/extraction_config_v8.yaml) records the model identifiers used for the study. For a live run, copy [.env.example](.env.example) to `.env` and set your API credentials. A live rerun also depends on availability of the original provider endpoints and models.

The runner expects parsed paper folders with `full.md` and MinerU-style structured-content files. Place your parsed papers in a local directory such as `data/papers`.

```bash
python scripts/run_all_papers_full_extraction.py --all-papers-dir data/papers --output-root analysis_outputs/extraction
```

Add `--inventory-only` to inspect the inputs without making model calls. The original publisher PDFs are omitted from the data release.

## Reproduce the analyses

The following commands use the deposited workbooks, [original training CSV](data/modeling/kinetic_modeling_dataset.csv), and frozen features. The CSV preserves the original numerical precision for tree-based models. Use Python 3.12 and install with the [reproduction constraints](scripts/analysis/reproduction-constraints.txt). Newer versions of scikit-learn can assign groups to different folds when group sizes are tied.

```bash
python -m pip install -r requirements.txt -c scripts/analysis/reproduction-constraints.txt
```

```bash
python scripts/analysis/prepare_inputs.py
python scripts/analysis/reproduce_summary.py
python scripts/analysis/reanalyse_existing_data.py
python scripts/analysis/nested_all_targets_20261010.py
python scripts/analysis/joint_unseen_cluster_substrate_20261010.py
python scripts/analysis/plot_model_results.py
```

The last command redraws [Figure S2](figures/modeling/Figure_S2_fixed_holdout_boxplots.pdf) and [Figure S3](figures/modeling/Figure_S3_nested_boxplots.pdf) from the deposited [fold metrics](data/modeling/fold_metrics/). The initial feature-generation script, [train_kinetics_model.py](scripts/analysis/train_kinetics_model.py), additionally requires PyTorch, fair-esm, and access to the original embedding checkpoint. Rerun `prepare_inputs.py` after regenerating features to restore the frozen inputs for the holdout analyses.

The [retrospective data](data/modeling/retrospective/) contain all 50 predictions, the training-mean baselines, and their source annotations. The [retrospective figures](figures/retrospective/) accompany that additional analysis. They use descriptive names independent of the manuscript SI numbering.

```bash
python scripts/analysis/retrospective/reproduce_predictions.py
python scripts/analysis/retrospective/plot_primary.py
python scripts/analysis/retrospective/plot_expanded.py
```

The analysis commands write their generated outputs to `analysis_outputs/reproduction/`. Verify the deposited files with `sha256sum -c data/SHA256SUMS.txt` from the repository root, or consult the [checksums](data/SHA256SUMS.txt).

## Tests and license

Run `python scripts/validate_project.py` to check the release layout, and `python -m pytest tests` to run the tests. The code is distributed under the [MIT License](LICENSE).
