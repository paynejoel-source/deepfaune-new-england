# DeepFaune New England

Credit split:

- upstream model and scientific work: the respective DeepFaune New England, MegaDetector, and PyTorch-Wildlife authors and organizations
- workflow design, packaging, and repository maintenance in this repo: `paynejoel-source`

Independent open-source workflow for Frigate-style wildlife video processing using MegaDetector and DeepFaune New England classification. This repository provides local automation, hourly processing, and structured reporting on top of the broader DFNE and PyTorch-Wildlife ecosystem.

This repository is publishable both as a public GitHub project and as a Python package/release artifact.

Core workflow features:

- MegaDetector via `PytorchWildlife`
- DeepFaune New England classification
- JSON per-clip and hourly reporting
- automatic copying of animal-positive clips into a designated folder
- clip-level error isolation so hourly runs continue past bad files
- automatic retention cleanup for generated outputs

## Overview

Technical design notes are in [docs/BUILD_STATUS.md](docs/BUILD_STATUS.md).
Methods and validation notes are in [METHODS.md](METHODS.md).

## Main Files

- Runtime entrypoint: [run_pipeline.py](run_pipeline.py)
- Example config: [config/pipeline_settings.example.yaml](config/pipeline_settings.example.yaml)
- Environment bootstrap: [setup_env.sh](setup_env.sh)
- Model setup: [docs/MODELS.md](docs/MODELS.md)
- Example outputs: [docs/examples](docs/examples)

## Requirements

- Linux environment
- Python 3
- `ffmpeg`
- `ffprobe`
- access to a Frigate-style clip tree or equivalent `.mp4` source layout
- local model files placed under `models/`

## Configuration

Create your local config from the example:

```bash
cp config/pipeline_settings.example.yaml config/pipeline_settings.yaml
```

Then edit `config/pipeline_settings.yaml` for your local mount paths, output paths, and preferred defaults.

The repository does not ship model weights. You will need to place the required detector and classifier weights into `models/` yourself.

## Quickstart

Package-style install from the repo root:

```bash
python -m pip install .
```

Then run the CLI with:

```bash
deepfaune-new-england --check
```

Install system prerequisites:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv
```

Bootstrap the Python environment:

```bash
./setup_env.sh
source .venv/bin/activate
```

Create your local config:

```bash
cp config/pipeline_settings.example.yaml config/pipeline_settings.yaml
```

Add the required model files under `models/` as described in [docs/MODELS.md](docs/MODELS.md).

Run a preflight validation check:

```bash
python run_pipeline.py --check
```

Export existing clip reports to a Camtrap DP package:

```bash
python run_pipeline.py --export-camtrap-dp
```

Validate existing clip reports against a labeled CSV or JSON file:

```bash
python run_pipeline.py --validate validation_cases.csv
```

Optional export tuning in the config:

- `CAMTRAP_OBSERVATION_MODE: "detection"` for one row per detected object
- `CAMTRAP_OBSERVATION_MODE: "clip_top"` for one row per clip using the top prediction
- `CAMTRAP_OBSERVATION_MODE: "species_summary"` for one row per species summary in each clip

Write demo reports without models or a mounted clip tree:

```bash
python run_pipeline.py --demo
```

Validate the pipeline on a local clip:

```bash
python run_pipeline.py --clip test_video/42.17.mp4
```

Run a dry-run hourly window:

```bash
python run_pipeline.py --hourly --camera back_yard --window-start 2026-03-21T16:00:00-04:00 --window-end 2026-03-21T17:00:00-04:00 --dry-run
```

## Basic Usage

Activate the environment:

```bash
source .venv/bin/activate
```

Run hourly mode:

```bash
python run_pipeline.py --hourly
```

Dry-run a specific hourly window:

```bash
python run_pipeline.py --hourly --camera back_yard --window-start 2026-03-21T16:00:00-04:00 --window-end 2026-03-21T17:00:00-04:00 --dry-run
```

Run tests:

```bash
./.venv/bin/python -m unittest discover -s tests
```

Review publish-safe example outputs:

- [docs/examples/clip_report_example.json](docs/examples/clip_report_example.json)
- [docs/examples/hourly_summary_example.json](docs/examples/hourly_summary_example.json)
- [docs/examples/validation_single_dog_case.csv](docs/examples/validation_single_dog_case.csv)
- [docs/examples/validation_single_dog_report.json](docs/examples/validation_single_dog_report.json)

Hourly runs also write a lightweight review bundle under `output/review/...` with:

- `review_summary.json`
- `review_summary.html`
- `output/validation/validation_report.json` for labeled validation runs

The single-dog validation example is intentionally narrow. It demonstrates the validation workflow on one real local clip and should not be interpreted as a meaningful estimate of field performance. In particular, Frigate clip capture can miss animal events, so the current example artifacts do not represent full detection opportunity or recall.

## Notes

- The tracked config file is the example config, not a live machine config.
- Generated output, model weights, virtual environments, and local caches are intentionally excluded from version control.
- Camtrap DP export is additive. It generates `deployments.csv`, `media.csv`, `observations.csv`, and `datapackage.json` from existing clip reports without changing the native JSON format.
- The export also writes `export_info.json` so the package records how it was generated and whether any species labels were left unresolved.

## Disclaimer and Credits

This repository is an independent workflow implementation built on top of the broader DeepFaune New England, MegaDetector, and PyTorch-Wildlife ecosystem. It is not the official upstream USGS or UVM DeepFaune New England repository or software release.

The engineering in this repository focuses on operational use cases such as:

- Frigate-style `.mp4` clip handling
- hourly automation
- structured JSON reporting
- local preflight validation and workflow support

Model weights, upstream model design, and the original DeepFaune New England scientific work belong to their respective upstream authors and organizations.

## Publishing

This repository includes Python package metadata in `pyproject.toml` and a GitHub Actions workflow for PyPI trusted publishing:

- `.github/workflows/publish-pypi.yml`

That makes the repository release-ready without requiring credentials to be stored in git.

Retention is controlled in [config/pipeline_settings.yaml](config/pipeline_settings.yaml) with:

```yaml
RETENTION_DAYS: 30
```
