# DeepFaune New England

Wildlife clip processing pipeline for mounted Beast/Frigate `.mp4` clips using:

- MegaDetector via `PytorchWildlife`
- DeepFaune New England classification
- JSON per-clip and hourly reporting
- automatic copying of animal-positive clips into a designated folder
- clip-level error isolation so hourly runs continue past bad files
- automatic retention cleanup for generated outputs

Created and maintained by Joel Payne.

## Overview

Technical design notes are in [docs/BUILD_STATUS.md](/home/joel/deepfaune_new_england/docs/BUILD_STATUS.md).

## Main Files

- Runtime entrypoint: [run_pipeline.py](/home/joel/deepfaune_new_england/run_pipeline.py)
- Example config: [config/pipeline_settings.example.yaml](/home/joel/deepfaune_new_england/config/pipeline_settings.example.yaml)
- Environment bootstrap: [setup_env.sh](/home/joel/deepfaune_new_england/setup_env.sh)
- Model setup: [docs/MODELS.md](/home/joel/deepfaune_new_england/docs/MODELS.md)

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

Add the required model files under `models/` as described in [docs/MODELS.md](/home/joel/deepfaune_new_england/docs/MODELS.md).

Run a preflight validation check:

```bash
python run_pipeline.py --check
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

## Notes

- The tracked config file is the example config, not a live machine config.
- Generated output, model weights, virtual environments, and local caches are intentionally excluded from version control.

Retention is controlled in [config/pipeline_settings.yaml](/home/joel/deepfaune_new_england/config/pipeline_settings.yaml) with:

```yaml
RETENTION_DAYS: 30
```
