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
