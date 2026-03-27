# DeepFaune New England Technical Notes

Last updated: 2026-03-27

## Summary

DeepFaune New England is a wildlife clip-processing pipeline for Frigate-style `.mp4` clips. It is designed to:

1. Read clips from a mounted clip tree
2. Detect animals with MegaDetector via `PytorchWildlife`
3. Classify detected animals with the DeepFaune New England model
4. Write detailed per-clip JSON reports
5. Support scheduler-driven hourly processing

## Current Capabilities

- Python environment bootstrap via [`setup_env.sh`](/home/joel/deepfaune_new_england/setup_env.sh)
- Single-clip processing via [`run_pipeline.py`](/home/joel/deepfaune_new_england/run_pipeline.py)
- Hourly orchestration mode with camera filtering and dry-run support
- Absolute UTC and local timestamps derived from Frigate-style clip filenames
- Per-clip and hourly JSON reports
- Clip-level error isolation so hourly runs continue after bad files
- Positive-clip copying based on classification confidence
- Whole-run file lock to prevent overlapping executions
- Retention cleanup for generated outputs
- `systemd` service and timer templates for scheduled execution

## Architecture

### Inference Flow

1. Select one or more `.mp4` clips
2. Extract JPG frames with `ffmpeg`
3. Read exact sampled frame timestamps with `ffprobe`
4. Run MegaDetector on extracted frames
5. Run DeepFaune New England classification on animal detections
6. Build a per-clip JSON report
7. Optionally aggregate reports into an hourly summary

### Detection

- MegaDetector v6
- Current variant: `MDV6-yolov9-c`
- Implementation: [`core/detector.py`](/home/joel/deepfaune_new_england/core/detector.py)

### Classification

- DeepFaune New England
- Implementation: [`core/classifier.py`](/home/joel/deepfaune_new_england/core/classifier.py)
- A local wrapper is used so the classifier loads downloaded model weights directly from the local `models/` directory

### Timestamp Handling

Two timestamp layers are used:

- Exact sampled frame offsets from `ffprobe`
- Absolute wall-clock timestamps derived from Frigate-style clip filenames such as `1774126221.93046-1774126800.07907.mp4`

This allows each frame to be reported with:

- `timestamp_seconds`
- `timestamp_utc`
- `timestamp_local`

## Hourly Mode

Hourly mode is implemented in [`run_pipeline.py`](/home/joel/deepfaune_new_england/run_pipeline.py).

Supported examples:

```bash
python run_pipeline.py --hourly
python run_pipeline.py --hourly --camera back_yard
python run_pipeline.py --hourly --window-start 2026-03-21T16:00:00-04:00 --window-end 2026-03-21T17:00:00-04:00 --dry-run
```

Hourly mode:

- selects clips whose parsed start timestamp falls inside a requested hour window
- can filter by camera name using the clip parent directory
- writes hourly summaries for both dry-run and normal processing
- writes per-clip reports during normal processing

## Outputs

### Per-Clip JSON

Each clip report includes:

- clip identifiers and paths
- clip start and end epoch values
- UTC and local timestamps
- frame sampling interval
- processing timing
- detector output
- classifier output where available
- summary counts and top frames

### Hourly JSON

Each hourly summary includes:

- requested window in UTC and local time
- optional camera filter
- clip counts and status counts
- total animal detections
- species summary counts
- selected clips and processed reports
- copied positive clips and top frames

## Positive Clip Copying

Positive clip copying is controlled by:

- `COPY_POSITIVE_CLIPS`
- `POSITIVE_CLIPS_DIR`
- `POSITIVE_CLIP_MIN_CONFIDENCE`

Only clips with animal detections and a sufficiently strong top classification are copied. Relative paths under `BEAST_MOUNT_PATH` are preserved.

## Concurrency and Retention

- Whole-run lock path is controlled by `LOCK_FILE`
- A concurrent run exits immediately with a clear error
- Generated outputs can be pruned automatically with `RETENTION_DAYS`

## Known Limits

- The published repository does not include model weights
- Local mount paths must be configured by each user
- Thresholds and frame sampling may still require tuning for specific camera placements or wildlife behavior
