# Model Setup

This repository does not include model weights.

You must provide the following files locally under `models/`:

- `models/MDV6-yolov9-c.pt`
- `models/dfne_weights_v1_0.pth`

## Expected Paths

The default config expects:

```yaml
DETECTOR_MODEL_PATH: "models/MDV6-yolov9-c.pt"
CLASSIFIER_MODEL_PATH: "models/dfne_weights_v1_0.pth"
```

## Detector Model

The detector is MegaDetector v6 using the `MDV6-yolov9-c` variant.

Place the detector checkpoint at:

```bash
models/MDV6-yolov9-c.pt
```

## Classifier Model

The classifier is the DeepFaune New England model.

Place the classifier checkpoint at:

```bash
models/dfne_weights_v1_0.pth
```

## Notes

- This repository intentionally does not redistribute model weights.
- Confirm the applicable licenses and redistribution terms before sharing model files.
- If you use different local paths, update `config/pipeline_settings.yaml` accordingly.
