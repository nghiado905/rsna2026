# RSNA 2025 Intracranial Aneurysm Detection — Local Pipelines

This directory contains an experimental extension of the MIC-DKFZ nnU-Net solution for the RSNA 2025 Intracranial Aneurysm Detection challenge. It adds:

- a Stage-1 Circle of Willis (CoW) region-of-interest detector;
- TopCoW vessel segmentation;
- vessel-guided image enhancement before aneurysm prediction;
- classification and localization evaluation utilities;
- export of voxel-space and physical coordinates in millimetres.

> **Important:** the current dataset and aneurysm checkpoint are **single-channel**. `dataset.json` defines only channel `0`, and the supplied plans contain intensity statistics only for channel `0`.

## Sources and Attribution

| Component | Source | Role in this project |
|---|---|---|
| Aneurysm model | [MIC-DKFZ RSNA 2025 solution](https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution) | nnU-Net base, dataset conversion, trainer, plans, and baseline inference |
| Stage-1 ROI model | [RSNA 2025 second-place solution](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution) and [BraveCoWCoW source code](https://github.com/PengchengShi1220/RSNA2025_Intracranial-Aneurysm-Detection) | Fast tri-axial vessel-region bounding-box prediction |
| Vessel segmentation | [fmusio/TopCoWSubmissions](https://github.com/fmusio/TopCoWSubmissions) | CTA/MRA CoW vessel segmentation code and weights |

This is a local experimental combination of those components, not an exact reproduction of any single source solution.

## Pipeline Overview

```text
DICOM series
    |
    +--> Pipeline 1: fixed 200 x 160 x 160 mm crop
    |       -> single-channel aneurysm model
    |       -> submission CSV + coordinate CSV
    |
    +--> Pipeline 2: Stage-1 vessel/CoW bounding box
    |       -> cropped image
    |       -> single-channel aneurysm model
    |       -> submission CSV + coordinate CSV
    |
    `--> Pipeline 3: Stage-1 bounding box
            -> cropped image
            -> TopCoW vessel mask
            -> vessel overlay on the same image channel
            -> single-channel aneurysm model
            -> submission CSV + coordinate CSV
```

Two concepts must be kept separate:

- **Stage-1 bounding box:** returns `(z_min, z_max, y_min, y_max, x_min, x_max)` for ROI cropping. It does not produce a saved 3D vessel mask.
- **TopCoW segmentation:** produces a 3D vessel mask that can be used to enhance the image.

## Installation

A dedicated Conda environment on Linux or WSL is recommended. Windows may work but is not the primary tested environment of the upstream solution.

```bash
conda create -n rsna2025 python=3.11 -y
conda activate rsna2025

# Install a CUDA-compatible PyTorch build first, then install the project dependencies.
pip install git+https://github.com/MIC-DKFZ/batchgeneratorsv2.git@07541d7eb5a4839aa4a5e494a123f3fe69ccfd4f
pip install -r requirements.txt
pip install -e .
```

### nnU-Net Environment Variables

Linux:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

PowerShell:

```powershell
$env:nnUNet_raw = "D:\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\nnUNet_preprocessed"
$env:nnUNet_results = "D:\nnUNet_results"
```

Place `nnUNet_preprocessed` on fast storage because preprocessing and training are I/O intensive.

## Model Weights

### Aneurysm Model

Download the checkpoint referenced by the [MIC-DKFZ solution](https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution) and arrange it as an nnU-Net result folder:

```text
nnUNet_results/
`-- Dataset004_iarsna_crop/
    `-- Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32/
        `-- fold_all/
            `-- checkpoint_epoch_1500.pth
```

### Stage-1 ROI Model

The Stage-1 model is published as [Dataset180 2D vessel box segmentation](https://www.kaggle.com/models/pengchengshi/dataset180_2d_vessel_box_seg_stable). Pass the training-output directory containing its fold and checkpoint to the relevant Stage-1 argument.

### TopCoW Vessel Models

TopCoW publishes its weights in this [Google Drive folder](https://drive.google.com/drive/folders/14u33bdB8MawGJ7Z4M5AjNi-i3dx60yWj?usp=sharing). CTA and MRA use separate models. Preserve the nnU-Net result structure for each model.

TopCoW expects LPS+ NIfTI images. Verify orientation, spacing, direction, and origin before applying a predicted mask to an image.

## Dataset Preparation

The expected Kaggle dataset layout is:

```text
rsna-intracranial-aneurysm-detection/
|-- series/
|   |-- <SeriesInstanceUID>/
|   |   `-- *.dcm
|   `-- ...
|-- segmentations/
|-- train.csv
`-- train_localizers.csv
```

The conversion pipeline:

1. loads and sorts each DICOM series;
2. creates a 3D volume;
3. crops the selected ROI;
4. flips the Y axis to match the original training convention;
5. converts localizers into label volumes;
6. writes `imagesTr`, `labelsTr`, `labels.csv`, `ids_mapping.json`, and `dataset.json`.

### Option A — Fixed Upstream Crop

`official_data_to_nnunet_default.py` uses an approximately `200 x 160 x 160 mm` fixed ROI:

```bash
python rsna-aneurysm-v1/dataset_conversion/kaggle_2025_rsna/official_data_to_nnunet_default.py \
  -i /data/rsna-intracranial-aneurysm-detection \
  -o "$nnUNet_raw/Dataset004_iarsna_crop" \
  --np 8
```

### Option B — Stage-1 Bounding-Box Crop

`official_data_to_nnunet.py` uses the Stage-1 prediction. If Stage-1 fails, `load_and_crop()` falls back to the fixed crop.

```bash
python rsna-aneurysm-v1/dataset_conversion/kaggle_2025_rsna/official_data_to_nnunet.py \
  -i /data/rsna-intracranial-aneurysm-detection \
  -o "$nnUNet_raw/Dataset004_iarsna_crop" \
  --np 8 \
  --stage1_model_dir /models/Dataset180_2D_vessel_box_seg/nnUNetTrainer__nnUNetPlans__2d \
  --stage1_checkpoint checkpoint_final.pth \
  --stage1_fold 0 \
  --device cuda
```

Each case still contains only one image channel:

```text
imagesTr/iarsna_0000_0000.nii  # channel 0: cropped image
labelsTr/iarsna_0000.nii       # aneurysm-location labels
```

## Vessel Segmentation and Image Overlay

Run the modality-appropriate TopCoW model to create a vessel mask for each cropped image. Image and mask identifiers must match, and their shapes and physical geometry must be aligned.

Example intermediate layout:

```text
cropped_images/
|-- iarsna_0000_0000.nii
`-- iarsna_0001_0000.nii

vessel_masks/
|-- iarsna_0000.nii.gz
`-- iarsna_0001.nii.gz
```

`overlay_on_images.py` does not create another channel. It multiplies intensity inside `mask > 0`, clips the volume, applies z-score normalization, and saves the result as channel `0`:

```bash
python rsna-aneurysm-v1/overlay_on_images.py \
  --images-dir /data/cropped_images \
  --mask-dir /data/vessel_masks \
  --output-dir /data/Dataset104_iarsna_crop_overlay/imagesTr \
  --multiply-factor 1.15 \
  --num-workers 8
```

## Planning, Preprocessing, and Training

```bash
nnUNetv2_extract_fingerprint -d 004 -np 16
nnUNetv2_plan_experiment -d 004 -pl nnUNetPlannerResEncM
```

Copy the manually adjusted plans file:

```text
rsna-aneurysm-v1/dataset_conversion/kaggle_2025_rsna/plans/nnUNetResEncUNetMPlans.json
```

to:

```text
$nnUNet_preprocessed/Dataset004_iarsna_crop/nnUNetResEncUNetMPlans.json
```

Preprocess:

```bash
nnUNetv2_preprocess \
  -d 004 \
  -np 16 \
  -c 3d_fullres_bs32 \
  -p nnUNetResEncUNetMPlans
```

Train:

```bash
nnUNet_n_proc_DA=16 nnUNetv2_train \
  004 3d_fullres_bs32 all \
  -tr Kaggle2025RSNATrainer \
  -num_gpus 4 \
  -p nnUNetResEncUNetMPlans
```

The upstream configuration used four GPUs with approximately 40 GB of VRAM each. Reduce batch size, patch size, augmentation workers, or GPU count when adapting it to smaller hardware.

For vessel-overlay training, create a new dataset ID, regenerate its fingerprint and plans, preprocess it again, and train using overlay images as `*_0000.nii`. Do not reuse intensity statistics from the original-image dataset.

## Inference Pipelines

### Pipeline 1 — Fixed-Crop Baseline

```bash
python rsna-aneurysm-v1/inference/kaggle_2025_rsna/inference_default.py \
  -i /data/series \
  -o /output/default/submission.csv \
  -m /models/Dataset004_iarsna_crop/Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32 \
  -c checkpoint_epoch_1500.pth \
  --fold "('all',)" \
  --disable_tta \
  --ids-mapping-json /data/Dataset004_iarsna_crop/ids_mapping.json
```

Flow: `DICOM -> fixed crop -> Y flip -> aneurysm model`.

### Pipeline 2 — Stage-1 Bounding-Box Crop

```bash
python rsna-aneurysm-v1/inference/kaggle_2025_rsna/inference_crop_classfication.py \
  -i /data/series \
  -o /output/stage1_crop/submission.csv \
  -m /models/Dataset004_iarsna_crop/Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32 \
  -c checkpoint_epoch_1500.pth \
  --fold "('all',)" \
  --stage1_model_dir /models/Dataset180_2D_vessel_box_seg/nnUNetTrainer__nnUNetPlans__2d \
  --stage1_checkpoint checkpoint_final.pth \
  --stage1_fold 0 \
  --ids-mapping-json /data/Dataset004_iarsna_crop/ids_mapping.json
```

Flow: `DICOM -> Stage-1 bounding box -> crop -> Y flip -> aneurysm model`.

The `[CROP] ... source=stage1` log confirms that Stage-1 was used. `source=fixed_fallback` indicates that Stage-1 failed and the fixed crop was used.

### Pipeline 3 — Stage-1 Crop with Vessel Guidance

```bash
python rsna-aneurysm-v1/inference/kaggle_2025_rsna/inference_crop_vessel_classification.py \
  -i /data/series \
  -o /output/vessel_overlay/submission.csv \
  -m /models/Dataset104_iarsna_crop_overlay/Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres \
  -c checkpoint_final.pth \
  --fold "('all',)" \
  --stage1-model-dir /models/Dataset180_2D_vessel_box_seg/nnUNetTrainer__nnUNetPlans__2d \
  --stage1-checkpoint checkpoint_final.pth \
  --stage1-fold 0 \
  --vessel-model-dir-ct /models/topcow_ct \
  --vessel-model-dir-mr /models/topcow_mr \
  --vessel-checkpoint checkpoint_final.pth \
  --vessel-fold "(0,1,2,3,4)" \
  --vessel-feature-mode overlay
```

Flow: `DICOM -> Stage-1 crop -> TopCoW mask -> vessel-enhanced image -> aneurysm model`.

## Evaluation

Evaluation is split into classification and coordinate-based detection/localization. Always evaluate all pipelines on the same held-out `SeriesInstanceUID` set.

### Classification Evaluation

Script:

```text
eval/classification/eval_rsna_metrics.py
```

The evaluator reads a submission CSV and a `train_localizers.csv`-style ground-truth file. It converts the localizers into 14 binary targets:

- `Aneurysm Present`;
- 13 anatomical location labels.

The primary score is a weighted mean of per-label ROC AUC values:

```text
weight(Aneurysm Present) = 13
weight(each location)    = 1
```

Therefore, `Aneurysm Present` contributes half of the total score weight. The script also reports per-label ROC AUC, PR AUC, precision, recall, F1, accuracy, support, and aggregate multilabel metrics.

Run from `rsna-aneurysm-v1`:

```bash
python eval/classification/eval_rsna_metrics.py \
  --submission /output/stage1_crop/submission.csv \
  --ground-truth /data/validation/train_localizers.csv \
  --mode submission \
  --threshold 0.5 \
  --output-dir /output/evaluation/classification
```

PowerShell example:

```powershell
python .\eval\classification\eval_rsna_metrics.py `
  --submission "D:\outputs\stage1_crop\submission.csv" `
  --ground-truth "D:\validation\train_localizers.csv" `
  --mode submission `
  --threshold 0.5 `
  --output-dir "D:\outputs\evaluation\classification"
```

#### Evaluation Modes

- `submission`: evaluates every UID present in the submission. A UID absent from `train_localizers.csv` is treated as a negative case. This is the recommended mode when the submission contains the complete held-out split.
- `overlap`: evaluates only UIDs shared by the submission and ground truth. Because localizer files commonly contain only positive annotations, this mode may exclude negative cases and produce a biased score.

`--threshold` affects binary metrics such as precision, recall, and F1. ROC AUC uses continuous probabilities and is independent of this threshold.

#### Required Classification Inputs

The submission CSV must contain:

```text
SeriesInstanceUID
Aneurysm Present
Other Posterior Circulation
Basilar Tip
Right Posterior Communicating Artery
Left Posterior Communicating Artery
Right Infraclinoid Internal Carotid Artery
Left Infraclinoid Internal Carotid Artery
Right Supraclinoid Internal Carotid Artery
Left Supraclinoid Internal Carotid Artery
Right Middle Cerebral Artery
Left Middle Cerebral Artery
Right Anterior Cerebral Artery
Left Anterior Cerebral Artery
Anterior Communicating Artery
```

The ground-truth CSV must contain at least:

```text
SeriesInstanceUID, location
```


### Detection and Localization Evaluation in Crop Millimetres

Script:

```text
eval/detection/test_case/crop_mm_batch.py
```

Run from `rsna-aneurysm-v1`:

```bash
python eval/detection/test_case/crop_mm_batch.py
```
PowerShell:

```powershell
python .\eval\detection\test_case\crop_mm_batch.py
```

## References

- [MIC-DKFZ — RSNA 2025 solution](https://github.com/MIC-DKFZ/kaggle-rsna-intracranial-aneurysm-detection-2025-solution)
- [BraveCoWCoW — second-place source code](https://github.com/PengchengShi1220/RSNA2025_Intracranial-Aneurysm-Detection)
- [Kaggle — second-place write-up](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/2nd-place-solution)
- [TopCoW inference code and weight instructions](https://github.com/fmusio/TopCoWSubmissions)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)

## License

Retain the license and attribution requirements of every upstream source. The MIC-DKFZ and BraveCoWCoW repositories were published under Apache-2.0 at the time of reference. Model weights and datasets may have separate terms that must be checked before redistribution or commercial use.
