"""Create nnU-Net splits_final.json with modality and multilabel stratification.

The script reads an nnU-Net dataset directory, maps case ids back to
SeriesInstanceUID through ids_mapping.json, reads the official RSNA metadata,
and writes a nnU-Net compatible splits_final.json.

Example:
    python dataset_conversion/kaggle_2025_rsna/create_nnunet_splits.py \
        --dataset-dir F:/Dataset/Dataset004_iarsna_fixed \
        --metadata-csv F:/final/train.csv \
        --output-json F:/Training/nnUNet_preprocessed/Dataset004_iarsna_fixed/splits_final.json \
        --num-folds 5 \
        --seed 12345
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_CLASSIFICATION_COLUMNS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="nnU-Net raw dataset directory containing imagesTr/, labelsTr/, and ids_mapping.json.",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="Official train.csv or dataset labels.csv containing SeriesInstanceUID, Modality, and label columns.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Output splits_final.json path. Defaults to <dataset-dir>/splits_final.json. "
            "For nnU-Net training, copy/write it to nnUNet_preprocessed/DatasetXXX/splits_final.json."
        ),
    )
    parser.add_argument("--num-folds", type=int, default=5, help="Number of folds.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--require-label-file",
        action="store_true",
        help="Use only cases that have a corresponding labelsTr/<case_id>.nii(.gz).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV summary with one row per case and assigned fold.",
    )
    return parser.parse_args()


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def case_id_from_image(path: Path) -> str:
    name = strip_nii_suffix(path)
    return name[:-5] if name.endswith("_0000") else name


def list_case_ids(images_dir: Path) -> list[str]:
    files = sorted(list(images_dir.glob("*.nii")) + list(images_dir.glob("*.nii.gz")))
    return [case_id_from_image(path) for path in files]


def label_exists(labels_dir: Path, case_id: str) -> bool:
    return (labels_dir / f"{case_id}.nii").exists() or (labels_dir / f"{case_id}.nii.gz").exists()


def load_id_to_uid(mapping_path: Path) -> dict[str, str]:
    with mapping_path.open("r", encoding="utf-8") as f:
        uid_to_id = json.load(f)
    return {case_id: uid for uid, case_id in uid_to_id.items()}


def load_metadata(metadata_csv: Path) -> dict[str, dict[str, str]]:
    with metadata_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return {row["SeriesInstanceUID"]: row for row in reader}


def as_int(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def build_case_rows(
    dataset_dir: Path,
    metadata_csv: Path,
    require_label_file: bool,
) -> list[dict]:
    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    id_to_uid = load_id_to_uid(dataset_dir / "ids_mapping.json")
    metadata = load_metadata(metadata_csv)

    rows = []
    missing_uid = []
    missing_metadata = []
    missing_label = []

    for case_id in list_case_ids(images_dir):
        uid = id_to_uid.get(case_id)
        if uid is None:
            missing_uid.append(case_id)
            continue
        meta = metadata.get(uid)
        if meta is None:
            missing_metadata.append(case_id)
            continue
        has_label_file = label_exists(labels_dir, case_id)
        if require_label_file and not has_label_file:
            missing_label.append(case_id)
            continue

        class_values = {col: as_int(meta, col) for col in DEFAULT_CLASSIFICATION_COLUMNS}
        aneurysm_present = as_int(meta, "Aneurysm Present", int(any(class_values.values())))
        modality = meta.get("Modality", "UNKNOWN") or "UNKNOWN"
        positive_locations = [col for col, value in class_values.items() if value > 0]

        rows.append(
            {
                "case_id": case_id,
                "series_uid": uid,
                "modality": modality,
                "aneurysm_present": aneurysm_present,
                "has_label_file": has_label_file,
                "positive_locations": positive_locations,
                "class_values": class_values,
            }
        )

    if missing_uid:
        print(f"[WARN] Missing ids_mapping entries: {len(missing_uid)}")
    if missing_metadata:
        print(f"[WARN] Missing metadata rows: {len(missing_metadata)}")
    if missing_label:
        print(f"[WARN] Skipped cases without labelsTr file: {len(missing_label)}")
    return rows


def case_features(row: dict) -> list[str]:
    features = [
        f"modality={row['modality']}",
        f"present={row['aneurysm_present']}",
        f"modality_present={row['modality']}:{row['aneurysm_present']}",
    ]
    for col, value in row["class_values"].items():
        if value > 0:
            features.append(f"label={col}")
            features.append(f"modality_label={row['modality']}:{col}")
    return features


def make_stratified_folds(rows: list[dict], num_folds: int, seed: int) -> list[list[str]]:
    rng = random.Random(seed)
    total = Counter()
    row_features = {}
    for row in rows:
        feats = case_features(row)
        row_features[row["case_id"]] = feats
        total.update(feats)

    base_size = len(rows) // num_folds
    remainder = len(rows) % num_folds
    fold_capacities = [base_size + (1 if fold_idx < remainder else 0) for fold_idx in range(num_folds)]
    target_per_fold = {key: value / num_folds for key, value in total.items()}
    fold_counts = [Counter() for _ in range(num_folds)]
    folds = [[] for _ in range(num_folds)]

    shuffled = rows[:]
    rng.shuffle(shuffled)
    def rarity_score(row: dict) -> float:
        feats = row_features[row["case_id"]]
        return sum(1.0 / max(total[feat], 1) for feat in feats)

    shuffled.sort(key=lambda row: (-rarity_score(row), row["modality"], row["case_id"]))

    for row in shuffled:
        feats = row_features[row["case_id"]]

        def score(fold_idx: int) -> tuple[float, int]:
            if len(folds[fold_idx]) >= fold_capacities[fold_idx]:
                return float("inf"), len(folds[fold_idx])

            projected_size_ratio = (len(folds[fold_idx]) + 1) / max(fold_capacities[fold_idx], 1)
            size_penalty = projected_size_ratio * 0.05
            feature_delta = 0.0
            for feat in feats:
                current = fold_counts[fold_idx][feat]
                projected = current + 1
                target = target_per_fold.get(feat, 0.0)
                scale = max(target, 1.0)
                feature_delta += ((projected - target) / scale) ** 2
                feature_delta -= ((current - target) / scale) ** 2
            return feature_delta + size_penalty, len(folds[fold_idx])

        best_fold = min(range(num_folds), key=score)
        folds[best_fold].append(row["case_id"])
        fold_counts[best_fold].update(feats)

    for fold in folds:
        fold.sort()
    return folds


def write_splits(folds: list[list[str]], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    all_cases = set().union(*[set(fold) for fold in folds])
    splits = []
    for fold_idx, val_cases in enumerate(folds):
        val_set = set(val_cases)
        train_cases = sorted(all_cases - val_set)
        splits.append({"train": train_cases, "val": sorted(val_cases)})
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=4)


def summarize(rows: list[dict], folds: list[list[str]]) -> None:
    by_case = {row["case_id"]: row for row in rows}
    print(f"Total cases: {len(rows)}")
    print(f"Total folds: {len(folds)}")
    print()
    for fold_idx, cases in enumerate(folds):
        fold_rows = [by_case[case_id] for case_id in cases]
        modality_count = Counter(row["modality"] for row in fold_rows)
        present_count = Counter(row["aneurysm_present"] for row in fold_rows)
        label_count = Counter()
        for row in fold_rows:
            for loc in row["positive_locations"]:
                label_count[loc] += 1
        print(f"Fold {fold_idx}: n={len(cases)}")
        print(f"  modality: {dict(sorted(modality_count.items()))}")
        print(f"  aneurysm_present: {dict(sorted(present_count.items()))}")
        print(f"  positive labels: {dict(sorted(label_count.items()))}")


def write_summary_csv(rows: list[dict], folds: list[list[str]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fold_by_case = {}
    for fold_idx, cases in enumerate(folds):
        for case_id in cases:
            fold_by_case[case_id] = fold_idx

    fieldnames = [
        "case_id",
        "series_uid",
        "fold",
        "modality",
        "aneurysm_present",
        "has_label_file",
        "positive_locations",
        *DEFAULT_CLASSIFICATION_COLUMNS,
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["case_id"]):
            out = {
                "case_id": row["case_id"],
                "series_uid": row["series_uid"],
                "fold": fold_by_case[row["case_id"]],
                "modality": row["modality"],
                "aneurysm_present": row["aneurysm_present"],
                "has_label_file": row["has_label_file"],
                "positive_locations": "|".join(row["positive_locations"]),
            }
            out.update(row["class_values"])
            writer.writerow(out)


def main() -> None:
    args = parse_args()
    output_json = args.output_json or (args.dataset_dir / "splits_final.json")

    rows = build_case_rows(args.dataset_dir, args.metadata_csv, args.require_label_file)
    if not rows:
        raise RuntimeError("No cases found for split generation.")
    if args.num_folds < 2:
        raise ValueError("--num-folds must be >= 2")
    if args.num_folds > len(rows):
        raise ValueError("--num-folds cannot be larger than number of cases.")

    folds = make_stratified_folds(rows, args.num_folds, args.seed)
    write_splits(folds, output_json)
    summarize(rows, folds)
    print()
    print(f"Wrote nnU-Net splits: {output_json}")

    if args.summary_csv is not None:
        write_summary_csv(rows, folds, args.summary_csv)
        print(f"Wrote split summary: {args.summary_csv}")


if __name__ == "__main__":
    main()
