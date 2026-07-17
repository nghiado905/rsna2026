from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution")
INPUT_DIR = REPO_ROOT / "output_last"
GT_CLASSIFICATION_CSV = REPO_ROOT / "train_localizers copy.csv"
GT_CROP_AUTOCROP_CSV = REPO_ROOT / "test_case_no_train" / "train_localizers_crop_coords.csv"
GT_CROP_DEFAULT_CSV = REPO_ROOT / "test_case_no_train" / "train_localizers_crop_coords_default_bbox.csv"
OUTPUT_BASE_DIR = INPUT_DIR / "coords_crop_mm_eval"

LOCATION_LABELS = [
    "Other Posterior Circulation",
    "Basilar Tip",
    "Right Posterior Communicating Artery",
    "Left Posterior Communicating Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Infraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Middle Cerebral Artery",
    "Left Middle Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Anterior Cerebral Artery",
    "Anterior Communicating Artery",
]

JOBS = [
    {
        "name": "autocrop_output_last",
        "submission_csv": INPUT_DIR / "autocrop_leak_submission.csv",
        "coords_csv": INPUT_DIR / "autocrop_leak_coors.csv",
        "gt_crop_csv": GT_CROP_AUTOCROP_CSV,
    },
    {
        "name": "vessel_output_last",
        "submission_csv": INPUT_DIR / "vessel_leak_submission.csv",
        "coords_csv": INPUT_DIR / "vessel_leak_coords.csv",
        "gt_crop_csv": GT_CROP_AUTOCROP_CSV,
    },
    {
        "name": "default_output_last",
        "submission_csv": INPUT_DIR / "default_leak_submission.csv",
        "coords_csv": INPUT_DIR / "default_leak_coords.csv",
        "gt_crop_csv": GT_CROP_DEFAULT_CSV,
    },
]


def load_gt_label_map(gt_csv: Path) -> dict[str, list[str]]:
    gt_df = pd.read_csv(gt_csv)
    gt_df["SeriesInstanceUID"] = gt_df["SeriesInstanceUID"].astype(str)
    gt_df["location"] = gt_df["location"].astype(str)
    return gt_df.groupby("SeriesInstanceUID")["location"].apply(list).to_dict()


def add_crop_mm_columns(df: pd.DataFrame, spacing_map: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(
        spacing_map,
        left_on="SeriesInstanceUID",
        right_index=True,
        how="left",
    ).copy()
    out["coord_x_crop_mm"] = out["coord_x_crop"] * out["spacing_x_mm"]
    out["coord_y_crop_mm"] = out["coord_y_crop"] * out["spacing_y_mm"]
    out["coord_z_crop_mm"] = out["coord_z_crop"] * out["spacing_z_mm"]
    return out


def evaluate_one(
    job_name: str,
    submission_csv: Path,
    coords_csv: Path,
    gt_crop_csv: Path,
    gt_label_map: dict[str, list[str]],
) -> tuple[dict, pd.DataFrame]:
    submission = pd.read_csv(submission_csv)
    coords = pd.read_csv(coords_csv)
    gt_crop = pd.read_csv(gt_crop_csv)

    submission["SeriesInstanceUID"] = submission["SeriesInstanceUID"].astype(str)
    coords["SeriesInstanceUID"] = coords["SeriesInstanceUID"].astype(str)
    coords["label"] = coords["label"].astype(str)
    gt_crop["SeriesInstanceUID"] = gt_crop["SeriesInstanceUID"].astype(str)
    gt_crop["location"] = gt_crop["location"].astype(str)

    spacing_map = gt_crop.drop_duplicates("SeriesInstanceUID").set_index("SeriesInstanceUID")[
        ["spacing_x_mm", "spacing_y_mm", "spacing_z_mm"]
    ]
    coords = add_crop_mm_columns(coords, spacing_map)

    submission["Pred_Label"] = submission[LOCATION_LABELS].idxmax(axis=1)
    submission["Actual_Labels"] = submission["SeriesInstanceUID"].map(
        lambda uid: gt_label_map.get(uid, [])
    )
    submission["Location_Match"] = submission.apply(
        lambda row: int(row["Pred_Label"] in row["Actual_Labels"]) if len(row["Actual_Labels"]) else 0,
        axis=1,
    )
    keep_series = set(
        submission.loc[submission["Location_Match"] == 1, "SeriesInstanceUID"].astype(str)
    )

    pred = coords[coords["SeriesInstanceUID"].isin(keep_series)].copy()
    gt = gt_crop[gt_crop["SeriesInstanceUID"].isin(keep_series)].copy()

    pred_groups: dict[tuple[str, str], np.ndarray] = {}
    for key, group in pred.groupby(["SeriesInstanceUID", "label"], sort=False):
        pred_groups[key] = group[
            ["coord_x_crop_mm", "coord_y_crop_mm", "coord_z_crop_mm", "max_prob"]
        ].to_numpy(dtype=float)

    matched_rows: list[dict] = []
    for _, gt_row in gt.iterrows():
        key = (gt_row["SeriesInstanceUID"], gt_row["location"])
        cand = pred_groups.get(key)
        if cand is None or len(cand) == 0:
            continue

        gt_xyz = np.array(
            [gt_row["coord_x_crop_mm"], gt_row["coord_y_crop_mm"], gt_row["coord_z_crop_mm"]],
            dtype=float,
        )
        pred_xyz = cand[:, :3]
        pred_score = cand[:, 3]
        dist_xyz = np.linalg.norm(pred_xyz - gt_xyz[None, :], axis=1)
        dist_xy = np.linalg.norm(pred_xyz[:, :2] - gt_xyz[None, :2], axis=1)
        j = int(np.argmin(dist_xyz))
        best_xyz = pred_xyz[j]
        abs_err = np.abs(best_xyz - gt_xyz)

        matched_rows.append(
            {
                "SeriesInstanceUID": key[0],
                "label": key[1],
                "pred_score": float(pred_score[j]),
                "gt_x_crop_mm": float(gt_xyz[0]),
                "gt_y_crop_mm": float(gt_xyz[1]),
                "gt_z_crop_mm": float(gt_xyz[2]),
                "pred_x_crop_mm": float(best_xyz[0]),
                "pred_y_crop_mm": float(best_xyz[1]),
                "pred_z_crop_mm": float(best_xyz[2]),
                "abs_err_x_mm": float(abs_err[0]),
                "abs_err_y_mm": float(abs_err[1]),
                "abs_err_z_mm": float(abs_err[2]),
                "dist_l2_xy_mm": float(dist_xy[j]),
                "dist_l2_3d_mm": float(dist_xyz[j]),
            }
        )

    detail = pd.DataFrame(matched_rows)
    summary = {
        "name": job_name,
        "n_gt": int(len(gt)),
        "n_matched": int(len(detail)),
        "n_unmatched": int(len(gt) - len(detail)),
        "match_rate": float(len(detail) / len(gt)) if len(gt) else np.nan,
        "MAE_x_mm": float(detail["abs_err_x_mm"].mean()) if len(detail) else np.nan,
        "MAE_y_mm": float(detail["abs_err_y_mm"].mean()) if len(detail) else np.nan,
        "MAE_z_mm": float(detail["abs_err_z_mm"].mean()) if len(detail) else np.nan,
        "MAE_L2_XY_mm": float(detail["dist_l2_xy_mm"].mean()) if len(detail) else np.nan,
        "MAE_L2_3D_mm": float(detail["dist_l2_3d_mm"].mean()) if len(detail) else np.nan,
    }
    return summary, detail


def main() -> None:
    out_dir = OUTPUT_BASE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    gt_label_map = load_gt_label_map(GT_CLASSIFICATION_CSV)
    summary_rows = []

    for job in JOBS:
        summary, detail = evaluate_one(
            job_name=job["name"],
            submission_csv=job["submission_csv"],
            coords_csv=job["coords_csv"],
            gt_crop_csv=job["gt_crop_csv"],
            gt_label_map=gt_label_map,
        )
        summary_rows.append(summary)
        detail.to_csv(out_dir / f"{job['name']}_matched_detail_mm.csv", index=False, encoding="utf-8-sig")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8-sig")
    print(summary_df.to_string(index=False))
    print(f"Saved: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
