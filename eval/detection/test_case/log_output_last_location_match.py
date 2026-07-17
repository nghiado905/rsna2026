from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_LAST_DIR = ROOT / "output_last"
GT_CROP_CSV = ROOT / "test_case_no_train" / "train_localizers_crop_coords.csv"

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

MODEL_FILES = [
    (
        "autocrop",
        OUTPUT_LAST_DIR / "autocrop_leak_submission.csv",
        OUTPUT_LAST_DIR / "autocrop_leak_coors.csv",
    ),
    (
        "vessel",
        OUTPUT_LAST_DIR / "vessel_leak_submission.csv",
        OUTPUT_LAST_DIR / "vessel_leak_coords.csv",
    ),
    (
        "default",
        OUTPUT_LAST_DIR / "default_leak_submission.csv",
        OUTPUT_LAST_DIR / "default_leak_coords.csv",
    ),
]


def _top1_label(row: pd.Series) -> tuple[str, float]:
    probs = row[LOCATION_LABELS].astype(float)
    label = str(probs.idxmax())
    prob = float(probs.max())
    return label, prob


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_LAST_DIR / f"location_match_logs_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = pd.read_csv(GT_CROP_CSV)
    gt["SeriesInstanceUID"] = gt["SeriesInstanceUID"].astype(str).str.strip()
    gt["location"] = gt["location"].astype(str).str.strip()

    gt_labels_by_series = (
        gt.groupby("SeriesInstanceUID")["location"]
        .apply(lambda s: sorted(set(str(v).strip() for v in s if str(v).strip())))
        .to_dict()
    )

    summary_rows: list[dict] = []

    for model_name, sub_csv, coords_csv in MODEL_FILES:
        sub = pd.read_csv(sub_csv)
        coords = pd.read_csv(coords_csv)

        sub["SeriesInstanceUID"] = sub["SeriesInstanceUID"].astype(str).str.strip()
        coords["SeriesInstanceUID"] = coords["SeriesInstanceUID"].astype(str).str.strip()
        coords["label"] = coords["label"].astype(str).str.strip()

        details: list[dict] = []
        location_match_count = 0
        mismatch_count = 0
        missing_pred_coord_count = 0

        for _, row in sub.iterrows():
            series_id = str(row["SeriesInstanceUID"]).strip()
            pred_label, pred_prob = _top1_label(row)
            gt_labels = gt_labels_by_series.get(series_id, [])
            location_match = pred_label in gt_labels

            pred_rows = coords[
                (coords["SeriesInstanceUID"] == series_id) & (coords["label"] == pred_label)
            ].copy()
            gt_rows = gt[
                (gt["SeriesInstanceUID"] == series_id) & (gt["location"] == pred_label)
            ].copy()

            rec = {
                "SeriesInstanceUID": series_id,
                "pred_label": pred_label,
                "pred_prob": pred_prob,
                "gt_labels": " | ".join(gt_labels),
                "location_match": bool(location_match),
                "pred_coord_found": False,
                "gt_coord_found": False,
                "coord_x_pred_crop": None,
                "coord_y_pred_crop": None,
                "coord_z_pred_crop": None,
                "coord_x_gt_crop": None,
                "coord_y_gt_crop": None,
                "coord_z_gt_crop": None,
                "abs_dx": None,
                "abs_dy": None,
                "abs_dz": None,
                "l2_3d": None,
            }

            if location_match:
                location_match_count += 1
            else:
                mismatch_count += 1

            if not pred_rows.empty:
                pred = pred_rows.iloc[0]
                rec["pred_coord_found"] = True
                rec["coord_x_pred_crop"] = float(pred["coord_x_crop"])
                rec["coord_y_pred_crop"] = float(pred["coord_y_crop"])
                rec["coord_z_pred_crop"] = float(pred["coord_z_crop"])
            else:
                missing_pred_coord_count += 1

            if location_match and not gt_rows.empty:
                gt_row = gt_rows.iloc[0]
                rec["gt_coord_found"] = True
                rec["coord_x_gt_crop"] = float(gt_row["coord_x_crop"])
                rec["coord_y_gt_crop"] = float(gt_row["coord_y_crop"])
                rec["coord_z_gt_crop"] = float(gt_row["coord_z_crop"])

            if rec["pred_coord_found"] and rec["gt_coord_found"]:
                dx = rec["coord_x_pred_crop"] - rec["coord_x_gt_crop"]
                dy = rec["coord_y_pred_crop"] - rec["coord_y_gt_crop"]
                dz = rec["coord_z_pred_crop"] - rec["coord_z_gt_crop"]
                rec["abs_dx"] = abs(dx)
                rec["abs_dy"] = abs(dy)
                rec["abs_dz"] = abs(dz)
                rec["l2_3d"] = (dx * dx + dy * dy + dz * dz) ** 0.5

            details.append(rec)

        details_df = pd.DataFrame(details)
        details_path = out_dir / f"{model_name}_location_match_detail.csv"
        details_df.to_csv(details_path, index=False)

        matched_with_coords = details_df[
            details_df["location_match"] & details_df["pred_coord_found"] & details_df["gt_coord_found"]
        ]

        summary_rows.append(
            {
                "model": model_name,
                "submission_csv": sub_csv.name,
                "coords_csv": coords_csv.name,
                "n_series": int(len(sub)),
                "n_location_match": int(location_match_count),
                "n_location_mismatch": int(mismatch_count),
                "n_missing_pred_coord": int(missing_pred_coord_count),
                "n_matched_with_coords": int(len(matched_with_coords)),
                "mean_abs_dx": float(matched_with_coords["abs_dx"].mean()) if len(matched_with_coords) else None,
                "mean_abs_dy": float(matched_with_coords["abs_dy"].mean()) if len(matched_with_coords) else None,
                "mean_abs_dz": float(matched_with_coords["abs_dz"].mean()) if len(matched_with_coords) else None,
                "mean_l2_3d": float(matched_with_coords["l2_3d"].mean()) if len(matched_with_coords) else None,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    print(f"Saved logs to: {out_dir}")
    for row in summary_rows:
        print(
            f"{row['model']}: match={row['n_location_match']} mismatch={row['n_location_mismatch']} "
            f"matched_with_coords={row['n_matched_with_coords']}"
        )


if __name__ == "__main__":
    main()
