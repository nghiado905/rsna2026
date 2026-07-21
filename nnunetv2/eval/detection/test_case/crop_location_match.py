import argparse
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_PRED_COORDS_CSV = Path(
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\output_last\autocrop_leak_vessel_coords.csv"
)


DEFAULT_GT_CROP_COORDS_CSV = Path(
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\test_case\train_localizers_crop_coords.csv"
)


DEFAULT_GT_CROP_COORDS_CSV_DEFAULT_BBOX = Path(
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\test_case_no_train\train_localizers_crop_coords_default_bbox.csv"
)


DEFAULT_LOCATION_MATCH_CSV = Path(
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\eval_outputs\20260327_025236\detailed_report.csv"
)


DEFAULT_OUT_DIR = Path(
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\LAST"
)

DEFAULT_GT_LABELS_CSV = Path(
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\train_localizers copy.csv"
)

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

OUTPUT_LAST_JOBS = [
    {
        "name": "autocrop_output_last",
        "submission_csv": Path(
            r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\output_last\autocrop_leak_submission.csv"
        ),
        "pred_coords_csv": Path(
            r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\output_last\autocrop_leak_coors.csv"
        ),
        "gt_crop_coords_csv": Path(
            r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\test_case_no_train\train_localizers_crop_coords.csv"
        ),
    },
    {
        "name": "vessel_output_last",
        "submission_csv": Path(
            r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\output_last\vessel_leak_submission.csv"
        ),
        "pred_coords_csv": Path(
            r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\output_last\vessel_leak_coords.csv"
        ),
        "gt_crop_coords_csv": Path(
            r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\test_case_no_train\train_localizers_crop_coords.csv"
        ),
    },
    {
        "name": "default_output_last",
        "submission_csv": Path(
            r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\output_last\default_leak_submission.csv"
        ),
        "pred_coords_csv": Path(
            r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\output_last\default_leak_coords.csv"
        ),
        "gt_crop_coords_csv": Path(
            r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\test_case_no_train\train_localizers_crop_coords_default_bbox.csv"
        ),
    },
]


def parse_args():
    
    p = argparse.ArgumentParser(
        description="Evaluate crop-space coordinates only on series marked as 'Location Match'."
    )
    
    p.add_argument("--pred-coords-csv", type=Path, default=DEFAULT_PRED_COORDS_CSV)
    p.add_argument("--gt-crop-coords-csv", type=Path, default=DEFAULT_GT_CROP_COORDS_CSV)
    p.add_argument(
        
        "--gt-crop-coords-csv-default-bbox",
        type=Path,
        default=DEFAULT_GT_CROP_COORDS_CSV_DEFAULT_BBOX,
    )
    
    p.add_argument("--location-match-csv", type=Path, default=DEFAULT_LOCATION_MATCH_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--batch-output-last",
        action="store_true",
        help="Run built-in batch evaluation for autocrop/vessel/default output_last files.",
    )
    p.add_argument(
        "--gt-labels-csv",
        type=Path,
        default=DEFAULT_GT_LABELS_CSV,
        help="Series-to-location GT CSV used to derive location matches from submission files in batch mode.",
    )
    p.add_argument(
        "--auto-select-gt",
        action="store_true",
        help="If set, use default-bbox GT automatically when pred coords filename contains 'default'.",
    )
    p.add_argument("--pred-series-col", type=str, default="SeriesInstanceUID")
    p.add_argument("--pred-label-col", type=str, default="label")
    p.add_argument("--pred-score-col", type=str, default="max_prob")
    p.add_argument("--pred-x-col", type=str, default="coord_x_crop")
    p.add_argument("--pred-y-col", type=str, default="coord_y_crop")
    p.add_argument("--pred-z-col", type=str, default="coord_z_crop")
    p.add_argument("--gt-series-col", type=str, default="SeriesInstanceUID")
    p.add_argument("--gt-label-col", type=str, default="location")
    p.add_argument("--gt-x-col", type=str, default="coord_x_crop")
    p.add_argument("--gt-y-col", type=str, default="coord_y_crop")
    p.add_argument("--gt-z-col", type=str, default="coord_z_crop")
    p.add_argument("--match-series-col", type=str, default="SeriesInstanceUID")
    p.add_argument("--match-note-col", type=str, default="Note")
    p.add_argument("--match-note-value", type=str, default="Location Match")
    p.add_argument("--pred-min-score", type=float, default=0.0)
    p.add_argument("--near-thresholds", type=float, nargs="*", default=[5.0, 10.0, 20.0])

    return p.parse_args()

def _load_gt_label_map(gt_labels_csv: Path) -> dict[str, list[str]]:
    gt_df = pd.read_csv(gt_labels_csv)
    gt_df["SeriesInstanceUID"] = gt_df["SeriesInstanceUID"].astype(str).str.strip()
    gt_df["location"] = gt_df["location"].astype(str).str.strip()
    return gt_df.groupby("SeriesInstanceUID")["location"].apply(list).to_dict()


def _keep_series_from_match_csv(match_df: pd.DataFrame, args) -> set[str]:
    wanted = str(args.match_note_value).strip().lower()
    return set(
        match_df.loc[
            match_df[args.match_note_col].astype(str).str.strip().str.lower() == wanted,
            args.match_series_col,
        ]
        .astype(str)
        .str.strip()
    )


def _keep_series_from_submission(submission_csv: Path, gt_label_map: dict[str, list[str]]) -> set[str]:
    sub = pd.read_csv(submission_csv)
    sub["SeriesInstanceUID"] = sub["SeriesInstanceUID"].astype(str).str.strip()
    sub["Pred_Label"] = sub[LOCATION_LABELS].idxmax(axis=1)
    sub["Actual_Labels"] = sub["SeriesInstanceUID"].map(lambda uid: gt_label_map.get(uid, []))
    sub["Location_Match"] = sub.apply(
        lambda row: int(row["Pred_Label"] in row["Actual_Labels"]) if len(row["Actual_Labels"]) else 0,
        axis=1,
    )
    return set(sub.loc[sub["Location_Match"] == 1, "SeriesInstanceUID"].astype(str).str.strip())


def evaluate_one(args, pred_coords_csv: Path, gt_crop_coords_csv: Path, keep_series: set[str]):
    pred = pd.read_csv(pred_coords_csv)
    gt_path = gt_crop_coords_csv
    if args.auto_select_gt and "default" in pred_coords_csv.name.lower():
        gt_path = args.gt_crop_coords_csv_default_bbox
    print(f"[INFO] Using GT crop coords: {gt_path}", flush=True)
    gt = pd.read_csv(gt_path)

    need_pred = [
        args.pred_series_col,
        args.pred_label_col,
        args.pred_score_col,
        args.pred_x_col,
        args.pred_y_col,
        args.pred_z_col,
    ]
    
    need_gt = [
        args.gt_series_col,
        args.gt_label_col,
        args.gt_x_col,
        args.gt_y_col,
        args.gt_z_col,
    ]
    
    for col in need_pred:
        if col not in pred.columns:
            raise ValueError(f"Prediction CSV missing column: {col}")
    for col in need_gt:
        if col not in gt.columns:
            raise ValueError(f"GT crop CSV missing column: {col}")
    print(f"[INFO] Location-match series kept: {len(keep_series)}", flush=True)

    pred[args.pred_series_col] = pred[args.pred_series_col].astype(str).str.strip()
    pred[args.pred_label_col] = pred[args.pred_label_col].astype(str).str.strip()
    gt[args.gt_series_col] = gt[args.gt_series_col].astype(str).str.strip()
    gt[args.gt_label_col] = gt[args.gt_label_col].astype(str).str.strip()

    pred = pred[pred[args.pred_score_col] >= args.pred_min_score].copy()
    pred = pred[pred[args.pred_series_col].isin(keep_series)].copy()
    gt = gt[gt[args.gt_series_col].isin(keep_series)].copy()
    print(f"[INFO] Pred rows after filter: {len(pred)}", flush=True)
    print(f"[INFO] GT rows after filter: {len(gt)}", flush=True)

    pred_groups = {}
    for key, g in pred.groupby([args.pred_series_col, args.pred_label_col], sort=False):
        pred_groups[key] = g[
            [args.pred_x_col, args.pred_y_col, args.pred_z_col, args.pred_score_col]
        ].to_numpy(dtype=float)

    rows = []
    unmatched = 0
    for _, r in gt.iterrows():
        key = (r[args.gt_series_col], r[args.gt_label_col])
        cand = pred_groups.get(key, None)
        gt_xyz = np.array([r[args.gt_x_col], r[args.gt_y_col], r[args.gt_z_col]], dtype=float)
        if cand is None or len(cand) == 0:
            unmatched += 1
            rows.append(
                {
                    "SeriesInstanceUID": key[0],
                    "label": key[1],
                    "matched": 0,
                    "gt_x_crop": gt_xyz[0],
                    "gt_y_crop": gt_xyz[1],
                    "gt_z_crop": gt_xyz[2],
                    "pred_x_crop": np.nan,
                    "pred_y_crop": np.nan,
                    "pred_z_crop": np.nan,
                    "pred_score": np.nan,
                    "abs_err_x": np.nan,
                    "abs_err_y": np.nan,
                    "abs_err_z": np.nan,
                    "dist_l2_xy": np.nan,
                    "dist_l2": np.nan,
                    "squared_dist_l2_xy": np.nan,
                    "squared_dist_l2": np.nan,
                }
            )
            continue

        pred_xyz = cand[:, :3]
        d_xyz = np.linalg.norm(pred_xyz - gt_xyz[None, :], axis=1)
        d_xy = np.linalg.norm(pred_xyz[:, :2] - gt_xyz[None, :2], axis=1)
        j = int(np.argmin(d_xyz))
        pxyz = pred_xyz[j]
        err = np.abs(pxyz - gt_xyz)
        rows.append(
            {
                "SeriesInstanceUID": key[0],
                "label": key[1],
                "matched": 1,
                "gt_x_crop": gt_xyz[0],
                "gt_y_crop": gt_xyz[1],
                "gt_z_crop": gt_xyz[2],
                "pred_x_crop": float(pxyz[0]),
                "pred_y_crop": float(pxyz[1]),
                "pred_z_crop": float(pxyz[2]),
                "pred_score": float(cand[j, 3]),
                "abs_err_x": float(err[0]),
                "abs_err_y": float(err[1]),
                "abs_err_z": float(err[2]),
                "dist_l2_xy": float(d_xy[j]),
                "dist_l2": float(d_xyz[j]),
                "squared_dist_l2_xy": float(d_xy[j] * d_xy[j]),
                "squared_dist_l2": float(d_xyz[j] * d_xyz[j]),
            }
        )

    detail = pd.DataFrame(rows)
    matched = detail[detail["matched"] == 1].copy()
    unmatched_df = detail[detail["matched"] == 0].copy()

    if len(matched) > 0:
        overall = pd.DataFrame(
            [
                {
                    "n_gt": int(len(detail)),
                    "n_matched": int(len(matched)),
                    "n_unmatched": int(unmatched),
                    "match_rate": float(len(matched) / len(detail)),
                    "MAE_x": float(matched["abs_err_x"].mean()),
                    "MAE_y": float(matched["abs_err_y"].mean()),
                    "MAE_z": float(matched["abs_err_z"].mean()),
                    "MAE_L2_XY": float(matched["dist_l2_xy"].mean()),
                    "MSE_L2_XY": float(matched["squared_dist_l2_xy"].mean()),
                    "RMSE_L2_XY": float(np.sqrt(matched["squared_dist_l2_xy"].mean())),
                    "MAE_L2_3D": float(matched["dist_l2"].mean()),
                    "MSE_L2_3D": float(matched["squared_dist_l2"].mean()),
                    "RMSE_L2_3D": float(np.sqrt(matched["squared_dist_l2"].mean())),
                }
            ]
        )
        per_series = (
            matched.groupby("SeriesInstanceUID", as_index=False)
            .agg(
                n_matched_points=("label", "size"),
                MAE_x=("abs_err_x", "mean"),
                MAE_y=("abs_err_y", "mean"),
                MAE_z=("abs_err_z", "mean"),
                MAE_L2_XY=("dist_l2_xy", "mean"),
                MSE_L2_XY=("squared_dist_l2_xy", "mean"),
                MAE_L2_3D=("dist_l2", "mean"),
                MSE_L2_3D=("squared_dist_l2", "mean"),
            )
            .sort_values(["n_matched_points", "MAE_L2_3D"], ascending=[False, True])
        )
        per_series["RMSE_L2_XY"] = np.sqrt(per_series["MSE_L2_XY"].values)
        per_series["RMSE_L2_3D"] = np.sqrt(per_series["MSE_L2_3D"].values)
        for thr in args.near_thresholds:
            thr_name = str(int(thr)) if float(thr).is_integer() else str(thr).replace(".", "p")
            overall[f"near_xy_le_{thr_name}"] = int((matched["dist_l2_xy"] <= thr).sum())
            overall[f"near_3d_le_{thr_name}"] = int((matched["dist_l2"] <= thr).sum())
            per_series[f"near_xy_le_{thr_name}"] = (
                matched.groupby("SeriesInstanceUID")["dist_l2_xy"]
                .apply(lambda s, t=thr: int((s <= t).sum()))
                .reindex(per_series["SeriesInstanceUID"])
                .values
            )
            per_series[f"near_3d_le_{thr_name}"] = (
                matched.groupby("SeriesInstanceUID")["dist_l2"]
                .apply(lambda s, t=thr: int((s <= t).sum()))
                .reindex(per_series["SeriesInstanceUID"])
                .values
            )
    else:
        overall = pd.DataFrame(
            [
                {
                    "n_gt": int(len(detail)),
                    "n_matched": 0,
                    "n_unmatched": int(unmatched),
                    "match_rate": 0.0,
                    "MAE_x": np.nan,
                    "MAE_y": np.nan,
                    "MAE_z": np.nan,
                    "MAE_L2_XY": np.nan,
                    "MSE_L2_XY": np.nan,
                    "RMSE_L2_XY": np.nan,
                    "MAE_L2_3D": np.nan,
                    "MSE_L2_3D": np.nan,
                    "RMSE_L2_3D": np.nan,
                }
            ]
        )
        per_series = pd.DataFrame(
            columns=[
                "SeriesInstanceUID",
                "n_matched_points",
                "MAE_x",
                "MAE_y",
                "MAE_z",
                "MAE_L2_XY",
                "MSE_L2_XY",
                "RMSE_L2_XY",
                "MAE_L2_3D",
                "MSE_L2_3D",
                "RMSE_L2_3D",
            ]
        )

    return detail, unmatched_df, overall, per_series


def _save_eval_outputs(base_out_dir: Path, detail: pd.DataFrame, unmatched_df: pd.DataFrame, overall: pd.DataFrame, per_series: pd.DataFrame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_out_dir / f"coords_crop_location_match_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "matched_detail.csv"
    unmatched_path = out_dir / "unmatched_gt_rows.csv"
    overall_path = out_dir / "overall_metrics.csv"
    per_series_path = out_dir / "per_series_metrics.csv"

    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    unmatched_df.to_csv(unmatched_path, index=False, encoding="utf-8-sig")
    overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    per_series.to_csv(per_series_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] out_dir: {out_dir}")
    print(f"[DONE] matched_detail: {detail_path}")
    print(f"[DONE] unmatched_gt_rows: {unmatched_path}")
    print(f"[DONE] overall: {overall_path}")
    print(f"[DONE] per_series: {per_series_path}")
    print(overall.to_string(index=False))
    return out_dir, overall_path


def main():
    args = parse_args()

    if args.batch_output_last:
        gt_label_map = _load_gt_label_map(args.gt_labels_csv)
        summary_rows = []
        batch_root = args.out_dir / f"batch_output_last_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        batch_root.mkdir(parents=True, exist_ok=True)

        for job in OUTPUT_LAST_JOBS:
            print(f"[BATCH] Running {job['name']}", flush=True)
            keep_series = _keep_series_from_submission(job["submission_csv"], gt_label_map)
            detail, unmatched_df, overall, per_series = evaluate_one(
                args=args,
                pred_coords_csv=job["pred_coords_csv"],
                gt_crop_coords_csv=job["gt_crop_coords_csv"],
                keep_series=keep_series,
            )
            job_out_dir, overall_path = _save_eval_outputs(
                batch_root / job["name"], detail, unmatched_df, overall, per_series
            )
            row = overall.iloc[0].to_dict()
            row["job"] = job["name"]
            row["overall_metrics_csv"] = str(overall_path)
            row["out_dir"] = str(job_out_dir)
            summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = batch_root / "summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[DONE] batch summary: {summary_path}")
        print(summary_df.to_string(index=False))
        return

    match_df = pd.read_csv(args.location_match_csv)
    if args.match_series_col not in match_df.columns:
        raise ValueError(f"Location-match CSV missing column: {args.match_series_col}")
    if args.match_note_col not in match_df.columns:
        raise ValueError(f"Location-match CSV missing column: {args.match_note_col}")
    keep_series = _keep_series_from_match_csv(match_df, args)
    detail, unmatched_df, overall, per_series = evaluate_one(
        args=args,
        pred_coords_csv=args.pred_coords_csv,
        gt_crop_coords_csv=args.gt_crop_coords_csv,
        keep_series=keep_series,
    )
    _save_eval_outputs(args.out_dir, detail, unmatched_df, overall, per_series)

if __name__ == "__main__":
    main()

