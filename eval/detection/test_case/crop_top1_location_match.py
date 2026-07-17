import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_PRED_COORDS_CSV = Path(
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\output_mask_80_e\output_last\autocrop_fake_vessel_coords.csv"
)
DEFAULT_GT_CROP_COORDS_CSV = Path(
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\test_case\train_localizers_crop_coords.csv"
)
DEFAULT_LOCATION_MATCH_CSV = Path(
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\eval_outputs\20260330_175118\detailed_report.csv"
)
DEFAULT_OUT_DIR = Path(
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\LAST"
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate crop-space coordinates using only the top-1 predicted label per series."
    )
    p.add_argument("--pred-coords-csv", type=Path, default=DEFAULT_PRED_COORDS_CSV)
    p.add_argument("--gt-crop-coords-csv", type=Path, default=DEFAULT_GT_CROP_COORDS_CSV)
    p.add_argument("--location-match-csv", type=Path, default=DEFAULT_LOCATION_MATCH_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
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
    p.add_argument("--near-thresholds", type=float, nargs="*", default=[5.0, 10.0, 20.0])
    return p.parse_args()


def main():
    args = parse_args()
    pred = pd.read_csv(args.pred_coords_csv)
    gt = pd.read_csv(args.gt_crop_coords_csv)
    match_df = pd.read_csv(args.location_match_csv)

    keep_series = set(
        match_df.loc[
            match_df[args.match_note_col].astype(str).str.strip().str.lower()
            == str(args.match_note_value).strip().lower(),
            args.match_series_col,
        ]
        .astype(str)
        .str.strip()
    )

    pred[args.pred_series_col] = pred[args.pred_series_col].astype(str).str.strip()
    pred[args.pred_label_col] = pred[args.pred_label_col].astype(str).str.strip()
    gt[args.gt_series_col] = gt[args.gt_series_col].astype(str).str.strip()
    gt[args.gt_label_col] = gt[args.gt_label_col].astype(str).str.strip()

    pred = pred[pred[args.pred_series_col].isin(keep_series)].copy()
    gt = gt[gt[args.gt_series_col].isin(keep_series)].copy()

    pred_top1 = (
        pred.sort_values(
            [args.pred_series_col, args.pred_score_col],
            ascending=[True, False],
        )
        .groupby(args.pred_series_col, as_index=False)
        .first()
    )
    top1_label_by_series = dict(
        zip(pred_top1[args.pred_series_col].astype(str), pred_top1[args.pred_label_col].astype(str))
    )

    gt = gt[
        gt.apply(
            lambda r: top1_label_by_series.get(str(r[args.gt_series_col]), None) == str(r[args.gt_label_col]),
            axis=1,
        )
    ].copy()

    rows = []
    unmatched = 0
    for _, r in gt.iterrows():
        series_uid = str(r[args.gt_series_col])
        label = str(r[args.gt_label_col])
        pred_row = pred_top1[pred_top1[args.pred_series_col] == series_uid]
        gt_xyz = np.array([r[args.gt_x_col], r[args.gt_y_col], r[args.gt_z_col]], dtype=float)

        if len(pred_row) == 0:
            unmatched += 1
            rows.append(
                {
                    "SeriesInstanceUID": series_uid,
                    "label": label,
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

        p = pred_row.iloc[0]
        pred_xyz = np.array([p[args.pred_x_col], p[args.pred_y_col], p[args.pred_z_col]], dtype=float)
        err = np.abs(pred_xyz - gt_xyz)
        dist_xy = float(np.linalg.norm(pred_xyz[:2] - gt_xyz[:2]))
        dist_3d = float(np.linalg.norm(pred_xyz - gt_xyz))
        rows.append(
            {
                "SeriesInstanceUID": series_uid,
                "label": label,
                "matched": 1,
                "gt_x_crop": gt_xyz[0],
                "gt_y_crop": gt_xyz[1],
                "gt_z_crop": gt_xyz[2],
                "pred_x_crop": float(pred_xyz[0]),
                "pred_y_crop": float(pred_xyz[1]),
                "pred_z_crop": float(pred_xyz[2]),
                "pred_score": float(p[args.pred_score_col]),
                "pred_top1_label": str(p[args.pred_label_col]),
                "abs_err_x": float(err[0]),
                "abs_err_y": float(err[1]),
                "abs_err_z": float(err[2]),
                "dist_l2_xy": dist_xy,
                "dist_l2": dist_3d,
                "squared_dist_l2_xy": dist_xy * dist_xy,
                "squared_dist_l2": dist_3d * dist_3d,
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
                n_points=("label", "size"),
                top1_label=("pred_top1_label", "first"),
                MAE_x=("abs_err_x", "mean"),
                MAE_y=("abs_err_y", "mean"),
                MAE_z=("abs_err_z", "mean"),
                MAE_L2_XY=("dist_l2_xy", "mean"),
                MSE_L2_XY=("squared_dist_l2_xy", "mean"),
                MAE_L2_3D=("dist_l2", "mean"),
                MSE_L2_3D=("squared_dist_l2", "mean"),
            )
            .sort_values(["n_points", "MAE_L2_3D"], ascending=[False, True])
        )
        per_series["RMSE_L2_XY"] = np.sqrt(per_series["MSE_L2_XY"].values)
        per_series["RMSE_L2_3D"] = np.sqrt(per_series["MSE_L2_3D"].values)
        for thr in args.near_thresholds:
            thr_name = str(int(thr)) if float(thr).is_integer() else str(thr).replace(".", "p")
            overall[f"near_xy_le_{thr_name}"] = int((matched["dist_l2_xy"] <= thr).sum())
            overall[f"near_3d_le_{thr_name}"] = int((matched["dist_l2"] <= thr).sum())
    else:
        overall = pd.DataFrame(
            [
                {
                    "n_gt": 0,
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
                "n_points",
                "top1_label",
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

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / f"coords_crop_top1_location_match_{ts}"
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


if __name__ == "__main__":
    main()
