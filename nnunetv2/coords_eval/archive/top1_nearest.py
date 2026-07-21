from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def _canon_series(value: object) -> str:
    s = str(value).strip()
    if not s:
        return s
    if s.isdigit():
        return str(int(s))
    return s.lstrip("0") or "0"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate one top-1 predicted point per series by matching it to the nearest GT point "
            "within the same series. Intended for datasets without trusted anatomy labels."
        )
    )
    p.add_argument("--pred-coords-csv", type=Path, required=True)
    p.add_argument("--gt-coords-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pred-series-col", type=str, default="SeriesInstanceUID")
    p.add_argument("--pred-label-col", type=str, default="label")
    p.add_argument("--pred-score-col", type=str, default="max_prob")
    p.add_argument("--pred-x-col", type=str, default="coord_x_crop")
    p.add_argument("--pred-y-col", type=str, default="coord_y_crop")
    p.add_argument("--pred-z-col", type=str, default="coord_z_crop")
    p.add_argument("--gt-series-col", type=str, default="SeriesInstanceUID")
    p.add_argument("--gt-label-col", type=str, default="location")
    p.add_argument("--gt-x-mm-col", type=str, default="coord_x_crop_mm")
    p.add_argument("--gt-y-mm-col", type=str, default="coord_y_crop_mm")
    p.add_argument("--gt-z-mm-col", type=str, default="coord_z_crop_mm")
    p.add_argument("--spacing-x-col", type=str, default="spacing_x_mm")
    p.add_argument("--spacing-y-col", type=str, default="spacing_y_mm")
    p.add_argument("--spacing-z-col", type=str, default="spacing_z_mm")
    p.add_argument("--pred-min-score", type=float, default=0.0)
    p.add_argument("--canon-series", action="store_true")
    return p.parse_args()


def _validate_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"{name} missing column: {col}")


def main() -> None:
    args = parse_args()

    pred = pd.read_csv(args.pred_coords_csv)
    gt = pd.read_csv(args.gt_coords_csv)

    _validate_columns(
        pred,
        [
            args.pred_series_col,
            args.pred_label_col,
            args.pred_score_col,
            args.pred_x_col,
            args.pred_y_col,
            args.pred_z_col,
        ],
        "Prediction CSV",
    )
    _validate_columns(
        gt,
        [
            args.gt_series_col,
            args.gt_label_col,
            args.gt_x_mm_col,
            args.gt_y_mm_col,
            args.gt_z_mm_col,
            args.spacing_x_col,
            args.spacing_y_col,
            args.spacing_z_col,
        ],
        "GT CSV",
    )

    pred = pred.copy()
    gt = gt.copy()
    pred[args.pred_series_col] = pred[args.pred_series_col].astype(str).str.strip()
    gt[args.gt_series_col] = gt[args.gt_series_col].astype(str).str.strip()
    pred[args.pred_label_col] = pred[args.pred_label_col].astype(str).str.strip()
    gt[args.gt_label_col] = gt[args.gt_label_col].astype(str).str.strip()

    if args.canon_series:
        pred[args.pred_series_col] = pred[args.pred_series_col].map(_canon_series)
        gt[args.gt_series_col] = gt[args.gt_series_col].map(_canon_series)

    pred = pred[pred[args.pred_score_col] >= args.pred_min_score].copy()

    common_series = sorted(
        set(pred[args.pred_series_col].unique()).intersection(set(gt[args.gt_series_col].unique()))
    )
    pred = pred[pred[args.pred_series_col].isin(common_series)].copy()
    gt = gt[gt[args.gt_series_col].isin(common_series)].copy()

    spacing_df = (
        gt[
            [
                args.gt_series_col,
                args.spacing_x_col,
                args.spacing_y_col,
                args.spacing_z_col,
            ]
        ]
        .drop_duplicates(subset=[args.gt_series_col])
        .copy()
    )
    if spacing_df[args.gt_series_col].duplicated().any():
        raise ValueError("GT spacing is not unique per series.")

    pred = pred.merge(
        spacing_df,
        left_on=args.pred_series_col,
        right_on=args.gt_series_col,
        how="left",
        validate="many_to_one",
    )
    if pred[[args.spacing_x_col, args.spacing_y_col, args.spacing_z_col]].isna().any().any():
        raise ValueError("Missing spacing after joining GT spacing into predictions.")

    pred["pred_x_mm"] = pred[args.pred_x_col].astype(float) * pred[args.spacing_x_col].astype(float)
    pred["pred_y_mm"] = pred[args.pred_y_col].astype(float) * pred[args.spacing_y_col].astype(float)
    pred["pred_z_mm"] = pred[args.pred_z_col].astype(float) * pred[args.spacing_z_col].astype(float)

    pred = pred.sort_values(
        by=[args.pred_series_col, args.pred_score_col, args.pred_label_col],
        ascending=[True, False, True],
        kind="mergesort",
    )
    pred_top1 = pred.drop_duplicates(subset=[args.pred_series_col], keep="first").copy()

    matched_rows: list[dict] = []
    per_series_rows: list[dict] = []

    for series_uid in common_series:
        pred_s = pred_top1[pred_top1[args.pred_series_col] == series_uid]
        gt_s = gt[gt[args.gt_series_col] == series_uid]
        if len(pred_s) == 0 or len(gt_s) == 0:
            continue

        pred_row = pred_s.iloc[0]
        gt_xyz = gt_s[[args.gt_x_mm_col, args.gt_y_mm_col, args.gt_z_mm_col]].to_numpy(dtype=float)
        pred_xyz = np.array(
            [
                float(pred_row["pred_x_mm"]),
                float(pred_row["pred_y_mm"]),
                float(pred_row["pred_z_mm"]),
            ],
            dtype=float,
        )

        dxyz = gt_xyz - pred_xyz[None, :]
        dists = np.linalg.norm(dxyz, axis=1)
        i_gt = int(np.argmin(dists))
        gt_row = gt_s.reset_index(drop=True).iloc[i_gt]

        dx = float(pred_row["pred_x_mm"] - gt_row[args.gt_x_mm_col])
        dy = float(pred_row["pred_y_mm"] - gt_row[args.gt_y_mm_col])
        dz = float(pred_row["pred_z_mm"] - gt_row[args.gt_z_mm_col])
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        abs_dz = abs(dz)
        l2_xy = float(np.sqrt(dx * dx + dy * dy))
        l2_3d = float(np.sqrt(dx * dx + dy * dy + dz * dz))

        matched_rows.append(
            {
                "SeriesInstanceUID": series_uid,
                "pred_label": pred_row[args.pred_label_col],
                "pred_score": float(pred_row[args.pred_score_col]),
                "gt_match_label": gt_row[args.gt_label_col],
                "n_gt_points": int(len(gt_s)),
                "pred_x_mm": float(pred_row["pred_x_mm"]),
                "pred_y_mm": float(pred_row["pred_y_mm"]),
                "pred_z_mm": float(pred_row["pred_z_mm"]),
                "gt_x_mm": float(gt_row[args.gt_x_mm_col]),
                "gt_y_mm": float(gt_row[args.gt_y_mm_col]),
                "gt_z_mm": float(gt_row[args.gt_z_mm_col]),
                "abs_err_x_mm": abs_dx,
                "abs_err_y_mm": abs_dy,
                "abs_err_z_mm": abs_dz,
                "dist_l2_xy_mm": l2_xy,
                "dist_l2_3d_mm": l2_3d,
            }
        )

        per_series_rows.append(
            {
                "SeriesInstanceUID": series_uid,
                "pred_label": pred_row[args.pred_label_col],
                "pred_score": float(pred_row[args.pred_score_col]),
                "gt_match_label": gt_row[args.gt_label_col],
                "n_gt_points": int(len(gt_s)),
                "MAE_x_mm": abs_dx,
                "MAE_y_mm": abs_dy,
                "MAE_z_mm": abs_dz,
                "MAE_L2_XY_mm": l2_xy,
                "MAE_L2_3D_mm": l2_3d,
            }
        )

    matched_df = pd.DataFrame(matched_rows)
    per_series_df = pd.DataFrame(per_series_rows)
    overall_df = pd.DataFrame(
        [
            {
                "n_series": int(len(per_series_df)),
                "n_pred_total": int(len(pred_top1[pred_top1[args.pred_series_col].isin(common_series)])),
                "n_gt_total": int(len(gt)),
                "MAE_x_mm": float(matched_df["abs_err_x_mm"].mean()) if len(matched_df) else np.nan,
                "MAE_y_mm": float(matched_df["abs_err_y_mm"].mean()) if len(matched_df) else np.nan,
                "MAE_z_mm": float(matched_df["abs_err_z_mm"].mean()) if len(matched_df) else np.nan,
                "MAE_L2_XY_mm": float(matched_df["dist_l2_xy_mm"].mean()) if len(matched_df) else np.nan,
                "MAE_L2_3D_mm": float(matched_df["dist_l2_3d_mm"].mean()) if len(matched_df) else np.nan,
            }
        ]
    )

    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_df.to_csv(out_dir / "matched_detail.csv", index=False, encoding="utf-8-sig")
    per_series_df.to_csv(out_dir / "per_series_metrics.csv", index=False, encoding="utf-8-sig")
    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")

    print(overall_df.to_string(index=False))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
