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
            "Evaluate predicted point sets against GT point sets using symmetric Chamfer distance "
            "within each SeriesInstanceUID."
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

    series_rows: list[dict] = []
    pooled_forward: list[float] = []
    pooled_backward: list[float] = []

    for series_uid in common_series:
        pred_s = pred[pred[args.pred_series_col] == series_uid].copy()
        gt_s = gt[gt[args.gt_series_col] == series_uid].copy()
        if len(pred_s) == 0 or len(gt_s) == 0:
            continue

        pred_xyz = pred_s[["pred_x_mm", "pred_y_mm", "pred_z_mm"]].to_numpy(dtype=float)
        gt_xyz = gt_s[[args.gt_x_mm_col, args.gt_y_mm_col, args.gt_z_mm_col]].to_numpy(dtype=float)
        dist_mat = np.linalg.norm(pred_xyz[:, None, :] - gt_xyz[None, :, :], axis=2)

        pred_to_gt = dist_mat.min(axis=1)
        gt_to_pred = dist_mat.min(axis=0)
        forward_mean = float(pred_to_gt.mean())
        backward_mean = float(gt_to_pred.mean())
        chamfer = forward_mean + backward_mean

        pooled_forward.extend(pred_to_gt.tolist())
        pooled_backward.extend(gt_to_pred.tolist())

        series_rows.append(
            {
                "SeriesInstanceUID": series_uid,
                "n_pred": int(len(pred_s)),
                "n_gt": int(len(gt_s)),
                "pred_to_gt_mean_mm": forward_mean,
                "gt_to_pred_mean_mm": backward_mean,
                "chamfer_mm": chamfer,
                "pred_to_gt_min_mm": float(pred_to_gt.min()),
                "pred_to_gt_max_mm": float(pred_to_gt.max()),
                "gt_to_pred_min_mm": float(gt_to_pred.min()),
                "gt_to_pred_max_mm": float(gt_to_pred.max()),
            }
        )

    series_df = pd.DataFrame(series_rows)
    overall_df = pd.DataFrame(
        [
            {
                "n_series": int(len(series_df)),
                "n_pred_total": int(len(pred)),
                "n_gt_total": int(len(gt)),
                "pred_to_gt_mean_mm": float(series_df["pred_to_gt_mean_mm"].mean()) if len(series_df) else np.nan,
                "gt_to_pred_mean_mm": float(series_df["gt_to_pred_mean_mm"].mean()) if len(series_df) else np.nan,
                "chamfer_mm": float(series_df["chamfer_mm"].mean()) if len(series_df) else np.nan,
                "pooled_pred_to_gt_mean_mm": float(np.mean(pooled_forward)) if pooled_forward else np.nan,
                "pooled_gt_to_pred_mean_mm": float(np.mean(pooled_backward)) if pooled_backward else np.nan,
                "pooled_chamfer_mm": (
                    float(np.mean(pooled_forward)) + float(np.mean(pooled_backward))
                    if pooled_forward and pooled_backward
                    else np.nan
                ),
            }
        ]
    )

    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    series_df.to_csv(out_dir / "per_series_chamfer.csv", index=False, encoding="utf-8-sig")
    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")

    print(overall_df.to_string(index=False))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
