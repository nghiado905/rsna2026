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
            "Series-level hit-rate evaluation in crop mm. "
            "A series is a hit if any predicted point is within threshold mm of any GT point."
        )
    )
    p.add_argument("--pred-coords-csv", type=Path, required=True)
    p.add_argument("--gt-coords-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--thresholds-mm", type=float, nargs="+", default=[15.0])
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

    pred[args.pred_series_col] = pred[args.pred_series_col].astype(str).str.strip()
    gt[args.gt_series_col] = gt[args.gt_series_col].astype(str).str.strip()
    pred[args.pred_label_col] = pred[args.pred_label_col].astype(str).str.strip()
    gt[args.gt_label_col] = gt[args.gt_label_col].astype(str).str.strip()

    if args.canon_series:
        pred[args.pred_series_col] = pred[args.pred_series_col].map(_canon_series)
        gt[args.gt_series_col] = gt[args.gt_series_col].map(_canon_series)

    common_series = sorted(
        set(pred[args.pred_series_col].unique()).intersection(set(gt[args.gt_series_col].unique()))
    )
    pred = pred[pred[args.pred_series_col].isin(common_series)].copy()
    gt = gt[gt[args.gt_series_col].isin(common_series)].copy()

    spacing_df = gt[
        [args.gt_series_col, args.spacing_x_col, args.spacing_y_col, args.spacing_z_col]
    ].drop_duplicates(subset=[args.gt_series_col])
    pred = pred.merge(
        spacing_df,
        left_on=args.pred_series_col,
        right_on=args.gt_series_col,
        how="left",
        validate="many_to_one",
    )
    pred["coord_x_crop_mm"] = pred[args.pred_x_col].astype(float) * pred[args.spacing_x_col].astype(float)
    pred["coord_y_crop_mm"] = pred[args.pred_y_col].astype(float) * pred[args.spacing_y_col].astype(float)
    pred["coord_z_crop_mm"] = pred[args.pred_z_col].astype(float) * pred[args.spacing_z_col].astype(float)

    thresholds = sorted({float(t) for t in args.thresholds_mm})
    series_rows: list[dict] = []
    summary_rows: list[dict] = []
    for series_uid in common_series:
        pred_s = pred[pred[args.pred_series_col] == series_uid].copy()
        gt_s = gt[gt[args.gt_series_col] == series_uid].copy()
        pred_xyz = pred_s[["coord_x_crop_mm", "coord_y_crop_mm", "coord_z_crop_mm"]].to_numpy(dtype=float)
        gt_xyz = gt_s[[args.gt_x_mm_col, args.gt_y_mm_col, args.gt_z_mm_col]].to_numpy(dtype=float)
        dist_mat = np.linalg.norm(pred_xyz[:, None, :] - gt_xyz[None, :, :], axis=2)
        min_dist = float(dist_mat.min()) if dist_mat.size else np.nan
        row = {
            "SeriesInstanceUID": series_uid,
            "min_pred_to_gt_dist_mm": min_dist,
        }
        for thr in thresholds:
            row[f"hit@{int(thr)}mm"] = int(min_dist <= thr)
        series_rows.append(row)

    series_df = pd.DataFrame(series_rows)
    for thr in thresholds:
        col = f"hit@{int(thr)}mm"
        summary_rows.append(
            {
                "threshold_mm": thr,
                "n_series": int(len(series_df)),
                "n_hit": int(series_df[col].sum()),
                "n_miss": int(len(series_df) - series_df[col].sum()),
                "SeriesHitRate": float(series_df[col].mean()) if len(series_df) else np.nan,
                "MeanMinDist_mm": float(series_df["min_pred_to_gt_dist_mm"].mean()) if len(series_df) else np.nan,
            }
        )

    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    series_df.to_csv(out_dir / "per_series_hit_detail.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")

    print(summary_df.to_string(index=False))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
