from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None


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
            "Series-only Hungarian matching in crop mm, then keep only matched pairs whose "
            "distance is <= threshold_mm. Report MAE on valid pairs and match rate."
        )
    )
    p.add_argument("--pred-coords-csv", type=Path, required=True)
    p.add_argument("--gt-coords-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--threshold-mm", type=float, default=15.0)
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


def _greedy_assignment(dist_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pairs: list[tuple[float, int, int]] = []
    for i in range(dist_mat.shape[0]):
        for j in range(dist_mat.shape[1]):
            pairs.append((float(dist_mat[i, j]), i, j))
    pairs.sort(key=lambda x: x[0])
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    row_idx: list[int] = []
    col_idx: list[int] = []
    for _, i, j in pairs:
        if i in used_rows or j in used_cols:
            continue
        used_rows.add(i)
        used_cols.add(j)
        row_idx.append(i)
        col_idx.append(j)
    return np.array(row_idx, dtype=int), np.array(col_idx, dtype=int)


def _assign_pairs(dist_mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    if dist_mat.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int), "empty"
    if linear_sum_assignment is not None:
        row_idx, col_idx = linear_sum_assignment(dist_mat)
        return np.asarray(row_idx, dtype=int), np.asarray(col_idx, dtype=int), "hungarian"
    row_idx, col_idx = _greedy_assignment(dist_mat)
    return row_idx, col_idx, "greedy"


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

    pred = pred[pred[args.pred_score_col] >= args.pred_min_score].copy()
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
    pred["pred_x_mm"] = pred[args.pred_x_col].astype(float) * pred[args.spacing_x_col].astype(float)
    pred["pred_y_mm"] = pred[args.pred_y_col].astype(float) * pred[args.spacing_y_col].astype(float)
    pred["pred_z_mm"] = pred[args.pred_z_col].astype(float) * pred[args.spacing_z_col].astype(float)

    matched_rows: list[dict] = []
    per_series_rows: list[dict] = []
    methods_used: set[str] = set()
    total_pred = 0
    total_gt = 0
    total_pairs = 0
    total_valid = 0

    for series_uid in common_series:
        pred_s = pred[pred[args.pred_series_col] == series_uid].reset_index(drop=True)
        gt_s = gt[gt[args.gt_series_col] == series_uid].reset_index(drop=True)
        if len(pred_s) == 0 or len(gt_s) == 0:
            continue

        pred_xyz = pred_s[["pred_x_mm", "pred_y_mm", "pred_z_mm"]].to_numpy(dtype=float)
        gt_xyz = gt_s[[args.gt_x_mm_col, args.gt_y_mm_col, args.gt_z_mm_col]].to_numpy(dtype=float)
        dist_mat = np.linalg.norm(pred_xyz[:, None, :] - gt_xyz[None, :, :], axis=2)
        row_idx, col_idx, method = _assign_pairs(dist_mat)
        methods_used.add(method)

        total_pred += len(pred_s)
        total_gt += len(gt_s)
        total_pairs += len(row_idx)

        valid_dists: list[float] = []
        valid_x: list[float] = []
        valid_y: list[float] = []
        valid_z: list[float] = []

        for i_pred, i_gt in zip(row_idx, col_idx):
            pred_row = pred_s.iloc[int(i_pred)]
            gt_row = gt_s.iloc[int(i_gt)]
            dx = float(pred_row["pred_x_mm"] - gt_row[args.gt_x_mm_col])
            dy = float(pred_row["pred_y_mm"] - gt_row[args.gt_y_mm_col])
            dz = float(pred_row["pred_z_mm"] - gt_row[args.gt_z_mm_col])
            l2 = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            is_valid = l2 <= args.threshold_mm
            if is_valid:
                total_valid += 1
                valid_x.append(abs(dx))
                valid_y.append(abs(dy))
                valid_z.append(abs(dz))
                valid_dists.append(l2)
            matched_rows.append(
                {
                    "SeriesInstanceUID": series_uid,
                    "match_method": method,
                    "pred_label": pred_row[args.pred_label_col],
                    "gt_label": gt_row[args.gt_label_col],
                    "pred_score": float(pred_row[args.pred_score_col]),
                    "pred_x_mm": float(pred_row["pred_x_mm"]),
                    "pred_y_mm": float(pred_row["pred_y_mm"]),
                    "pred_z_mm": float(pred_row["pred_z_mm"]),
                    "gt_x_mm": float(gt_row[args.gt_x_mm_col]),
                    "gt_y_mm": float(gt_row[args.gt_y_mm_col]),
                    "gt_z_mm": float(gt_row[args.gt_z_mm_col]),
                    "abs_err_x_mm": abs(dx),
                    "abs_err_y_mm": abs(dy),
                    "abs_err_z_mm": abs(dz),
                    "dist_l2_3d_mm": l2,
                    "valid_under_threshold": int(is_valid),
                }
            )

        per_series_rows.append(
            {
                "SeriesInstanceUID": series_uid,
                "match_method": method,
                "n_pred": int(len(pred_s)),
                "n_gt": int(len(gt_s)),
                "n_pairs": int(len(row_idx)),
                "n_valid_pairs": int(len(valid_dists)),
                "match_rate_vs_gt": float(len(valid_dists) / len(gt_s)),
                "MAE_x_mm_valid": float(np.mean(valid_x)) if valid_x else np.nan,
                "MAE_y_mm_valid": float(np.mean(valid_y)) if valid_y else np.nan,
                "MAE_z_mm_valid": float(np.mean(valid_z)) if valid_z else np.nan,
                "MAE_L2_3D_mm_valid": float(np.mean(valid_dists)) if valid_dists else np.nan,
            }
        )

    matched_df = pd.DataFrame(matched_rows)
    per_series_df = pd.DataFrame(per_series_rows)
    valid_df = matched_df[matched_df["valid_under_threshold"] == 1].copy()
    overall_df = pd.DataFrame(
        [
            {
                "threshold_mm": float(args.threshold_mm),
                "n_series": int(len(common_series)),
                "n_pred_total": int(total_pred),
                "n_gt_total": int(total_gt),
                "n_pairs_total": int(total_pairs),
                "n_valid_pairs_total": int(total_valid),
                "match_rate_vs_gt": float(total_valid / total_gt) if total_gt else np.nan,
                "pair_valid_rate": float(total_valid / total_pairs) if total_pairs else np.nan,
                "MAE_x_mm_valid": float(valid_df["abs_err_x_mm"].mean()) if len(valid_df) else np.nan,
                "MAE_y_mm_valid": float(valid_df["abs_err_y_mm"].mean()) if len(valid_df) else np.nan,
                "MAE_z_mm_valid": float(valid_df["abs_err_z_mm"].mean()) if len(valid_df) else np.nan,
                "MAE_L2_3D_mm_valid": float(valid_df["dist_l2_3d_mm"].mean()) if len(valid_df) else np.nan,
                "match_method_used": ",".join(sorted(methods_used)),
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
