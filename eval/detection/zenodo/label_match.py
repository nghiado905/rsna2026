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
            "Evaluate coordinate predictions by matching on SeriesInstanceUID + label, "
            "then computing distance metrics on matched label pairs."
        )
    )
    p.add_argument("--pred-coords-csv", type=Path, required=True)
    p.add_argument("--gt-coords-csv", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--pred-series-col", type=str, default="SeriesInstanceUID")
    p.add_argument("--pred-label-col", type=str, default="label")
    p.add_argument("--pred-score-col", type=str, default="max_prob")
    p.add_argument("--pred-x-col", type=str, default="coord_x_crop_mm")
    p.add_argument("--pred-y-col", type=str, default="coord_y_crop_mm")
    p.add_argument("--pred-z-col", type=str, default="coord_z_crop_mm")
    p.add_argument("--gt-series-col", type=str, default="SeriesInstanceUID")
    p.add_argument("--gt-label-col", type=str, default="location")
    p.add_argument("--gt-x-col", type=str, default="coord_x_crop_mm")
    p.add_argument("--gt-y-col", type=str, default="coord_y_crop_mm")
    p.add_argument("--gt-z-col", type=str, default="coord_z_crop_mm")
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
            args.gt_x_col,
            args.gt_y_col,
            args.gt_z_col,
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

    pred_keys = set(zip(pred[args.pred_series_col], pred[args.pred_label_col]))
    gt_keys = set(zip(gt[args.gt_series_col], gt[args.gt_label_col]))
    common_keys = pred_keys.intersection(gt_keys)

    matched_rows: list[dict] = []
    per_key_rows: list[dict] = []
    methods_used: set[str] = set()
    common_series = sorted({sid for sid, _ in common_keys})
    total_pred = 0
    total_gt = 0
    total_matched = 0

    for series_uid, label in sorted(common_keys):
        pred_s = pred[
            (pred[args.pred_series_col] == series_uid) & (pred[args.pred_label_col] == label)
        ].copy()
        gt_s = gt[
            (gt[args.gt_series_col] == series_uid) & (gt[args.gt_label_col] == label)
        ].copy()
        if len(pred_s) == 0 or len(gt_s) == 0:
            continue

        pred_xyz = pred_s[[args.pred_x_col, args.pred_y_col, args.pred_z_col]].to_numpy(dtype=float)
        gt_xyz = gt_s[[args.gt_x_col, args.gt_y_col, args.gt_z_col]].to_numpy(dtype=float)
        dist_mat = np.linalg.norm(pred_xyz[:, None, :] - gt_xyz[None, :, :], axis=2)
        row_idx, col_idx, method = _assign_pairs(dist_mat)
        methods_used.add(method)

        total_pred += len(pred_s)
        total_gt += len(gt_s)
        total_matched += len(row_idx)

        key_err_x: list[float] = []
        key_err_y: list[float] = []
        key_err_z: list[float] = []
        key_l2_xy: list[float] = []
        key_l2_3d: list[float] = []

        pred_s = pred_s.reset_index(drop=True)
        gt_s = gt_s.reset_index(drop=True)
        for i_pred, i_gt in zip(row_idx, col_idx):
            pred_row = pred_s.iloc[int(i_pred)]
            gt_row = gt_s.iloc[int(i_gt)]
            dx = float(pred_row[args.pred_x_col] - gt_row[args.gt_x_col])
            dy = float(pred_row[args.pred_y_col] - gt_row[args.gt_y_col])
            dz = float(pred_row[args.pred_z_col] - gt_row[args.gt_z_col])
            abs_dx = abs(dx)
            abs_dy = abs(dy)
            abs_dz = abs(dz)
            l2_xy = float(np.sqrt(dx * dx + dy * dy))
            l2_3d = float(np.sqrt(dx * dx + dy * dy + dz * dz))

            key_err_x.append(abs_dx)
            key_err_y.append(abs_dy)
            key_err_z.append(abs_dz)
            key_l2_xy.append(l2_xy)
            key_l2_3d.append(l2_3d)

            matched_rows.append(
                {
                    "SeriesInstanceUID": series_uid,
                    "label": label,
                    "match_method": method,
                    "pred_score": float(pred_row[args.pred_score_col]),
                    "pred_x": float(pred_row[args.pred_x_col]),
                    "pred_y": float(pred_row[args.pred_y_col]),
                    "pred_z": float(pred_row[args.pred_z_col]),
                    "gt_x": float(gt_row[args.gt_x_col]),
                    "gt_y": float(gt_row[args.gt_y_col]),
                    "gt_z": float(gt_row[args.gt_z_col]),
                    "abs_err_x": abs_dx,
                    "abs_err_y": abs_dy,
                    "abs_err_z": abs_dz,
                    "dist_l2_xy": l2_xy,
                    "dist_l2_3d": l2_3d,
                }
            )

        per_key_rows.append(
            {
                "SeriesInstanceUID": series_uid,
                "label": label,
                "match_method": method,
                "n_pred": int(len(pred_s)),
                "n_gt": int(len(gt_s)),
                "n_matched": int(len(row_idx)),
                "match_rate_vs_gt": float(len(row_idx) / len(gt_s)),
                "MAE_x": float(np.mean(key_err_x)) if key_err_x else np.nan,
                "MAE_y": float(np.mean(key_err_y)) if key_err_y else np.nan,
                "MAE_z": float(np.mean(key_err_z)) if key_err_z else np.nan,
                "MAE_L2_XY": float(np.mean(key_l2_xy)) if key_l2_xy else np.nan,
                "MAE_L2_3D": float(np.mean(key_l2_3d)) if key_l2_3d else np.nan,
            }
        )

    matched_df = pd.DataFrame(matched_rows)
    per_key_df = pd.DataFrame(per_key_rows)
    overall_df = pd.DataFrame(
        [
            {
                "n_series": int(len(common_series)),
                "n_series_label_keys": int(len(common_keys)),
                "n_pred_total": int(total_pred),
                "n_gt_total": int(total_gt),
                "n_matched_total": int(total_matched),
                "match_rate_vs_gt": float(total_matched / total_gt) if total_gt else 0.0,
                "MAE_x": float(matched_df["abs_err_x"].mean()) if len(matched_df) else np.nan,
                "MAE_y": float(matched_df["abs_err_y"].mean()) if len(matched_df) else np.nan,
                "MAE_z": float(matched_df["abs_err_z"].mean()) if len(matched_df) else np.nan,
                "MAE_L2_XY": float(matched_df["dist_l2_xy"].mean()) if len(matched_df) else np.nan,
                "MAE_L2_3D": float(matched_df["dist_l2_3d"].mean()) if len(matched_df) else np.nan,
                "match_method_used": ",".join(sorted(methods_used)),
            }
        ]
    )

    out_dir = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_df.to_csv(out_dir / "matched_detail.csv", index=False, encoding="utf-8-sig")
    per_key_df.to_csv(out_dir / "per_key_metrics.csv", index=False, encoding="utf-8-sig")
    overall_df.to_csv(out_dir / "overall_metrics.csv", index=False, encoding="utf-8-sig")

    print(overall_df.to_string(index=False))
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
