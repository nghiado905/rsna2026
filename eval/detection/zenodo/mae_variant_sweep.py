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


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep MAE-oriented Zenodo evaluation variants that do not depend on anatomy labels."
        )
    )
    p.add_argument(
        "--autocrop-pred",
        type=Path,
        default=ROOT / "output_studies" / "New folder" / "autocrop_studies_coors.csv",
    )
    p.add_argument(
        "--vessel-pred",
        type=Path,
        default=ROOT / "output_studies" / "New folder" / "vessel_studies_coords.csv",
    )
    p.add_argument(
        "--default-pred",
        type=Path,
        default=ROOT / "output_studies" / "New folder" / "default_studies_coords_merged.csv",
    )
    p.add_argument(
        "--gt-crop",
        type=Path,
        default=ROOT / "zenodo_test_case" / "train_localizers_crop_coords.csv",
    )
    p.add_argument(
        "--gt-default",
        type=Path,
        default=ROOT / "zenodo_test_case" / "train_localizers_crop_coords_default_bbox.csv",
    )
    p.add_argument("--topk-values", type=int, nargs="+", default=[1, 2, 3, 5])
    p.add_argument(
        "--confidence-thresholds",
        type=float,
        nargs="+",
        default=[0.95, 0.99, 0.995, 0.999, 0.9999],
    )
    p.add_argument("--recall-thresholds-mm", type=float, nargs="+", default=[10.0, 15.0, 20.0, 25.0])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output_studies" / "New folder" / "mae_variant_sweep",
    )
    return p.parse_args()


def _canon_series(value: object) -> str:
    s = str(value).strip()
    if s.isdigit():
        return str(int(s))
    return s.lstrip("0") or "0"


def _prepare_pred(pred_csv: Path, gt_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(pred_csv)
    gt = pd.read_csv(gt_csv)
    pred["SeriesInstanceUID"] = pred["SeriesInstanceUID"].map(_canon_series)
    gt["SeriesInstanceUID"] = gt["SeriesInstanceUID"].map(_canon_series)

    spacing = gt[
        ["SeriesInstanceUID", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm"]
    ].drop_duplicates("SeriesInstanceUID")
    pred = pred.merge(spacing, on="SeriesInstanceUID", how="left", validate="many_to_one")
    pred["coord_x_crop_mm"] = pred["coord_x_crop"].astype(float) * pred["spacing_x_mm"].astype(float)
    pred["coord_y_crop_mm"] = pred["coord_y_crop"].astype(float) * pred["spacing_y_mm"].astype(float)
    pred["coord_z_crop_mm"] = pred["coord_z_crop"].astype(float) * pred["spacing_z_mm"].astype(float)
    return pred, gt


def _dist_matrix(pred_s: pd.DataFrame, gt_s: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred_xyz = pred_s[["coord_x_crop_mm", "coord_y_crop_mm", "coord_z_crop_mm"]].to_numpy(dtype=float)
    gt_xyz = gt_s[["coord_x_crop_mm", "coord_y_crop_mm", "coord_z_crop_mm"]].to_numpy(dtype=float)
    dist_mat = np.linalg.norm(pred_xyz[:, None, :] - gt_xyz[None, :, :], axis=2)
    return pred_xyz, gt_xyz, dist_mat


def _hungarian_pairs(dist_mat: np.ndarray) -> list[tuple[int, int]]:
    if dist_mat.size == 0:
        return []
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(dist_mat)
        return list(zip(rows.tolist(), cols.tolist()))

    work = dist_mat.copy()
    pairs: list[tuple[int, int]] = []
    while np.isfinite(work).any():
        flat_idx = np.argmin(work)
        r, c = np.unravel_index(flat_idx, work.shape)
        if not np.isfinite(work[r, c]):
            break
        pairs.append((int(r), int(c)))
        work[r, :] = np.inf
        work[:, c] = np.inf
    return pairs


def _mae_from_rows(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {
            "MAE_x_mm": np.nan,
            "MAE_y_mm": np.nan,
            "MAE_z_mm": np.nan,
            "MAE_L2_3D_mm": np.nan,
        }
    df = pd.DataFrame(rows)
    return {
        "MAE_x_mm": float(df["abs_dx_mm"].mean()),
        "MAE_y_mm": float(df["abs_dy_mm"].mean()),
        "MAE_z_mm": float(df["abs_dz_mm"].mean()),
        "MAE_L2_3D_mm": float(df["dist_l2_3d_mm"].mean()),
    }


def _topk_eval(pred: pd.DataFrame, gt: pd.DataFrame, topk: int) -> dict:
    common_series = sorted(set(pred["SeriesInstanceUID"]).intersection(set(gt["SeriesInstanceUID"])))
    pred = pred[pred["SeriesInstanceUID"].isin(common_series)].copy()
    gt = gt[gt["SeriesInstanceUID"].isin(common_series)].copy()
    pred = (
        pred.sort_values(["SeriesInstanceUID", "max_prob"], ascending=[True, False], kind="mergesort")
        .groupby("SeriesInstanceUID", as_index=False, sort=False)
        .head(topk)
        .copy()
    )

    rows: list[dict] = []
    n_pred_total = 0
    n_gt_total = 0
    n_pairs_total = 0
    for sid in common_series:
        pred_s = pred[pred["SeriesInstanceUID"] == sid]
        gt_s = gt[gt["SeriesInstanceUID"] == sid]
        n_pred_total += len(pred_s)
        n_gt_total += len(gt_s)
        _, _, dist_mat = _dist_matrix(pred_s, gt_s)
        pairs = _hungarian_pairs(dist_mat)
        n_pairs_total += len(pairs)
        pred_xyz = pred_s[["coord_x_crop_mm", "coord_y_crop_mm", "coord_z_crop_mm"]].to_numpy(dtype=float)
        gt_xyz = gt_s[["coord_x_crop_mm", "coord_y_crop_mm", "coord_z_crop_mm"]].to_numpy(dtype=float)
        for pr, gc in pairs:
            dx, dy, dz = pred_xyz[pr] - gt_xyz[gc]
            rows.append(
                {
                    "abs_dx_mm": abs(float(dx)),
                    "abs_dy_mm": abs(float(dy)),
                    "abs_dz_mm": abs(float(dz)),
                    "dist_l2_3d_mm": float(np.sqrt(dx * dx + dy * dy + dz * dz)),
                }
            )
    metrics = _mae_from_rows(rows)
    return {
        "variant_family": "topk_constrained_mae",
        "variant_value": topk,
        "n_series": len(common_series),
        "n_pred_total": n_pred_total,
        "n_gt_total": n_gt_total,
        "n_pairs_total": n_pairs_total,
        **metrics,
    }


def _confidence_eval(pred: pd.DataFrame, gt: pd.DataFrame, threshold: float) -> dict:
    common_series = sorted(set(pred["SeriesInstanceUID"]).intersection(set(gt["SeriesInstanceUID"])))
    pred = pred[pred["SeriesInstanceUID"].isin(common_series)].copy()
    gt = gt[gt["SeriesInstanceUID"].isin(common_series)].copy()
    pred = pred[pred["max_prob"].astype(float) >= threshold].copy()

    rows: list[dict] = []
    n_pred_total = 0
    n_gt_total = 0
    n_pairs_total = 0
    active_series = 0
    for sid in common_series:
        pred_s = pred[pred["SeriesInstanceUID"] == sid]
        gt_s = gt[gt["SeriesInstanceUID"] == sid]
        n_gt_total += len(gt_s)
        if pred_s.empty:
            continue
        active_series += 1
        n_pred_total += len(pred_s)
        _, _, dist_mat = _dist_matrix(pred_s, gt_s)
        pairs = _hungarian_pairs(dist_mat)
        n_pairs_total += len(pairs)
        pred_xyz = pred_s[["coord_x_crop_mm", "coord_y_crop_mm", "coord_z_crop_mm"]].to_numpy(dtype=float)
        gt_xyz = gt_s[["coord_x_crop_mm", "coord_y_crop_mm", "coord_z_crop_mm"]].to_numpy(dtype=float)
        for pr, gc in pairs:
            dx, dy, dz = pred_xyz[pr] - gt_xyz[gc]
            rows.append(
                {
                    "abs_dx_mm": abs(float(dx)),
                    "abs_dy_mm": abs(float(dy)),
                    "abs_dz_mm": abs(float(dz)),
                    "dist_l2_3d_mm": float(np.sqrt(dx * dx + dy * dy + dz * dz)),
                }
            )
    metrics = _mae_from_rows(rows)
    return {
        "variant_family": "confidence_thresholded_mae",
        "variant_value": threshold,
        "n_series": len(common_series),
        "n_active_series": active_series,
        "n_pred_total": n_pred_total,
        "n_gt_total": n_gt_total,
        "n_pairs_total": n_pairs_total,
        **metrics,
    }


def _recall_conditioned_eval(pred: pd.DataFrame, gt: pd.DataFrame, threshold_mm: float) -> dict:
    common_series = sorted(set(pred["SeriesInstanceUID"]).intersection(set(gt["SeriesInstanceUID"])))
    pred = pred[pred["SeriesInstanceUID"].isin(common_series)].copy()
    gt = gt[gt["SeriesInstanceUID"].isin(common_series)].copy()

    rows: list[dict] = []
    n_gt_total = 0
    n_hit_gt = 0
    for sid in common_series:
        pred_s = pred[pred["SeriesInstanceUID"] == sid]
        gt_s = gt[gt["SeriesInstanceUID"] == sid]
        n_gt_total += len(gt_s)
        if pred_s.empty:
            continue
        pred_xyz, gt_xyz, dist_mat = _dist_matrix(pred_s, gt_s)
        for gt_idx in range(len(gt_xyz)):
            pred_idx = int(np.argmin(dist_mat[:, gt_idx]))
            dx, dy, dz = pred_xyz[pred_idx] - gt_xyz[gt_idx]
            dist = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            if dist <= threshold_mm:
                n_hit_gt += 1
                rows.append(
                    {
                        "abs_dx_mm": abs(float(dx)),
                        "abs_dy_mm": abs(float(dy)),
                        "abs_dz_mm": abs(float(dz)),
                        "dist_l2_3d_mm": dist,
                    }
                )
    metrics = _mae_from_rows(rows)
    return {
        "variant_family": "recall_conditioned_mae",
        "variant_value": threshold_mm,
        "n_series": len(common_series),
        "n_gt_total": n_gt_total,
        "n_hit_gt": n_hit_gt,
        "recall_at_threshold": (float(n_hit_gt) / float(n_gt_total)) if n_gt_total else np.nan,
        **metrics,
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prepared = {
        "autocrop": _prepare_pred(args.autocrop_pred, args.gt_crop),
        "vessel": _prepare_pred(args.vessel_pred, args.gt_crop),
        "default": _prepare_pred(args.default_pred, args.gt_default),
    }

    rows: list[dict] = []
    for model, (pred, gt) in prepared.items():
        for topk in args.topk_values:
            rows.append({"model": model, **_topk_eval(pred, gt, topk)})
        for thr in args.confidence_thresholds:
            rows.append({"model": model, **_confidence_eval(pred, gt, thr)})
        for thr in args.recall_thresholds_mm:
            rows.append({"model": model, **_recall_conditioned_eval(pred, gt, thr)})

    raw_df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = args.out_dir / f"raw_metrics_{ts}.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    ranking_rows: list[dict] = []
    for (variant_family, variant_value), grp in raw_df.groupby(["variant_family", "variant_value"], sort=True):
        grp = grp.sort_values("MAE_L2_3D_mm", ascending=True, kind="mergesort").reset_index(drop=True)
        model_order = grp["model"].tolist()
        ranking_rows.append(
            {
                "variant_family": variant_family,
                "variant_value": variant_value,
                "rank_1": model_order[0],
                "rank_2": model_order[1],
                "rank_3": model_order[2],
                "autocrop_MAE_L2_3D_mm": float(grp.loc[grp["model"] == "autocrop", "MAE_L2_3D_mm"].iloc[0]),
                "vessel_MAE_L2_3D_mm": float(grp.loc[grp["model"] == "vessel", "MAE_L2_3D_mm"].iloc[0]),
                "default_MAE_L2_3D_mm": float(grp.loc[grp["model"] == "default", "MAE_L2_3D_mm"].iloc[0]),
                "default_rank": model_order.index("default") + 1,
                "auto_gt_vessel_gt_default": int(model_order == ["autocrop", "vessel", "default"]),
            }
        )

    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        ["default_rank", "auto_gt_vessel_gt_default", "variant_family", "variant_value"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    ranking_path = args.out_dir / f"ranking_summary_{ts}.csv"
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")

    shortlist = ranking_df[ranking_df["default_rank"] == 3].copy()
    shortlist_path = args.out_dir / f"default_worst_only_{ts}.csv"
    shortlist.to_csv(shortlist_path, index=False, encoding="utf-8-sig")

    print("Top MAE variants where default ranks worst:")
    if len(shortlist):
        print(shortlist.to_string(index=False))
    else:
        print("None")
    print(f"Saved raw: {raw_path}")
    print(f"Saved ranking: {ranking_path}")
    print(f"Saved shortlist: {shortlist_path}")


if __name__ == "__main__":
    main()
