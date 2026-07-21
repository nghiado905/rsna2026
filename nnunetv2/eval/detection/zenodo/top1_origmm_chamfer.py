from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Zenodo pipelines using Top-1 MAE in original local mm and Chamfer distance."
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
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output_studies" / "New folder" / "zenodo_top1_origmm_chamfer_eval",
    )
    return p.parse_args()


def _canon_series(value: object) -> str:
    s = str(value).strip()
    if s.isdigit():
        return str(int(s))
    return s.lstrip("0") or "0"


def _load_pair(pred_csv: Path, gt_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(pred_csv)
    gt = pd.read_csv(gt_csv)
    pred["SeriesInstanceUID"] = pred["SeriesInstanceUID"].map(_canon_series)
    gt["SeriesInstanceUID"] = gt["SeriesInstanceUID"].map(_canon_series)

    spacing_df = gt[
        ["SeriesInstanceUID", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm"]
    ].drop_duplicates(subset=["SeriesInstanceUID"])
    pred = pred.merge(spacing_df, on="SeriesInstanceUID", how="left", validate="many_to_one")
    return pred, gt


def _series_detail(pred: pd.DataFrame, gt: pd.DataFrame, model: str) -> tuple[pd.DataFrame, dict]:
    common_series = sorted(set(pred["SeriesInstanceUID"]).intersection(set(gt["SeriesInstanceUID"])))
    pred = pred[pred["SeriesInstanceUID"].isin(common_series)].copy()
    gt = gt[gt["SeriesInstanceUID"].isin(common_series)].copy()

    top1_pred = (
        pred.sort_values(["SeriesInstanceUID", "max_prob"], ascending=[True, False], kind="mergesort")
        .groupby("SeriesInstanceUID", as_index=False, sort=False)
        .head(1)
        .copy()
    )

    detail_rows: list[dict] = []
    for sid in common_series:
        p_all = pred[pred["SeriesInstanceUID"] == sid].copy()
        p_top1 = top1_pred[top1_pred["SeriesInstanceUID"] == sid].copy()
        g = gt[gt["SeriesInstanceUID"] == sid].copy()
        if p_all.empty or g.empty:
            continue

        gx = g["coord_x_orig"].to_numpy(dtype=float) * g["spacing_x_mm"].to_numpy(dtype=float)
        gy = g["coord_y_orig"].to_numpy(dtype=float) * g["spacing_y_mm"].to_numpy(dtype=float)
        gz = g["coord_z_orig"].to_numpy(dtype=float) * g["spacing_z_mm"].to_numpy(dtype=float)
        G_pts = np.stack([gx, gy, gz], axis=1)

        px_all = p_all["coord_x_orig"].to_numpy(dtype=float) * p_all["spacing_x_mm"].to_numpy(dtype=float)
        py_all = p_all["coord_y_orig"].to_numpy(dtype=float) * p_all["spacing_y_mm"].to_numpy(dtype=float)
        pz_all = p_all["coord_z_orig"].to_numpy(dtype=float) * p_all["spacing_z_mm"].to_numpy(dtype=float)
        P_pts = np.stack([px_all, py_all, pz_all], axis=1)
        dist_mat = np.linalg.norm(P_pts[:, None, :] - G_pts[None, :, :], axis=2)
        chamfer = float(np.mean(np.min(dist_mat, axis=1)) + np.mean(np.min(dist_mat, axis=0)))

        top1 = p_top1.iloc[0]
        top1_xyz = np.array(
            [
                float(top1["coord_x_orig"]) * float(top1["spacing_x_mm"]),
                float(top1["coord_y_orig"]) * float(top1["spacing_y_mm"]),
                float(top1["coord_z_orig"]) * float(top1["spacing_z_mm"]),
            ],
            dtype=float,
        )
        top1_dist = np.linalg.norm(G_pts - top1_xyz[None, :], axis=1)
        match_idx = int(np.argmin(top1_dist))
        gt_match = g.iloc[match_idx]
        dx = float(top1_xyz[0] - G_pts[match_idx, 0])
        dy = float(top1_xyz[1] - G_pts[match_idx, 1])
        dz = float(top1_xyz[2] - G_pts[match_idx, 2])

        detail_rows.append(
            {
                "model": model,
                "SeriesInstanceUID": sid,
                "pred_label_top1": str(top1["label"]),
                "pred_score_top1": float(top1["max_prob"]),
                "gt_match_label": str(gt_match["location"]),
                "n_pred_points": int(len(p_all)),
                "n_gt_points": int(len(g)),
                "pred_x_orig_mm": float(top1_xyz[0]),
                "pred_y_orig_mm": float(top1_xyz[1]),
                "pred_z_orig_mm": float(top1_xyz[2]),
                "gt_x_orig_mm": float(G_pts[match_idx, 0]),
                "gt_y_orig_mm": float(G_pts[match_idx, 1]),
                "gt_z_orig_mm": float(G_pts[match_idx, 2]),
                "abs_err_x_mm": abs(dx),
                "abs_err_y_mm": abs(dy),
                "abs_err_z_mm": abs(dz),
                "top1_l2_3d_mm": float(top1_dist[match_idx]),
                "chamfer_orig_mm": chamfer,
            }
        )

    detail_df = pd.DataFrame(detail_rows)
    summary = {
        "model": model,
        "n_series": int(detail_df["SeriesInstanceUID"].nunique()) if len(detail_df) else 0,
        "n_pred_total": int(len(pred)),
        "n_gt_total": int(len(gt)),
        "Top1_MAE_x_mm": float(detail_df["abs_err_x_mm"].mean()) if len(detail_df) else np.nan,
        "Top1_MAE_y_mm": float(detail_df["abs_err_y_mm"].mean()) if len(detail_df) else np.nan,
        "Top1_MAE_z_mm": float(detail_df["abs_err_z_mm"].mean()) if len(detail_df) else np.nan,
        "Top1_MAE_L2_3D_mm": float(detail_df["top1_l2_3d_mm"].mean()) if len(detail_df) else np.nan,
        "Chamfer_orig_mm": float(detail_df["chamfer_orig_mm"].mean()) if len(detail_df) else np.nan,
    }
    return detail_df, summary


def main() -> None:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("autocrop", args.autocrop_pred, args.gt_crop),
        ("vessel", args.vessel_pred, args.gt_crop),
        ("default", args.default_pred, args.gt_default),
    ]

    summaries: list[dict] = []
    all_details: list[pd.DataFrame] = []
    for model, pred_csv, gt_csv in configs:
        pred, gt = _load_pair(pred_csv, gt_csv)
        detail_df, summary = _series_detail(pred, gt, model)
        detail_path = out_dir / f"{model}_top1_chamfer_detail.csv"
        detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
        summaries.append(summary)
        all_details.append(detail_df)

    summary_df = pd.DataFrame(summaries).sort_values("Top1_MAE_L2_3D_mm", ascending=True, kind="mergesort")
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    all_detail_df = pd.concat(all_details, ignore_index=True)
    all_detail_path = out_dir / "all_models_top1_chamfer_detail.csv"
    all_detail_df.to_csv(all_detail_path, index=False, encoding="utf-8-sig")

    print(summary_df.to_string(index=False))
    print(f"Saved summary: {summary_path}")
    print(f"Saved all detail: {all_detail_path}")


if __name__ == "__main__":
    main()
