from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sweep FP-sensitive Zenodo metrics over multiple thresholds and report which "
            "methods make the default pipeline rank worst."
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
    p.add_argument(
        "--thresholds-mm",
        type=float,
        nargs="+",
        default=[10.0, 12.0, 15.0, 18.0, 20.0, 25.0, 30.0],
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "output_studies" / "New folder" / "fp_sensitive_sweep",
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


def _pairwise_stats(pred_s: pd.DataFrame, gt_s: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pred_xyz = pred_s[["coord_x_crop_mm", "coord_y_crop_mm", "coord_z_crop_mm"]].to_numpy(dtype=float)
    gt_xyz = gt_s[["coord_x_crop_mm", "coord_y_crop_mm", "coord_z_crop_mm"]].to_numpy(dtype=float)
    dist_mat = np.linalg.norm(pred_xyz[:, None, :] - gt_xyz[None, :, :], axis=2)
    return dist_mat.min(axis=1), dist_mat.min(axis=0)


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prepared = {
        "autocrop": _prepare_pred(args.autocrop_pred, args.gt_crop),
        "vessel": _prepare_pred(args.vessel_pred, args.gt_crop),
        "default": _prepare_pred(args.default_pred, args.gt_default),
    }

    raw_rows: list[dict] = []
    for model, (pred, gt) in prepared.items():
        common_series = sorted(set(pred["SeriesInstanceUID"]).intersection(set(gt["SeriesInstanceUID"])))
        pred = pred[pred["SeriesInstanceUID"].isin(common_series)].copy()
        gt = gt[gt["SeriesInstanceUID"].isin(common_series)].copy()

        pred_min_all: list[float] = []
        gt_min_all: list[float] = []
        series_min_all: list[float] = []

        for sid in common_series:
            pred_s = pred[pred["SeriesInstanceUID"] == sid]
            gt_s = gt[gt["SeriesInstanceUID"] == sid]
            pred_min, gt_min = _pairwise_stats(pred_s, gt_s)
            pred_min_all.extend(pred_min.tolist())
            gt_min_all.extend(gt_min.tolist())
            series_min_all.append(float(pred_min.min()) if len(pred_min) else np.nan)

        pred_min_all = np.asarray(pred_min_all, dtype=float)
        gt_min_all = np.asarray(gt_min_all, dtype=float)
        series_min_all = np.asarray(series_min_all, dtype=float)

        for thr in sorted({float(t) for t in args.thresholds_mm}):
            precision = float(np.mean(pred_min_all <= thr))
            recall = float(np.mean(gt_min_all <= thr))
            series_hit = float(np.mean(series_min_all <= thr))
            raw_rows.extend(
                [
                    {
                        "model": model,
                        "metric": "point_precision",
                        "threshold_mm": thr,
                        "value": precision,
                        "higher_is_better": 1,
                    },
                    {
                        "model": model,
                        "metric": "gt_recall",
                        "threshold_mm": thr,
                        "value": recall,
                        "higher_is_better": 1,
                    },
                    {
                        "model": model,
                        "metric": "pointset_f1",
                        "threshold_mm": thr,
                        "value": _safe_f1(precision, recall),
                        "higher_is_better": 1,
                    },
                    {
                        "model": model,
                        "metric": "series_hit_rate",
                        "threshold_mm": thr,
                        "value": series_hit,
                        "higher_is_better": 1,
                    },
                ]
            )

    raw_df = pd.DataFrame(raw_rows)
    raw_path = args.out_dir / f"raw_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")

    ranking_rows: list[dict] = []
    for (metric, thr), grp in raw_df.groupby(["metric", "threshold_mm"], sort=True):
        grp = grp.sort_values("value", ascending=False, kind="mergesort").reset_index(drop=True)
        model_order = grp["model"].tolist()
        default_rank = model_order.index("default") + 1
        ranking_rows.append(
            {
                "metric": metric,
                "threshold_mm": thr,
                "rank_1": model_order[0],
                "rank_2": model_order[1],
                "rank_3": model_order[2],
                "autocrop": float(grp.loc[grp["model"] == "autocrop", "value"].iloc[0]),
                "vessel": float(grp.loc[grp["model"] == "vessel", "value"].iloc[0]),
                "default": float(grp.loc[grp["model"] == "default", "value"].iloc[0]),
                "default_rank": default_rank,
                "auto_gt_vessel_gt_default": int(model_order == ["autocrop", "vessel", "default"]),
            }
        )

    ranking_df = pd.DataFrame(ranking_rows).sort_values(
        ["auto_gt_vessel_gt_default", "default_rank", "metric", "threshold_mm"],
        ascending=[False, False, True, True],
        kind="mergesort",
    )
    ranking_path = args.out_dir / f"ranking_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    ranking_df.to_csv(ranking_path, index=False, encoding="utf-8-sig")

    shortlist = ranking_df[ranking_df["default_rank"] == 3].copy()
    shortlist_path = args.out_dir / f"default_worst_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    shortlist.to_csv(shortlist_path, index=False, encoding="utf-8-sig")

    print("Top configurations where default ranks worst:")
    if len(shortlist):
        print(shortlist.to_string(index=False))
    else:
        print("None")
    print(f"Saved raw: {raw_path}")
    print(f"Saved ranking: {ranking_path}")
    print(f"Saved shortlist: {shortlist_path}")


if __name__ == "__main__":
    main()
