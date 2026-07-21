import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_fscore_support, roc_auc_score, roc_curve

SUB_PATHS = [
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\output_last\autocrop_leak_submission.csv",
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\output_last\vessel_leak_submission.csv",
    r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\output_last\default_leak_submission.csv",
]
GT_PATH = r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\train_localizers copy.csv"

GLOBAL_LABEL = "Aneurysm Present"
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

ALL_LABELS = [GLOBAL_LABEL] + LOCATION_LABELS
RSNA_WEIGHTS = [13] + [1] * 13
OUTPUT_DIR = "eval_outputs"
THRESH = 0.58
AP_MODE = "original"
BLEND_ALPHA = 0.5
TOPK = 3


class ParticipantVisibleError(Exception):
    """Error surfaced to participants when submission formatting is invalid."""


def prepare_ground_truth(gt_path, target_uids):
    """Pivot ground-truth to one-hot with global aneurysm flag."""
    df = pd.read_csv(gt_path)

    gt_pivot = df.pivot_table(
        index="SeriesInstanceUID",
        columns="location",
        aggfunc="size",
        fill_value=0,
    ).reindex(columns=LOCATION_LABELS, fill_value=0)

    gt_pivot[GLOBAL_LABEL] = (gt_pivot.sum(axis=1) > 0).astype(int)
    gt_onehot = gt_pivot.reindex(target_uids, fill_value=0)

    gt_text_list = []
    for uid in target_uids:
        labels = gt_onehot.loc[uid][gt_onehot.loc[uid] > 0].index.tolist()
        gt_text_list.append(", ".join(labels) if labels else "Negative (No Aneurysm)")

    gt_df = gt_onehot.reset_index().rename(columns={"index": "SeriesInstanceUID"})
    return gt_df, gt_text_list


def weighted_multilabel_auc(y_true: np.ndarray, y_scores: np.ndarray, class_weights=None):
    """Compute weighted macro AUC exactly as official scorer."""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_classes = y_true.shape[1]

    try:
        individual_aucs = roc_auc_score(y_true, y_scores, average=None)
    except ValueError:
        raise ParticipantVisibleError("AUC could not be calculated from given predictions.") from None

    if class_weights is None:
        weights_array = np.ones(n_classes)
    else:
        weights_array = np.asarray(class_weights)

    if len(weights_array) != n_classes:
        raise ValueError(
            f"Number of weights ({len(weights_array)}) must match number of classes ({n_classes})"
        )
    if np.any(weights_array < 0):
        raise ValueError("All class weights must be non-negative")
    if np.sum(weights_array) == 0:
        raise ValueError("At least one class weight must be positive")

    weights_array = weights_array / np.sum(weights_array)
    return float(np.sum(individual_aucs * weights_array)), individual_aucs


def plot_roc_curves(y_true: np.ndarray, y_scores: np.ndarray, labels, output_path="roc_curves.png"):
    """Draw per-class ROC curves."""
    plt.figure(figsize=(10, 8))
    for i, label in enumerate(labels):
        try:
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
        except ValueError:
            plt.plot([0, 1], [0, 1], "--", color="gray", alpha=0.2, label=f"{label} (no positives/negatives)")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-class ROC curves")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved ROC: {output_path}")


def adjust_aneurysm_present(sub_df: pd.DataFrame, mode: str, alpha: float, topk: int) -> pd.DataFrame:
    out = sub_df.copy()
    loc_scores = out[LOCATION_LABELS].to_numpy(dtype=float)
    old_ap = out[GLOBAL_LABEL].to_numpy(dtype=float)
    max_loc = loc_scores.max(axis=1)
    k = max(1, min(int(topk), loc_scores.shape[1]))
    topk_mean = np.sort(loc_scores, axis=1)[:, -k:].mean(axis=1)

    if mode == "original":
        new_ap = old_ap
    elif mode == "max_loc":
        new_ap = max_loc
    elif mode == "max_with_old":
        new_ap = np.maximum(old_ap, max_loc)
    elif mode == "blend_max":
        new_ap = alpha * old_ap + (1.0 - alpha) * max_loc
    elif mode == "mean_topk":
        new_ap = topk_mean
    elif mode == "blend_topk":
        new_ap = alpha * old_ap + (1.0 - alpha) * topk_mean
    else:
        raise ValueError(f"Unsupported AP_MODE: {mode}")

    out[GLOBAL_LABEL] = np.clip(new_ap, 0.0, 1.0)
    return out


def evaluate_one_submission(sub_path: str, batch_out_dir: str):
    print(f"Reading Submission: {sub_path}")
    sub_df = pd.read_csv(sub_path)
    sub_df.columns = sub_df.columns.str.strip()
    sub_df["SeriesInstanceUID"] = sub_df["SeriesInstanceUID"].astype(str).str.strip()

    sub_cols = [c for c in sub_df.columns if c in ALL_LABELS]
    if len(sub_cols) != len(ALL_LABELS):
        missing = set(ALL_LABELS) - set(sub_cols)
        raise ParticipantVisibleError(f"Thieu cot: {missing}")

    sub_df = adjust_aneurysm_present(sub_df, AP_MODE, BLEND_ALPHA, TOPK)
    print(f"[CONFIG] AP_MODE={AP_MODE} THRESH={THRESH} BLEND_ALPHA={BLEND_ALPHA} TOPK={TOPK}")

    dup_mask = sub_df["SeriesInstanceUID"].duplicated(keep=False)
    dup_count = int(dup_mask.sum())
    unique_uid_count = int(sub_df["SeriesInstanceUID"].nunique())
    print(f"[CHECK] submission rows={len(sub_df)} unique_series={unique_uid_count} duplicate_rows={dup_count}")
    if dup_count > 0:
        dup_uids = sub_df.loc[dup_mask, "SeriesInstanceUID"].drop_duplicates().astype(str).tolist()[:10]
        print(f"[WARN] Duplicate SeriesInstanceUID found in submission, sample={dup_uids}")

    ordered_cols = ALL_LABELS.copy()
    target_uids = sub_df["SeriesInstanceUID"].unique()
    sol_df, gt_text_list = prepare_ground_truth(GT_PATH, target_uids)

    sol_df = sol_df[["SeriesInstanceUID"] + ordered_cols].sort_values("SeriesInstanceUID").reset_index(drop=True)
    sub_df = sub_df[["SeriesInstanceUID"] + ordered_cols].sort_values("SeriesInstanceUID").reset_index(drop=True)
    sol_df["SeriesInstanceUID"] = sol_df["SeriesInstanceUID"].astype(str).str.strip()
    sub_df["SeriesInstanceUID"] = sub_df["SeriesInstanceUID"].astype(str).str.strip()

    uid_match = sol_df["SeriesInstanceUID"].equals(sub_df["SeriesInstanceUID"])
    print(f"[CHECK] after sort, GT/submission SeriesInstanceUID aligned={uid_match}")
    if not uid_match:
        mismatch_idx = sol_df["SeriesInstanceUID"] != sub_df["SeriesInstanceUID"]
        mismatch_count = int(mismatch_idx.sum())
        print(f"[WARN] UID alignment mismatch rows={mismatch_count}")
        sample = pd.DataFrame(
            {
                "GT_UID": sol_df.loc[mismatch_idx, "SeriesInstanceUID"],
                "SUB_UID": sub_df.loc[mismatch_idx, "SeriesInstanceUID"],
            }
        ).head(10)
        print(sample.to_string(index=False))
    else:
        print(f"[CHECK] aligned rows={len(sol_df)}")

    y_pred = sub_df[ordered_cols].values
    sorted_gt_text = sub_df["SeriesInstanceUID"].map(dict(zip(target_uids, gt_text_list))).tolist()
    print(f"-> So ca danh gia: {len(sub_df)}")

    sub_name = Path(sub_path).stem
    out_dir = os.path.join(batch_out_dir, sub_name)
    os.makedirs(out_dir, exist_ok=True)
    adjusted_submission_path = os.path.join(out_dir, "submission_adjusted.csv")
    sub_df.to_csv(adjusted_submission_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {adjusted_submission_path}")

    try:
        final_score, individual_aucs = weighted_multilabel_auc(
            sol_df[ordered_cols].values,
            y_pred,
            class_weights=RSNA_WEIGHTS,
        )
    except ParticipantVisibleError:
        print("Warning: some classes lack positive/negative samples; using 0.5 for those AUCs.")
        per_class = []
        for i in range(len(ordered_cols)):
            try:
                per_class.append(roc_auc_score(sol_df[ordered_cols].values[:, i], y_pred[:, i]))
            except ValueError:
                per_class.append(0.5)
        per_class = np.array(per_class)
        weights = np.asarray(RSNA_WEIGHTS, dtype=float)
        weights = weights / weights.sum()
        final_score = float(np.sum(per_class * weights))
        individual_aucs = per_class

    roc_path = os.path.join(out_dir, "roc_curves.png")
    plot_roc_curves(sol_df[ordered_cols].values, y_pred, ordered_cols, output_path=roc_path)

    report_data = []
    for idx, row in sub_df.iterrows():
        uid = row["SeriesInstanceUID"]
        actual_txt = sorted_gt_text[idx]
        loc_probs = row[LOCATION_LABELS]
        top_pred_name = loc_probs.idxmax()
        top_pred_prob = loc_probs.max()
        ap_prob = row[GLOBAL_LABEL]

        correct_flag = int(top_pred_name in actual_txt)
        status = "Negative Case" if "Negative" in actual_txt else ("Location Match" if correct_flag else "Mismatch")
        report_data.append(
            {
                "SeriesInstanceUID": uid,
                "Actual_Labels": actual_txt,
                "Pred_Aneurysm_Prob": f"{ap_prob:.4f}",
                "Pred_Label": top_pred_name,
                "Pred_Label_Prob": f"{top_pred_prob:.4f}",
                "MatchFlag": correct_flag,
                "Note": status,
            }
        )

    df_report = pd.DataFrame(report_data)
    detailed_path = os.path.join(out_dir, "detailed_report.csv")
    df_report.to_csv(detailed_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {detailed_path}")

    gt_array = sol_df[ordered_cols].values
    gt_bin = (gt_array > 0).astype(int)
    y_true_bin = gt_bin
    y_pred_bin = (y_pred >= THRESH).astype(int)
    top_pred_labels = df_report["Pred_Label"].values
    label_summary = []
    for i, label in enumerate(ordered_cols):
        gt_pos = int(gt_bin[:, i].sum())
        top_count = int((top_pred_labels == label).sum())
        top_correct = int(((top_pred_labels == label) & (gt_bin[:, i] == 1)).sum())

        pred_bin_col = y_pred_bin[:, i]
        true_bin_col = gt_bin[:, i]
        tp = int(((true_bin_col == 1) & (pred_bin_col == 1)).sum())
        tn = int(((true_bin_col == 0) & (pred_bin_col == 0)).sum())
        fp = int(((true_bin_col == 0) & (pred_bin_col == 1)).sum())
        fn = int(((true_bin_col == 1) & (pred_bin_col == 0)).sum())

        precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_i = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0.0

        label_summary.append(
            {
                "Label": label,
                "GT_Positive": gt_pos,
                "TopPred_Count": top_count,
                "TopPred_Correct": top_correct,
                "AUC": individual_aucs[i],
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "Precision": precision_i,
                "Recall": recall_i,
                "F1": f1_i,
            }
        )
    df_summary = pd.DataFrame(label_summary)
    summary_path = os.path.join(out_dir, "label_stats.csv")
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {summary_path}")

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true_bin,
        y_pred_bin,
        average=None,
        zero_division=0,
    )
    df_prf = pd.DataFrame({"Label": ordered_cols, "Precision": prec, "Recall": rec, "F1": f1})
    prf_path = os.path.join(out_dir, "label_prf.csv")
    df_prf.to_csv(prf_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {prf_path}")

    overall_rows = []
    for avg_name in ["micro", "macro", "weighted"]:
        avg_prec, avg_rec, avg_f1, _ = precision_recall_fscore_support(
            y_true_bin,
            y_pred_bin,
            average=avg_name,
            zero_division=0,
        )
        overall_rows.append(
            {
                "Average": avg_name,
                "Precision": avg_prec,
                "Recall": avg_rec,
                "F1": avg_f1,
            }
        )

    df_overall = pd.DataFrame(overall_rows)
    overall_path = os.path.join(out_dir, "overall_metrics.csv")
    df_overall.to_csv(overall_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {overall_path}")

    print("\n" + "=" * 60)
    print(f"{'LABEL':<45} | {'AUC':<8} | {'PREC':<8} | {'RECALL':<8} | {'F1':<8}")
    print("-" * 95)
    for label, auc_val, p_val, r_val, f1_val in zip(ordered_cols, individual_aucs, prec, rec, f1):
        print(f"{label:<45} | {auc_val:.4f} | {p_val:.4f} | {r_val:.4f} | {f1_val:.4f}")
    print("-" * 95)
    print(f"FINAL OFFICIAL SCORE: {final_score:.6f}")
    for _, metric_row in df_overall.iterrows():
        print(
            f"{metric_row['Average'].capitalize()} Precision: {metric_row['Precision']:.4f} | "
            f"{metric_row['Average'].capitalize()} Recall: {metric_row['Recall']:.4f} | "
            f"{metric_row['Average'].capitalize()} F1: {metric_row['F1']:.4f}"
        )
    print("=" * 60)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_out_dir = os.path.join(OUTPUT_DIR, ts)
    os.makedirs(batch_out_dir, exist_ok=True)
    print(f"[BATCH OUT] {batch_out_dir}")
    for sub_path in SUB_PATHS:
        print("\n" + "#" * 100)
        evaluate_one_submission(sub_path, batch_out_dir)


if __name__ == "__main__":
    main()
