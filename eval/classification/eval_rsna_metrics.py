import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    hamming_loss,
    jaccard_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


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
RSNA_WEIGHTS = np.array([13] + [1] * len(LOCATION_LABELS), dtype=float)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RSNA aneurysm predictions with official weighted AUC and extra metrics."
    )
    parser.add_argument("--submission", required=True, help="Path to submission CSV.")
    parser.add_argument("--ground-truth", required=True, help="Path to train_localizers CSV.")
    parser.add_argument(
        "--mode",
        choices=["submission", "overlap"],
        default="submission",
        help="submission: official-like reindex on submission UIDs; overlap: inner join only.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary metrics such as precision/recall/F1.",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_rsna_metrics_outputs",
        help="Base output directory.",
    )
    return parser.parse_args()


def print_section(title: str):
    line = "=" * 100
    print("")
    print(line)
    print(title)
    print(line)


def print_table(df: pd.DataFrame, float_fmt: str = "{:.6f}"):
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: float_fmt.format(x))
    print(display_df.to_string(index=False))


def print_eval_style_summary(
    per_label_rows: list[dict],
    official_weighted_auc: float,
    macro_p: float,
    macro_r: float,
    macro_f1: float,
    micro_p: float,
    micro_r: float,
    micro_f1: float,
):
    print("")
    print("=" * 60)
    print(f"{'LABEL':<45} | {'AUC':<8} | {'PREC':<8} | {'RECALL':<8} | {'F1':<8}")
    print("-" * 95)
    for row in per_label_rows:
        print(
            f"{row['label']:<45} | "
            f"{row['roc_auc']:.4f} | "
            f"{row['precision']:.4f} | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f}"
        )
    print("-" * 95)
    print(f"FINAL OFFICIAL SCORE: {official_weighted_auc:.6f}")
    print(f"Macro Precision: {macro_p:.4f} | Macro Recall: {macro_r:.4f} | Macro F1: {macro_f1:.4f}")
    print(f"Micro Precision: {micro_p:.4f} | Micro Recall: {micro_r:.4f} | Micro F1: {micro_f1:.4f}")
    print("=" * 60)


def prepare_ground_truth(gt_path: str) -> pd.DataFrame:
    df = pd.read_csv(gt_path)
    gt_pivot = df.pivot_table(
        index="SeriesInstanceUID",
        columns="location",
        aggfunc="size",
        fill_value=0,
    ).reindex(columns=LOCATION_LABELS, fill_value=0)
    gt_pivot = (gt_pivot > 0).astype(int)
    gt_pivot[GLOBAL_LABEL] = (gt_pivot.sum(axis=1) > 0).astype(int)
    gt_df = gt_pivot.reset_index()
    gt_df["SeriesInstanceUID"] = gt_df["SeriesInstanceUID"].astype(str).str.strip()
    return gt_df[["SeriesInstanceUID"] + ALL_LABELS]


def prepare_submission(sub_path: str) -> pd.DataFrame:
    sub_df = pd.read_csv(sub_path)
    sub_df.columns = sub_df.columns.str.strip()
    sub_df["SeriesInstanceUID"] = sub_df["SeriesInstanceUID"].astype(str).str.strip()

    missing = [col for col in ["SeriesInstanceUID"] + ALL_LABELS if col not in sub_df.columns]
    if missing:
        raise ValueError(f"Submission missing required columns: {missing}")

    return sub_df[["SeriesInstanceUID"] + ALL_LABELS].copy()


def build_eval_frames(gt_df: pd.DataFrame, sub_df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mode == "submission":
        target_uids = sub_df["SeriesInstanceUID"].unique()
        sol_df = gt_df.set_index("SeriesInstanceUID").reindex(target_uids, fill_value=0).reset_index()
        pred_df = sub_df.copy()
    else:
        merged = gt_df.merge(sub_df, on="SeriesInstanceUID", how="inner", suffixes=("_true", "_pred"))
        sol_df = merged[["SeriesInstanceUID"] + [f"{c}_true" for c in ALL_LABELS]].copy()
        pred_df = merged[["SeriesInstanceUID"] + [f"{c}_pred" for c in ALL_LABELS]].copy()
        sol_df.columns = ["SeriesInstanceUID"] + ALL_LABELS
        pred_df.columns = ["SeriesInstanceUID"] + ALL_LABELS

    sol_df = sol_df.sort_values("SeriesInstanceUID").reset_index(drop=True)
    pred_df = pred_df.sort_values("SeriesInstanceUID").reset_index(drop=True)
    return sol_df, pred_df


def build_case_report(sol_df: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    report_rows = []
    for _, row in pred_df.iterrows():
        uid = row["SeriesInstanceUID"]
        gt_row = sol_df.loc[sol_df["SeriesInstanceUID"] == uid, ALL_LABELS].iloc[0]
        actual_labels = [label for label in ALL_LABELS if int(gt_row[label]) == 1]
        actual_text = ", ".join(actual_labels) if actual_labels else "Negative (No Aneurysm)"

        loc_probs = row[LOCATION_LABELS]
        top_pred_name = str(loc_probs.idxmax())
        top_pred_prob = float(loc_probs.max())
        ap_prob = float(row[GLOBAL_LABEL])

        correct_flag = int(top_pred_name in actual_labels)
        status = (
            "Negative Case"
            if not actual_labels
            else ("Location Match" if correct_flag else "Mismatch")
        )

        report_rows.append(
            {
                "SeriesInstanceUID": uid,
                "Actual_Labels": actual_text,
                "Pred_Aneurysm_Prob": round(ap_prob, 6),
                "Pred_Label": top_pred_name,
                "Pred_Label_Prob": round(top_pred_prob, 6),
                "MatchFlag": correct_flag,
                "Note": status,
            }
        )
    return pd.DataFrame(report_rows)


def safe_auc_per_label(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aucs = []
    valid = []
    for i in range(y_true.shape[1]):
        col_true = y_true[:, i]
        if np.unique(col_true).size < 2:
            aucs.append(0.5)
            valid.append(False)
        else:
            aucs.append(roc_auc_score(col_true, y_scores[:, i]))
            valid.append(True)
    return np.array(aucs, dtype=float), np.array(valid, dtype=bool)


def plot_roc_curves(y_true: np.ndarray, y_scores: np.ndarray, labels: list[str], output_path: str):
    plt.figure(figsize=(11, 8))
    for i, label in enumerate(labels):
        if np.unique(y_true[:, i]).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-class ROC curves")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_pr_curves(y_true: np.ndarray, y_scores: np.ndarray, labels: list[str], output_path: str):
    plt.figure(figsize=(11, 8))
    for i, label in enumerate(labels):
        if np.unique(y_true[:, i]).size < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{label} (PR AUC={pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Per-class Precision-Recall curves")
    plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()

    gt_df = prepare_ground_truth(args.ground_truth)
    sub_df = prepare_submission(args.submission)
    sol_df, pred_df = build_eval_frames(gt_df, sub_df, args.mode)

    y_true = sol_df[ALL_LABELS].values.astype(int)
    y_scores = pred_df[ALL_LABELS].values.astype(float)
    y_pred = (y_scores >= args.threshold).astype(int)

    per_label_auc, auc_valid_mask = safe_auc_per_label(y_true, y_scores)
    weights = RSNA_WEIGHTS / RSNA_WEIGHTS.sum()
    official_weighted_auc = float(np.sum(per_label_auc * weights))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    samples_p, samples_r, samples_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="samples", zero_division=0
    )

    exact_match_ratio = float((y_true == y_pred).all(axis=1).mean())
    hamming_accuracy = float(1.0 - hamming_loss(y_true, y_pred))
    jaccard_samples = float(jaccard_score(y_true, y_pred, average="samples", zero_division=0))
    pr_auc_per_label = []
    for i in range(y_true.shape[1]):
        if np.unique(y_true[:, i]).size < 2:
            pr_auc_per_label.append(0.0)
        else:
            pr_auc_per_label.append(average_precision_score(y_true[:, i], y_scores[:, i]))
    pr_auc_per_label = np.array(pr_auc_per_label, dtype=float)

    acc_per_label = [accuracy_score(y_true[:, i], y_pred[:, i]) for i in range(y_true.shape[1])]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = os.path.splitext(os.path.basename(args.submission))[0]
    out_dir = os.path.join(args.output_dir, f"{run_name}_{args.mode}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    overall_df = pd.DataFrame(
        [
            {
                "submission": args.submission,
                "ground_truth": args.ground_truth,
                "mode": args.mode,
                "threshold": args.threshold,
                "rows_scored": len(pred_df),
                "unique_submission_uids": sub_df["SeriesInstanceUID"].nunique(),
                "unique_gt_uids": gt_df["SeriesInstanceUID"].nunique(),
                "overlap_uids": gt_df["SeriesInstanceUID"].isin(sub_df["SeriesInstanceUID"]).sum(),
                "official_weighted_auc": official_weighted_auc,
                "macro_precision": macro_p,
                "macro_recall": macro_r,
                "macro_f1": macro_f1,
                "micro_precision": micro_p,
                "micro_recall": micro_r,
                "micro_f1": micro_f1,
                "weighted_precision": weighted_p,
                "weighted_recall": weighted_r,
                "weighted_f1": weighted_f1,
                "samples_precision": samples_p,
                "samples_recall": samples_r,
                "samples_f1": samples_f1,
                "exact_match_ratio": exact_match_ratio,
                "hamming_accuracy": hamming_accuracy,
                "jaccard_samples": jaccard_samples,
            }
        ]
    )

    per_label_rows = []
    for i, label in enumerate(ALL_LABELS):
        true_col = y_true[:, i]
        pred_col = y_pred[:, i]
        tp = int(((true_col == 1) & (pred_col == 1)).sum())
        tn = int(((true_col == 0) & (pred_col == 0)).sum())
        fp = int(((true_col == 0) & (pred_col == 1)).sum())
        fn = int(((true_col == 1) & (pred_col == 0)).sum())
        per_label_rows.append(
            {
                "label": label,
                "support_pos": int(support[i]),
                "pred_pos": int(pred_col.sum()),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "precision": precision[i],
                "recall": recall[i],
                "f1": f1[i],
                "accuracy": acc_per_label[i],
                "roc_auc": per_label_auc[i],
                "pr_auc": pr_auc_per_label[i],
                "auc_valid": bool(auc_valid_mask[i]),
                "weight": RSNA_WEIGHTS[i],
            }
        )

    overall_path = os.path.join(out_dir, "overall_metrics.csv")
    per_label_path = os.path.join(out_dir, "per_label_metrics.csv")
    detailed_path = os.path.join(out_dir, "detailed_report.csv")
    overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_label_rows).to_csv(per_label_path, index=False, encoding="utf-8-sig")
    detailed_df = build_case_report(sol_df, pred_df)
    detailed_df.to_csv(detailed_path, index=False, encoding="utf-8-sig")

    plot_roc_curves(y_true, y_scores, ALL_LABELS, os.path.join(out_dir, "roc_curves.png"))
    plot_pr_curves(y_true, y_scores, ALL_LABELS, os.path.join(out_dir, "pr_curves.png"))

    print_section("FILES")
    paths_df = pd.DataFrame(
        [
            {"Item": "Submission", "Path": args.submission},
            {"Item": "Ground truth", "Path": args.ground_truth},
            {"Item": "Overall metrics CSV", "Path": overall_path},
            {"Item": "Per-label metrics CSV", "Path": per_label_path},
            {"Item": "Detailed report CSV", "Path": detailed_path},
            {"Item": "ROC curves", "Path": os.path.join(out_dir, "roc_curves.png")},
            {"Item": "PR curves", "Path": os.path.join(out_dir, "pr_curves.png")},
        ]
    )
    print_table(paths_df, float_fmt="{}")

    print_section("OVERALL METRICS")
    
    overall_display = pd.DataFrame(
        [
            {"Metric": "rows_scored", "Value": float(len(pred_df))},
            {"Metric": "official_weighted_auc", "Value": official_weighted_auc},
            {"Metric": "macro_precision", "Value": macro_p},
            {"Metric": "macro_recall", "Value": macro_r},
            {"Metric": "macro_f1", "Value": macro_f1},
            {"Metric": "micro_precision", "Value": micro_p},
            {"Metric": "micro_recall", "Value": micro_r},
            {"Metric": "micro_f1", "Value": micro_f1},
            {"Metric": "weighted_precision", "Value": weighted_p},
            {"Metric": "weighted_recall", "Value": weighted_r},
            {"Metric": "weighted_f1", "Value": weighted_f1},
            {"Metric": "samples_precision", "Value": samples_p},
            {"Metric": "samples_recall", "Value": samples_r},
            {"Metric": "samples_f1", "Value": samples_f1},
            {"Metric": "exact_match_ratio", "Value": exact_match_ratio},
            {"Metric": "hamming_accuracy", "Value": hamming_accuracy},
            {"Metric": "jaccard_samples", "Value": jaccard_samples},
        ]
    )
    
    print_table(overall_display)

    print_section("PER-LABEL SUMMARY")
    
    per_label_display = pd.DataFrame(per_label_rows)[
        ["label", "support_pos", "pred_pos", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    ]
    
    print_table(per_label_display)

    print_eval_style_summary(
        per_label_rows,
        official_weighted_auc,
        macro_p,
        macro_r,
        macro_f1,
        micro_p,
        micro_r,
        micro_f1,
    )

if __name__ == "__main__":
    main()

