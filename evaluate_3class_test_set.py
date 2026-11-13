import os
import sys
import argparse

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize

from src.models import TFFaultNet, TFFaultNetBackbone

DEFAULT_CONFIG = "configs/experiment_3class.json"


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)



LOCK_PATH = "results/FINAL_TEST_EVALUATION.lock"


def wilson_ci(successes, n, z=1.96):
    """95% Wilson score interval for a binomial proportion (assumes independent samples)."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="3-Class TF-FaultNet Evaluation")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"Path to experiment config JSON (default: {DEFAULT_CONFIG})")
    parser.add_argument("--allow-rerun", action="store_true",
                        help="Override the one-time lock. Only with explicit approval from the project owner.")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    batch_size = cfg["training"]["batch_size"]
    ckpt_path  = cfg["checkpoint_selection"]["checkpoint_path"]
    npz_path   = "results/heldout_test_set_3class.npz"

    print("=" * 80, flush=True)
    print("      EVALUATING 3-CLASS MODEL ON HELD-OUT TEST SET", flush=True)
    print(f"      Config: {args.config}", flush=True)
    print("=" * 80, flush=True)

    if os.path.exists(LOCK_PATH) and not args.allow_rerun:
        with open(LOCK_PATH, encoding="utf-8") as fh:
            print(fh.read(), flush=True)
        sys.exit(f"REFUSED: the held-out test set was already evaluated ({LOCK_PATH}). "
                 "It is scored once per locked model.")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"{npz_path} not found.")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} not found.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load Quarantined Test Set
    test_data = np.load(npz_path, allow_pickle=True)
    test_audio = test_data["audio"]               # (N_test, 44100)
    test_specs = test_data["spectrogram_images"]  # (N_test, 1, 64, 173)
    test_labels = test_data["labels"]             # (N_test,)
    test_filenames = test_data["filenames"]       # (N_test,)
    test_folders = test_data["original_folders"]  # (N_test,)
    class_names = test_data["class_names"].tolist()

    n_samples = len(test_labels)
    n_classes = len(class_names)
    print(f"Loaded Held-Out Test Set: {n_samples} samples across {n_classes} Macro Classes:", flush=True)
    for c_idx, c_name in enumerate(class_names):
        count = np.sum(test_labels == c_idx)
        print(f"  • Class {c_idx} [{c_name}]: {count} samples ({count/n_samples*100:.1f}%)", flush=True)

    # 2. Load Models (num_classes inferred from the data, not hardcoded)
    model_full = TFFaultNet(num_classes=n_classes).to(device)
    model_full.load_state_dict(torch.load(ckpt_path, map_location=device))
    model_full.eval()

    model_backbone = TFFaultNetBackbone(model_full).to(device)
    model_backbone.eval()

    # 3. INFERENCE MODE A: RAW AUDIO WAVEFORM INFERENCE
    print("\n" + "-" * 80, flush=True)
    print("INFERENCE MODE A: Full End-to-End Pipeline (Raw Audio Waveforms)", flush=True)
    print("-" * 80, flush=True)

    audio_preds = []
    audio_probs = []
    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_audio = torch.from_numpy(test_audio[i:i+batch_size]).to(device)
            logits = model_full(b_audio)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            audio_preds.extend(preds)
            audio_probs.extend(probs)

    t_audio = (time.perf_counter() - t_start) * 1000.0 / n_samples
    audio_preds = np.array(audio_preds)
    audio_probs = np.array(audio_probs)

    acc_audio = accuracy_score(test_labels, audio_preds)
    p_audio, r_audio, f1_audio, _ = precision_recall_fscore_support(test_labels, audio_preds, average='macro')
    print(f"  ✓ Waveform Inference Complete: {t_audio:.2f} ms / sample", flush=True)
    print(f"  • 3-Class Test Accuracy: {acc_audio * 100:.2f}% | Macro F1: {f1_audio * 100:.2f}%", flush=True)

    # 4. INFERENCE MODE B: 2D SPECTROGRAM IMAGE INFERENCE SEPARATELY
    print("\n" + "-" * 80, flush=True)
    print("INFERENCE MODE B: 2D Spectrogram Image Backbone Inference (Separately)", flush=True)
    print("-" * 80, flush=True)

    spec_preds = []
    spec_probs = []
    t_start = time.perf_counter()

    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            b_spec = torch.from_numpy(test_specs[i:i+batch_size]).to(device)
            logits = model_backbone(b_spec)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            spec_preds.extend(preds)
            spec_probs.extend(probs)

    t_spec = (time.perf_counter() - t_start) * 1000.0 / n_samples
    spec_preds = np.array(spec_preds)
    spec_probs = np.array(spec_probs)

    acc_spec = accuracy_score(test_labels, spec_preds)
    p_spec, r_spec, f1_spec, _ = precision_recall_fscore_support(test_labels, spec_preds, average='macro')
    print(f"  ✓ Spectrogram Image Inference Complete: {t_spec:.2f} ms / sample", flush=True)
    print(f"  • 3-Class Test Accuracy: {acc_spec * 100:.2f}% | Macro F1: {f1_spec * 100:.2f}%", flush=True)

    matches = np.sum(audio_preds == spec_preds)
    print(f"  • Dual Pipeline Consistency Check: {matches}/{n_samples} ({matches/n_samples*100:.2f}% identical outputs)", flush=True)

    # 5. ROC & PRECISION-RECALL CURVE COMPUTATION (3 CLASSES)
    Y_test = label_binarize(test_labels, classes=[0, 1, 2])
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    precision = dict()
    recall = dict()
    pr_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], audio_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], audio_probs[:, i])
        pr_auc[i] = average_precision_score(Y_test[:, i], audio_probs[:, i])

    # Micro & Macro averages
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), audio_probs.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), audio_probs.ravel())
    pr_auc["micro"] = average_precision_score(Y_test, audio_probs, average="micro")

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    pr_auc["macro"] = np.mean([pr_auc[i] for i in range(n_classes)])

    # 6. COMPREHENSIVE PERFORMANCE TABLE
    print("\n" + "=" * 90, flush=True)
    print(f"FINAL 3-CLASS PERFORMANCE EVALUATION ON HELD-OUT TEST SET (N={n_samples})", flush=True)
    print("=" * 90, flush=True)

    prec_arr, rec_arr, f1_arr, supp_arr = precision_recall_fscore_support(test_labels, audio_preds, average=None)

    table_rows = []
    print(f"{'Idx':<5} | {'Macro Health Condition':<35} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<9} | {'Support'}", flush=True)
    print("-" * 96, flush=True)

    for i in range(n_classes):
        n_true_i = int(np.sum(test_labels == i))
        n_hit_i = int(np.sum((test_labels == i) & (audio_preds == i)))
        row = {
            "class_index": i,
            "class_name": class_names[i],
            "precision": float(prec_arr[i]),
            "recall": float(rec_arr[i]),
            "recall_wilson95": wilson_ci(n_hit_i, n_true_i),
            "f1_score": float(f1_arr[i]),
            "roc_auc": float(roc_auc[i]),
            "pr_auc": float(pr_auc[i]),
            "support": int(supp_arr[i])
        }
        table_rows.append(row)
        print(f"{i:<5} | {class_names[i]:<35} | {prec_arr[i]*100:>8.2f}%  | {rec_arr[i]*100:>8.2f}%  | {f1_arr[i]*100:>8.2f}%  | {roc_auc[i]:>8.5f} | {supp_arr[i]:>6}", flush=True)

    print("-" * 96, flush=True)
    print(f"{'ALL':<5} | {'Macro Average (3 Macro Classes)':<35} | {p_audio*100:>8.2f}%  | {r_audio*100:>8.2f}%  | {f1_audio*100:>8.2f}%  | {roc_auc['macro']:>8.5f} | {n_samples:>6}", flush=True)
    print(f"{'ALL':<5} | {'Top-1 3-Class Test Accuracy':<35} | {acc_audio*100:>8.2f}% ({np.sum(audio_preds == test_labels)}/{n_samples} Correct)", flush=True)
    print("=" * 96, flush=True)

    # 7. GENERATE 3-CLASS CONFUSION MATRIX
    cm = confusion_matrix(test_labels, audio_preds)
    cm_path = "results/visualizations/test_set_confusion_matrix_3class.png"
    plot_3class_confusion_matrix(cm, class_names, acc_audio, f1_audio, cm_path)

    # 8. GENERATE 3-CLASS ROC & PR CURVES
    roc_pr_path = "results/visualizations/test_set_roc_pr_curves_3class.png"
    plot_3class_roc_pr(fpr, tpr, roc_auc, precision, recall, pr_auc, class_names, roc_pr_path, n_samples)

    # 9. GENERATE SEPARATE SPECTROGRAM INFERENCE DEMO
    demo_path = "results/visualizations/test_sample_spectrograms_3class.png"
    plot_3class_sample_spectrograms(test_specs, test_labels, audio_preds, audio_probs, class_names, demo_path)

    # 10. SAVE METRICS AND REPORTS
    df_metrics = pd.DataFrame(table_rows)
    df_metrics.to_csv("results/test_roc_auc_metrics_3class.csv", index=False)

    # Per-sample predictions (for error analysis without re-running the test set)
    pred_rows = pd.DataFrame({
        "filename": [os.path.basename(str(f).replace("\\", "/")) for f in test_filenames],
        "original_folder": test_folders,
        "true_label": test_labels,
        "true_name": [class_names[i] for i in test_labels],
        "pred_label": audio_preds,
        "pred_name": [class_names[i] for i in audio_preds],
        "correct": audio_preds == test_labels,
        **{f"prob_{i}_{class_names[i]}": audio_probs[:, i] for i in range(n_classes)},
        "pred_label_spectrogram_path": spec_preds,
    })
    pred_rows.to_csv("results/test_predictions_3class.csv", index=False)
    errors = pred_rows[~pred_rows["correct"]]
    print(f">>> Per-sample predictions saved to: results/test_predictions_3class.csv "
          f"({len(errors)} errors)", flush=True)
    if len(errors):
        print(errors[["filename", "original_folder", "true_name", "pred_name"]].to_string(index=False), flush=True)

    n_correct = int(np.sum(audio_preds == test_labels))
    report_3class = {
        "dataset": "Base_de_Dados 3-Class Macro Taxonomy",
        "evaluation_partition": "Held-out test split (file-level, stratified; session independence not verified)",
        "sample_count": n_samples,
        "classes": class_names,
        "confusion_matrix_rows_true_cols_pred": cm.tolist(),
        "n_correct": n_correct,
        "metrics": {
            "test_accuracy": float(acc_audio),
            "test_accuracy_wilson95": wilson_ci(n_correct, n_samples),
            "macro_precision": float(p_audio),
            "macro_recall": float(r_audio),
            "macro_f1_score": float(f1_audio),
            "macro_roc_auc": float(roc_auc["macro"]),
            "micro_roc_auc": float(roc_auc["micro"]),
            "macro_pr_auc": float(pr_auc["macro"]),
            "micro_pr_auc": float(pr_auc["micro"])
        },
        "per_class": table_rows,
        "exported_models": {
            "torchscript_jit": "results/checkpoints/best_tf_faultnet_3class_jit.pt",
            "onnx_backbone": "results/checkpoints/best_tf_faultnet_3class_backbone.onnx",
            "pytorch_weights": ckpt_path
        },
        "provenance": {
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_sha256": sha256_file(ckpt_path),
            "config": args.config,
            "training_config": cfg["training"],
            "seed": cfg["seed"],
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__,
            "git_commit": os.popen("git rev-parse HEAD").read().strip() or None,
            "spectrogram_path_agreement": f"{matches}/{n_samples}",
            "waveform_ms_per_sample_batched": round(t_audio, 3)
        }
    }

    report_path = "results/heldout_test_evaluation_report_3class.json"
    with open(report_path, "w") as f:
        json.dump(report_3class, f, indent=2)

    with open(LOCK_PATH, "w", encoding="utf-8") as fh:
        json.dump({"evaluated_at": report_3class["provenance"]["evaluated_at"],
                   "checkpoint_sha256": report_3class["provenance"]["checkpoint_sha256"],
                   "test_accuracy": float(acc_audio), "macro_f1": float(f1_audio),
                   "git_commit": report_3class["provenance"]["git_commit"]}, fh, indent=2)

    print(f"\n>>> Full 3-Class JSON report saved to: {report_path}", flush=True)
    print(f">>> Test-set lock written: {LOCK_PATH}", flush=True)
    print(f">>> Full 3-Class CSV metrics saved to: results/test_roc_auc_metrics_3class.csv", flush=True)


def plot_3class_confusion_matrix(cm, class_names, acc, f1, output_path, n_samples=None):
    if n_samples is None:
        n_samples = int(cm.sum())
    plt.figure(figsize=(9, 7.5), dpi=300)
    # Shorten class labels for display
    display_names = ["Healthy Baseline", "Continuous Friction", "Impulsive Shocks"]
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', cbar=True,
        xticklabels=display_names, yticklabels=display_names,
        linewidths=1.0, linecolor='white', annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(f"3-Class Held-Out Test Set Confusion Matrix (N={n_samples})\nAccuracy: {acc*100:.2f}% | Macro F1: {f1*100:.2f}%", 
              fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Predicted Condition", fontsize=11, fontweight='bold')
    plt.ylabel("Ground Truth Condition", fontsize=11, fontweight='bold')
    plt.xticks(fontsize=10.5)
    plt.yticks(fontsize=10.5, rotation=0)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f">>> 3-Class confusion matrix saved to: {output_path}", flush=True)


def plot_3class_roc_pr(fpr, tpr, roc_auc, precision, recall, pr_auc, class_names, output_path, n_samples=322):
    from matplotlib.patches import FancyBboxPatch
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(18, 7.5), dpi=300)
    display_names = ["Healthy Baseline", "Continuous Friction & Wear", "Impulsive Shocks & Structural"]
    short_names   = ["Healthy", "Friction/Wear", "Impulsive/Struct."]
    class_colors  = ["#2ca02c", "#1f77b4", "#d62728"]
    class_markers = ["o", "s", "D"]

    # ═══════════════ PANEL 1: ROC CURVES ═══════════════
    # Shaded AUC fill for each class
    for i in range(3):
        ax_roc.fill_between(fpr[i], 0, tpr[i], alpha=0.08, color=class_colors[i])

    # Plot curves
    ax_roc.plot(fpr["micro"], tpr["micro"], label=f"Micro-Avg (AUC = {roc_auc['micro']:.5f})",
                color="#FF00FF", linestyle=":", linewidth=2.8, zorder=5)
    ax_roc.plot(fpr["macro"], tpr["macro"], label=f"Macro-Avg (AUC = {roc_auc['macro']:.5f})",
                color="navy", linestyle="--", linewidth=2.5, zorder=5)

    for i in range(3):
        ax_roc.plot(fpr[i], tpr[i], color=class_colors[i], linewidth=2.2,
                    label=f"Class {i}: {short_names[i]} (AUC = {roc_auc[i]:.5f})", zorder=4)
        # No "optimal threshold" marker: thresholds must never be chosen on test data
        # (predictions use argmax).

    ax_roc.plot([0, 1], [0, 1], "k--", linewidth=1.0, alpha=0.4, label="Chance (AUC = 0.50)")
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.05])
    ax_roc.set_xlabel("False Positive Rate (1 − Specificity)", fontsize=11, fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, fontweight="bold")
    ax_roc.set_title(f"(a) 3-Class ROC Curves\nMacro-AUC: {roc_auc['macro']:.5f} | Micro-AUC: {roc_auc['micro']:.5f}",
                     fontsize=12, fontweight="bold", pad=14)
    ax_roc.grid(True, linestyle="--", alpha=0.35)
    ax_roc.legend(loc="lower right", fontsize=8.8, frameon=True, framealpha=0.92, edgecolor='gray')

    # ---- ZOOMED INSET for ROC (top-left high-performance corner) ----
    axins = inset_axes(ax_roc, width="45%", height="45%", loc='center',
                       bbox_to_anchor=(-0.05, 0.02, 1, 1), bbox_transform=ax_roc.transAxes)
    for i in range(3):
        axins.fill_between(fpr[i], 0, tpr[i], alpha=0.12, color=class_colors[i])
        axins.plot(fpr[i], tpr[i], color=class_colors[i], linewidth=2.0)

    axins.plot(fpr["micro"], tpr["micro"], color="#FF00FF", linestyle=":", linewidth=2.0)
    axins.plot(fpr["macro"], tpr["macro"], color="navy", linestyle="--", linewidth=2.0)
    axins.set_xlim([-0.005, 0.08])
    axins.set_ylim([0.92, 1.005])
    axins.set_title("Zoomed: FPR ∈ [0, 0.08]", fontsize=8, fontweight="bold", pad=3)
    axins.grid(True, linestyle=":", alpha=0.4)
    axins.tick_params(labelsize=7)
    mark_inset(ax_roc, axins, loc1=1, loc2=3, fc="none", ec="0.5", linestyle="--", linewidth=0.8)

    # ═══════════════ PANEL 2: PR CURVES ═══════════════
    # Macro-average PR curve: mean of each class's *interpolated* precision,
    # p_interp(r) = max precision at any recall >= r (the standard PR envelope).
    # Linear interpolation over raw points is wrong here: at recall = 1.0 sklearn also returns
    # low-precision points, which dragged the average below every class curve.
    all_recall_pts = np.linspace(0, 1, 500)
    mean_precision_arr = np.zeros_like(all_recall_pts)
    for i in range(3):
        order = np.argsort(recall[i], kind="stable")
        rec_sorted, prec_sorted = recall[i][order], precision[i][order]
        envelope = np.maximum.accumulate(prec_sorted[::-1])[::-1]
        idx = np.clip(np.searchsorted(rec_sorted, all_recall_pts, side="left"), 0, len(rec_sorted) - 1)
        mean_precision_arr += envelope[idx]
    mean_precision_arr /= 3

    # Shaded fills for each class
    for i in range(3):
        sorted_idx = np.argsort(recall[i])
        ax_pr.fill_between(recall[i][sorted_idx], 0, precision[i][sorted_idx], alpha=0.08, color=class_colors[i], step='post')

    ax_pr.plot(recall["micro"], precision["micro"], label=f"Micro-Avg (AP = {pr_auc['micro']:.5f})",
               color="#FF00FF", linestyle=":", linewidth=2.8, zorder=5)
    ax_pr.plot(all_recall_pts, mean_precision_arr, label=f"Macro-Avg (AP = {pr_auc['macro']:.5f})",
               color="navy", linestyle="--", linewidth=2.5, zorder=5)

    for i in range(3):
        ax_pr.plot(recall[i], precision[i], color=class_colors[i], linewidth=2.2,
                   label=f"Class {i}: {short_names[i]} (AP = {pr_auc[i]:.5f})", zorder=4)

    # Baseline prevalence lines for each class
    n_total = sum(np.sum(fpr[i] >= 0) for i in range(3))  # rough proxy, not used for prevalence
    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.05])
    ax_pr.set_xlabel("Recall (True Positive Rate)", fontsize=11, fontweight="bold")
    ax_pr.set_ylabel("Precision (Positive Predictive Value)", fontsize=11, fontweight="bold")
    ax_pr.set_title(f"(b) 3-Class Precision-Recall (PR) Curves\nMacro-AP: {pr_auc['macro']:.5f} | Micro-AP: {pr_auc['micro']:.5f}",
                    fontsize=12, fontweight="bold", pad=14)
    ax_pr.grid(True, linestyle="--", alpha=0.35)
    ax_pr.legend(loc="lower left", fontsize=8.8, frameon=True, framealpha=0.92, edgecolor='gray')

    # ---- ZOOMED INSET for PR (high-performance corner) ----
    axins_pr = inset_axes(ax_pr, width="45%", height="45%", loc='center',
                          bbox_to_anchor=(0.0, 0.02, 1, 1), bbox_transform=ax_pr.transAxes)
    for i in range(3):
        axins_pr.plot(recall[i], precision[i], color=class_colors[i], linewidth=2.0)
    axins_pr.plot(recall["micro"], precision["micro"], color="#FF00FF", linestyle=":", linewidth=2.0)
    axins_pr.plot(all_recall_pts, mean_precision_arr, color="navy", linestyle="--", linewidth=2.0)
    axins_pr.set_xlim([0.92, 1.005])
    axins_pr.set_ylim([0.92, 1.005])
    axins_pr.set_title("Zoomed: R & P ∈ [0.92, 1.0]", fontsize=8, fontweight="bold", pad=3)
    axins_pr.grid(True, linestyle=":", alpha=0.4)
    axins_pr.tick_params(labelsize=7)
    mark_inset(ax_pr, axins_pr, loc1=1, loc2=3, fc="none", ec="0.5", linestyle="--", linewidth=0.8)

    plt.suptitle(f"TF-FaultNet 3-Class Macro Taxonomy · Held-Out Test Set (N={n_samples})",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.subplots_adjust(top=0.84, bottom=0.10, left=0.06, right=0.97, wspace=0.18)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f">>> 3-Class ROC and PR curves saved to: {output_path}", flush=True)


def plot_3class_sample_spectrograms(specs, true_labels, preds, probs, class_names, output_path):
    """
    Renders sample 2D spectrogram image inferences for the 3 classes.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=300)
    display_names = ["Healthy Baseline", "Continuous Friction & Wear", "Impulsive Shocks & Structural"]

    # Pick 1 representative sample from each class
    for c in range(3):
        ax = axes[c]
        c_idxs = np.where(true_labels == c)[0]
        sample_idx = c_idxs[0]
        spec_img = specs[sample_idx, 0] # (64, 173)
        confidence = probs[sample_idx, preds[sample_idx]] * 100

        ax.imshow(spec_img, aspect='auto', origin='lower', cmap='magma', extent=[0, 1.0, 0, 64])
        is_correct = preds[sample_idx] == c
        ax.set_title(f"Class {c}: {display_names[c]}\nPred: {display_names[preds[sample_idx]]} ({confidence:.1f}%)\n"
                     f"[{'CORRECT' if is_correct else 'WRONG'}]",
                     fontsize=10.5, fontweight="bold", color="#2ca02c" if is_correct else "#d62728", pad=8)
        ax.set_xlabel("Time (0 to 1.0s)", fontsize=9.5, fontweight="bold")
        if c == 0:
            ax.set_ylabel("Mel Frequency Bins (0-64)", fontsize=9.5, fontweight="bold")

    plt.suptitle("Held-Out Test Set 2D Spectrogram Image Inference Demo (Separate Image Input)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f">>> 3-Class sample spectrogram image inference demo saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
