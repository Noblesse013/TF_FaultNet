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
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import preload_all_audio, MemoryFaultDataset
from src.audio_features import LogMelSpectrogramExtractor
from src.models import TFFaultNet, TFFaultNetBackbone
from src.ablation_models import ABLATION_VARIANTS, build_model
from src.split_manifest import load_split_manifest, rows_for_split
from src.train_utils import train_epoch, eval_epoch, recalibrate_batchnorm, batchnorm_buffers


DEFAULT_CONFIG = "configs/experiment_3class.json"


def load_config(path: str) -> dict:
    """Load and return the experiment config JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Config not found: {path}. "
            "Pass --config <path> or ensure configs/experiment_3class.json exists."
        )
    with open(path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return cfg

# Define the 3 Macro Classes and the exact mapping from the 12 original subfolders
MACRO_CLASSES = [
    "Healthy_Baseline",
    "Continuous_Friction_and_Wear",
    "Impulsive_Shocks_and_Structural"
]

FOLDER_TO_3CLASS = {
    # Class 0: Healthy Baseline (Normal)
    "Normal": 0,
    
    # Class 1: Continuous Friction & Wear (alternator-belt slippage / diffuse material loss;
    #          suffix _P1 / _P1P4 = plus misfire in cylinder P1 / cylinders P1+P4)
    "Escorregamento": 1,
    "Escorregamento_P1": 1,
    "Escorregamento_P1P4": 1,
    "Perda_material": 1,
    "Perda_material_P1": 1,
    "Perda_material_P1P4": 1,
    
    # Class 2: Impulsive Shocks & Structural (concentrated belt material loss, any misfire state;
    #          Sem_P1 / Sem_P1P4 = misfire only, cylinder P1 / cylinders P1+P4, no belt fault)
    "Perda_concentrada": 2,
    "Perda_concentrada_P1": 2,
    "Perda_concentrada_P1P4": 2,
    "Sem_P1": 2,
    "Sem_P1P4": 2
}

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility over speed: same config + code + seed -> same run on the same GPU/driver.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gather_3class_dataset(data_dir="Base_de_Dados"):
    subdirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    filepaths = []
    labels_3class = []
    original_classes = []
    
    for d in subdirs:
        if d not in FOLDER_TO_3CLASS:
            continue
        c3 = FOLDER_TO_3CLASS[d]
        wavs = sorted(glob.glob(os.path.join(data_dir, d, "*.wav")))
        for w in wavs:
            filepaths.append(w)
            labels_3class.append(c3)
            original_classes.append(d)
            
    return np.array(filepaths), np.array(labels_3class), np.array(original_classes), MACRO_CLASSES



def main():
    parser = argparse.ArgumentParser(description="3-Class TF-FaultNet Training Pipeline")
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG,
        help=f"Path to experiment config JSON (default: {DEFAULT_CONFIG})"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the config seed (used for multi-seed validation sweeps)."
    )
    parser.add_argument(
        "--bn-recal", choices=["on", "off"], default=None,
        help="Override training.bn_recalibration: re-estimate BatchNorm statistics on clean "
             "training data before each validation pass."
    )
    parser.add_argument(
        "--run-dir", default=None,
        help="Development/trial mode: write checkpoint, history and plot into this folder only. "
             "The held-out test set is NOT touched and models are NOT exported."
    )
    parser.add_argument(
        "--variant", default="M0", choices=list(ABLATION_VARIANTS),
        help="Model variant (ablation study). M0 = the real TFFaultNet. A1-A5 require --run-dir."
    )
    args = parser.parse_args()
    if args.variant != "M0" and args.run_dir is None:
        parser.error("ablation variants A1-A5 can only be trained in trial mode (--run-dir)")

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    sched_cfg = train_cfg["scheduler_params"]

    # Pull every hyperparameter from config — no magic numbers below this line.
    seed          = args.seed if args.seed is not None else cfg["seed"]
    num_classes   = model_cfg["num_classes"]
    target_len    = model_cfg["input_shape"][0]
    n_mels        = model_cfg["spectrogram"]["n_mels"]
    sr            = model_cfg["spectrogram"]["sr"]
    lr            = train_cfg["learning_rate"]
    weight_decay  = train_cfg["weight_decay"]
    batch_size    = train_cfg["batch_size"]
    epochs        = train_cfg["epochs"]
    label_smooth  = train_cfg["label_smoothing"]
    warmup_epochs = sched_cfg["warmup_epochs"]
    eta_min       = sched_cfg["eta_min"]
    max_grad_norm = train_cfg["max_grad_norm"]
    bn_recal      = train_cfg.get("bn_recalibration", False)
    if args.bn_recal is not None:
        bn_recal = args.bn_recal == "on"
    manifest_path = cfg["data_split"]["manifest"]
    ckpt_path     = cfg["checkpoint_selection"]["checkpoint_path"]

    trial_mode = args.run_dir is not None
    if trial_mode:
        os.makedirs(args.run_dir, exist_ok=True)
        ckpt_path = os.path.join(args.run_dir, "best_checkpoint.pt")
        history_path = os.path.join(args.run_dir, "training_history.csv")
        fig_path = os.path.join(args.run_dir, "train_val_loss_convergence.png")
    else:
        history_path = "results/training_convergence_history_3class.csv"
        fig_path = "results/visualizations/train_val_loss_convergence_3class.png"

    print("=" * 80, flush=True)
    print("       3-CLASS MACRO TAXONOMY TRAINING PIPELINE", flush=True)
    print(f"       Config: {args.config} | Seed: {seed} | Variant: {args.variant} | "
          f"Mode: {'TRIAL (' + args.run_dir + ')' if trial_mode else 'FINAL'}", flush=True)
    print("=" * 80, flush=True)

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Hardware Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    os.makedirs("results/checkpoints", exist_ok=True)
    os.makedirs("results/visualizations", exist_ok=True)

    # 1. Load the frozen manifest. Never recreate the split during training.
    manifest_rows = load_split_manifest(manifest_path)
    split_parts = {name: rows_for_split(manifest_rows, name) for name in ("train", "val", "test")}
    ordered_rows = [
        row for split in ("train", "val", "test")
        for row in manifest_rows if row["split_multiclass_3class"] == split
    ]
    filepaths = np.array([row["filepath"] for row in ordered_rows])
    labels = np.array([row["macro_class_index"] for row in ordered_rows], dtype=np.int64)
    original_folders = np.array([row["original_subfolder"] for row in ordered_rows])
    class_names = MACRO_CLASSES
    split_sizes = {name: len(split_parts[name][0]) for name in split_parts}
    train_idx = np.arange(0, split_sizes["train"])
    val_idx = np.arange(split_sizes["train"], split_sizes["train"] + split_sizes["val"])
    test_idx = np.arange(split_sizes["train"] + split_sizes["val"], len(ordered_rows))
    total_samples = len(filepaths)
    num_classes = len(class_names)
    print(f"Loaded {total_samples} audio samples mapped into {num_classes} Macro Classes:", flush=True)
    for c_idx, c_name in enumerate(class_names):
        count = np.sum(labels == c_idx)
        print(f"  • Class {c_idx} [{c_name}]: {count} samples ({count/total_samples*100:.1f}%)", flush=True)

    print("\nPreloading audio waveforms into memory...", flush=True)
    audio_matrix = preload_all_audio(filepaths, target_len=target_len)

    # 2. Verify the frozen train/validation/test partitions.
    set_train = set(train_idx)
    set_val = set(val_idx)
    set_test = set(test_idx)

    assert len(set_train.intersection(set_val)) == 0, "LEAKAGE DETECTED: Train and Val overlap!"
    assert len(set_train.intersection(set_test)) == 0, "LEAKAGE DETECTED: Train and Test overlap!"
    assert len(set_val.intersection(set_test)) == 0, "LEAKAGE DETECTED: Val and Test overlap!"
    assert len(set_train) + len(set_val) + len(set_test) == total_samples, "Sample count mismatch!"

    print("\n" + "-" * 80, flush=True)
    print("3-CLASS DATASET SPLIT AUDIT (ZERO DATA LEAKAGE):", flush=True)
    print(f"  • Training Set:    {len(train_idx)} samples ({len(train_idx)/total_samples*100:.1f}%) -> Used for backprop", flush=True)
    print(f"  • Validation Set:  {len(val_idx)} samples ({len(val_idx)/total_samples*100:.1f}%) -> Used for convergence & checkpointing", flush=True)
    print(f"  • Held-Out Test:   {len(test_idx)} samples ({len(test_idx)/total_samples*100:.1f}%) -> Held out (not used for training or tuning)", flush=True)
    print("-" * 80, flush=True)

    if not trial_mode:
        # Extract 2D Log-Mel spectrogram images for the test set
        spec_extractor = LogMelSpectrogramExtractor(sr=sr, n_mels=n_mels).to(device)
        test_audio_tensor = torch.from_numpy(audio_matrix[test_idx]).to(device)
        with torch.no_grad():
            test_specs = spec_extractor(test_audio_tensor).cpu().numpy() # (N_test, 1, 64, 173)

        # Save held-out 3-class test set (relative paths only)
        np.savez_compressed(
            "results/heldout_test_set_3class.npz",
            audio=audio_matrix[test_idx],
            spectrogram_images=test_specs,
            labels=labels[test_idx],
            filenames=np.array([os.path.relpath(p) for p in filepaths[test_idx]]),
            original_folders=original_folders[test_idx],
            class_names=np.array(class_names)
        )
        print(f"Held-out 3-class test set saved to: results/heldout_test_set_3class.npz ({len(test_idx)} samples)", flush=True)
    else:
        print("Trial mode: held-out test set not touched.", flush=True)

    # 3. DataLoaders
    train_ds = MemoryFaultDataset(audio_matrix[train_idx], labels[train_idx], augment=True)
    val_ds = MemoryFaultDataset(audio_matrix[val_idx], labels[val_idx], augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True)
    if bn_recal:
        # Separate generators: these extra passes must not consume the global RNG, so the
        # training trajectory (shuffling, augmentation) stays identical to a run without them.
        recal_loader = DataLoader(MemoryFaultDataset(audio_matrix[train_idx], labels[train_idx], augment=False),
                                  batch_size=batch_size, shuffle=False, pin_memory=True,
                                  generator=torch.Generator().manual_seed(seed))
        val_loader_recal = DataLoader(val_ds, batch_size=batch_size, shuffle=False, pin_memory=True,
                                      generator=torch.Generator().manual_seed(seed))

    # 4. Model — all hyperparameters sourced from config.
    model = build_model(args.variant, num_classes=num_classes).to(device)  # M0 -> TFFaultNet
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model variant {args.variant}: {ABLATION_VARIANTS[args.variant][0]} ({n_params:,} parameters)", flush=True)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Linear warmup for `warmup_epochs`, then cosine decay to eta_min; stepped once per batch.
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    min_factor = eta_min / lr

    def lr_factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    print(f"LR schedule: warmup {warmup_epochs} epochs ({warmup_steps} steps) -> cosine to {eta_min}; "
          f"grad-norm clip {max_grad_norm}; BatchNorm recalibration {'ON' if bn_recal else 'OFF'}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"TRAINING 3-CLASS TF-FAULTNET CONVERGENCE ({epochs} EPOCHS)", flush=True)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Val Loss':<10} | {'Train Acc':<10} | {'Val Acc':<10} | {'Val F1':<10} | {'Status'}", flush=True)
    print("=" * 80, flush=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_f1": [],
        "lr": [],
        "grad_norm_mean": [],
        "grad_norm_max": [],
        # Validation with the BatchNorm running statistics as left by training (= the old
        # behaviour). Equal to val_loss/val_acc/val_f1 when BatchNorm recalibration is off.
        "val_loss_raw": [],
        "val_acc_raw": [],
        "val_f1_raw": []
    }

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    clean_ckpt_path = ckpt_path

    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']  # LR at the start of this epoch
        grad_stats = {}
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device,
                                            max_grad_norm=max_grad_norm, scheduler=scheduler,
                                            stats=grad_stats)
        val_loss_raw, val_acc_raw, val_f1_raw, _, _ = eval_epoch(model, val_loader, criterion, device)
        raw_bn_state = None
        if bn_recal:
            # Keep the training-time BatchNorm statistics so training continues exactly as without
            # recalibration; the recalibrated statistics are only used for validation/checkpoint.
            raw_bn_state = batchnorm_buffers(model)
            recalibrate_batchnorm(model, recal_loader, device)
            val_loss, val_acc, val_f1, _, _ = eval_epoch(model, val_loader_recal, criterion, device)
        else:
            val_loss, val_acc, val_f1 = val_loss_raw, val_acc_raw, val_f1_raw
        history["val_loss_raw"].append(val_loss_raw)
        history["val_acc_raw"].append(val_acc_raw)
        history["val_f1_raw"].append(val_f1_raw)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)
        history["grad_norm_mean"].append(grad_stats["grad_norm_mean"])
        history["grad_norm_max"].append(grad_stats["grad_norm_max"])

        status_str = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save(model.state_dict(), clean_ckpt_path)  # includes recalibrated BN stats if on
            status_str = f"★ Saved (Val Loss: {val_loss:.4f})"

        if raw_bn_state is not None:
            model.load_state_dict(raw_bn_state, strict=False)

        print(f"{epoch:<6} | {train_loss:<11.4f} | {val_loss:<10.4f} | {train_acc*100:>8.2f}%  | {val_acc*100:>8.2f}%  | {val_f1*100:>8.2f}%  | {status_str}", flush=True)

    print("=" * 80, flush=True)
    print(f"3-Class Convergence Reached! Optimal Epoch {best_epoch}: Val Loss = {best_val_loss:.4f}, Val Acc = {best_val_acc*100:.2f}%", flush=True)

    # Save convergence history
    df_history = pd.DataFrame(history)
    df_history.to_csv(history_path, index=False)

    # 5. PLOT SINGLE-FIGURE CONVERGENCE (TRAIN AND VAL LOSS IN ONE PLOT)
    plot_convergence_curves_3class(df_history, best_epoch, fig_path)

    if trial_mode:
        summary = {"seed": seed, "variant": args.variant, "n_params": n_params,
                   "best_epoch": best_epoch, "best_val_loss": best_val_loss,
                   "best_val_acc": best_val_acc, "best_val_f1": best_val_f1,
                   "learning_rate": lr, "warmup_epochs": warmup_epochs,
                   "max_grad_norm": max_grad_norm, "bn_recalibration": bn_recal,
                   "config": args.config}
        with open(os.path.join(args.run_dir, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\n>>> Trial run finished. Outputs in {args.run_dir} (no export, test untouched).", flush=True)
        return

    # 6. EXPORT 3-CLASS PRODUCTION MODELS (TORCHSCRIPT JIT & ONNX)
    print("\n" + "=" * 80, flush=True)
    print("EXPORTING 3-CLASS MODELS TO PRODUCTION RUNTIMES", flush=True)
    print("=" * 80, flush=True)

    cpu_model = TFFaultNet(num_classes=num_classes)
    cpu_model.load_state_dict(torch.load(clean_ckpt_path, map_location="cpu"))
    cpu_model.eval()

    # A. TorchScript JIT Export (Full End-to-End Audio Waveform)
    jit_path = "results/checkpoints/best_tf_faultnet_3class_jit.pt"
    dummy_audio = torch.randn(1, 44100)
    try:
        traced_jit = torch.jit.trace(cpu_model, dummy_audio)
        traced_jit.save(jit_path)
        jit_size = os.path.getsize(jit_path) / (1024 * 1024)
        print(f"  ✓ Exported Full TorchScript JIT Model:  {jit_path} ({jit_size:.2f} MB)", flush=True)
    except Exception as e:
        print(f"  ✗ TorchScript JIT export failed: {e}", flush=True)

    # B. ONNX 2D Backbone Export (Spectrogram Images)
    onnx_path = "results/checkpoints/best_tf_faultnet_3class_backbone.onnx"
    backbone = TFFaultNetBackbone(cpu_model).eval()
    dummy_spec = torch.randn(1, 1, 64, 173)
    try:
        torch.onnx.export(
            backbone,
            dummy_spec,
            onnx_path,
            export_params=True,
            opset_version=14,
            dynamo=False,
            do_constant_folding=True,
            input_names=['spectrogram_image'],
            output_names=['class_logits']
        )
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ✓ Exported 2D Backbone ONNX Model:      {onnx_path} ({onnx_size:.2f} MB)", flush=True)
    except Exception as e:
        print(f"  ✗ ONNX export failed: {e}", flush=True)

    print("\n>>> 3-Class training and model export successfully finished!", flush=True)


def plot_convergence_curves_3class(df, best_epoch, output_path):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    epochs = df["epoch"].values
    train_loss = df["train_loss"].values
    val_loss = df["val_loss"].values
    train_acc = df["train_acc"].values * 100
    val_acc = df["val_acc"].values * 100

    # ---------------- PANEL 1: LOSS (BOTH IN ONE FIG) ----------------
    ax_loss.plot(epochs, train_loss, label="Training Loss", color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
    ax_loss.plot(epochs, val_loss, label="Validation Loss", color="#d62728", linewidth=2.5, marker="s", markersize=4)
    ax_loss.fill_between(epochs, train_loss, val_loss, color="gray", alpha=0.12, label="Generalization Margin")

    best_loss_val = val_loss[best_epoch - 1]
    ax_loss.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_loss.scatter([best_epoch], [best_loss_val], color="#2ca02c", s=140, zorder=5, 
                    edgecolor="black", linewidth=1.5, label=f"Best Checkpoint (Epoch {best_epoch})")
    ax_loss.annotate(f"Optimal Checkpoint\nLoss: {best_loss_val:.4f}",
                     xy=(best_epoch, best_loss_val),
                     xytext=(best_epoch + 1.2, best_loss_val + 0.12),
                     arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
                     fontsize=9.5, fontweight="bold", color="#2ca02c",
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", lw=1))

    ax_loss.set_title("(a) 3-Class Training vs. Validation Loss Convergence (In One Plot)", fontsize=12, fontweight="bold", pad=12)
    ax_loss.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_loss.set_ylabel("Cross-Entropy Loss (Label Smoothing=0.05)", fontsize=11, fontweight="bold")
    ax_loss.set_xticks(np.arange(1, len(epochs) + 1, 2))
    ax_loss.set_ylim(0, max(max(train_loss), max(val_loss)) * 1.15)
    ax_loss.grid(True, linestyle="--", alpha=0.6)
    ax_loss.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5)

    # ---------------- PANEL 2: ACCURACY ----------------
    ax_acc.plot(epochs, train_acc, label="Training Accuracy", color="#1f77b4", linewidth=2.5, marker="o", markersize=4)
    ax_acc.plot(epochs, val_acc, label="Validation Accuracy", color="#d62728", linewidth=2.5, marker="s", markersize=4)

    best_acc_val = val_acc[best_epoch - 1]
    ax_acc.axvline(best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.8)
    ax_acc.scatter([best_epoch], [best_acc_val], color="#2ca02c", s=140, zorder=5,
                   edgecolor="black", linewidth=1.5, label=f"Best Model Acc: {best_acc_val:.2f}%")
    ax_acc.annotate(f"Val Acc: {best_acc_val:.2f}%\nVal F1: {df['val_f1'].values[best_epoch-1]*100:.2f}%",
                    xy=(best_epoch, best_acc_val),
                    xytext=(best_epoch - 7.5, best_acc_val - 12.0),
                    arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=1.5),
                    fontsize=9.5, fontweight="bold", color="#2ca02c",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2ca02c", lw=1))

    ax_acc.set_title("(b) 3-Class Training vs. Validation Accuracy Trajectory", fontsize=12, fontweight="bold", pad=12)
    ax_acc.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax_acc.set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight="bold")
    ax_acc.set_xticks(np.arange(1, len(epochs) + 1, 2))
    ax_acc.set_ylim(40, 105)
    ax_acc.grid(True, linestyle="--", alpha=0.6)
    ax_acc.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9.5)

    plt.suptitle("TF-FaultNet 3-Class Convergence (70/15/15 file-level split; validation with recalibrated BatchNorm)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Master 3-class convergence curve saved to: {output_path}", flush=True)

if __name__ == "__main__":
    main()
