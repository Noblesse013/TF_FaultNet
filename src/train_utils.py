"""
src/train_utils.py
------------------
Shared training loop functions for the 3-class TF-FaultNet pipeline.
"""
import numpy as np
import torch
try:
    from sklearn.metrics import accuracy_score, f1_score
except Exception:
    def accuracy_score(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(y_true == y_pred))

    def f1_score(y_true, y_pred, average='macro'):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        classes = np.unique(np.concatenate([y_true, y_pred]))
        f1_list = []
        for c in classes:
            tp = np.sum((y_true == c) & (y_pred == c))
            fp = np.sum((y_true != c) & (y_pred == c))
            fn = np.sum((y_true == c) & (y_pred != c))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_list.append(f1)
        if average == 'macro':
            return float(np.mean(f1_list)) if len(f1_list) > 0 else 0.0
        return f1_list



def train_epoch(model, loader, criterion, optimizer, device,
                max_grad_norm=None, scheduler=None, stats=None):
    """One training epoch. Returns (avg_loss, accuracy).

    Training loop order (§10):
    1. forward propagation
    2. compute loss
    3. zero gradients
    4. backward propagation
    5. (optional) gradient-norm clipping
    6. optimizer step
    7. (optional) per-step learning-rate scheduler step

    max_grad_norm: if set, gradients are rescaled so their global L2 norm is at most this value.
    scheduler:     if set, stepped once per optimizer step (used for warmup + cosine decay).
    stats:         if a dict is passed, it receives the mean and max pre-clipping gradient norm.
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    grad_norms = []

    for audio, labels in loader:
        audio, labels = audio.to(device), labels.to(device)
        # §10 order: forward → loss → zero_grad → backward → step
        outputs = model(audio)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        # Total gradient norm before clipping (inf disables clipping but still measures it).
        clip_to = max_grad_norm if max_grad_norm is not None else float("inf")
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_to)
        grad_norms.append(float(grad_norm))
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * len(labels)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += len(labels)

    if stats is not None:
        stats["grad_norm_mean"] = float(np.mean(grad_norms))
        stats["grad_norm_max"] = float(np.max(grad_norms))
    return total_loss / total, correct / total


def batchnorm_buffers(model):
    """Copy of every BatchNorm running statistic (to restore later with load_state_dict(strict=False))."""
    keys = ("running_mean", "running_var", "num_batches_tracked")
    return {k: v.detach().clone() for k, v in model.state_dict().items() if k.endswith(keys)}


def recalibrate_batchnorm(model, loader, device):
    """Re-estimate every BatchNorm layer's running mean/variance from `loader`.

    The running statistics are reset and replaced by an exact (cumulative) average over one
    pass of `loader`, using the current weights. Only BatchNorm buffers change; weights,
    gradients and optimizer state are untouched. Use clean (non-augmented) TRAINING data only.

    Pass a loader built with its own torch.Generator so this pass does not consume the global
    RNG (which would change the shuffling of later training epochs).
    """
    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        return
    saved_momentum = [m.momentum for m in bn_layers]
    for m in bn_layers:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average = exact mean over the pass
    was_training = model.training
    # Only BatchNorm layers in train mode (so they accumulate statistics). Dropout stays in eval
    # mode: it would otherwise draw from the global RNG and change later training epochs.
    model.eval()
    for m in bn_layers:
        m.train()
    with torch.no_grad():
        for audio, _ in loader:
            model(audio.to(device))
    for m, mom in zip(bn_layers, saved_momentum):
        m.momentum = mom
    model.train(was_training)


def eval_epoch(model, loader, criterion, device):
    """One evaluation epoch. Returns (avg_loss, accuracy, macro_f1, all_preds, all_targets)."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for audio, labels in loader:
            audio, labels = audio.to(device), labels.to(device)
            outputs = model(audio)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * len(labels)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return total_loss / len(all_targets), acc, macro_f1, all_preds, all_targets
