"""
Ablation variants of TF-FaultNet (3-class).

M0 is the real `src.models.TFFaultNet`, so the full model in the ablation study is exactly the
model trained in the final run. A1-A5 each remove one component, using `FlexibleFaultNet3Class`,
whose layers mirror TFFaultNet (same layers, same creation order).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.audio_features import LogMelSpectrogramExtractor, get_mel_filterbank
from src.models import TFFaultNet, SqueezeExcitation

# id -> (name, description, flags). M0 flags are listed for documentation only.
ABLATION_VARIANTS = {
    "M0": ("Full TF-FaultNet (proposed)",
           "SE attention + dual avg/max pooling + per-spectrogram normalisation + residual connections",
           dict(use_se=True, pooling_mode="dual", use_instance_norm=True, use_residual=True)),
    "A1": ("A1: No SE channel attention",
           "Removes the Squeeze-and-Excitation gating from all three residual blocks",
           dict(use_se=False, pooling_mode="dual", use_instance_norm=True, use_residual=True)),
    "A2": ("A2: Average pooling only",
           "Global average pooling only (no global max pooling); classifier input 256",
           dict(use_se=True, pooling_mode="avg_only", use_instance_norm=True, use_residual=True)),
    "A3": ("A3: Max pooling only",
           "Global max pooling only (no global average pooling); classifier input 256",
           dict(use_se=True, pooling_mode="max_only", use_instance_norm=True, use_residual=True)),
    "A4": ("A4: No per-spectrogram normalisation",
           "Raw log-mel input without the per-recording zero-mean / unit-variance step",
           dict(use_se=True, pooling_mode="dual", use_instance_norm=False, use_residual=True)),
    "A5": ("A5: No residual connections",
           "Plain stacked convolution blocks without shortcut connections",
           dict(use_se=True, pooling_mode="dual", use_instance_norm=True, use_residual=False)),
}


class LogMelNoNorm(nn.Module):
    """Log-Mel extractor WITHOUT per-recording instance normalisation (ablation A4).
    Same STFT / mel settings as LogMelSpectrogramExtractor."""

    def __init__(self, sr=44100, n_fft=1024, hop_length=256, n_mels=64, f_min=20.0, f_max=22050.0):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        self.register_buffer("mel_filterbank", get_mel_filterbank(sr, n_fft, n_mels, f_min, f_max))

    def forward(self, x):
        stft = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.n_fft,
                          window=self.window, return_complex=True, center=True, pad_mode="reflect")
        power = torch.abs(stft) ** 2
        mel_spec = torch.matmul(self.mel_filterbank, power)
        return torch.log(torch.clamp(mel_spec, min=1e-6)).unsqueeze(1)


class AblationResBlock(nn.Module):
    """Mirror of src.models.ConvBlock2D with switchable SE and residual shortcut."""

    def __init__(self, in_c, out_c, stride=1, use_se=True, use_residual=True):
        super().__init__()
        self.use_se = use_se
        self.use_residual = use_residual
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        if use_se:
            self.se = SqueezeExcitation(out_c)
        if use_residual:
            if stride != 1 or in_c != out_c:
                self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                                              nn.BatchNorm2d(out_c))
            else:
                self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        if self.use_residual:
            out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class FlexibleFaultNet3Class(nn.Module):
    """TF-FaultNet with one component switched off (A1-A5)."""

    def __init__(self, num_classes=3, sr=44100, n_mels=64, n_fft=1024, hop_length=256,
                 use_se=True, pooling_mode="dual", use_instance_norm=True, use_residual=True):
        super().__init__()
        self.pooling_mode = pooling_mode
        extractor = LogMelSpectrogramExtractor if use_instance_norm else LogMelNoNorm
        self.spec_extractor = extractor(sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.layer1 = AblationResBlock(32, 64, 2, use_se, use_residual)
        self.layer2 = AblationResBlock(64, 128, 2, use_se, use_residual)
        self.layer3 = AblationResBlock(128, 256, 2, use_se, use_residual)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max = nn.AdaptiveMaxPool2d((1, 1))
        feat_dim = 512 if pooling_mode == "dual" else 256
        self.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(feat_dim, 256), nn.ReLU(inplace=True),
                                        nn.Dropout(0.2), nn.Linear(256, num_classes))

    def forward(self, x):
        feat = self.layer3(self.layer2(self.layer1(self.init_conv(self.spec_extractor(x)))))
        avg_p = self.global_pool(feat).flatten(1)
        max_p = self.global_max(feat).flatten(1)
        if self.pooling_mode == "dual":
            pooled = torch.cat([avg_p, max_p], dim=1)
        elif self.pooling_mode == "avg_only":
            pooled = avg_p
        elif self.pooling_mode == "max_only":
            pooled = max_p
        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")
        return self.classifier(pooled)


def build_model(variant, num_classes=3):
    """M0 -> the real TFFaultNet; A1-A5 -> FlexibleFaultNet3Class with one component removed."""
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {list(ABLATION_VARIANTS)}")
    if variant == "M0":
        return TFFaultNet(num_classes=num_classes)
    return FlexibleFaultNet3Class(num_classes=num_classes, **ABLATION_VARIANTS[variant][2])
