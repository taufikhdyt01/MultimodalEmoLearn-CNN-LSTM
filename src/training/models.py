"""
Model Architectures for Emotion Recognition
=============================================
CNN, FCNN, Early Fusion (Input-Level), Intermediate Fusion (Feature-Level),
Late Fusion (Decision-Level) — PyTorch implementation.
Includes Transfer Learning variants (ResNet18).
"""

import torch
import torch.nn as nn
from torchvision import models as tv_models


class EmotionCNN(nn.Module):
    """CNN untuk fitur penampilan (citra wajah 224x224x3).

    4 Conv blocks (32->64->128->256) + 2 FC layers (512->256).
    """

    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224x3 -> 112x112x32
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 2: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 3: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 4: 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )

        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return self.head(x)

    def extract_features(self, x):
        """Extract 256-dim feature vector (untuk fusion)."""
        x = self.features(x)
        return self.classifier(x)


class EmotionCNN1D(nn.Module):
    """1D-CNN untuk fitur geometrik mentah (koordinat 68 landmark sebagai sequence).

    Input shape: (B, 2, 68) — 2 channel (x, y) × 68 titik landmark.
    Arahan dosen: CNN dengan fitur geometrik baseline = hanya koordinat titik.

    4 Conv1d blocks (32->64->128->256) + GAP + FC head — mirror EmotionCNN tapi 1D.
    """

    def __init__(self, num_classes=7, num_points=68, in_channels=2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 68 -> 34
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.25),

            # Block 2: 34 -> 17
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.25),

            # Block 3: 17 -> 8
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.25),

            # Block 4: 8 -> 4
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # global average pool -> (B, 256, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # (B, 256)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )

        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        # Accept (B, 136) flat → reshape to (B, 68, 2) → permute to (B, 2, 68)
        if x.dim() == 2:
            x = x.view(x.size(0), -1, 2).transpose(1, 2)
        x = self.features(x)
        x = self.classifier(x)
        return self.head(x)

    def extract_features(self, x):
        """Extract 128-dim feature vector (untuk fusion)."""
        if x.dim() == 2:
            x = x.view(x.size(0), -1, 2).transpose(1, 2)
        x = self.features(x)
        return self.classifier(x)


class EmotionCNN1D_FACS(nn.Module):
    """1D-CNN untuk FACS-decomposed Euclidean distance features.

    Input shape: (B, 1, D) where D = jumlah FACS-AU distances (default 28).
    Arahan dosen: "CNN dengan fitur geometrik yang didekomposisi - mencari jarak
    euclid dari acuan FACS - Facial Action Coding System".

    3 Conv1d blocks (32->64->128) + GAP + FC head — kernel kecil k=3 sesuai
    ukuran sequence yang lebih pendek dari raw landmark sequence (28 vs 68).
    """

    def __init__(self, num_classes=7, in_dim=28, in_channels=1):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: D -> D
            nn.Conv1d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32), nn.ReLU(),
            nn.Dropout(0.25),

            # Block 2: D -> D/2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.25),

            # Block 3: D/2 -> 1 (GAP)
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
        )

        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        # Accept (B, D) flat → unsqueeze to (B, 1, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.classifier(x)
        return self.head(x)

    def extract_features(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


class MultiStackedCNN_Bachtiar(nn.Module):
    """Replikasi arsitektur 'Multi-stacked CNN' dari Bachtiar et al. (2024)
    "Engagement Level Detection Using Facial Extraction and Multi-stacked
    Convolutional Neural Network in e-Learning Settings" (EECCIS 2024).

    Spesifikasi paper: 3 CNN layers + Batch Normalization + 2 linear layers
    dengan dropout. Best hyperparams: batch=16, lr=0.001, epoch=10.

    Input: feature vector hand-crafted (FL + AU + HE + DE, e.g. 1490-dim atau
    subset). Treated sebagai 1D sequence dengan 1 channel.

    Channels & kernels diestimasi sesuai konvensi paper (Fig. 3) — paper tidak
    spesifikasi exact dim, jadi pakai default reasonable:
      Conv1d(1, 32, k=5) -> BN -> Conv1d(32, 64, k=5) -> Conv1d(64, 128, k=3)
      -> Flatten + GAP -> Linear(128, 64) -> Dropout(0.3) -> Linear(64, num_classes)
    """

    def __init__(self, num_classes=7, in_dim=259):
        super().__init__()
        self.in_dim = in_dim

        # 3 CNN layers, dengan BN antara conv1 & conv2 (sesuai Fig. 3)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)

        # 2 linear layers + dropout (Fig. 3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Accept (B, D) → unsqueeze to (B, 1, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.gap(x).squeeze(-1)  # (B, 128)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class EmotionFCNN(nn.Module):
    """FCNN untuk fitur geometrik (68 landmarks = 136 fitur).

    5 Dense layers (256->512->512->256->128).
    """

    def __init__(self, input_dim=136, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )

        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        return self.head(x)

    def extract_features(self, x):
        """Extract 128-dim feature vector (untuk fusion)."""
        return self.features(x)


class EmotionEarlyFusion(nn.Module):
    """Early Fusion (Input-Level Fusion): landmark heatmap di-stack sebagai channel
    tambahan ke citra wajah, diproses oleh single CNN.

    Input: 4-channel tensor (R, G, B, landmark_heatmap) 224x224.
    Arsitektur sama dengan EmotionCNN, hanya first Conv2d menerima 4 channel.

    Referensi: HAE-Net (Mo et al., MMM 2020) — landmark heatmap sebagai additional channel.
    """

    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224x4 -> 112x112x32  (first conv accepts 4 channels)
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 2: 112x112x32 -> 56x56x64
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 3: 56x56x64 -> 28x28x128
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            # Block 4: 28x28x128 -> 14x14x256
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )

        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return self.head(x)


class _SpatialGate(nn.Module):
    """Spatial gating module untuk Gated Early Fusion.

    Input: 4-channel tensor (RGB + heatmap) (B, 4, H, W)
    Output: (rgb_gated, heatmap_gated, gate_map_2ch) di mana
        rgb_gated = RGB ⊙ g_rgb (broadcast g_rgb dari (B,1,H,W) ke 3 channel)
        heatmap_gated = heatmap ⊙ g_heatmap
        gate_map_2ch = (B, 2, H, W) untuk inspeksi/visualisasi

    Gate dihitung dari concat input via small Conv2d (3x3 → 1x1 → Sigmoid).
    Gate per pixel ∈ [0,1] per modality (RGB & heatmap punya gate independen,
    bukan softmax — supaya bisa simultaneous-attend atau simultaneous-suppress
    di region tertentu).
    """

    def __init__(self, hidden_ch: int = 8):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Conv2d(4, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 2, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x_concat):
        # x_concat: (B, 4, H, W)
        gates = self.gate_net(x_concat)  # (B, 2, H, W)
        g_rgb = gates[:, 0:1, :, :]      # (B, 1, H, W)
        g_heat = gates[:, 1:2, :, :]     # (B, 1, H, W)
        rgb = x_concat[:, :3, :, :] * g_rgb
        heatmap = x_concat[:, 3:4, :, :] * g_heat
        return torch.cat([rgb, heatmap], dim=1), gates


class EmotionEarlyFusionGated(nn.Module):
    """Gated Early Fusion (scratch): spatial sigmoid gating sebelum CNN backbone.

    Berbeda dengan EmotionEarlyFusion (concat) yang treat semua 4 channel sama,
    versi gated punya `_SpatialGate` learnable module yang memprediksi spatial
    weight per modality berdasarkan input itu sendiri. Model bisa attend ke
    region berbeda untuk RGB vs heatmap.

    Backbone CNN sama persis dengan EmotionEarlyFusion supaya fair-compare.
    Parameter overhead: ~314 (Conv 4→8 3x3 + Conv 8→2 1x1 + biases).
    """

    def __init__(self, num_classes=7, gate_hidden_ch: int = 8):
        super().__init__()
        self.gate = _SpatialGate(hidden_ch=gate_hidden_ch)

        self.features = nn.Sequential(
            nn.Conv2d(4, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x, return_gates: bool = False):
        x_gated, gates = self.gate(x)
        feat = self.features(x_gated)
        feat = self.classifier(feat)
        logits = self.head(feat)
        if return_gates:
            return logits, gates
        return logits


class EmotionEarlyFusionTransferGated(nn.Module):
    """Gated Early Fusion TL: spatial gating + ResNet18 pretrained backbone.

    First conv ResNet18 di-extend ke 4 channel dengan inisialisasi yang sama
    dengan EmotionEarlyFusionTransfer (RGB di-copy dari pretrained, 4th channel
    diinisialisasi dari mean RGB pretrained weights). Gate ditambahkan sebelum
    conv pertama.
    """

    def __init__(self, num_classes=7, pretrained=True, gate_hidden_ch: int = 8):
        super().__init__()
        self.gate = _SpatialGate(hidden_ch=gate_hidden_ch)

        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet18(weights=weights)
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            in_channels=4, out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=(old_conv.bias is not None),
        )
        if pretrained:
            with torch.no_grad():
                new_conv.weight[:, :3, :, :] = old_conv.weight
                new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)
        resnet.conv1 = new_conv
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x, return_gates: bool = False):
        x_gated, gates = self.gate(x)
        feat = self.features(x_gated)
        feat = self.classifier(feat)
        logits = self.head(feat)
        if return_gates:
            return logits, gates
        return logits


class EmotionEarlyFusionTransfer(nn.Module):
    """Early Fusion TL: ResNet18 pretrained dengan first conv dimodifikasi
    menerima 4-channel input (RGB + landmark heatmap).

    Strategy:
    - First conv (Conv2d) diubah dari in_channels=3 ke in_channels=4.
    - Weight 3 channel pertama (RGB) di-copy dari pretrained.
    - Weight channel ke-4 (heatmap) diinisialisasi dari rata-rata 3 channel
      pretrained → memberikan starting point yang reasonable.
    """

    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()

        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet18(weights=weights)

        # Modify first conv: 3 -> 4 channels
        old_conv = resnet.conv1  # Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv = nn.Conv2d(
            in_channels=4, out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride,
            padding=old_conv.padding, bias=(old_conv.bias is not None),
        )

        if pretrained:
            with torch.no_grad():
                # Copy RGB channel weights
                new_conv.weight[:, :3, :, :] = old_conv.weight
                # Initialize 4th channel from mean of RGB (reasonable starting point)
                new_conv.weight[:, 3:4, :, :] = old_conv.weight.mean(dim=1, keepdim=True)

        resnet.conv1 = new_conv
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # output: 512-dim

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )

        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return self.head(x)


class IntermediateFusion(nn.Module):
    """Intermediate Fusion (Feature-Level Fusion): CNN + FCNN digabung di level fitur.

    Referensi: Boulahia et al. (2021) - "Early, intermediate and late fusion strategies
    for robust deep learning-based multimodal action recognition."

    Image stream (CNN) -> 256-dim feature
    Landmark stream (FCNN) -> 128-dim feature
    Concatenate -> 384-dim -> Dense(512->256) -> 7 classes
    """

    def __init__(self, num_classes=7, landmark_dim=136):
        super().__init__()

        # Image stream (CNN - 3 blocks, lighter than standalone CNN)
        self.image_features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout2d(0.25),
        )
        self.image_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )

        # Landmark stream
        self.landmark_features = nn.Sequential(
            nn.Linear(landmark_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )

        # Post-fusion: 256 (image) + 128 (landmark) = 384
        self.fusion_head = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, landmark):
        img_feat = self.image_features(image)
        img_feat = self.image_fc(img_feat)
        lm_feat = self.landmark_features(landmark)
        fused = torch.cat([img_feat, lm_feat], dim=1)
        return self.fusion_head(fused)


# ============== TRANSFER LEARNING ==============

class EmotionCNNTransfer(nn.Module):
    """Transfer Learning CNN menggunakan ResNet18 pretrained di ImageNet.

    ResNet18 dipilih karena:
    - Ringan (~11M params) tapi efektif untuk image classification
    - Pretrained weights dari ImageNet memberikan fitur visual yang kaya
    - Cocok untuk dataset kecil (~7000 images) yang tidak cukup untuk train from scratch

    Strategy: Fine-tune seluruh network dengan learning rate kecil.
    """

    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()

        # Load ResNet18 pretrained
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet18(weights=weights)

        # Ambil semua layer kecuali FC terakhir
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # output: 512-dim

        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )

        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return self.head(x)

    def extract_features(self, x):
        """Extract 256-dim feature vector (untuk fusion)."""
        x = self.features(x)
        return self.classifier(x)


class EmotionCNNTransferFER(nn.Module):
    """Transfer Learning CNN: ResNet18 pretrained ImageNet → fine-tuned FER2013 → fine-tune dataset.

    Two-stage transfer learning:
    1. ImageNet → general visual features
    2. FER2013 → emotion-specific features
    3. Fine-tune pada dataset sendiri

    Args:
        num_classes: jumlah kelas target
        fer_weights_path: path ke weights yang sudah di-pretrain di FER2013
    """

    def __init__(self, num_classes=7, fer_weights_path=None):
        super().__init__()

        # Load ResNet18 pretrained ImageNet
        weights = tv_models.ResNet18_Weights.DEFAULT
        resnet = tv_models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )

        self.head = nn.Linear(256, num_classes)

        # Load FER2013 pre-trained weights if provided
        if fer_weights_path is not None:
            self._load_fer_weights(fer_weights_path, num_classes)

    def _load_fer_weights(self, fer_weights_path, num_classes):
        """Load weights from FER2013 pre-trained model, handle head mismatch."""
        state_dict = torch.load(fer_weights_path, map_location="cpu", weights_only=True)
        # Remove head weights if num_classes differs (FER2013 has 7 classes)
        if state_dict.get("head.weight", None) is not None:
            if state_dict["head.weight"].shape[0] != num_classes:
                del state_dict["head.weight"]
                del state_dict["head.bias"]
        self.load_state_dict(state_dict, strict=False)
        print(f"  Loaded FER2013 pre-trained weights from {fer_weights_path}")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return self.head(x)

    def extract_features(self, x):
        x = self.features(x)
        return self.classifier(x)


class IntermediateFusionTransferFER(nn.Module):
    """Intermediate Fusion: ResNet18 (FER2013 pretrained) + FCNN.

    Same architecture as IntermediateFusionTransfer but CNN uses FER2013 weights.
    """

    def __init__(self, num_classes=7, landmark_dim=136, fer_weights_path=None):
        super().__init__()

        # Image stream (ResNet18 → load FER2013 weights for features + classifier)
        weights = tv_models.ResNet18_Weights.DEFAULT
        resnet = tv_models.resnet18(weights=weights)
        self.image_features = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )

        # Load FER2013 weights for image stream if provided
        if fer_weights_path is not None:
            self._load_fer_image_weights(fer_weights_path)

        # Landmark stream (from scratch)
        self.landmark_features = nn.Sequential(
            nn.Linear(landmark_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )

        # Post-fusion: 256 + 128 = 384
        self.fusion_head = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def _load_fer_image_weights(self, fer_weights_path):
        """Load only image features + classifier weights from FER2013 model."""
        state_dict = torch.load(fer_weights_path, map_location="cpu", weights_only=True)
        # Map FER model keys to this model's keys
        new_state = {}
        for k, v in state_dict.items():
            if k.startswith("features."):
                new_state[k.replace("features.", "image_features.")] = v
            elif k.startswith("classifier."):
                new_state[k.replace("classifier.", "image_fc.")] = v
        self.load_state_dict(new_state, strict=False)
        print(f"  Loaded FER2013 image weights from {fer_weights_path}")

    def forward(self, image, landmark):
        img_feat = self.image_features(image)
        img_feat = self.image_fc(img_feat)
        lm_feat = self.landmark_features(landmark)
        fused = torch.cat([img_feat, lm_feat], dim=1)
        return self.fusion_head(fused)


class IntermediateFusionTransfer(nn.Module):
    """Intermediate Fusion dengan Transfer Learning CNN (ResNet18) + FCNN.

    Image stream: ResNet18 pretrained → 256-dim
    Landmark stream: FCNN → 128-dim
    Concatenate → 384-dim → Dense(512→256) → num_classes
    """

    def __init__(self, num_classes=7, landmark_dim=136, pretrained=True):
        super().__init__()

        # Image stream (ResNet18 pretrained)
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = tv_models.resnet18(weights=weights)
        self.image_features = nn.Sequential(*list(resnet.children())[:-1])
        self.image_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
        )

        # Landmark stream (sama dengan IntermediateFusion)
        self.landmark_features = nn.Sequential(
            nn.Linear(landmark_dim, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        )

        # Post-fusion: 256 (image) + 128 (landmark) = 384
        self.fusion_head = nn.Sequential(
            nn.Linear(384, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, image, landmark):
        img_feat = self.image_features(image)
        img_feat = self.image_fc(img_feat)
        lm_feat = self.landmark_features(landmark)
        fused = torch.cat([img_feat, lm_feat], dim=1)
        return self.fusion_head(fused)