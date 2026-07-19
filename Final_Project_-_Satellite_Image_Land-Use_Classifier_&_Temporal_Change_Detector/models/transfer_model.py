"""
Transfer-learning wrapper around torchvision's ResNet-18 or EfficientNet-B0.

Exposes:
  - forward(x)                 -> class logits
  - get_embedding(x)            -> 512-d penultimate-layer embedding (feature vector)
  - freeze_backbone() / unfreeze_last_blocks() for the two-phase fine-tuning strategy
"""
import torch
import torch.nn as nn
import torchvision.models as tvm


class TransferModel(nn.Module):
    EMBED_DIM = 512

    def __init__(self, backbone="resnet18", num_classes=10, pretrained=True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "resnet18":
            net = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
            in_features = net.fc.in_features  # 512
            net.fc = nn.Identity()
            self.backbone = net
            self.embed_dim = in_features
            self.last_block_names = ["layer4"]  # last conv block to unfreeze in phase 2
        elif backbone == "efficientnet_b0":
            net = tvm.efficientnet_b0(weights=tvm.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
            in_features = net.classifier[1].in_features  # 1280
            net.classifier = nn.Identity()
            self.backbone = net
            self.embed_dim = in_features
            self.last_block_names = ["features.7", "features.8"]  # last 2 blocks
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Project to a common 512-d embedding space regardless of backbone,
        # so the change-detection module always works with 512-d vectors.
        self.embed_proj = (
            nn.Identity() if self.embed_dim == self.EMBED_DIM
            else nn.Linear(self.embed_dim, self.EMBED_DIM)
        )
        self.classifier_head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(self.EMBED_DIM, num_classes),
        )

    def get_embedding(self, x):
        feats = self.backbone(x)
        return self.embed_proj(feats)

    def forward(self, x):
        emb = self.get_embedding(x)
        return self.classifier_head(emb)

    # ---- two-phase fine-tuning helpers -------------------------------------------------
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.embed_proj.parameters():
            p.requires_grad = True
        for p in self.classifier_head.parameters():
            p.requires_grad = True

    def unfreeze_last_blocks(self):
        """Unfreezes only the last 2 convolutional blocks of the backbone,
        per the required Phase 2 strategy."""
        named = dict(self.backbone.named_modules())
        for block_name in self.last_block_names:
            module = named.get(block_name)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True

    def trainable_param_groups(self, base_lr):
        """Returns param groups; call after freeze/unfreeze so only
        requires_grad=True params get an entry."""
        return [{"params": [p for p in self.parameters() if p.requires_grad], "lr": base_lr}]


if __name__ == "__main__":
    m = TransferModel("resnet18")
    dummy = torch.randn(2, 3, 64, 64)
    print("logits:", m(dummy).shape)
    print("embedding:", m.get_embedding(dummy).shape)
