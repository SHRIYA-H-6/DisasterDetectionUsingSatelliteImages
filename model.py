"""
Single-stage, multi-task disaster damage model: a pseudo-Siamese ResNet-18.

"Pseudo-Siamese" = two encoder branches with the *same* architecture but
*independently* learned weights (unlike a true Siamese net, which shares one
set of weights across both inputs) -- one branch encodes the pre-disaster
image, the other the post-disaster image. Their 512-d embeddings are combined
via concatenation + a difference vector, then fed through two classification
heads sharing that combined representation: disaster type (4-way) and
severity (3-way).
"""
import torch
import torch.nn as nn
import torchvision.models as models

import config


def _make_resnet18_encoder(pretrained=True):
    """ResNet-18 with its final fc layer removed, returning 512-d embeddings."""
    backbone = None
    if pretrained:
        try:
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:
            print(f"[model.py] Could not download pretrained ImageNet weights ({e}). "
                  f"Falling back to random initialization -- fine-tune longer to compensate.")
    if backbone is None:
        backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()  # -> outputs the 512-d pooled feature
    return backbone


class PseudoSiameseResNet(nn.Module):
    """Two independent ResNet-18 encoders (pre/post) + dual classification heads."""

    EMBED_DIM = 512

    def __init__(self, num_disaster_classes=len(config.DISASTER_TYPES),
                 num_severity_classes=len(config.SEVERITY_LEVELS),
                 pretrained=True, dropout=0.3):
        super().__init__()
        self.pre_encoder = _make_resnet18_encoder(pretrained)
        self.post_encoder = _make_resnet18_encoder(pretrained)

        combined_dim = self.EMBED_DIM * 3  # [pre, post, post-pre difference]
        self.shared_fc = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.disaster_head = nn.Linear(512, num_disaster_classes)
        self.severity_head = nn.Linear(512, num_severity_classes)

    def forward(self, pre_img, post_img):
        pre_emb = self.pre_encoder(pre_img)
        post_emb = self.post_encoder(post_img)
        diff_emb = post_emb - pre_emb
        combined = torch.cat([pre_emb, post_emb, diff_emb], dim=1)
        shared = self.shared_fc(combined)
        disaster_logits = self.disaster_head(shared)
        severity_logits = self.severity_head(shared)
        return disaster_logits, severity_logits


def build_model(pretrained=True):
    return PseudoSiameseResNet(pretrained=pretrained)


if __name__ == "__main__":
    model = build_model(pretrained=True)
    pre = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    post = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    d_logits, s_logits = model(pre, post)
    print("disaster_logits", d_logits.shape, "severity_logits", s_logits.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params:,}")
