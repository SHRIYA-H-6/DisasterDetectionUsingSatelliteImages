"""Models: pseudo-Siamese multitask EfficientNet-B0 classifier and a U-Net for damage segmentation."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from . import config as C


def efficientnet_b0(pretrained=True):
    # ImageNet weights from the lukemelas/EfficientNet-PyTorch GitHub release.
    return EfficientNet.from_pretrained("efficientnet-b0") if pretrained else EfficientNet.from_name("efficientnet-b0")


class MultitaskSiameseEfficientNet(nn.Module):
    """Two separate (pseudo-Siamese) EfficientNet-B0 encoders for the pre- and post-event images.

    Pooled features f_pre, f_post and their difference are concatenated and fed to a shared
    dropout-regularised layer, then to two softmax heads: disaster type and severity.
    """

    # Progressive fine-tuning: (first epoch, unfreeze encoder blocks from this index; None = frozen, 0 = all)
    UNFREEZE_SCHEDULE = [(1, None), (4, 11), (8, 5), (12, 0)]

    def __init__(self, pretrained=True, dropout=0.4, hidden=512):
        super().__init__()
        self.encoder_pre = efficientnet_b0(pretrained)
        self.encoder_post = efficientnet_b0(pretrained)
        feat = self.encoder_pre._fc.in_features  # 1280
        for enc in (self.encoder_pre, self.encoder_post):
            enc._fc = nn.Identity()
            enc._dropout = nn.Identity()
        self.shared = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(3 * feat, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.type_head = nn.Linear(hidden, len(C.DISASTER_TYPES))
        self.severity_head = nn.Linear(hidden, len(C.SEVERITIES))
        self.unfrozen_from = None
        self.set_unfrozen(None)

    def encode(self, encoder, x):
        return F.adaptive_avg_pool2d(encoder.extract_features(x), 1).flatten(1)

    def forward(self, pre, post):
        f_pre, f_post = self.encode(self.encoder_pre, pre), self.encode(self.encoder_post, post)
        h = self.shared(torch.cat([f_pre, f_post, f_post - f_pre], dim=1))
        return self.type_head(h), self.severity_head(h)

    def backbone_parameters(self):
        return list(self.encoder_pre.parameters()) + list(self.encoder_post.parameters())

    def head_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder_")]

    def set_unfrozen(self, from_block):
        """Freeze both encoders, then unfreeze blocks >= from_block (plus the conv head).

        from_block=None keeps the encoders fully frozen; 0 also unfreezes the stem.
        """
        self.unfrozen_from = from_block
        for enc in (self.encoder_pre, self.encoder_post):
            for p in enc.parameters():
                p.requires_grad = False
            if from_block is None:
                continue
            for block in enc._blocks[from_block:]:
                for p in block.parameters():
                    p.requires_grad = True
            for module in (enc._conv_head, enc._bn1) + ((enc._conv_stem, enc._bn0) if from_block == 0 else ()):
                for p in module.parameters():
                    p.requires_grad = True

    def train(self, mode=True):
        super().train(mode)
        if mode:  # frozen layers keep their pretrained BatchNorm statistics
            for enc in (self.encoder_pre, self.encoder_post):
                for m in enc.modules():
                    if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                        m.eval()
        return self

    def unfreeze_for_epoch(self, epoch):
        """Apply the progressive schedule; returns a description if the frozen set changed."""
        target = None
        for start, from_block in self.UNFREEZE_SCHEDULE:
            if epoch >= start:
                target = from_block
        if target != self.unfrozen_from:
            self.set_unfrozen(target)
            return "all encoder layers" if target == 0 else f"encoder blocks {target}-15 + conv head"
        return None


class ConvBlock(nn.Sequential):
    def __init__(self, cin, cout, dropout=0.0):
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                  nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]
        if dropout:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) on the channel-stacked pre/post pair (6 input channels).

    Outputs per-pixel logits for background / intact structure / affected (damaged or landslide).
    """

    def __init__(self, in_channels=6, n_classes=3, base=32, dropout=0.2):
        super().__init__()
        ch = [base, base * 2, base * 4, base * 8, base * 16]
        self.downs = nn.ModuleList()
        cin = in_channels
        for i, c in enumerate(ch):
            self.downs.append(ConvBlock(cin, c, dropout if i >= 3 else 0.0))
            cin = c
        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for c in reversed(ch[:-1]):
            self.ups.append(nn.ConvTranspose2d(cin, c, 2, stride=2))
            self.up_blocks.append(ConvBlock(2 * c, c))
            cin = c
        self.out = nn.Conv2d(cin, n_classes, 1)

    def forward(self, pre, post):
        x = torch.cat([pre, post], dim=1)
        skips = []
        for i, down in enumerate(self.downs):
            x = down(x)
            if i < len(self.downs) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for up, block, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            x = block(torch.cat([up(x), skip], dim=1))
        return self.out(x)
