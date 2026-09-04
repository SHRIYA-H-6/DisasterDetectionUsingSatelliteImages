"""
Run the trained pseudo-Siamese model on a single real pre/post image pair.

Usage:
    python predict.py --pre path/to/pre.png --post path/to/post.png
"""
import argparse

import torch
import torch.nn.functional as F

import config
from dataset import get_transform
from evaluate import load_checkpoint
import cv2


def load_image_tensor(path, transform):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return transform(img)


@torch.no_grad()
def predict_pair(model, pre_path, post_path, device):
    transform = get_transform(train=False)
    pre_t = load_image_tensor(pre_path, transform).unsqueeze(0).to(device)
    post_t = load_image_tensor(post_path, transform).unsqueeze(0).to(device)

    d_logits, s_logits = model(pre_t, post_t)
    d_probs = F.softmax(d_logits, dim=1).squeeze(0).cpu().tolist()
    s_probs = F.softmax(s_logits, dim=1).squeeze(0).cpu().tolist()

    disaster_idx = int(torch.tensor(d_probs).argmax())
    severity_idx = int(torch.tensor(s_probs).argmax())

    return {
        "disaster_type": config.DISASTER_TYPES[disaster_idx],
        "disaster_probs": dict(zip(config.DISASTER_TYPES, d_probs)),
        "severity": config.SEVERITY_LEVELS[severity_idx],
        "severity_probs": dict(zip(config.SEVERITY_LEVELS, s_probs)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre", required=True, help="Path to the pre-disaster image")
    parser.add_argument("--post", required=True, help="Path to the post-disaster image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(device=device)

    result = predict_pair(model, args.pre, args.post, device)
    print(f"Predicted disaster type: {result['disaster_type']}")
    print(f"  probabilities: {result['disaster_probs']}")
    print(f"Predicted severity: {result['severity']}")
    print(f"  probabilities: {result['severity_probs']}")


if __name__ == "__main__":
    main()
