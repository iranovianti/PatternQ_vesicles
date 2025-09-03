import os, glob, json, argparse
import numpy as np
from PIL import Image
import torch

from dataset import prepare_image
from model_defs import RegressorModel

def parse_model_info(basename, model_idx=0, imsize_idx=1):
    parts = basename.replace(".pth", "").split("_")
    arch = parts[model_idx]
    image_size = int(parts[imsize_idx])
    return arch, image_size

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to image file or directory")
    parser.add_argument("--model", required=True, help="Path to .pth file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    basename = os.path.basename(args.model)
    model_name, image_size = parse_model_info(basename, model_idx=1, imsize_idx=2)

    representation_dim = 512
    model = RegressorModel(
        arch=model_name,
        pretrained='custom',
        representation_dim=representation_dim,
        custom_weight_path=args.model,
        device=device,
        mode='eval'
    ).to(device).eval()

    if os.path.isdir(args.input):
        paths = sorted(glob.glob(os.path.join(args.input, "*.png")))
    elif os.path.isfile(args.input):
        paths = [args.input]
    else:
        raise SystemExit(f"Input not found: {args.input}")

    if not paths:
        raise SystemExit(f"No images found under {args.input}")

    images = [np.array(Image.open(p)) for p in paths]
    tensors = [prepare_image(img, image_size) for img in images]

    scores = {}
    with torch.no_grad():
        for path, t in zip(paths, tensors):
            out = model(t.unsqueeze(0).to(device).float())
            score = out.cpu().numpy()[0][0]
            scores[os.path.basename(path)] = float(score)

    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)

    print(f"Saved {len(scores)} scores to {args.output}")

if __name__ == "__main__":
    main()
