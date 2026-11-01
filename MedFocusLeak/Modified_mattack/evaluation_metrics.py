import os
import torch
import numpy as np
from PIL import Image
import hydra
from tqdm import tqdm
from typing import Dict, List

from config_schema import MainConfig
from utils import (
    hash_training_config,
    setup_wandb,
    log_wandb,
    finish_wandb,
    ensure_dir,
    get_output_paths,
)

VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPEG"]


def to_tensor(pic):
    """Convert PIL.Image to PyTorch Tensor in range [0,1]."""
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.float32) / 255.0


def load_and_preprocess_image(image_path: str) -> torch.Tensor:
    """Load image from path and preprocess it."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_tensor = to_tensor(img)
        return img_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def calculate_metrics(original: torch.Tensor, adversarial: torch.Tensor) -> Dict[str, float]:
    """Calculate normalized L1 and L2 metrics between original and adversarial images."""
    if original.shape != adversarial.shape:
        adversarial = torch.nn.functional.interpolate(
            adversarial.unsqueeze(0),
            size=original.shape[1:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    perturbation = adversarial - original
    n_pixels = torch.numel(perturbation)
    l1_norm = torch.abs(perturbation).sum() / n_pixels
    l2_norm = torch.sqrt(torch.sum(perturbation ** 2) / n_pixels)

    return {
        "normalized_l1": l1_norm.item(),
        "normalized_l2": l2_norm.item(),
    }


def save_metrics(metrics: List[Dict[str, float]], output_file: str):
    """Save metrics to a file."""
    ensure_dir(os.path.dirname(output_file))

    avg_l1 = sum(m["normalized_l1"] for m in metrics) / len(metrics)
    avg_l2 = sum(m["normalized_l2"] for m in metrics) / len(metrics)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"Average Normalized L1: {avg_l1:.6f}\n")
        f.write(f"Average Normalized L2: {avg_l2:.6f}\n\n")
        f.write("Filename | Normalized L1 | Normalized L2\n")
        f.write("=" * 60 + "\n")

        for m in metrics:
            f.write(f"{m['filename']} | {m['normalized_l1']:.6f} | {m['normalized_l2']:.6f}\n")


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    setup_wandb(cfg, tags=["perturbation_evaluation"])

    config_hash = hash_training_config(cfg)
    print(f"Using training output for config hash: {config_hash}")

    paths = get_output_paths(cfg, config_hash)
    adv_images_dir = paths["output_dir"]

    if not os.path.exists(adv_images_dir):
        print(f"Error: Adversarial images directory {adv_images_dir} does not exist")
        return

    metrics_dir = os.path.join(cfg.data.output, "metrics", config_hash)
    ensure_dir(metrics_dir)
    metrics_file = os.path.join(metrics_dir, "perturbation_metrics.txt")

    all_metrics = []
    total_l1 = 0.0
    total_l2 = 0.0
    count = 0

    print("Evaluating perturbations...")
    for root, _, files in os.walk(adv_images_dir):
        for file in tqdm(files):
            if any(file.lower().endswith(ext.lower()) for ext in VALID_IMAGE_EXTENSIONS):
                try:
                    adv_path = os.path.join(root, file)
                    folder = os.path.basename(root)
                    filename_base = os.path.splitext(file)[0]

                    orig_found = False
                    for ext in VALID_IMAGE_EXTENSIONS:
                        orig_path = os.path.join(cfg.data.cle_data_path, folder, filename_base + ext)
                        if os.path.exists(orig_path):
                            orig_found = True
                            break

                    if orig_found:
                        orig_tensor = load_and_preprocess_image(orig_path)
                        adv_tensor = load_and_preprocess_image(adv_path)

                        if orig_tensor is not None and adv_tensor is not None:
                            metrics = calculate_metrics(orig_tensor, adv_tensor)
                            metrics["filename"] = file
                            all_metrics.append(metrics)

                            total_l1 += metrics["normalized_l1"]
                            total_l2 += metrics["normalized_l2"]
                            count += 1

                            log_wandb(
                                cfg,
                                {
                                    f"metrics/{file}/normalized_l1": metrics["normalized_l1"],
                                    f"metrics/{file}/normalized_l2": metrics["normalized_l2"],
                                    "running_avg_l1": total_l1 / count,
                                    "running_avg_l2": total_l2 / count,
                                },
                            )
                    else:
                        print(f"Original image not found for {filename_base}, skipping")

                except Exception as e:
                    print(f"Error processing {file}: {e}")

    if all_metrics:
        save_metrics(all_metrics, metrics_file)

        log_wandb(
            cfg,
            {
                "final_avg_l1": total_l1 / count,
                "final_avg_l2": total_l2 / count,
                "total_evaluated": count,
            },
        )

        print(f"\nEvaluation complete:")
        print(f"Average Normalized L1: {total_l1 / count:.6f}")
        print(f"Average Normalized L2: {total_l2 / count:.6f}")
        print(f"Results saved to: {metrics_file}")
    else:
        print("No images were successfully evaluated")

    finish_wandb(cfg)


if __name__ == "__main__":
    main()
