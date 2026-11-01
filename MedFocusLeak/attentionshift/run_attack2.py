import os
import argparse
import torch
from PIL import Image
from tqdm import tqdm

from surrogate import MODEL_CLASS_MAP
from Attack2 import AttentionPerturber

# --- Default Configuration ---
DEFAULT_IMAGE_DIR = "Imgs/"
DEFAULT_MASK_DIR = "masks/"
DEFAULT_RESULT_DIR = "results/"

# --- PGD attack parameters ---
DEFAULT_NUM_STEPS = 300
DEFAULT_EPSILON = 16 / 255.0
DEFAULT_ALPHA = 1 / 255.0

# --- Manual Model Configuration Dictionary ---
ENSEMBLE_CONFIG = {
    "laion/CLIP-ViT-G-14-laion2B-s12B-b42K": {
        "class": MODEL_CLASS_MAP["laion/CLIP-ViT-G-14-laion2B-s12B-b42K"],
        "input_size": (224, 224),
        "patch_size": 14,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    },
    "openai/clip-vit-large-patch14-336": {
        "class": MODEL_CLASS_MAP["openai/clip-vit-large-patch14-336"],
        "input_size": (336, 336),
        "patch_size": 14,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    },
    "openai/clip-vit-base-patch16": {
        "class": MODEL_CLASS_MAP["openai/clip-vit-base-patch16"],
        "input_size": (224, 224),
        "patch_size": 16,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    },
    "openai/clip-vit-base-patch32": {
        "class": MODEL_CLASS_MAP["openai/clip-vit-base-patch32"],
        "input_size": (224, 224),
        "patch_size": 32,
        "image_mean": [0.48145466, 0.4578275, 0.40821073],
        "image_std": [0.26862954, 0.26130258, 0.27577711],
    },
}


def _find_mask(mask_dir: str, rel_dir: str, image_name: str) -> str:
    mask_base, _ = os.path.splitext(image_name)
    candidates = [
        os.path.join(mask_dir, rel_dir, mask_base + ".png"),
        os.path.join(mask_dir, rel_dir, image_name),
        os.path.join(mask_dir, "1", mask_base + ".png"),
        os.path.join(mask_dir, "1", image_name),
        os.path.join(mask_dir, mask_base + ".png"),
        os.path.join(mask_dir, image_name),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return ""


def run_attack(
    image_dir: str = DEFAULT_IMAGE_DIR,
    mask_dir: str = DEFAULT_MASK_DIR,
    result_dir: str = DEFAULT_RESULT_DIR,
    num_steps: int = DEFAULT_NUM_STEPS,
    epsilon: float = DEFAULT_EPSILON,
    alpha: float = DEFAULT_ALPHA,
):
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        return
    os.makedirs(result_dir, exist_ok=True)

    print("Loading ensemble models... This will take a while and use significant VRAM.")
    loaded_extractors = []
    try:
        for model_name, config in ENSEMBLE_CONFIG.items():
            print(f" - Loading {model_name}...")
            extractor_class = config["class"]
            extractor = extractor_class(model_name=model_name)
            extractor.manual_config = config
            loaded_extractors.append(extractor)
    except Exception as e:
        print(f"Failed to load models. Error: {e}")
        del loaded_extractors
        torch.cuda.empty_cache()
        return

    print(f"\nSuccessfully loaded {len(loaded_extractors)} models into memory.")

    # Collect all images recursively inside image_dir
    image_entries = []
    for root, _, files in os.walk(image_dir):
        rel_dir = os.path.relpath(root, image_dir)
        if rel_dir == ".":
            rel_dir = ""
        for f in sorted(files):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                image_entries.append((rel_dir, f))

    if not image_entries:
        print(f"No images found in '{image_dir}'. Exiting.")
        del loaded_extractors
        torch.cuda.empty_cache()
        return

    print(f"\nFound {len(image_entries)} images to process. Starting attack loop...")
    for rel_dir, image_name in tqdm(image_entries, desc="Processing Images"):
        print(f"\n--- Attacking {os.path.join(rel_dir, image_name)} ---")
        try:
            image_path = os.path.join(image_dir, rel_dir, image_name)
            mask_path = _find_mask(mask_dir, rel_dir, image_name)
            if not mask_path:
                print(f"Warning: Mask not found for {image_name}. Skipping.")
                continue

            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path)

            attacker = AttentionPerturber(
                extractors=loaded_extractors,
                image=image,
                mask=mask,
            )

            perturbed_image = attacker.perturb(
                num_steps=num_steps, alpha=alpha, epsilon=epsilon
            )

            out_subdir = os.path.join(result_dir, rel_dir) if rel_dir else result_dir
            os.makedirs(out_subdir, exist_ok=True)
            output_path = os.path.join(out_subdir, image_name)
            perturbed_image.save(output_path)
            print(f"Successfully saved attacked image to {output_path}")

        except Exception as e:
            print(f"An error occurred while processing {image_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n--- All images processed. Cleaning up models from memory. ---")
    del loaded_extractors
    torch.cuda.empty_cache()
    print("Cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Attention-shift background perturbation.")
    parser.add_argument("--image_dir", type=str, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--mask_dir", type=str, default=DEFAULT_MASK_DIR)
    parser.add_argument("--result_dir", type=str, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--num_steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()

    run_attack(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        result_dir=args.result_dir,
        num_steps=args.num_steps,
        epsilon=args.epsilon,
        alpha=args.alpha,
    )
