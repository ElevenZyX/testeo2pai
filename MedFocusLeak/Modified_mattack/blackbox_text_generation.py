import os
import json
import shutil
import pandas as pd
from PIL import Image
from typing import List, Tuple, Optional
import hydra
import torch
from tqdm import tqdm

from config_schema import MainConfig
from utils import (
    hash_training_config,
    setup_wandb,
    log_wandb,
    finish_wandb,
    ensure_dir,
    get_output_paths,
    safe_model_tag,
)

VALID_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".JPEG"]


class ImageDescriptionGenerator:
    """
    Local Vision-Language Model evaluator using HuggingFace Transformers.
    Default target model: Qwen/Qwen2.5-VL-7B-Instruct (no external API required).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        prompt: str = "Describe the clinical findings and abnormalities in this medical scan.",
        max_new_tokens: int = 256,
    ):
        self.model_name = model_name
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

        print(f"[VLM Evaluator] Loading local VLM: {model_name} ...")
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLMClass
        except ImportError:
            from transformers import AutoModelForVision2Seq as VLMClass

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        self.model = VLMClass.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"[VLM Evaluator] Model {model_name} loaded successfully.")

    def generate_description(self, image_path: str, custom_prompt: Optional[str] = None) -> str:
        prompt_text = custom_prompt or self.prompt
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(
            text=[text_input],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        # Collapse newlines for single-line table compatibility while keeping full text
        return " ".join(output_text.strip().splitlines())


def save_descriptions(descriptions: List[Tuple[str, str]], output_file: str):
    """Save image descriptions to file."""
    ensure_dir(os.path.dirname(output_file))
    with open(output_file, "w", encoding="utf-8") as f:
        for filename, desc in descriptions:
            f.write(f"{filename}: {desc}\n")


def _find_matching_image(base_dir: str, subfolder: str, filename_base: str) -> Optional[str]:
    """Find matching clean or target image across valid extensions and candidate subfolders."""
    candidate_folders = [subfolder, "1", ""]
    for folder in candidate_folders:
        for ext in VALID_IMAGE_EXTENSIONS:
            cand = os.path.join(base_dir, folder, filename_base + ext) if folder else os.path.join(base_dir, filename_base + ext)
            if os.path.exists(cand):
                return cand
    return None


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    setup_wandb(cfg, tags=["vlm_evaluation"])

    config_hash = hash_training_config(cfg)
    print(f"Using training output for config hash: {config_hash}")

    paths = get_output_paths(cfg, config_hash)
    ensure_dir(paths["desc_output_dir"])

    model_tag = safe_model_tag(cfg.blackbox.model_name)
    allowed_modalities = (
        [m.lower() for m in cfg.data.modalities]
        if getattr(cfg.data, "modalities", None)
        else None
    )

    try:
        generator = ImageDescriptionGenerator(
            model_name=cfg.blackbox.model_name,
            prompt=getattr(
                cfg.blackbox,
                "prompt",
                "Describe the clinical findings and abnormalities in this medical scan.",
            ),
            max_new_tokens=getattr(cfg.blackbox, "max_new_tokens", 256),
        )

        tgt_descriptions = []
        adv_descriptions = []
        export_records = []

        # Create export bundle directory for easy download of images + text from cluster
        export_dir = os.path.join(cfg.data.output, "export_bundle", config_hash)
        export_clean_img_dir = os.path.join(export_dir, "clean_images")
        export_adv_img_dir = os.path.join(export_dir, "adversarial_images")
        ensure_dir(export_clean_img_dir)
        ensure_dir(export_adv_img_dir)

        print("Processing images with local VLM...")
        for root, _, files in os.walk(paths["output_dir"]):
            subfolder = os.path.basename(root)
            # If modalities filter is set and subfolder explicitly names a modality, filter it
            if (
                allowed_modalities
                and subfolder.lower() not in allowed_modalities
                and subfolder.lower() not in ["1", "nips17", os.path.basename(paths["output_dir"]).lower()]
            ):
                print(f"Skipping modality folder '{subfolder}' (allowed: {cfg.data.modalities})")
                continue

            for file in tqdm(sorted(files), desc=f"Folder: {subfolder}"):
                if any(file.lower().endswith(ext.lower()) for ext in VALID_IMAGE_EXTENSIONS):
                    try:
                        adv_path = os.path.join(root, file)
                        filename_base = os.path.splitext(os.path.basename(adv_path))[0]

                        # Prefer clean source image for y_clean (Paper Section A.6), fallback to target image
                        ref_path = _find_matching_image(cfg.data.cle_data_path, subfolder, filename_base)
                        if ref_path is None:
                            ref_path = _find_matching_image(cfg.data.tgt_data_path, subfolder, filename_base)

                        if ref_path is not None:
                            clean_desc = generator.generate_description(ref_path)
                            adv_desc = generator.generate_description(adv_path)

                            rel_key = f"{subfolder}/{file}" if subfolder != os.path.basename(paths["output_dir"]) else file
                            tgt_descriptions.append((rel_key, clean_desc))
                            adv_descriptions.append((rel_key, adv_desc))

                            # Copy images into export_bundle for easy download
                            ensure_dir(os.path.join(export_clean_img_dir, subfolder))
                            ensure_dir(os.path.join(export_adv_img_dir, subfolder))
                            exported_clean = os.path.join(export_clean_img_dir, subfolder, os.path.basename(ref_path))
                            exported_adv = os.path.join(export_adv_img_dir, subfolder, file)
                            shutil.copy2(ref_path, exported_clean)
                            shutil.copy2(adv_path, exported_adv)

                            export_records.append(
                                {
                                    "id": rel_key,
                                    "modality": subfolder,
                                    "clean_image_path": exported_clean,
                                    "adversarial_image_path": exported_adv,
                                    "clean_findings": clean_desc,
                                    "adversarial_findings": adv_desc,
                                }
                            )

                            log_wandb(
                                cfg,
                                {
                                    f"descriptions/{rel_key}/clean": clean_desc,
                                    f"descriptions/{rel_key}/adversarial": adv_desc,
                                },
                            )
                        else:
                            print(f"Reference image not found for {filename_base}, skipping.")

                    except Exception as e:
                        print(f"Error processing {file}: {e}")

        # Save descriptions in txt format for judge evaluation
        save_descriptions(
            tgt_descriptions,
            os.path.join(paths["desc_output_dir"], f"target_{model_tag}.txt"),
        )
        save_descriptions(
            adv_descriptions,
            os.path.join(paths["desc_output_dir"], f"adversarial_{model_tag}.txt"),
        )

        # Save downloadable CSV and JSON with image paths + generated texts
        if export_records:
            export_csv = os.path.join(export_dir, f"vlm_generations_{model_tag}.csv")
            export_json = os.path.join(export_dir, f"vlm_generations_{model_tag}.json")
            pd.DataFrame(export_records).to_csv(export_csv, index=False)
            with open(export_json, "w", encoding="utf-8") as f:
                json.dump(export_records, f, indent=2, ensure_ascii=False)
            print(f"Export bundle (images + text) saved to: {export_dir}")

        print(f"Descriptions saved to {paths['desc_output_dir']}")

    finally:
        finish_wandb(cfg)


if __name__ == "__main__":
    main()
