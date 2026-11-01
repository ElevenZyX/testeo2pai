"""
prepare_split.py
----------------
Prepara automaticamente los datos de una modalidad (ej. Mammography o MRI) a partir de Sample_Images/<SPLIT>:
1. Copia las imagenes limpias a work_<SPLIT>/clean_images/1/
2. Genera las mascaras binarias (foreground=blanco, background=negro) en work_<SPLIT>/masks/1/
3. Si no existe data_<SPLIT>.csv, genera los hallazgos clinicos base (findings) usando Qwen/Qwen2.5-VL-7B-Instruct
   y guarda data_<SPLIT>.csv con las columnas 'images' y 'findings'.
"""

import os
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm


def generate_foreground_mask(img: Image.Image) -> Image.Image:
    """
    Segmenta automaticamente la region anatomica (foreground=255, background=0)
    para mamografias y resonancias magneticas (MRI).
    """
    gray = img.convert("L")
    # Suavizado para eliminar ruido de fondo
    blurred = gray.filter(ImageFilter.GaussianBlur(radius=3))
    arr = np.array(blurred, dtype=np.float32)

    # Umbral adaptativo (Otsu simplificado / percentil)
    p_low, p_high = np.percentile(arr, 5), np.percentile(arr, 95)
    threshold = max(20.0, p_low + 0.18 * (p_high - p_low))
    mask_np = (arr > threshold).astype(np.uint8) * 255

    # Asegurar que al menos un 10% sea background y un 10% sea foreground
    fg_ratio = (mask_np > 128).mean()
    if fg_ratio > 0.90 or fg_ratio < 0.05:
        median_val = np.median(arr)
        mask_np = (arr > median_val).astype(np.uint8) * 255

    mask_img = Image.fromarray(mask_np, mode="L")
    # Operaciones morfologicas basicas (cierre + apertura) con MaxFilter/MinFilter
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
    mask_img = mask_img.filter(ImageFilter.MinFilter(5))
    return mask_img


def prepare_modality(
    split: str,
    repo_dir: str,
    work_dir: str,
    output_csv: str,
    vlm_model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
):
    sample_dir = os.path.join(repo_dir, "Sample_Images", split)
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"No se encontro la carpeta de imagenes: {sample_dir}")

    clean_dir = os.path.join(work_dir, "clean_images", "1")
    mask_dir = os.path.join(work_dir, "masks", "1")
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    image_files = sorted(
        [
            f
            for f in os.listdir(sample_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    if not image_files:
        raise RuntimeError(f"No hay imagenes en {sample_dir}")

    clean_paths = []
    print(f"[Prepare {split}] Copiando {len(image_files)} imagenes y generando mascaras...")
    for fname in image_files:
        src_path = os.path.join(sample_dir, fname)
        dst_clean = os.path.join(clean_dir, fname)
        shutil.copy2(src_path, dst_clean)
        clean_paths.append(dst_clean)

        base_name, _ = os.path.splitext(fname)
        dst_mask = os.path.join(mask_dir, base_name + ".png")
        if not os.path.exists(dst_mask):
            img = Image.open(src_path).convert("RGB")
            mask = generate_foreground_mask(img)
            mask.save(dst_mask)

    # Si ya existe data_<SPLIT>.csv con findings validos, actualizar rutas de 'images' y terminar
    if os.path.exists(output_csv):
        df = pd.read_csv(output_csv)
        if "findings" in df.columns and df["findings"].notna().all() and len(df) == len(clean_paths):
            df["images"] = clean_paths
            df.to_csv(output_csv, index=False)
            print(f"[Prepare {split}] CSV existente verificado en: {output_csv}")
            return

    # Generar los hallazgos clinicos originales (y_clean) usando Qwen/Qwen2.5-VL-7B-Instruct
    print(f"[Prepare {split}] Generando findings iniciales con {vlm_model_name} ...")
    import sys
    sys.path.insert(0, os.path.join(repo_dir, "MedFocusLeak", "Modified_mattack"))
    from blackbox_text_generation import ImageDescriptionGenerator

    modality_prompt = (
        f"You are an expert radiologist. Provide a concise 3-4 sentence clinical findings report "
        f"for this {split} scan, describing tissue density, anatomical structures, and any potential lesions or abnormalities."
    )
    generator = ImageDescriptionGenerator(
        model_name=vlm_model_name,
        prompt=modality_prompt,
        max_new_tokens=200,
    )

    rows = []
    for img_path in tqdm(clean_paths, desc=f"Generando findings ({split})"):
        finding_text = generator.generate_description(img_path, custom_prompt=modality_prompt)
        rows.append(
            {
                "images": img_path,
                "modality": split,
                "findings": finding_text,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"[Prepare {split}] Creado {output_csv} con {len(df)} filas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara imagenes, mascaras y CSV inicial para una modalidad.")
    parser.add_argument("--split", type=str, required=True, help="Nombre de la modalidad en Sample_Images (ej. Mammography o MRI)")
    parser.add_argument("--repo_dir", type=str, default=".", help="Ruta raiz del repositorio")
    parser.add_argument("--work_dir", type=str, required=True, help="Directorio de trabajo work_<SPLIT>")
    parser.add_argument("--output_csv", type=str, required=True, help="Ruta donde guardar data_<SPLIT>.csv")
    parser.add_argument("--vlm_model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Modelo VLM para generar findings base")
    args = parser.parse_args()

    prepare_modality(
        split=args.split,
        repo_dir=os.path.abspath(args.repo_dir),
        work_dir=os.path.abspath(args.work_dir),
        output_csv=os.path.abspath(args.output_csv),
        vlm_model_name=args.vlm_model,
    )

