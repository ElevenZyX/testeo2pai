"""
Pre-descarga en HF_HOME (~/hf_cache) todos los modelos requeridos por MedFocusLeak:
- 4 modelos CLIP surrogates
- Qwen/Qwen2.5-VL-7B-Instruct (modelo victima VLM a evaluar)
- Verifica o descarga Qwen/Qwen2.5-14B-Instruct (editor adversarial) y Qwen/Qwen3-32B (juez MTR)
"""

import os
from huggingface_hub import snapshot_download

MODELS_TO_DOWNLOAD = [
    # Surrogates CLIP (Pasos 3, 4 y 5)
    "openai/clip-vit-base-patch32",
    "openai/clip-vit-base-patch16",
    "openai/clip-vit-large-patch14-336",
    "laion/CLIP-ViT-G-14-laion2B-s12B-b42K",
    # VLM Victima a evaluar (Paso 6)
    "Qwen/Qwen2.5-VL-7B-Instruct",
    # LLMs del cluster (Paso 1 y Paso 7 - si ya estan en HF_HOME se verifican instantaneamente)
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen3-32B",
]


def main():
    token = os.environ.get("HF_TOKEN", None)
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/hf_cache"))
    print(f"Usando HF_HOME: {hf_home}")

    for repo_id in MODELS_TO_DOWNLOAD:
        print(f"\n---> Verificando / Descargando: {repo_id} ...")
        try:
            path = snapshot_download(
                repo_id=repo_id,
                token=token,
                resume_download=True,
            )
            print(f"[OK] {repo_id} listo en: {path}")
        except Exception as e:
            print(f"[WARN] Error al verificar/descargar {repo_id}: {e}")


if __name__ == "__main__":
    main()

