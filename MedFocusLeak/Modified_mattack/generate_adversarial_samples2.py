import os
import json
import hashlib
import random
import torchvision.transforms as transforms
import numpy as np
import torch
import torchvision
from PIL import Image
import os
import hydra
from config_schema import MainConfig
from functools import partial
from typing import List, Dict, Optional
from torch import nn
from tqdm import tqdm

from cropping import CustomBackgroundCrop

from surrogates import (
    ClipB16FeatureExtractor,
    ClipL336FeatureExtractor,
    ClipB32FeatureExtractor,
    ClipLaionFeatureExtractor,
    EnsembleFeatureLoss,
    EnsembleFeatureExtractor,
)

from utils import hash_training_config, ensure_dir

BACKBONE_MAP: Dict[str, type] = {
    "L336": ClipL336FeatureExtractor,
    "B16": ClipB16FeatureExtractor,
    "B32": ClipB32FeatureExtractor,
    "Laion": ClipLaionFeatureExtractor,
}


def get_models(cfg: MainConfig):
    """Get models based on configuration.

    Args:
        cfg: Configuration object containing model settings

    Returns:
        Tuple of (feature_extractor, list of models)

    Raises:
        ValueError: If ensemble=False but multiple backbones specified
    """
    if not cfg.model.ensemble and len(cfg.model.backbone) > 1:
        raise ValueError("When ensemble=False, only one backbone can be specified")

    models = []
    for backbone_name in cfg.model.backbone:
        if backbone_name not in BACKBONE_MAP:
            raise ValueError(
                f"Unknown backbone: {backbone_name}. Valid options are: {list(BACKBONE_MAP.keys())}"
            )
        model_class = BACKBONE_MAP[backbone_name]
        model = model_class().eval().to(cfg.model.device).requires_grad_(False)
        models.append(model)

    if cfg.model.ensemble:
        ensemble_extractor = EnsembleFeatureExtractor(models)
    else:
        ensemble_extractor = models[0]  # Use single model directly

    return ensemble_extractor, models


def get_ensemble_loss(cfg: MainConfig, models: List[nn.Module]):
    ensemble_loss = EnsembleFeatureLoss(models)
    return ensemble_loss


def set_environment(seed=2023):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Transform PIL.Image to PyTorch Tensor
def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


# Dataset with image paths
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)


@hydra.main(version_base=None, config_path="config", config_name="ensemble_3models")
def main(cfg: MainConfig):
    set_environment()

    ensemble_extractor, models = get_models(cfg)
    ensemble_loss = get_ensemble_loss(cfg, models)

    transform_fn = transforms.Compose(
        [
            transforms.Resize(
                cfg.model.input_res,
                interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(cfg.model.input_res),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.Lambda(lambda img: to_tensor(img)),
        ]
    )

    clean_data = ImageFolderWithPaths(cfg.data.cle_data_path, transform=transform_fn)
    target_data = ImageFolderWithPaths(cfg.data.tgt_data_path, transform=transform_fn)
    mask_data = ImageFolderWithPaths(cfg.data.mask_data_path, transform=transform_fn)

    data_loader_imagenet = torch.utils.data.DataLoader(
        clean_data, batch_size=cfg.data.batch_size, shuffle=False
    )
    data_loader_target = torch.utils.data.DataLoader(
        target_data, batch_size=cfg.data.batch_size, shuffle=False
    )
    data_loader_mask = torch.utils.data.DataLoader(
        mask_data, batch_size=cfg.data.batch_size, shuffle=False
    )

    print("Using source crop:", cfg.model.use_source_crop)
    print("Using target crop:", cfg.model.use_target_crop)

    source_crop = (
        CustomBackgroundCrop(k=cfg.model.k, non_overlapping=cfg.model.non_overlap, target_size=cfg.model.input_res)
        if cfg.model.use_source_crop
        else None
    )
    target_crop = (torch.nn.Identity())
    
    for i, ((image_org, _, path_org), (image_tgt, _, path_tgt), (image_mask, _, path_mask)) in enumerate(
        zip(data_loader_imagenet, data_loader_target, data_loader_mask)
    ):
        if cfg.data.batch_size * (i + 1) > cfg.data.num_samples:
            break

        print(f"\nProcessing image {i+1}/{cfg.data.num_samples//cfg.data.batch_size}")

        attack_imgpair(
            cfg=cfg,
            ensemble_extractor=ensemble_extractor,
            ensemble_loss=ensemble_loss,
            source_crop=source_crop,
            img_index=i,
            image_org=image_org,
            path_org=path_org,
            image_tgt=image_tgt,
            target_crop=target_crop,
            image_mask=image_mask
        )


def attack_imgpair(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[CustomBackgroundCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    path_org: List[str],
    image_tgt: torch.Tensor,
    image_mask: torch.Tensor
):
    image_org, image_tgt, image_mask = image_org.to(cfg.model.device), image_tgt.to(cfg.model.device), image_mask.to(cfg.model.device)
    attack_type = cfg.attack
    attack_fn = {
        "fgsm": fgsm_attack,
        "mifgsm": mifgsm_attack,
        "pgd": pgd_attack,
    }[attack_type]
    adv_image = attack_fn(
        cfg=cfg,
        ensemble_extractor=ensemble_extractor,
        ensemble_loss=ensemble_loss,
        source_crop=source_crop,
        target_crop=target_crop,
        img_index=img_index,
        image_org=image_org,
        image_tgt=image_tgt,
        image_mask=image_mask
    )

    # Get config hash for output directory
    config_hash = hash_training_config(cfg)

    # Save images
    for path_idx in range(len(path_org)):
        folder, name = (
            path_org[path_idx].split("/")[-2],
            path_org[path_idx].split("/")[-1],
        )
        # Use config hash in output path
        folder_to_save = os.path.join(cfg.data.output, "img", config_hash, folder)
        ensure_dir(folder_to_save)

        base_name, _ = os.path.splitext(name)
        torchvision.utils.save_image(
            adv_image[path_idx], os.path.join(folder_to_save, base_name + ".png")
        )


def fgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[CustomBackgroundCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
    image_mask: torch.Tensor
):
    """
    Perform FGSM attack on the image to generate adversarial examples.
    """
    # Initialize perturbation
    delta = torch.zeros_like(image_org, requires_grad=True)

    # Progress bar for optimization
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        # Forward pass
        adv_image = image_org + delta
        adv_features = ensemble_extractor(adv_image)

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)
        metrics["global_similarity"] = global_sim.item()

        if cfg.model.use_source_crop and source_crop is not None:
            # Apply background crop to each image in the batch
            batch_size = adv_image.shape[0]
            local_cropped_batch = []
            
            for batch_idx in range(batch_size):
                local_cropped = source_crop(adv_image[batch_idx], image_mask[batch_idx])
                local_cropped_batch.append(local_cropped)
            
            # Stack the cropped images
            local_cropped = torch.stack(local_cropped_batch)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = local_sim
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = global_sim

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        # Update delta using FGSM
        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(grad),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    # Create final adversarial image
    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


def mifgsm_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[CustomBackgroundCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
    image_mask: torch.Tensor
):
    """
    Perform MI-FGSM attack on the image to generate adversarial examples.
    """
    # Initialize perturbation and momentum
    delta = torch.zeros_like(image_org, requires_grad=True)
    momentum = torch.zeros_like(image_org, requires_grad=False)

    # Progress bar for optimization
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        # Forward pass
        adv_image = image_org + delta
        adv_features = ensemble_extractor(adv_image)

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)
        metrics["global_similarity"] = global_sim.item()

        if cfg.model.use_source_crop and source_crop is not None:
            # Apply background crop to each image in the batch
            batch_size = adv_image.shape[0]
            local_cropped_batch = []
            
            for batch_idx in range(batch_size):
                local_cropped = source_crop(adv_image[batch_idx], image_mask[batch_idx])
                local_cropped_batch.append(local_cropped)
            
            # Stack the cropped images
            local_cropped = torch.stack(local_cropped_batch)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = local_sim
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = global_sim

        grad = torch.autograd.grad(loss, delta, create_graph=False)[0]

        # MI-FGSM update
        momentum = momentum * 0.9 + grad
        delta.data = torch.clamp(
            delta + cfg.optim.alpha * torch.sign(momentum),
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    # Create final adversarial image
    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


def pgd_attack(
    cfg: MainConfig,
    ensemble_extractor: nn.Module,
    ensemble_loss: nn.Module,
    source_crop: Optional[CustomBackgroundCrop],
    target_crop: Optional[transforms.RandomResizedCrop],
    img_index: int,
    image_org: torch.Tensor,
    image_tgt: torch.Tensor,
    image_mask: torch.Tensor
):
    """
    Perform PGD attack on the image to generate adversarial examples.
    """
    # Initialize perturbation
    delta = torch.zeros_like(image_org, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=cfg.optim.alpha)

    # Progress bar for optimization
    pbar = tqdm(range(cfg.optim.steps), desc=f"Attack progress")

    # Main optimization loop
    for epoch in pbar:

        with torch.no_grad():
            ensemble_loss.set_ground_truth(target_crop(image_tgt))

        # Forward pass
        adv_image = image_org + delta
        adv_features = ensemble_extractor(adv_image)

        # Calculate metrics
        metrics = {
            "max_delta": torch.max(torch.abs(delta)).item(),
            "mean_delta": torch.mean(torch.abs(delta)).item(),
        }

        # Calculate loss based on configuration
        global_sim = ensemble_loss(adv_features)
        metrics["global_similarity"] = global_sim.item()

        if cfg.model.use_source_crop and source_crop is not None:
            # Apply background crop to each image in the batch
            batch_size = adv_image.shape[0]
            local_cropped_batch = []
            
            for batch_idx in range(batch_size):
                local_cropped = source_crop(adv_image[batch_idx], image_mask[batch_idx])
                local_cropped_batch.append(local_cropped)
            
            # Stack the cropped images
            local_cropped = torch.stack(local_cropped_batch)
            local_features = ensemble_extractor(local_cropped)
            local_sim = ensemble_loss(local_features)
            loss = -local_sim  # Negative because we want to maximize
            metrics["local_similarity"] = local_sim.item()
        else:
            # Otherwise use global similarity as loss
            loss = -global_sim

        optimizer.zero_grad()
        loss.backward()

        # PGD update
        optimizer.step()
        delta.data = torch.clamp(
            delta,
            min=-cfg.optim.epsilon,
            max=cfg.optim.epsilon,
        )

    # Create final adversarial image
    adv_image = image_org + delta
    adv_image = torch.clamp(adv_image / 255.0, 0.0, 1.0)

    return adv_image


if __name__ == "__main__":
    main()