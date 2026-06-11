#!/usr/bin/env python3
"""
Prepare SYNTHIA dataset for ControlNet training.

This script converts SYNTHIA dataset structure to Cityscapes format:
- Remaps SYNTHIA class IDs (0-22) to Cityscapes trainIds (0-18)
- Generates captions using BLIP model
- Creates metadata.jsonl for dataset loader

Usage:
    python prepare_synthia.py \
        --synthia_root C:\\data\\raw\\synthia \
        --output_root C:\\data\\synthia_prepared \
        --device cuda
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration


# SYNTHIA to Cityscapes trainId mapping
# Based on: labels.py using the 19-class Cityscapes format
SYNTHIA_TO_CITYSCAPES_MAPPING = {
    0: (255, "void"),        # void → ignore
    1: (10, "sky"),          # sky
    2: (2, "building"),      # building
    3: (0, "road"),          # road
    4: (1, "sidewalk"),      # sidewalk
    5: (4, "fence"),         # fence
    6: (8, "vegetation"),    # vegetation
    7: (5, "pole"),          # pole
    8: (13, "car"),          # car
    9: (7, "traffic_sign"),  # traffic sign
    10: (11, "person"),      # pedestrian → person
    11: (18, "bicycle"),     # bicycle
    12: (17, "motorcycle"),  # motorcycle
    13: (255, "parking"),    # parking → ignore
    14: (255, "road_work"),  # road_work → ignore
    15: (6, "traffic_light"),# traffic light
    16: (9, "terrain"),      # terrain
    17: (12, "rider"),       # rider
    18: (14, "truck"),       # truck
    19: (15, "bus"),         # bus
    20: (16, "train"),       # train
    21: (3, "wall"),         # wall
    22: (255, "lanemarking"),# lanemarking → ignore
}

# Cityscapes 19 classes color palette (from labels.py)
CITYSCAPES_COLORS = {
    0: (128, 64, 128),      # road
    1: (244, 35, 232),      # sidewalk
    2: (70, 70, 70),        # building
    3: (102, 102, 156),     # wall
    4: (190, 153, 153),     # fence
    5: (153, 153, 153),     # pole
    6: (250, 170, 30),      # traffic light
    7: (220, 220, 0),       # traffic sign
    8: (107, 142, 35),      # vegetation
    9: (152, 251, 152),     # terrain
    10: (70, 130, 180),     # sky
    11: (220, 20, 60),      # person
    12: (255, 0, 0),        # rider
    13: (0, 0, 142),        # car
    14: (0, 0, 70),         # truck
    15: (0, 60, 100),       # bus
    16: (0, 80, 100),       # train
    17: (0, 0, 230),        # motorcycle
    18: (119, 11, 32),      # bicycle
    255: (0, 0, 0),         # ignore (void)
}


def load_blip_model(device: str):
    """Load BLIP model and processor.
    
    Args:
        device: torch device (cuda/cpu)
    
    Returns:
        Tuple of (processor, model)
    """
    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)
    model.eval()
    return processor, model


def generate_captions(
    image_paths: list,
    processor,
    model,
    device: str,
    batch_size: int = 20
) -> list:
    """Generate captions for a list of images using BLIP.
    
    Args:
        image_paths: List of image file paths
        processor: BLIP processor
        model: BLIP model
        device: torch device
        batch_size: Batch size for processing
    
    Returns:
        List of generated captions
    """
    captions = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Generating captions"):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load images
        try:
            images = [Image.open(p).convert("RGB") for p in batch_paths]
        except Exception as e:
            print(f"Error loading images: {e}")
            captions.extend(["a synthetic street scene"] * len(batch_paths))
            continue
        
        # Generate captions
        try:
            inputs = processor(
                images,
                return_tensors="pt"
            ).to(device, torch.float16 if "cuda" in device else torch.float32)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_length=50)
            
            batch_captions = processor.batch_decode(out, skip_special_tokens=True)
            captions.extend(batch_captions)
        except Exception as e:
            print(f"Error generating captions: {e}")
            captions.extend(["a synthetic street scene"] * len(batch_paths))
    
    return captions

def convert_labels_to_trainids(labels_path: str) -> np.ndarray:
    """Convert SYNTHIA LABELS PNG to Cityscapes trainIds.
    
    SYNTHIA stores class ID in channel 2 of a 3-channel uint16 PNG.
    Channel 0: unused (0)
    Channel 1: instance ID
    Channel 2: class ID (0-21)
    
    Args:
        labels_path: Path to SYNTHIA LABELS PNG file
    
    Returns:
        trainIds array (H, W) with values 0-18 and 255 for ignore
    """
    # Read SYNTHIA labels (uint16, 3 channels)
    labels_img = cv2.imread(labels_path, cv2.IMREAD_UNCHANGED)
    
    if labels_img is None:
        raise ValueError(f"Cannot read labels from {labels_path}")
    
    # Extract class ID from channel 2
    if len(labels_img.shape) == 3 and labels_img.shape[2] == 3:
        class_ids = labels_img[:, :, 2].astype(np.uint8)  # Channel 2 = class ID
    else:
        # Fallback for unexpected format
        class_ids = labels_img if len(labels_img.shape) == 2 else labels_img[:, :, 0]
        class_ids = class_ids.astype(np.uint8)
    
    # Create trainIds array
    trainids = np.full_like(class_ids, 255, dtype=np.uint8)
    
    # Map each SYNTHIA class to Cityscapes trainId
    for synthia_id, (cityscapes_trainid, _) in SYNTHIA_TO_CITYSCAPES_MAPPING.items():
        trainids[class_ids == synthia_id] = cityscapes_trainid
    
    return trainids


def create_color_visualization(trainids: np.ndarray) -> np.ndarray:
    """Create RGB visualization from trainIds.
    
    Args:
        trainids: trainIds array (H, W)
    
    Returns:
        RGB image (H, W, 3)
    """
    color_img = np.zeros((trainids.shape[0], trainids.shape[1], 3), dtype=np.uint8)
    
    for trainid, color in CITYSCAPES_COLORS.items():
        mask = trainids == trainid
        color_img[mask] = color
    
    return color_img


def prepare_synthia(
    synthia_root: str,
    output_root: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """Prepare SYNTHIA dataset for ControlNet training.
    
    Converts SYNTHIA format to Cityscapes format and generates captions.
    
    Args:
        synthia_root: Root directory of SYNTHIA dataset (C:\data\raw\synthia)
        output_root: Output directory for prepared dataset
        device: torch device for BLIP inference
    """
    
    # Setup paths
    synthia_root = Path(synthia_root)
    output_root = Path(output_root)
    
    rgb_dir = synthia_root / "RGB"
    gt_labels_dir = synthia_root / "GT" / "LABELS"
    gt_color_dir = synthia_root / "GT" / "COLOR"
    
    # Create output directories
    output_img_dir = output_root / "leftImg8bit" / "train"
    output_gt_dir = output_root / "gtFine" / "train"
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_gt_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"SYNTHIA root: {synthia_root}")
    print(f"Output root: {output_root}")
    print(f"Device: {device}")
    
    # Get list of images
    image_files = sorted([f for f in rgb_dir.glob("*.png") if f.is_file()])
    print(f"Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("No images found! Check your SYNTHIA path.")
        return
    
    # Load BLIP model
    processor, model = load_blip_model(device)
    
    # Prepare metadata
    metadata = []
    image_paths_for_captions = []
    image_filenames = []
    
    # Convert labels and collect images for captioning
    print("\nConverting annotations...")
    for rgb_file in tqdm(image_files, desc="Converting labels"):
        # Get corresponding label file
        label_file = gt_labels_dir / rgb_file.name
        color_file = gt_color_dir / rgb_file.name
        
        if not label_file.exists():
            print(f"Warning: Label not found for {rgb_file.name}")
            continue
        
        # Convert labels to trainIds
        try:
            trainids = convert_labels_to_trainids(str(label_file))
        except Exception as e:
            print(f"Error converting {rgb_file.name}: {e}")
            continue
        
        # Create RGB visualization
        try:
            color_img = create_color_visualization(trainids)
        except Exception as e:
            print(f"Error creating color visualization for {rgb_file.name}: {e}")
            color_img = None
        
        # Save trainIds
        trainid_output = output_gt_dir / rgb_file.name.replace(".png", "_trainIds.png")
        cv2.imwrite(str(trainid_output), trainids)
        
        # Save color visualization
        if color_img is not None:
            color_output = output_gt_dir / rgb_file.name.replace(".png", "_color.png")
            cv2.imwrite(str(color_output), cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR))
        
        # Copy RGB image
        rgb_output = output_img_dir / rgb_file.name
        image = cv2.imread(str(rgb_file))
        cv2.imwrite(str(rgb_output), image)
        
        # Track for captioning
        image_paths_for_captions.append(str(rgb_output))
        image_filenames.append(rgb_file.name)
    
    print(f"Converted {len(image_filenames)} images")
    
    # Generate captions
    print("\nGenerating captions with BLIP...")
    captions = generate_captions(
        image_paths_for_captions,
        processor,
        model,
        device,
        batch_size=20
    )
    
    # Create metadata.jsonl
    print("\nCreating metadata.jsonl...")
    metadata_file = output_root / "metadata.jsonl"
    
    with open(metadata_file, "w") as f:
        for filename, caption, image_path in zip(image_filenames, captions, image_paths_for_captions):
            # Store relative path from output root
            rel_image_path = Path(image_path).relative_to(output_root)
            rel_gt_path = Path(str(output_gt_dir / filename.replace(".png", "_trainIds.png"))).relative_to(output_root)
            
            entry = {
                "image": str(rel_image_path),
                "text": caption,
                "conditioning_image": str(rel_gt_path),
                "syn_or_real": 1  # 1 = synthetic (SYNTHIA)
            }
            f.write(json.dumps(entry) + "\n")
    
    print(f"\nMetadata saved to {metadata_file}")
    print(f"Total entries: {len(image_filenames)}")
    
    # Summary
    print("\n" + "="*60)
    print("SYNTHIA PREPARATION COMPLETE")
    print("="*60)
    print(f"Images: {output_img_dir}")
    print(f"Annotations: {output_gt_dir}")
    print(f"Metadata: {metadata_file}")
    print("\nYou can now use this dataset by:")
    print(f"  --train_data_dir {output_root}")


def main():
    parser = argparse.ArgumentParser(description="Prepare SYNTHIA dataset for ControlNet training")
    parser.add_argument(
        "--synthia_root",
        type=str,
        required=True,
        help="Path to SYNTHIA dataset root (C:\\data\\raw\\synthia)"
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Output directory for prepared dataset (default: C:\\data\\synthia_prepared)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="torch device (cuda/cpu). Default: auto-detect"
    )
    
    args = parser.parse_args()
    
    if args.output_root is None:
        # Default output location
        synthia_path = Path(args.synthia_root)
        args.output_root = str(synthia_path.parent / "synthia_prepared")
    
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    prepare_synthia(
        synthia_root=args.synthia_root,
        output_root=args.output_root,
        device=args.device
    )


if __name__ == "__main__":
    main()