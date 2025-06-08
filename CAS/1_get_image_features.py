import os
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import open_clip
import json
import argparse


def parse_arguments():
    """
    Parse command line arguments for image feature extraction

    Optional Parameters:
    --input_folder: Path to image directory (default: './test')
    --output_file: Output JSON file path (default: './image_features.json')
    --backbone: CLIP model backbone (default: 'ViT-B-32')
    --pretrained_path: Pretrained weights path (default: './open_clip_pytorch_model.bin')
    --device: Computation device (default: auto-select)
    --desc: Progress bar description (default: 'Processing images')
    --extensions: Supported image extensions (default: '.jpg,.jpeg,.png')
    """
    parser = argparse.ArgumentParser(description='Extract image features using CLIP models')

    # Add optional parameters with proper default values
    parser.add_argument('--input_folder', type=str,
                        default=r'.\test',
                        help='Path to directory containing images')
    parser.add_argument('--output_file', type=str,
                        default=r'.\image_features.json',
                        help='Output JSON file path for features')
    parser.add_argument('--backbone', type=str,
                        default='ViT-B-32',
                        help='CLIP model backbone architecture (e.g., ViT-B-32, RN50)')
    parser.add_argument('--pretrained_path', type=str,
                        default=r'.\open_clip_pytorch_model.bin',
                        help='Path to pretrained weights file')
    parser.add_argument('--device', type=str,
                        default=None,
                        help='Computation device (cuda or cpu), auto-selects if None')
    parser.add_argument('--desc', type=str,
                        default='Processing images',
                        help='Progress bar description text')
    parser.add_argument('--extensions', type=str,
                        default='.jpg,.jpeg,.png',
                        help='Supported image file extensions (comma-separated)')

    return parser.parse_args()


def extract_image_features(image_path, model, preprocess, device):
    """Extract features for a single image"""
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image_input)

    return image_features.cpu().numpy()


def process_image_folder(folder_path, model, preprocess, device,
                         desc='Processing images', extensions=('.jpg', '.jpeg', '.png')):
    """Process all images in a folder recursively"""
    image_features_dict = {}
    all_image_files = []

    # Collect all image files
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if any(file_name.lower().endswith(ext) for ext in extensions):
                all_image_files.append(os.path.join(root, file_name))

    # Process each image
    for image_path in tqdm(all_image_files, desc=desc):
        try:
            features = extract_image_features(image_path, model, preprocess, device)
            image_name = os.path.basename(image_path)
            image_features_dict[image_name] = features.tolist()
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    return image_features_dict


if __name__ == "__main__":
    args = parse_arguments()

    # Set device with proper default
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process extensions
    extensions = tuple(ext.strip().lower() for ext in args.extensions.split(','))
    print(f"Supported extensions: {', '.join(extensions)}")

    # Load model
    print(f"Loading model: backbone={args.backbone}, weights={args.pretrained_path}")
    model, _, preprocess = open_clip.create_model_and_transforms(args.backbone, pretrained=False)
    model.load_state_dict(torch.load(args.pretrained_path, map_location=device))
    model = model.to(device).eval()

    # Extract features
    print(f"Processing folder: {args.input_folder}")
    features_dict = process_image_folder(
        folder_path=args.input_folder,
        model=model,
        preprocess=preprocess,
        device=device,
        desc=args.desc,
        extensions=extensions
    )

    # Save results
    print(f"Saving features to: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(features_dict, f, ensure_ascii=False, indent=4)

    print(f"Done! Processed {len(features_dict)} images")