import json
import os
import torch
import open_clip
from tqdm import tqdm
import argparse


def parse_arguments():
    """
    Parse command-line arguments.

    Optional arguments:
    --image_features: Path to the image features JSON file (default: '.\image_features.json')
    --dictionary_folder: Path to the dictionary JSON folder (default: '.\dictionary_split')
    --output_path: Path to the output JSON file (default: '.\closest_features.json')
    --backbone: CLIP model backbone (default: 'ViT-B-32')
    --pretrained_path: Path to the pretrained weights (default: '.\open_clip_pytorch_model.bin')
    --device: Computing device (default: auto select between cuda or cpu)
    """
    parser = argparse.ArgumentParser(description='Match images with visual concepts.')

    parser.add_argument('--image_features', type=str,
                        default=r'.\image_features.json',
                        help='Path to the image features JSON file.')
    parser.add_argument('--dictionary_folder', type=str,
                        default=r'.\dictionary_split',
                        help='Path to the folder containing dictionary JSON files.')
    parser.add_argument('--output_path', type=str,
                        default=r'.\closest_features.json',
                        help='Path to save the output JSON file.')
    parser.add_argument('--backbone', type=str, default='ViT-B-32',
                        help='CLIP model backbone (e.g., ViT-B-32, RN50, etc.)')
    parser.add_argument('--pretrained_path', type=str,
                        default=r'。\open_clip_pytorch_model.bin',
                        help='Path to the pretrained weights file.')
    parser.add_argument('--device', type=str, default=None,
                        help='Computing device (cuda/cpu); defaults to auto selection.')

    return parser.parse_args()


def setup_clip_model(backbone, pretrained_path, device=None):
    """Set up the CLIP model and preprocessing functions."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: backbone={backbone}, weights={pretrained_path}")

    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=False)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model = model.to(device).eval()  # Set model to evaluation mode

    return model, preprocess, device


def ensure_directory_exists(file_path):
    """Ensure the directory for a file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def main():
    args = parse_arguments()

    # Set up model (currently not used in computation, kept for potential future use)
    _, _, device = setup_clip_model(args.backbone, args.pretrained_path, args.device)

    # Load image features
    print(f"Loading image features from: {args.image_features}")
    with open(args.image_features, 'r', encoding='utf-8') as f:
        image_features_dict = json.load(f)

    # Load all dictionary JSON files
    print(f"Loading dictionary files from: {args.dictionary_folder}")
    json_files = [f for f in os.listdir(args.dictionary_folder) if f.endswith('.json')]

    if not json_files:
        print(f"Error: No JSON files found in {args.dictionary_folder}")
        return

    # Initialize the result dictionary
    result_dict = {image_name: {} for image_name in image_features_dict.keys()}

    # Process all dictionary files
    for json_file in tqdm(json_files, desc="Processing dictionary files"):
        json_file_path = os.path.join(args.dictionary_folder, json_file)

        with open(json_file_path, 'r', encoding='utf-8') as f:
            secondary_features_dict = json.load(f)

        # Find the most similar concept in the current dictionary for each image
        for image_name, image_features in tqdm(image_features_dict.items(),
                                               desc=f"Processing {json_file}", leave=False):
            image_features_tensor = torch.tensor(image_features).to(device)
            max_similarity = -1
            closest_secondary_attribute = None

            # Iterate through all concept features
            for secondary_attribute, secondary_features in secondary_features_dict.items():
                secondary_features_tensor = torch.tensor(secondary_features).to(device)
                similarity = torch.cosine_similarity(
                    image_features_tensor, secondary_features_tensor, dim=0
                ).mean().item()

                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_secondary_attribute = secondary_attribute

            # Save the most similar concept
            if closest_secondary_attribute:
                key = os.path.splitext(json_file)[0]
                result_dict[image_name][key] = closest_secondary_attribute

    # Ensure output directory exists
    ensure_directory_exists(args.output_path)

    # Save results
    print(f"Saving results to: {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=4)

    print(f"Matching complete! Processed {len(image_features_dict)} images and {len(json_files)} dictionary files.")


if __name__ == "__main__":
    main()
