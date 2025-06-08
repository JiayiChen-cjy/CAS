import os
import torch
import numpy as np
from tqdm import tqdm
import open_clip
import json
import argparse


def parse_arguments():
    """
    Parse command-line arguments.

    Optional arguments:
    --npy_file: Path to the input .npy file (default: './visual_concepts_phrased.npy')
    --image_features: Path to the image features JSON file (default: './image_features.json')
    --text_features_output: Output path for text features (default: './text_features.json')
    --similarity_output: Output path for similarity results (default: './dictionary.json')
    --backbone: CLIP model backbone (default: 'ViT-B-32')
    --pretrained_path: Path to pretrained weights (default: './open_clip_pytorch_model.bin')
    --device: Computation device (default: auto-select cuda or cpu)
    """
    parser = argparse.ArgumentParser(description='Build a text-image feature dictionary.')

    parser.add_argument('--npy_file', type=str,
                        default=r'.\visual_concepts_phrased.npy',
                        help='Path to the .npy file containing visual concepts.')
    parser.add_argument('--image_features', type=str,
                        default=r'.\image_features.json',
                        help='Path to the image features JSON file.')
    parser.add_argument('--text_features_output', type=str,
                        default=r'.\text_features.json',
                        help='Output path for text features.')
    parser.add_argument('--similarity_output', type=str,
                        default=r'.\dictionary.json',
                        help='Output path for similarity dictionary.')
    parser.add_argument('--backbone', type=str, default='ViT-B-32',
                        help='CLIP model backbone (e.g., ViT-B-32, RN50, etc.)')
    parser.add_argument('--pretrained_path', type=str,
                        default=r'.\open_clip_pytorch_model.bin',
                        help='Path to the pretrained weights file.')
    parser.add_argument('--device', type=str, default=None,
                        help='Computation device (cuda/cpu), auto-selected by default.')

    return parser.parse_args()


def setup_clip_model(backbone, pretrained_path, device=None):
    """Set up CLIP model and preprocessing."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    print(f"Loading model: backbone={backbone}, weights={pretrained_path}")

    model, _, preprocess = open_clip.create_model_and_transforms(backbone, pretrained=False)
    model.load_state_dict(torch.load(pretrained_path, map_location=device))
    model = model.to(device).eval()  # Set to evaluation mode

    return model, preprocess, device


def extract_text_features(text, model, device):
    """Extract text features using CLIP model."""
    with torch.no_grad():
        text_features = model.encode_text(open_clip.tokenize([text]).to(device))
    return text_features.cpu().numpy()


def process_npy_file(npy_file_path, model, device):
    """Process .npy file and extract text features."""
    dictionary = np.load(npy_file_path, allow_pickle=True).item()
    text_features_dict = {}

    total_texts = sum(len(secondary_list) for secondary_list in dictionary.values())

    with tqdm(total=total_texts, desc="Processing text features") as pbar:
        for primary_key, secondary_list in dictionary.items():
            for text in secondary_list:
                text_features = extract_text_features(text, model, device)
                text_features_dict[text] = text_features.tolist()
                pbar.update(1)

    return text_features_dict


def calculate_similarity(text_features_dict, image_features_dict, device):
    """Compute cosine similarity between text and image features."""
    similarity_dict = {}

    with tqdm(total=len(text_features_dict), desc="Computing similarity") as pbar:
        for text, text_features in text_features_dict.items():
            text_features_tensor = torch.tensor(text_features).squeeze().to(device)
            max_similarity = -np.inf
            best_image_features = None

            for image_name, image_features in image_features_dict.items():
                image_features_tensor = torch.tensor(image_features).squeeze().to(device)
                similarity = torch.cosine_similarity(text_features_tensor, image_features_tensor, dim=0).mean().item()

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_image_features = image_features

            similarity_dict[text] = best_image_features if best_image_features is not None else []

            pbar.update(1)

    return similarity_dict


def ensure_directory_exists(file_path):
    """Ensure the directory of the file path exists."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def save_json(data, file_path, desc="Saving JSON file"):
    """Save data to a JSON file with progress indication."""
    with tqdm(total=1, desc=desc) as pbar:
        ensure_directory_exists(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        pbar.update(1)


def main():
    args = parse_arguments()

    # Set up model
    model, _, device = setup_clip_model(args.backbone, args.pretrained_path, args.device)

    # Load image features
    print(f"Loading image features: {args.image_features}")
    with open(args.image_features, 'r', encoding='utf-8') as f:
        image_features_dict = json.load(f)

    # Extract text features
    print(f"Processing text file: {args.npy_file}")
    text_features_dict = process_npy_file(args.npy_file, model, device)

    # Save text features
    save_json(text_features_dict, args.text_features_output, "Saving text features")

    # Calculate similarity
    similarity_dict = calculate_similarity(text_features_dict, image_features_dict, device)

    # Save similarity results
    save_json(similarity_dict, args.similarity_output, "Saving similarity dictionary")

    print(f"Done! Processed {len(text_features_dict)} text concepts.")


if __name__ == "__main__":
    main()
