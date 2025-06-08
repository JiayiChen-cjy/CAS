import os
import json
import shutil
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter


def parse_arguments():
    """
    Parse command-line arguments for image classification.

    Optional arguments:
    --json_folder: Path to the folder containing JSON files (default: './json_folder')
    --image_folder: Path to the folder containing images (default: './image_folder')
    --output_base_folder: Base output path for classified images (default: './output')
    --top_k: Number of top secondary attributes to select per primary attribute (default: 10)
    """
    parser = argparse.ArgumentParser(description='Classify images based on JSON attributes.')

    parser.add_argument('--json_folder', type=str,
                        default='./json_folder',
                        help='Path to the folder containing JSON files.')
    parser.add_argument('--image_folder', type=str,
                        default='./image_folder',
                        help='Path to the folder containing original images.')
    parser.add_argument('--output_base_folder', type=str,
                        default='./output',
                        help='Base output path for classified images.')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of top secondary attributes to select per primary attribute.')

    return parser.parse_args()


def ensure_directory_exists(path):
    """Ensure the specified directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def main():
    args = parse_arguments()

    # Ensure directories exist
    ensure_directory_exists(args.output_base_folder)
    ensure_directory_exists(args.json_folder)
    ensure_directory_exists(args.image_folder)

    # Process each JSON file in the folder
    for json_file in os.listdir(args.json_folder):
        if json_file.endswith('.json'):
            json_path = os.path.join(args.json_folder, json_file)
            json_name = os.path.splitext(json_file)[0]
            output_folder = os.path.join(args.output_base_folder, json_name)
            ensure_directory_exists(output_folder)

            # Load JSON data
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Count secondary attributes
            secondary_counts = defaultdict(Counter)
            for image, attributes in data.items():
                for primary, secondary in attributes.items():
                    secondary_counts[primary][secondary] += 1

            # Select top K secondary attributes
            selected_images = defaultdict(list)
            for primary, counter in secondary_counts.items():
                top_secondaries = counter.most_common(args.top_k)
                for secondary, _ in top_secondaries:
                    for image, attributes in data.items():
                        if attributes.get(primary) == secondary:
                            selected_images[(primary, secondary)].append(image)

            # Calculate total steps for progress bar
            total_steps = sum(len(images) for images in selected_images.values())

            # Process images with progress tracking
            with tqdm(total=total_steps, desc=f"Processing {json_name}", unit="img") as pbar:
                for (primary, secondary), images in selected_images.items():
                    primary_dir = os.path.join(output_folder, primary)
                    secondary_dir = os.path.join(primary_dir, secondary)
                    ensure_directory_exists(primary_dir)
                    ensure_directory_exists(secondary_dir)

                    for image in images:
                        src = os.path.join(args.image_folder, image)
                        dst = os.path.join(secondary_dir, image)
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                        pbar.update(1)

    print("Image classification completed!")


if __name__ == "__main__":
    main()