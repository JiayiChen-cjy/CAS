import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import json
from tqdm import tqdm
import argparse


def parse_arguments():
    """
    Parse command-line arguments.

    Arguments:
    --model_path: Path to trained ResNet50 model (required)
    --parent_folder: Root folder containing subfolders to process (required)
    --output_directory: Directory to save accuracy results (required)
    --class_mapping: Path to class-to-index mapping JSON file (required)
    --device: Computation device (auto-selects CUDA if available by default)
    """
    parser = argparse.ArgumentParser(description='Evaluate image classification accuracy.')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained ResNet50 model (.pth file)')
    parser.add_argument('--parent_folder', type=str, required=True,
                        help='Root directory containing class subfolders')
    parser.add_argument('--output_directory', type=str, required=True,
                        help='Directory to save accuracy JSON files')
    parser.add_argument('--class_mapping', type=str, required=True,
                        help='Path to class-to-index mapping JSON file')
    parser.add_argument('--device', type=str, default=None,
                        help='Computation device (cuda/cpu), auto-selects by default')

    return parser.parse_args()


def load_model(model_path, device=None):
    """Load trained ResNet50 model"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 100)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device).eval()
    print(f"Model loaded on {device} device")
    return model


def predict_image(image_path, model, device):
    """Predict class for a single image"""
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()


def evaluate_accuracy_recursive(folder_path, model, class_mapping, device):
    """Recursively calculate classification accuracy"""
    correct = 0
    total = 0

    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        if os.path.isdir(item_path):
            # Process subdirectory
            sub_accuracy = evaluate_accuracy_recursive(item_path, model, class_mapping, device)
            correct += sub_accuracy
            total += 1
        elif item.lower().endswith(('.jpeg', '.jpg', '.png')):
            # Process image file
            class_name = item.split('_')[0]
            predicted_class = predict_image(item_path, model, device)
            actual_class = class_mapping.get(class_name, -1)

            if predicted_class == actual_class:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0


def process_folders(parent_folder, model, class_mapping, device, output_json_path):
    """Process folders and save accuracies to JSON"""
    all_accuracies = {}
    folders = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]

    with tqdm(total=len(folders), desc='Processing folders') as pbar:
        for folder_name in folders:
            folder_path = os.path.join(parent_folder, folder_name)
            accuracy = evaluate_accuracy_recursive(folder_path, model, class_mapping, device)
            parent_name = os.path.basename(parent_folder)

            if parent_name not in all_accuracies:
                all_accuracies[parent_name] = {}

            all_accuracies[parent_name][folder_name] = f"{accuracy * 100:.2f}%"
            pbar.update(1)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(all_accuracies, f, indent=4)

    print(f"Saved accuracies to {output_json_path}")


def main():
    args = parse_arguments()

    # Auto-select device if not specified
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and class mapping
    model = load_model(args.model_path, device)
    with open(args.class_mapping, 'r') as f:
        class_mapping = json.load(f)

    # Ensure output directory exists
    os.makedirs(args.output_directory, exist_ok=True)

    # Process each subfolder in parent directory
    for subfolder in os.listdir(args.parent_folder):
        subfolder_path = os.path.join(args.parent_folder, subfolder)
        if os.path.isdir(subfolder_path):
            output_path = os.path.join(args.output_directory, f"{subfolder}.json")
            print(f"\nProcessing: {subfolder}")
            process_folders(subfolder_path, model, class_mapping, device, output_path)

    print("\nEvaluation completed!")


# Image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

if __name__ == "__main__":
    main()