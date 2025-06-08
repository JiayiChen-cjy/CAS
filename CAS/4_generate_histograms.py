import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import argparse


def parse_arguments():
    """
    Parse command-line arguments

    Optional arguments:
    --json_folder: Path to the folder containing matching result JSON files (default: '.\closest_features_split')
    --output_dir: Path to the output directory (default: '.\Histogram')
    --min_count: Minimum number of occurrences for an attribute to be shown (default: 1)
    --max_bars: Maximum number of bars shown per chart (default: 20)
    --figsize: Figure size (width,height) (default: '10,5')
    --rotation: Rotation angle for x-axis labels (default: 45)
    """
    parser = argparse.ArgumentParser(description='Generate bar charts for attribute distribution')

    parser.add_argument('--json_folder', type=str,
                        default=r'.\closest_features_split',
                        help='Path to the folder containing matching result JSON files')
    parser.add_argument('--output_dir', type=str,
                        default=r'.\Histogram',
                        help='Output directory path')
    parser.add_argument('--min_count', type=int, default=1,
                        help='Minimum number of occurrences to display an attribute')
    parser.add_argument('--max_bars', type=int, default=20,
                        help='Maximum number of bars shown per chart')
    parser.add_argument('--figsize', type=str, default='10,5',
                        help='Figure size (width,height)')
    parser.add_argument('--rotation', type=int, default=45,
                        help='Rotation angle for x-axis labels')

    return parser.parse_args()


def ensure_directory_exists(directory):
    """Ensure the given directory exists; create if it doesn't."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def process_json_file(json_file_path, output_dir, min_count, max_bars, figsize, rotation):
    """Process a single JSON file and generate bar charts"""
    width, height = map(float, figsize.split(','))

    with open(json_file_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    attribute_count = defaultdict(lambda: defaultdict(int))

    # Count the occurrences of attributes
    for image_name, attributes in json_data.items():
        for primary_attribute, secondary_attribute in attributes.items():
            primary_attribute = primary_attribute.replace('.json', '')
            attribute_count[primary_attribute][secondary_attribute] += 1

    base_name = os.path.splitext(os.path.basename(json_file_path))[0]

    # Create a subdirectory for the current file's output
    file_output_dir = os.path.join(output_dir, base_name)
    ensure_directory_exists(file_output_dir)

    # Save counts to a JSON file
    output_json_path = os.path.join(file_output_dir, f'{base_name}.json')
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(attribute_count, f, ensure_ascii=False, indent=4)

    # Generate bar charts
    for primary_attribute, secondary_count in attribute_count.items():
        filtered_secondary_count = {k: v for k, v in secondary_count.items() if v >= min_count}
        sorted_secondary_count = dict(
            sorted(filtered_secondary_count.items(), key=lambda item: item[1], reverse=True)[:max_bars]
        )

        if sorted_secondary_count:
            fig, ax = plt.subplots(figsize=(width, height))
            bars = ax.bar(sorted_secondary_count.keys(), sorted_secondary_count.values())

            # Add text labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

            ax.set_xlabel('Secondary Attribute')
            ax.set_ylabel('Count')
            ax.set_title(
                f'Primary Attribute: {primary_attribute}\nOccurrence Distribution (min={min_count}, top={max_bars})')
            plt.xticks(rotation=rotation)
            plt.tight_layout()

            # Save chart
            img_name = f'{primary_attribute}.png'
            plt.savefig(os.path.join(file_output_dir, img_name), bbox_inches='tight')
            plt.close()

    return len(attribute_count)


def main():
    args = parse_arguments()

    ensure_directory_exists(args.output_dir)

    json_files = [f for f in os.listdir(args.json_folder) if f.endswith('.json')]

    if not json_files:
        print(f"Error: No JSON files found in {args.json_folder}")
        return

    total_charts = 0
    for json_file in tqdm(json_files, desc="Processing files"):
        json_file_path = os.path.join(args.json_folder, json_file)
        charts_created = process_json_file(
            json_file_path,
            args.output_dir,
            args.min_count,
            args.max_bars,
            args.figsize,
            args.rotation
        )
        total_charts += charts_created

    print(f"Processing complete! {len(json_files)} files processed, {total_charts} charts generated")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
