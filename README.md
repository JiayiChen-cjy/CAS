# Compositional Attribute Imbalance in Vision Datasets

> **Compositional Attribute Imbalance in Vision Datasets**  
> Yanbiao Ma*†, Jiayi Chen*, Andi Zhang, Wei Dai

# 📝 Abstract

Visual attribute imbalance is a common yet underexplored issue in image classification, significantly impacting model performance and generalization. In this work, we first define the firstlevel and second-level attributes of images and then introduce a CLIP-based framework to construct a visual attribute dictionary, enabling automatic evaluation of image attributes. By systematically analyzing both single-attribute imbalance and compositional attribute imbalance, we reveal how the rarity of attributes affects model performance. To tackle these challenges, we propose adjusting the sampling probability of samples based on the rarity of their compositional attributes. This strategy is further integrated with various data augmentation techniques (such as
CutMix, Fmix, and SaliencyMix) to enhance the model's ability to represent rare attributes. Extensive experiments on benchmark datasets demonstrate that our method effectively mitigates attribute imbalance, thereby improving the robustness and fairness of deep neural networks. Our research highlights the importance of modeling visual attribute distributions and provides a scalable solution for long-tail image classification tasks.

<!-- --- -->

<!-- ## 🔑 Key words

**Compositional Attribute Imbalance**, **Compositional Attribute Scarcity**, **Sampling Strategy**, **Data Augmentation** -->

---

<!-- # ⚙️ Engineering -->

# 🌐 Environment

** Download dependency**

```bash
conda create -n CAS python=3.9
conda activate CAS
pip install -r requirements.txt
```

---

# 🚀 Experiment

---

## 🗂️ 1. CLIP Image Feature Extraction

### Description

This script uses the Open-CLIP model to extract feature vectors from all images in a specified folder (recursively) and saves the results to a JSON file. The feature vectors can be used for tasks such as image retrieval, classification, similarity computation, etc.

### Optional Parameters

| Parameter           | Type   | Description                                                                     | Default Value                   |
| ------------------- | ------ | ------------------------------------------------------------------------------- | ------------------------------- |
| `--input_folder`    | string | Path to the folder containing images.                                           | `.\test`                        |
| `--output_file`     | string | Path to the output JSON file for saving features.                               | `.\image_features.json`         |
| `--backbone`        | string | CLIP model backbone (e.g., 'ViT-B-32', 'RN50').                                 | `'ViT-B-32'`                    |
| `--pretrained_path` | string | Path to the pretrained weights file.                                            | `.\open_clip_pytorch_model.bin` |
| `--device`          | string | Computation device ('cuda' or 'cpu'). If not specified, uses CUDA if available. | Auto-select                     |
| `--desc`            | string | Description text for the progress bar.                                          | `'Processing images'`           |
| `--extensions`      | string | Supported image file extensions (comma-separated, e.g., '.jpg,.jpeg,.png').     | `'.jpg,.jpeg,.png'`             |

### Run Script

```bash

python 1_get_image_features.py [optional parameters]
```

Examples:

```bash

# Run with default parameters
python 1_get_image_features.py

# Custom input and output paths
python 1_get_image_features.py --input_folder "path/to/your/images" --output_file "output/features.json"

# Use RN50 model with custom weights
python 1_get_image_features.py --backbone "RN50" --pretrained_path "path/to/RN50_weights.bin"

# Specify device as CPU and change progress bar description
python 1_get_image_features.py --device "cpu" --desc "Extracting features on CPU"

# Add support for additional image formats (e.g., .bmp and .tiff)
python 1_get_image_features.py --extensions ".jpg,.jpeg,.png,.bmp,.tiff"

```

### Output

- A JSON file containing feature vectors, structured as:

```json
{
"image1.jpg": [0.123, -0.456, 0.789],
"image2.png": [0.234, -0.567, 0.890]
}
```

- Console output showing progress and statistics:

```
Using device: cuda
Loading model: backbone=ViT-B-32, weights=.\open_clip_pytorch_model.bin
Processing folder: .\test
Processing images: 100%|██████████| 1000/1000 [05:23<00:00, 3.12 images/s]
Saving features to: .\image_features.json
Done! Processed 1000 images.
```

### Output File Description

| File Path             | Format | Description                                                                                                     |
| --------------------- | ------ | --------------------------------------------------------------------------------------------------------------- |
| `image_features.json` | JSON   | Contains feature vectors for all images, with filenames as keys and feature vectors (list of floats) as values. |

### Notes

1. **Model Compatibility**: Ensure that the weights file specified by `--pretrained_path` matches the model backbone specified by `--backbone`.

2. **Image Formats**: By default, the script supports JPG, JPEG, and PNG formats. Use the `--extensions` parameter to add others.

3. **Hardware Requirements**:

- GPU is recommended for large-scale image processing.

- CPU mode is suitable for small-scale testing.

4. **Error Handling**: The script skips images that cannot be processed and logs the error.

5. **Feature Dimension**: The output dimension depends on the backbone (e.g., ViT-B-32 outputs 512-dimensional vectors).

---

## 📚 2. Build Concept-Image Dictionary

### Description

This script processes visual concepts from a .npy file, extracts their text features using CLIP, and finds the best matching image features from a precomputed set. The output is a dictionary mapping text concepts to the most similar image features.

### Optional Parameters

| Parameter                | Type   | Description                                                        | Default Value                   |
| ------------------------ | ------ | ------------------------------------------------------------------ | ------------------------------- |
| `--npy_file`             | string | Path to .npy file containing visual concepts                       | `.\visual_concepts_phrased.npy` |
| `--image_features`       | string | Path to JSON file containing image features                        | `.\image_features.json`         |
| `--text_features_output` | string | Output path for text features JSON file                            | `.\text_features.json`          |
| `--similarity_output`    | string | Output path for similarity dictionary JSON file                    | `.\dictionary.json`             |
| `--backbone`             | string | CLIP model architecture (e.g., 'ViT-B-32', 'RN50')                 | `ViT-B-32`                      |
| `--pretrained_path`      | string | Path to pretrained weights file                                    | `.\open_clip_pytorch_model.bin` |
| `--device`               | string | Computation device ('cuda' or 'cpu') - auto-selects if unspecified | Auto-select                     |

### Run Script

```bash
python 2_build_dictionary.py [optional parameters]
```

Examples:

```bash
# Run with default parameters
python 2_build_dictionary.py

# Custom input/output paths
python 2_build_dictionary.py --npy_file "data/my_concepts.npy" --image_features "features/image_features.json"

# Different model configuration
python 2_build_dictionary.py --backbone "RN50" --pretrained_path "models/custom_weights.bin"

# Force CPU processing
python 2_build_dictionary.py --device "cpu"
```

### Processing Steps

1. **Setup CLIP Model**: Loads the specified CLIP model with given weights
2. **Load Image Features**: Reads precomputed image features from JSON file
3. **Extract Text Features**: Processes .npy file to extract text features for each concept
4. **Calculate Similarity**: For each text concept, finds the best matching image features using cosine similarity
5. **Save Results**: Outputs text features and similarity dictionary as JSON files

### Output Files

| File Path            | Format | Description                                                               |
| -------------------- | ------ | ------------------------------------------------------------------------- |
| `text_features.json` | JSON   | Dictionary with text concepts as keys and their feature vectors as values |
| `dictionary.json`    | JSON   | Dictionary mapping text concepts to best matching image features          |

### Console Output

```
Equipment used: CUDA
Load model: backbone=ViT-B-32, weights=F:Long_tail_learningclip_modelopen_clip_pytorch_model.bin
Load image features: F:Desktopinfodictionary_featureimage_features.json
Working with text files: F:Long_tail_learningclip_conceptvisual_concepts_phrased.npy
Processed text features: 100%|██████████| 1000/1000 [01:23<00:00, 12.00 texts/s]
Save text features: 100%|██████████| 1/1 [00:00<00:00, 5.00it/s]
Calculate similarity: 100%|██████████| 1000/1000 [05:23<00:00, 3.12 texts/s]
Save the similarity dictionary: 100%|██████████| 1/1 [00:00<00:00, 5.00it/s]
Finish! Processed 1000 text concepts
```

### Notes

1. **Input Requirements**:
   - `.npy` file should contain a dictionary of visual concepts
   - Image features JSON should match the format from `1_get_image_features.py`
2. **Performance**:
   - Similarity calculation is O(N\*M) where N=texts, M=images
   - For large datasets, consider using batch processing or approximate nearest neighbors
3. **Output Format**:

   ```json
   {
     "concept1": [0.123, -0.456, 0.789],
     "concept2": [0.234, -0.567, 0.890]
   }
   ```

4. **Error Handling**:
   - Automatically creates missing output directories
   - Progress bars show real-time processing status。

---

## 🔍 3. Match Images with Visual Concepts

### Description

This script matches images with visual concepts by comparing their CLIP feature vectors. For each image, it finds the most relevant concept from each dictionary file in a specified folder.

### Optional Parameters

| Parameter             | Type   | Description                                                        | Default Value                   |
| --------------------- | ------ | ------------------------------------------------------------------ | ------------------------------- |
| `--image_features`    | string | Path to JSON file containing image features                        | `.\image_features.json`         |
| `--dictionary_folder` | string | Path to folder containing concept dictionary JSON files            | `.\dictionary_split`            |
| `--output_path`       | string | Output JSON file path for matching results                         | `.\closest_features.json`       |
| `--backbone`          | string | CLIP model architecture (e.g., 'ViT-B-32', 'RN50')                 | `ViT-B-32`                      |
| `--pretrained_path`   | string | Path to pretrained weights file                                    | `.\open_clip_pytorch_model.bin` |
| `--device`            | string | Computation device ('cuda' or 'cpu') - auto-selects if unspecified | Auto-select                     |

### Run Script

```bash
python 3_match_image_and_concept.py [optional parameters]
```

Examples:

```bash
# Run with default parameters
python 3_match_image_and_concept.py

# Custom input/output paths
python 3_match_image_and_concept.py --image_features "data/image_features.json" --dictionary_folder "dictionaries/" --output_path "results/matches.json"

# Different model configuration
python 3_match_image_and_concept.py --backbone "RN50" --pretrained_path "models/custom_weights.bin"

# Force CPU processing
python 3_match_image_and_concept.py --device "cpu"
```

### Processing Steps

1. **Load Image Features**: Reads precomputed image features from JSON file
2. **Load Concept Dictionaries**: Processes all JSON files in the dictionary folder
3. **Match Concepts**: For each image and each dictionary, finds the most similar concept using cosine similarity
4. **Save Results**: Outputs matching results as a JSON file

### Output Format

```json
{
  "image1.jpg": {
    "dictionary1": "concept_a",
    "dictionary2": "concept_b"
  },
  "image2.png": {
    "dictionary1": "concept_c",
    "dictionary2": "concept_d"
  }
}
```

### Console Output

```
Equipment used: CUDA
Load model: backbone=ViT-B-32, weights=F:Long_tail_learningclip_modelopen_clip_pytorch_model.bin
Load image features: F:DesktopinfominiImageNet_image_features.json
Load the dictionary folder: F:Desktopinfodictionarydictionary_split
Processing dictionary files: 100%|██████████| 50/50 [15:23<00:00, 18.46s/file]
Save the result to: F:DesktopinfominiImageNet_closest_features.json
Match complete! 1000 images and 50 dictionary files were processed
```

### Notes

1. **Input Requirements**:
   - Image features JSON should match the format from `1_get_image_features.py`
   - Dictionary JSON files should be in the format: `{"concept1": [feature_vector], "concept2": [feature_vector], ...}`
2. **Performance**:
   - Processing time increases with number of images × number of dictionaries × number of concepts
   - GPU acceleration significantly improves performance for large datasets
3. **Output Structure**:

   - Top-level keys are image filenames
   - Second-level keys are dictionary filenames (without .json extension)
   - Values are the most similar concepts from each dictionary

4. **Error Handling**:
   - Automatically creates missing output directories
   - Skips non-JSON files in dictionary folder
   - Provides detailed progress indicators

---

## 📊 4. Generate Attribute Distribution Histograms

### Description

This script processes JSON files containing image-attribute matching results and generates histograms showing the distribution of secondary attributes for each primary attribute category.

### Optional Parameters

| Parameter       | Type    | Description                                                 | Default Value              |
| --------------- | ------- | ----------------------------------------------------------- | -------------------------- |
| `--json_folder` | string  | Path to folder containing JSON files with matching results  | `.\closest_features_split` |
| `--output_dir`  | string  | Output directory for generated charts and statistics        | `.\Histogram`              |
| `--min_count`   | integer | Minimum occurrence count for attributes to be displayed     | `1`                        |
| `--max_bars`    | integer | Maximum number of bars to show in each chart                | `20`                       |
| `--figsize`     | string  | Chart dimensions (width,height) in inches (comma-separated) | `10,5`                     |
| `--rotation`    | integer | Rotation angle for x-axis labels (degrees)                  | `45`                       |

### Run Script

```bash
python 4_generate_histograms.py [optional parameters]
```

Examples:

```bash
# Run with default parameters
python 4_generate_histograms.py

# Custom input/output paths
python 4_generate_histograms.py --json_folder "data/matches" --output_dir "results/histograms"

# Filter attributes with minimum count
python 4_generate_histograms.py --min_count 5

# Adjust chart appearance
python 4_generate_histograms.py --max_bars 15 --figsize "12,6" --rotation 30
```

### Processing Steps

1. **Load Matching Data**: Reads JSON files containing image-attribute matches
2. **Count Attribute Occurrences**: For each primary attribute category, counts occurrences of secondary attributes
3. **Generate Statistics**: Saves attribute counts to JSON files
4. **Create Histograms**: Generates bar charts showing attribute distributions
5. **Save Results**: Outputs all files to organized directory structure

### Output Structure

```
output_dir/
├── file1/                  # Directory for each input JSON file
│   ├── file1.json          # Attribute count statistics
│   ├── primary_attr1.png   # Histogram for primary attribute 1
│   ├── primary_attr2.png   # Histogram for primary attribute 2
│   └── ...
├── file2/
│   ├── file2.json
│   ├── primary_attr1.png
│   └── ...
└── ...
```

### Chart Features

1. **Attribute Filtering**: Only shows attributes meeting `--min_count` threshold
2. **Top-N Display**: Limits charts to top `--max_bars` attributes
3. **Value Labels**: Displays exact count on each bar
4. **Customizable Layout**: Adjustable size and label rotation
5. **Clean Formatting**: Optimized spacing and readability

### Console Output

```
Processed files: 100%|██████████| 10/10 [02:30<00:00, 15.00s/file]
The calculation is complete! A total of 10 files were processed and 150 charts were generated
The results have been saved to: F:DesktopinfominiImageNet_Histogram
```

### Notes

1. **Input Requirements**:
   - JSON files should have the format: `{"image.jpg": {"primary_attr1": "secondary_attr", ...}, ...}`
2. **Performance**:
   - Processing time depends on number of files and attributes
   - Larger `--max_bars` values increase chart generation time
3. **Output Optimization**:
   - Use `--min_count` to filter rare attributes
   - Adjust `--max_bars` to focus on most frequent attributes
   - Increase `--figsize` for charts with many attributes
4. **Error Handling**:
   - Automatically creates missing directories
   - Skips non-JSON files
   - Handles empty input files gracefully

---

## 📂 5.Image Classification by Visual Attributes

### Description

This script organizes images into a hierarchical folder structure based on visual attributes extracted from JSON files. For each JSON file, it:

1. Analyzes primary and secondary visual attributes
2. Selects top K most frequent secondary attributes per primary category
3. Copies images into `output_base_folder/JSON_file_name/primary_attribute/secondary_attribute/` folders

### Optional Parameters

| Parameter              | Type   | Description                                                        | Default Value    |
| ---------------------- | ------ | ------------------------------------------------------------------ | ---------------- |
| `--json_folder`        | string | Path to folder containing JSON files with attribute data           | `./json_folder`  |
| `--image_folder`       | string | Path to folder containing original images                          | `./image_folder` |
| `--output_base_folder` | string | Base output path for classified images                             | `./output`       |
| `--top_k`              | int    | Number of top secondary attributes to select per primary attribute | `10`             |

### Run Script

```bash
python 5_classify_images.py [optional parameters]
```

Examples:

```bash
# Run with default parameters
python 5_classify_images.py

# Custom paths and top 15 attributes
python 5_classify_images.py \
    --json_folder "data/attribute_files" \
    --image_folder "./raw" \
    --output_base_folder "images/classified" \
    --top_k 15

# Windows paths with different top value
python 5_classify_images.py ^
    --json_folder ".\json_input" ^
    --image_folder ".\source" ^
    --output_base_folder "F:\images\classified" ^
    --top_k 20
```

### Processing Steps

1. **Load JSON Attribute Files**: Processes all JSON files in specified folder
2. **Analyze Attributes**: For each JSON file:
   - Counts occurrences of secondary attributes per primary attribute
   - Selects top K most frequent secondary attributes
3. **Organize Images**: Creates folder hierarchy and copies images:
   - `output_base_folder/JSON_file_name/`
   - → `primary_attribute/`
   - → → `secondary_attribute/`
   - → → → `image_files.jpg`
4. **Progress Tracking**: Shows real-time progress with file counts

### Output Structure

```
output_base_folder/
├── json_file_1/
│   ├── color/
│   │   ├── red/
│   │   │   ├── image1.jpg
│   │   │   └── image5.png
│   │   └── blue/
│   │       ├── image3.jpg
│   │       └── image7.png
│   └── texture/
│       ├── smooth/
│       └── rough/
└── json_file_2/
    ├── shape/
    └── material/
```

### Console Output Example

```
Created directory: F:\output\json_file_1
Processing json_file_1: 100%|██████████| 1500/1500 [02:15<00:00, 11.1 images/s]
Processing json_file_2: 100%|██████████| 1200/1200 [01:45<00:00, 11.4 images/s]
Image classification completed!
```

### Notes

1. **Input Requirements**:
   - JSON files must contain `{image_name: {primary_attr: secondary_attr}}` structure
   - Image names in JSON must match filenames in image folder
2. **Performance**:

   - Processing speed depends on number of images and attributes
   - File copy operations are the most time-consuming step
   - Progress bar shows real-time throughput (images/second)

3. **Output Characteristics**:

   - Creates mirror directory structure of JSON files
   - Only copies images belonging to top K secondary attributes
   - Preserves original filenames and extensions

4. **Error Handling**:

   - Automatically creates required output directories
   - Skips missing source images (no error interruption)
   - Handles special characters in attribute names
   - Validates JSON file format before processing

5. **Customization**:
   - Adjust `--top_k` to balance granularity vs. folder size
   - Works with any attribute hierarchy (not limited to visual concepts)
   - Handles multiple JSON files in batch processing mode

---

## 📊 6.Image Classification Accuracy Evaluator

### Description

This script evaluates classification accuracy of a trained ResNet50 model on image datasets organized in hierarchical folder structures. For each folder hierarchy, it calculates per-class accuracy percentages and saves results as JSON files.

### Arguments

| Parameter            | Type   | Description                                                        | Required | Default |
| -------------------- | ------ | ------------------------------------------------------------------ | -------- | ------- |
| `--model_path`       | string | Path to trained ResNet50 model (.pth file)                         | Yes      | -       |
| `--parent_folder`    | string | Root directory containing class subfolders                         | Yes      | -       |
| `--output_directory` | string | Directory to save accuracy JSON files                              | Yes      | -       |
| `--class_mapping`    | string | Path to class-to-index mapping JSON file                           | Yes      | -       |
| `--device`           | string | Computation device ('cuda' or 'cpu') - auto-selects if unspecified | No       | Auto    |

### Run Script

```bash
python 6_test_accuracy.py [arguments]
```

Examples:

```bash
# Basic usage with required parameters
python 6_test_accuracy.py \
    --model_path "models/resnet50_miniImageNet.pth" \
    --parent_folder "datasets/class" \
    --output_directory "results/accuracy" \
    --class_mapping "mappings/class_to_idx.json"

# Force CPU processing
python 6_test_accuracy.py \
    --model_path "models/resnet50_miniImageNet.pth" \
    --parent_folder "datasets/class" \
    --output_directory "results/accuracy" \
    --class_mapping "mappings/class_to_idx.json" \
    --device "cpu"

# Custom device selection
python 6_test_accuracy.py \
    --model_path "models/resnet50_miniImageNet.pth" \
    --parent_folder "datasets/class" \
    --output_directory "results/accuracy" \
    --class_mapping "mappings/class_to_idx.json" \
    --device "cuda:0"
```

### Processing Steps

1. **Load Model**: Loads trained ResNet50 with custom classification head
2. **Load Class Mapping**: Reads class-to-index mapping from JSON file
3. **Process Folders**: Recursively processes each class subfolder hierarchy
4. **Classify Images**: Predicts class for each image using the trained model
5. **Calculate Accuracy**: Compares predictions with true class labels
6. **Save Results**: Outputs hierarchical accuracy results as JSON files

### Output Format

```json
{
  "parent_folder1": {
    "subfolder1": "85.25%",
    "subfolder2": "92.10%"
  },
  "parent_folder2": {
    "subfolder1": "78.33%",
    "subfolder2": "89.47%"
  }
}
```

### Console Output Example

```
Using device: cuda
Model loaded on cuda device
Processing: novel_classes
Processing folders: 100%|██████████| 20/20 [05:32<00:00, 16.65s/folder]
Saved accuracies to results/accuracy/novel_classes.json

Processing: base_classes
Processing folders: 100%|██████████| 40/40 [12:18<00:00, 18.47s/folder]
Saved accuracies to results/accuracy/base_classes.json

Evaluation completed!
```

### Notes

1. **Input Requirements**:

   - Image formats: .jpeg, .jpg, .png (case-insensitive)
   - Folder structure: `parent_folder/class_subfolder/images`
   - Class mapping format: `{"class_name": index, ...}`

2. **Performance Considerations**:

   - GPU acceleration recommended for large datasets
   - Processing time increases with:
     - Number of images
     - Depth of folder hierarchy
     - Number of classes

3. **Output Structure**:

   - Creates one JSON file per top-level folder in parent_directory
   - Filename format: `[top_level_folder_name].json`
   - Results include hierarchical accuracy percentages

4. **Special Cases**:

   - Automatically creates output directories
   - Handles nested folder structures recursively
   - Skips non-image files in directories
   - Provides progress indicators for long operations

5. **Error Handling**:
   - Verifies model file exists before loading
   - Checks class mapping contains all required classes
   - Handles invalid device specifications gracefully
   - Skips empty directories with warning messages

<!-- ---

## Learning Trajectory (Updating...)

When I completed this project, I was a third-year undergraduate student. 🌿 I will share my learning trajectory and how to efficiently and comprehensively develop expertise in a specific field. 🌊 I believe that the most effective approach is to start by identifying high-quality review articles from top-tier journals. 📚 After forming a comprehensive understanding of the field, I recommend selecting detailed papers from the references cited in these outstanding reviews, focusing on those that align with the direction of our current work for in-depth study. 🔍 This process resembles a leaf with its veins hollowed out — our process of understanding is akin to a flood flowing through the leaf, with the central vein serving as the core from which knowledge selectively branches out in all directions. 🚀

- **2023Tpami** "Deep Long-Tailed Learning: A Survey" [Paper](https://arxiv.org/pdf/2304.00685)——Review on Long-Tailed Learning

- **2024Tpami** "Vision-Language Models for Vision Tasks: A Survey" [Paper](https://arxiv.org/pdf/2304.00685) & [Github](https://github.com/jingyi0000/VLM_survey)——Review on Vision-Language Large Models

- **2024Tpami** "Federated Learning for Generalization, Robustness, Fairness: A Survey and Benchmark" [Paper](https://arxiv.org/pdf/2311.06750) & [Github](https://github.com/WenkeHuang/MarsFL)——Review on Federated Learning

- **2021CVPR** "Model-Contrastive Federated Learning" [Paper](https://arxiv.org/pdf/2103.16257) & [Github](https://github.com/QinbinLi/MOON)——MOON(Alignment of Local and Global Model Representations)

- **2022AAAI** "FedProto: Federated Prototype Learning across Heterogeneous Clients" [Paper](https://arxiv.org/pdf/2105.00243)——FedProto(Alignment of Local and Global Prototype Representations)

- **2023FGCS** "FedProc: Prototypical contrastive federated learning on non-IID data" [Paper](https://arxiv.org/pdf/2109.12273)——FedProc(Alignment of Local and Global Prototype Representations)
- **2020ICML** "SCAFFOLD:Stochastic Controlled Averaging for Federated Learning" [Paper](https://arxiv.org/pdf/1910.06378)——SCAFFOLD(Alignment of Local and Global Optimization Directions)
- **2021ICLR** "FEDERATED LEARNING BASED ON DYNAMIC REGULARIZATION" [Paper](https://arxiv.org/pdf/2111.04263)——FedDyn(Alignment of Local and Global Losses)

- **2022NeurIPS** "Preservation of the Global Knowledge by Not-True Distillation in Federated Learning" [Paper](https://arxiv.org/pdf/2106.03097)——FedNTD(Alignment of Unseen Local Losses with Global Losses)

- **2021ICLR** "ADAPTIVE FEDERATED OPTIMIZATION" [Paper](https://arxiv.org/pdf/2003.00295)——FedOpt(Server-Side Aggregation Optimization)

- **2024CVPR** "Fair Federated Learning under Domain Skew with Local Consistency and Domain Diversity"[Paper](https://arxiv.org/pdf/2405.16585) & [Github](https://github.com/yuhangchen0/FedHEAL)——FedHEAL(Alignment of Local and Global Model Representations)

- **2023WACV** "Federated Domain Generalization for Image Recognition via Cross-Client Style Transfer"[Paper](https://arxiv.org/pdf/2210.00912) & [Github](https://chenjunming.ml/proj/CCST)——CCST(Alignment of Local and Global Optimization Directions)

- **2023TMC** "FedFA: Federated Learning with Feature Anchors to Align Features and Classifiers for Heterogeneous Data" [Paper](https://arxiv.org/pdf/2211.09299)——FedFA(Alignment of Features and Classifiers)

- **2024AAAI** "CLIP-Guided Federated Learning on Heterogeneous and Long-Tailed Data" [Paper](https://arxiv.org/pdf/2312.08648)——CLIP As Backbond For FL

- **2023CVPR** "Rethinking Federated Learning with Domain Shift: A Prototype View" [Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Huang_Rethinking_Federated_Learning_With_Domain_Shift_A_Prototype_View_CVPR_2023_paper.pdf) & [Github](https://github.com/WenkeHuang/RethinkFL/tree/main)——Cross-Domain Prototype Loss Alignment

- **2023ICLR** "FEDFA: FEDERATED FEATURE AUGMENTATION" [Paper](https://arxiv.org/pdf/2301.12995) & [Github](https://github.com/tfzhou/FedFA)——Class Prototype Gaussian Enhancement

- **2021ICLR** "FEDMIX: APPROXIMATION OF MIXUP UNDER MEAN AUGMENTED FEDERATED LEARNING" [Paper](https://arxiv.org/pdf/2107.00233)——Mixup For FL

- **2021PMLR** "Data-Free Knowledge Distillation for Heterogeneous Federated Learning" [Paper](https://arxiv.org/pdf/2105.10056)——Data-Free Knowledge Distillation For FL

- **2017ICML** "Communication-Efficient Learning of Deep Networks from Decentralized Data" [Paper](https://arxiv.org/pdf/1602.05629)——FedAvg(Average aggregation)

- **2025ICLR** "Pursuing Better Decision Boundaries for Long-Tailed Object Detection via Category Information Amount" [Paper](https://arxiv.org/pdf/2502.03852)——IGAM Loss(Revise decision boundaries)

- **2025Entropy** "Trade-Offs Between Richness and Bias of Augmented Data in Long-Tailed Recognition" [Paper](https://www.mdpi.com/1099-4300/27/2/201)——EIG(Effectiveness of distributed gain) -->

---

<!-- ## 💡 Citation

If you find our work useful, please cite it using the following BibTeX format:

```bibtex
@article{ma2025geometric,
  title={Geometric Knowledge-Guided Localized Global Distribution Alignment for Federated Learning},
  author={Ma, Yanbiao and Dai, Wei and Huang, Wenke and Chen, Jiayi},
  journal={arXiv preprint arXiv:2503.06457},
  year={2025},
  url={https://arxiv.org/pdf/2503.06457}
}
``` -->

## 📧 Contact

**For any questions or help, feel welcome to write an email to <br> 22012100031@stu.xidian.edu.cn**
