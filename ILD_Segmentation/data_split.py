import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import shutil
import collections

# Define paths
data_path = "/home/huchengpeng/workspace/MedSAM-main/ILD/data/rawdata"
img_path = os.path.join(data_path, "imgs")
gt_path = os.path.join(data_path, "gts")
lung_path = os.path.join(data_path, "lungs")
output_path = "/home/huchengpeng/workspace/MedSAM-main/ILD/data"  # Save visualization here

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

# Get list of files
img_files = sorted(os.listdir(img_path))
gt_files = sorted(os.listdir(gt_path))
lung_files = sorted(os.listdir(lung_path))

# Function to process labels correctly
def process_labels(gt_data, lung_data):
    """
    Process labels: 
    1. Add 1 to all non-zero labels
    2. Set lung regions (not overlapping with original labels) to 1
    """
    # Create a copy to avoid modifying original data
    processed_gt = gt_data.copy()
    
    # Step 1: Add 1 to all non-zero labels
    processed_gt[gt_data > 0] = gt_data[gt_data > 0] + 1
    
    # Step 2: Set lung regions (not overlapping with original labels) to 1
    # Lung regions where there were no original labels (background)
    lung_background_mask = (lung_data > 0) & (gt_data == 0)
    processed_gt[lung_background_mask] = 1
    
    return processed_gt

# Label mapping (after +1 transformation)
label_mapping = {
    1: "normal",
    2: "GGO",
    3: "reticulation",
    4: "consolidation",
    5: "honeycombing",
}

# 1. Check the total number of processed npy files
total_files = len(img_files)
print(f"Total number of processed image files: {total_files}")
print(f"Total number of processed ground truth files: {len(gt_files)}")

if total_files == 0:
    print("No files to process.")
    exit()

# 2. Randomly select one image and its corresponding gt for visualization
random_index = random.randint(0, total_files - 1)
random_img_file = img_files[random_index]
random_gt_file = random_img_file

# Load the selected image, gt and lung mask
img_data = np.load(os.path.join(img_path, random_img_file))
gt_data = np.load(os.path.join(gt_path, random_gt_file))
lung_data = np.load(os.path.join(lung_path, random_img_file))  # Use same filename
# Process labels correctly
gt_data = process_labels(gt_data, lung_data)

# 3. Use matplotlib to save visualization with legend
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_data)
plt.title(f"Image: {random_img_file}")
plt.axis("off")

ax = plt.subplot(1, 2, 2)
# Define specific colors for each label
colors = ['black', 'blue', 'cyan', 'yellow', 'red', 'orange']  # 0: black, 1: blue, 2: cyan, etc.
from matplotlib.colors import ListedColormap
cmap = ListedColormap(colors[:max(label_mapping.keys())+1])
im = ax.imshow(gt_data, cmap=cmap, vmin=0, vmax=max(label_mapping.keys()))
plt.title(f"Ground Truth: {random_gt_file}")
plt.axis("off")

# Create legend with correct colors
patches = [mpatches.Patch(color=colors[label_id], label=f"{label_name}")
           for label_id, label_name in label_mapping.items()]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

visualization_filename = os.path.join(output_path, "sanity_check_visualization.png")
plt.savefig(visualization_filename, bbox_inches='tight')
print(f"Visualization saved to {visualization_filename}")
plt.close()

# 4. Count slices containing each lesion type
slice_counts = {label_id: 0 for label_id in label_mapping}
slice_lesion_map = collections.defaultdict(list)

print("Counting slice statistics...")
for gt_file in tqdm(gt_files):
    gt_data = np.load(os.path.join(gt_path, gt_file))
    lung_data = np.load(os.path.join(lung_path, gt_file))  # Load corresponding lung mask
    # Process labels correctly
    gt_data = process_labels(gt_data, lung_data)
    unique_labels = np.unique(gt_data)
    has_lesion = False
    for label_id in unique_labels:
        if label_id in slice_counts:
            slice_counts[label_id] += 1
            slice_lesion_map[gt_file].append(label_id)
            has_lesion = True
    if not has_lesion:
        slice_lesion_map[gt_file].append(1) # for background slices (now label 1)

print("\nSlice Statistics (number of slices containing each lesion):")
for label_id, count in slice_counts.items():
    lesion_name = label_mapping.get(label_id, "Unknown")
    print(f"Label {label_id} ({lesion_name}): {count} slices")

# 5. Dataset splitting configuration
# Configure train/validation/test split ratios
train_ratio = 0.8
valid_ratio = 0.1
test_ratio = 0.1

# Validate ratios
assert abs(train_ratio + valid_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"

print(f"\nSplitting dataset with ratios - Train: {train_ratio}, Valid: {valid_ratio}, Test: {test_ratio}")

X = list(slice_lesion_map.keys())

primary_labels = []
for f in X:
    labels_present = slice_lesion_map[f]
    if not labels_present or all(l == 1 for l in labels_present):
        primary_labels.append(1) # Background (now label 1)
        continue
    # Filter out background label 1 for finding rarest lesion
    actual_lesions = [l for l in labels_present if l != 1]
    if not actual_lesions:
        primary_labels.append(1)
        continue
    rarest_label = min(actual_lesions, key=lambda label: slice_counts[label])
    primary_labels.append(rarest_label)

# Split dataset based on configured ratios
if valid_ratio > 0:
    # First split: separate test set
    X_temp, X_test, y_temp, _ = train_test_split(
        X, primary_labels, test_size=test_ratio, random_state=42, stratify=primary_labels
    )
    # Second split: separate train and validation from remaining data
    valid_size_adjusted = valid_ratio / (train_ratio + valid_ratio)
    X_train, X_valid, _, _ = train_test_split(
        X_temp, y_temp, test_size=valid_size_adjusted, random_state=42, stratify=y_temp
    )
else:
    # Only split into train and test
    X_train, X_test, _, _ = train_test_split(
        X, primary_labels, test_size=test_ratio, random_state=42, stratify=primary_labels
    )
    X_valid = []

print(f"Train set size: {len(X_train)}")
print(f"Validation set size: {len(X_valid)}")
print(f"Test set size: {len(X_test)}")

# 6. Save the split datasets
def save_split(file_list, split_name):
    if not file_list:
        print(f"No files for {split_name} set. Skipping.")
        return
    split_dir = os.path.join(output_path, split_name)
    img_dir = os.path.join(split_dir, "imgs")
    gt_dir = os.path.join(split_dir, "gts")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    for filename in tqdm(file_list, desc=f"Saving {split_name} set"):
        # Source paths
        src_img_path = os.path.join(img_path, filename)
        src_gt_path = os.path.join(gt_path, filename)
        # Destination paths
        dst_img_path = os.path.join(img_dir, filename)
        dst_gt_path = os.path.join(gt_dir, filename)

        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        if os.path.exists(src_gt_path):
            # Load, transform with lung mask, and save ground truth
            gt_data = np.load(src_gt_path)
            lung_data = np.load(os.path.join(lung_path, filename))  # Load corresponding lung mask
            gt_data = process_labels(gt_data, lung_data)  # Process labels correctly
            np.save(dst_gt_path, gt_data)

save_split(X_train, "train")
save_split(X_valid, "valid")
save_split(X_test, "test")

print("\nDataset splitting and saving complete.")