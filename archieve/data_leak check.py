import os

# Load your train, val, test file lists
def load_split_files(split_file):
    """Load image and annotation paths from .dat file"""
    images = []
    with open(split_file, 'r') as f:
        for line in f:
            img_path, ann_path = line.strip().split()
            images.append(img_path)
    return images

# Load all splits
train_images = load_split_files('data/jazzmus/splits/train_0.txt')
val_images = load_split_files('data/jazzmus/splits/val_0.txt')
test_images = load_split_files('data/jazzmus/splits/test_0.txt')

print(f"Train size: {len(train_images)}")
print(f"Val size: {len(val_images)}")
print(f"Test size: {len(test_images)}")

# Check for overlaps
train_set = set(train_images)
val_set = set(val_images)
test_set = set(test_images)

train_val_overlap = train_set.intersection(val_set)
train_test_overlap = train_set.intersection(test_set)
val_test_overlap = val_set.intersection(test_set)

print(f"\n=== Data Leakage Check ===")
print(f"Train-Val overlap: {len(train_val_overlap)} samples")
print(f"Train-Test overlap: {len(train_test_overlap)} samples")
print(f"Val-Test overlap: {len(val_test_overlap)} samples")

if len(train_val_overlap) > 0:
    print(f"\n⚠️ WARNING: Train and Val share samples!")
    print(f"Examples: {list(train_val_overlap)[:5]}")