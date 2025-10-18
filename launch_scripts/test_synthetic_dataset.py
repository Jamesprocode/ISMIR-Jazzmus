"""Test script to check if PRAIG/JAZZMUS_Synthetic exists and its structure"""
from datasets import load_dataset

try:
    print("Attempting to load PRAIG/JAZZMUS_Synthetic...")
    dataset = load_dataset("PRAIG/JAZZMUS_Synthetic", split="train")
    print(f"✓ Dataset loaded successfully!")
    print(f"Number of samples: {len(dataset)}")
    print(f"\nDataset features: {dataset.features}")
    print(f"\nFirst sample keys: {dataset[0].keys()}")

    # Check structure
    first_sample = dataset[0]
    print(f"\nFirst sample structure:")
    for key, value in first_sample.items():
        print(f"  {key}: {type(value)}")

except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print(f"\nTrying to check if dataset exists...")
    from huggingface_hub import list_datasets
    datasets = [d.id for d in list_datasets(author="PRAIG")]
    print(f"\nAvailable PRAIG datasets:")
    for d in datasets:
        print(f"  - {d}")
