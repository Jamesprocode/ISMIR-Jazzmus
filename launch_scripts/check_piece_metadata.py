"""
Check if HuggingFace JAZZMUS dataset contains piece title/name metadata.
This script examines the annotation structure to see if we can identify
piece names or versions, which would allow proper piece-level splitting.
"""

import json
import ast
from datasets import load_dataset
import re

def extract_title_from_musicxml(musicxml_str):
    """Extract title from MusicXML string."""
    # Look for movement-title tag (JAZZMUS uses this)
    match = re.search(r'<movement-title>(.*?)</movement-title>', musicxml_str)
    if match:
        return match.group(1).strip()

    # Fallback to work-title tag
    match = re.search(r'<work-title>(.*?)</work-title>', musicxml_str)
    if match:
        return match.group(1).strip()

    return None


def extract_composer_from_musicxml(musicxml_str):
    """Extract composer from MusicXML string."""
    match = re.search(r'<creator type="composer">(.*?)</creator>', musicxml_str)
    if match:
        return match.group(1).strip()
    return None


def check_dataset_metadata(hf_name="PRAIG/JAZZMUS", num_samples=10):
    """Check first N samples for piece metadata."""

    print("="*60)
    print(f"Loading {num_samples} samples from {hf_name}...")
    print("="*60)

    dataset = load_dataset(hf_name, split=f"train[:{num_samples}]")

    print(f"✓ Loaded {len(dataset)} samples\n")

    piece_titles = {}
    has_musicxml = False

    for idx in range(len(dataset)):
        print(f"\n--- Sample {idx} ---")

        # Get annotation
        annotation = dataset[idx]["annotation"]

        # Parse if string
        if isinstance(annotation, str):
            try:
                annotation = json.loads(annotation)
            except json.JSONDecodeError:
                annotation = ast.literal_eval(annotation)

        # Check top-level keys
        print(f"Top-level keys: {list(annotation.keys())}")

        # Check for MusicXML in the 'encodings' dictionary
        if 'encodings' in annotation and 'musicxml' in annotation['encodings']:
            has_musicxml = True
            musicxml = annotation['encodings']['musicxml']
            print(f"  ✓ Found MusicXML in encodings")

            # Try to extract title and composer
            title = extract_title_from_musicxml(musicxml)
            composer = extract_composer_from_musicxml(musicxml)

            if title:
                print(f"  ✓ Title: '{title}'")
                if composer:
                    print(f"  ✓ Composer: '{composer}'")
                    piece_titles[idx] = f"{title} - {composer}"
                else:
                    piece_titles[idx] = title
            else:
                print(f"  ✗ MusicXML exists but no title found in <work-title> or <movement-title>")
                # Show first 500 chars to see structure
                print(f"  MusicXML snippet (first 500 chars):")
                print(f"  {musicxml[:500]}...")
        else:
            print(f"  ✗ No 'musicxml' key found in encodings")
            if 'encodings' in annotation:
                print(f"  Available encoding keys: {list(annotation['encodings'].keys())}")

        # Show what's in systems (for debugging)
        if 'systems' in annotation and len(annotation['systems']) > 0:
            first_system = annotation['systems'][0]
            print(f"  First system keys: {list(first_system.keys())}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Has MusicXML: {has_musicxml}")
    print(f"Pieces with titles: {len(piece_titles)}/{num_samples}")

    if piece_titles:
        print("\nFound titles:")
        for idx, title in piece_titles.items():
            print(f"  Image {idx}: {title}")

        # Check for duplicates (versions)
        from collections import Counter
        title_counts = Counter(piece_titles.values())
        duplicates = {title: count for title, count in title_counts.items() if count > 1}

        if duplicates:
            print("\n✓ Found duplicate titles (possible versions):")
            for title, count in duplicates.items():
                print(f"  '{title}': {count} copies")
        else:
            print("\n✗ No duplicate titles found in this sample")

    return has_musicxml, piece_titles


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_name", type=str, default="PRAIG/JAZZMUS")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to check")
    args = parser.parse_args()

    check_dataset_metadata(args.hf_name, args.num_samples)
