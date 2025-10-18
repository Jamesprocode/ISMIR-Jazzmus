"""
Test script to process a single image from synthetic dataset and debug bounding box format.
This will help identify and fix the bbox mismatch with Kern format.
"""
import json
import ast
from PIL import Image, ImageDraw
from datasets import load_dataset


def visualize_bbox(image, bbox, label="", color="red"):
    """Draw bounding box on image for visualization"""
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    # Draw rectangle
    draw.rectangle(
        [(bbox['fromX'], bbox['fromY']), (bbox['toX'], bbox['toY'])],
        outline=color,
        width=3
    )

    # Draw label if provided
    if label:
        draw.text((bbox['fromX'], bbox['fromY'] - 20), label, fill=color)

    return img_copy


def process_single_image(idx=0):
    """Process a single image from the synthetic dataset"""
    print(f"Loading dataset...")
    dataset = load_dataset("PRAIG/JAZZMUS_Synthetic", split="train")

    print(f"\nDataset loaded. Total samples: {len(dataset)}")
    print(f"Processing image {idx}...\n")

    # Get the image and annotation
    image = dataset[idx]["image"]
    annotation_data = dataset[idx]["annotation"]

    print(f"Image size: {image.size}")
    print(f"Annotation type: {type(annotation_data)}\n")

    # Parse annotation if it's a string
    if isinstance(annotation_data, str):
        try:
            regions = json.loads(annotation_data)
        except json.JSONDecodeError:
            try:
                regions = ast.literal_eval(annotation_data)
            except Exception as e:
                print(f"Error parsing annotation: {e}")
                return
    else:
        regions = annotation_data

    print(f"Annotation keys: {regions.keys()}\n")

    # Print complete annotation structure
    print("=" * 60)
    print("COMPLETE ANNOTATION STRUCTURE:")
    print("=" * 60)
    print(json.dumps(regions, indent=2))
    print("=" * 60)
    print()

    # Save annotation to file for inspection
    with open("test_annotation_structure.json", "w") as f:
        json.dump(regions, f, indent=2)
    print("Full annotation saved to: test_annotation_structure.json\n")

    # Check for systems
    if "systems" not in regions:
        print(f"ERROR: No 'systems' key found!")
        print(f"Available keys: {list(regions.keys())}")
        return

    print(f"Number of systems found: {len(regions['systems'])}\n")

    # Process first system to test bbox
    if len(regions['systems']) > 0:
        system = regions['systems'][0]
        print("=" * 60)
        print("SYSTEM 0 DETAILS:")
        print("=" * 60)
        print(f"System keys: {system.keys()}\n")

        if "bounding_box" in system:
            bbox = system["bounding_box"]
            print(f"Bounding box: {bbox}")
            print(f"\nBounding box breakdown:")
            print(f"  fromX: {bbox.get('fromX', 'MISSING')}")
            print(f"  toX:   {bbox.get('toX', 'MISSING')}")
            print(f"  fromY: {bbox.get('fromY', 'MISSING')}")
            print(f"  toY:   {bbox.get('toY', 'MISSING')}")

            # Check if coordinates are valid
            if all(k in bbox for k in ['fromX', 'toX', 'fromY', 'toY']):
                width = bbox['toX'] - bbox['fromX']
                height = bbox['toY'] - bbox['fromY']
                print(f"\nCalculated dimensions:")
                print(f"  Width:  {width} px")
                print(f"  Height: {height} px")

                # Check if bbox is within image bounds
                img_width, img_height = image.size
                print(f"\nImage dimensions: {img_width} x {img_height}")

                if bbox['fromX'] < 0 or bbox['fromY'] < 0:
                    print("  WARNING: Negative coordinates!")
                if bbox['toX'] > img_width or bbox['toY'] > img_height:
                    print("  WARNING: Bbox extends beyond image!")
                    print(f"    toX ({bbox['toX']}) > img_width ({img_width}): {bbox['toX'] > img_width}")
                    print(f"    toY ({bbox['toY']}) > img_height ({img_height}): {bbox['toY'] > img_height}")

                # Expected format for PIL crop: (left, top, right, bottom)
                # Testing different interpretations of the bounding box

                print(f"\n" + "="*60)
                print("TESTING DIFFERENT BBOX INTERPRETATIONS:")
                print("="*60)

                # Interpretation 1: Direct mapping (fromX, fromY, toX, toY) -> (left, top, right, bottom)
                bbox1 = (bbox['fromX'], bbox['fromY'], bbox['toX'], bbox['toY'])
                print(f"\nInterpretation 1 - Direct: (fromX, fromY, toX, toY)")
                print(f"  Crop: {bbox1}")

                # Interpretation 2: Maybe fromY/toY are swapped?
                bbox2 = (bbox['fromX'], bbox['toY'], bbox['toX'], bbox['fromY'])
                print(f"\nInterpretation 2 - Y swapped: (fromX, toY, toX, fromY)")
                print(f"  Crop: {bbox2}")

                # Interpretation 3: Maybe X and Y are swapped entirely?
                bbox3 = (bbox['fromY'], bbox['fromX'], bbox['toY'], bbox['toX'])
                print(f"\nInterpretation 3 - X/Y swapped: (fromY, fromX, toY, toX)")
                print(f"  Crop: {bbox3}")

                # Interpretation 4: Maybe it's (top, left, bottom, right)?
                bbox4 = (bbox['fromY'], bbox['fromX'], bbox['toY'], bbox['toX'])
                print(f"\nInterpretation 4 - Same as 3")
                print(f"  Crop: {bbox4}")

                # Try cropping with all interpretations
                print(f"\n" + "="*60)
                print("SAVING CROPPED VERSIONS:")
                print("="*60)

                try:
                    # Save original with bbox drawn (using direct interpretation)
                    viz_img = visualize_bbox(image, bbox, label="System 0 - Direct", color="red")
                    viz_img.save("test_bbox_visualization.png")
                    print("\n  Saved: test_bbox_visualization.png (original with bbox)")

                    # Try all interpretations
                    for i, crop_coords in enumerate([bbox1, bbox2, bbox3, bbox4], 1):
                        try:
                            cropped = image.crop(crop_coords)
                            filename = f"test_bbox_cropped_v{i}.png"
                            cropped.save(filename)
                            print(f"  Saved: {filename} - size {cropped.size} using coords {crop_coords}")
                        except Exception as e:
                            print(f"  ERROR v{i}: {e}")

                except Exception as e:
                    print(f"\nERROR: {e}")
            else:
                print("\nERROR: Missing required bbox keys!")
        else:
            print("ERROR: No 'bounding_box' in system!")

        # Check for **kern encoding
        if "**kern" in system:
            kern_content = system["**kern"]
            print(f"\n**kern encoding found ({len(kern_content)} characters)")
            print(f"\n{'='*60}")
            print("COMPLETE **KERN CONTENT:")
            print(f"{'='*60}")
            print(kern_content)
            print(f"{'='*60}")

            # Also save to file
            with open("test_kern_output.txt", "w") as f:
                f.write(kern_content)
            print("\nKern content saved to: test_kern_output.txt")
        else:
            print("\nWARNING: No '**kern' encoding found!")

    # Print all systems summary
    print("\n" + "=" * 60)
    print("ALL SYSTEMS SUMMARY:")
    print("=" * 60)
    for i, system in enumerate(regions['systems']):
        has_bbox = "bounding_box" in system
        has_kern = "**kern" in system
        bbox_str = ""
        if has_bbox:
            b = system["bounding_box"]
            bbox_str = f" bbox: ({b.get('fromX')}, {b.get('fromY')}) -> ({b.get('toX')}, {b.get('toY')})"
        print(f"System {i}: bbox={has_bbox}, kern={has_kern}{bbox_str}")


if __name__ == "__main__":
    # Test with first image
    process_single_image(idx=0)

    print("\n" + "=" * 60)
    print("Test complete! Check the saved images:")
    print("  - test_bbox_visualization.png")
    print("  - test_bbox_cropped.png")
    print("=" * 60)
