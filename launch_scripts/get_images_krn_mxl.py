from pathlib import Path
import json, requests
import argparse
from tqdm import tqdm
from jazzmus.dataset.generate_synthetic_score import render_and_clean_lyrics
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from io import BytesIO
import sys
import warnings


def process_single_json(
    f,
    override_existing=False,
    add_synthetic_jazz=False,
    add_synthetic_classical=False,
    musescore_jazz_style_path=None,
    musescore_path=None,
):
    # read the file
    with open(f) as file:
        data = json.load(file)

    # download images
    image_path = f.with_suffix(".jpg")
    if override_existing or not image_path.exists():
        # download image and store it in the same folder with the name of the file
        image_url = data["original"]
        if "http://localhost:8182/" in image_url:
            image_url = image_url.replace(
                "http://localhost:8182/", "https://muret.dlsi.ua.es/images/"
            )
        # download the image
        response = requests.get(image_url)
        # Use context manager to catch the DecompressionBombWarning occurs
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", Image.DecompressionBombWarning)

            img = Image.open(BytesIO(response.content))

            # Check if DecompressionBombWarning was issued
            if any(
                issubclass(warning.category, Image.DecompressionBombWarning)
                for warning in w
            ):
                print(f"DecompressionBombWarning for file: {f}")
                # downscale the image by a factor of 2
                img.thumbnail((img.width // 2, img.height // 2))

        # convert it to jpg, IIIF server automatically converts to jpg when storing info
        img.save(image_path, "JPEG")

    # export kern
    kern_path = f.with_suffix(".krn")
    if override_existing or not kern_path.exists():
        # save kern content for render purposes
        kern = data["encodings"]["**kern"]
        with open(kern_path, "w", encoding="utf-8") as krn:
            krn.write(kern)

    # export musicxml
    musicxml_path = f.with_suffix(".musicxml")
    if override_existing or not musicxml_path.exists():
        # save musicxml content for render purposes
        musicxml = data["encodings"]["musicxml"]
        with open(musicxml_path, "w", encoding="utf-8") as music:
            music.write(musicxml)

    # create synthetic images
    # jazz
    if add_synthetic_jazz:
        assert musescore_jazz_style_path is not None
        assert musescore_path is not None
        synthetic_jazz_svg_path = f.parent / Path(str(f.stem) + "_synjazz.svg")
        if override_existing or not synthetic_jazz_svg_path.exists():
            render_and_clean_lyrics(
                "musescore",
                musescore_jazz_style_path,
                "jazz",
                musescore_path,
                musicxml_path,
                synthetic_jazz_svg_path,
            )

    # classical
    if add_synthetic_classical:
        assert musescore_jazz_style_path is not None
        assert musescore_path is not None
        synthetic_classical_svg_path = f.parent / Path(
            str(f.stem) + "_synclassical.svg"
        )
        if override_existing or not synthetic_classical_svg_path.exists():
            render_and_clean_lyrics(
                "musescore",
                musescore_jazz_style_path,
                "classical",
                musescore_path,
                musicxml_path,
                synthetic_classical_svg_path,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download images and store them in the same folder"
    )
    parser.add_argument(
        "path", type=str, help="Path to the folder containing the json files"
    )
    parser.add_argument(
        "--override_existing",
        action="store_true",
        help="Override existing files if they exist already",
    )
    parser.add_argument(
        "--add-synthetic-jazz",
        action="store_true",
        help="Create the syntheti svg in jazz font",
    )
    parser.add_argument(
        "--add-synthetic-classical",
        action="store_true",
        help="Create the synthetic svg in classical font",
    )
    parser.add_argument(
        "--musescore-jazz-style-path",
        help="Path to the MuseScore style file.",
        type=str,
        default=r"C:\Program Files\MuseScore 4\styles\MuseJazz.mss",
    )
    parser.add_argument(
        "--musescore-path",
        help="Path to the MuseScore executable.",
        type=str,
        default=r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    )
    args = parser.parse_args()

    # select only the json files
    files = list(Path(args.path).glob("*.json"))
    # for f in tqdm(files):
    #     process_single_json(f, args.override_existing, args.add_synthetic_jazz, args.add_synthetic_classical, args.musescore_jazz_style_path, args.musescore_path)
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(
                    process_single_json,
                    f,
                    args.override_existing,
                    args.add_synthetic_jazz,
                    args.add_synthetic_classical,
                    args.musescore_jazz_style_path,
                    args.musescore_path,
                )
                for f in files
            ]

            # Using tqdm to monitor progress as futures complete
            for _ in tqdm(
                as_completed(futures), total=len(futures), desc="Processing files"
            ):
                pass
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught, shutting down...")
        executor.shutdown(wait=False)
        sys.exit(1)
