from pathlib import Path
from jazzmus.utils.la_inference import run_la_inference

# Convert SVG to PNG
import io
from cairosvg import svg2png
from PIL import Image

# read the available SVG files
svg_files = list(Path("./data/jazzmus_dataset_synthetic").glob("*.svg"))

# convert each SVG file to PNG
from tqdm import tqdm

for svgfile in tqdm(svg_files, desc="Converting SVG to PNG"):
    pngfile = str(svgfile.with_suffix(".png"))

    with open(svgfile, mode="r") as input_file:
        svg_stream = input_file.read()
        # See https://github.com/Kozea/CairoSVG/issues/300 for the reason why we have to replace inherit with visible here
        svg_stream = svg_stream.replace("overflow=\"inherit\"", "overflow=\"visible\"")
        svg2png(bytestring=svg_stream, background_color="transparent", write_to=pngfile)
            

png_files = list(Path("./data/jazzmus_dataset_synthetic").glob("*.png"))

run_la_inference(png_files, "./data/jazzmus_dataset_synthetic_temp")

# rename the files to match the real data ids
synthetic_files = list(Path("./data/jazzmus_dataset_synthetic_temp").glob("*.jpg"))
synthetic_files = [e for e in synthetic_files if "result" not in e.stem]
real_files = list(Path("./data/jazzmus_dataset_regions").glob("*.jpg"))

synthetic_pieces = set([e.stem.split("_syn")[0] for e in synthetic_files])
real_pieces = set([e.stem[:-7] for e in real_files])

for piece in synthetic_pieces:
    assert piece in real_pieces, f"Piece {piece} is not in the real dataset"

for synthetic_piece in synthetic_pieces:
    # get all synthetic files for this piece
    corresponding_synthetic_files = [e for e in synthetic_files if e.stem.split("_syn")[0] == synthetic_piece]
    # order them by id, i.e., the last number before the extension
    corresponding_synthetic_files.sort(key=lambda e: int(e.stem.split("_")[-1]))

    # get all real files for this piece
    corresponding_real_files = [e for e in real_files if e.stem[:-7] == synthetic_piece]
    # order them by id, i.e., the last number before the extension
    corresponding_real_files.sort(key=lambda e: int(e.stem.split("_")[-1]))

    for kind in ["synclassical", "synjazz"]:
        syn_kind = [e for e in corresponding_synthetic_files if kind in e.stem]
        if len(syn_kind) != len(corresponding_real_files):
            print(f"The number of synthetic {len(syn_kind)} and real files {len(corresponding_real_files)} do not match")
            print("skipping piece", synthetic_piece)
        else:
            # rename the synthetic files to match the real ones and move to folder "data/jazzmus_dataset_regions_renamed"
            for synthetic_file, real_file in zip(syn_kind, corresponding_real_files):
                synthetic_file.rename(real_file.parent.parent/ "jazzmus_dataset_synthetic_regions"/ Path(real_file.stem + "_"+ kind + ".jpg"))


